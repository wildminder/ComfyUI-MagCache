import numpy as np
import torch
from unittest.mock import patch
import comfy.model_management as mm
from comfy.ldm.flux.layers import timestep_embedding

from .ratios import MAG_RATIOS


def magcache_flux_forward(self, img: torch.Tensor, img_ids: torch.Tensor, txt: torch.Tensor, txt_ids: torch.Tensor, timesteps: torch.Tensor, y: torch.Tensor, guidance: torch.Tensor = None, control = None, transformer_options={}, **kwargs):
    opts = transformer_options.get("magcache_options", {})
    enable_magcache = opts.get("enable_magcache", False)
    mag_ratios = opts.get("mag_ratios", None)
    magcache_thresh = opts.get("magcache_thresh", 0.0)
    K = opts.get("K", 0)
    cond_or_uncond_key = transformer_options.get("cond_or_uncond", [0])[0]

    # Initialize state
    if not hasattr(self, 'magcache_state'):
        self.magcache_state = {
            0: {'accumulated_ratio': 1.0, 'accumulated_err': 0.0, 'accumulated_steps': 0, 'previous_residual': None, 'cnt': 0},
            1: {'accumulated_ratio': 1.0, 'accumulated_err': 0.0, 'accumulated_steps': 0, 'previous_residual': None, 'cnt': 0}
        }
    state = self.magcache_state[cond_or_uncond_key]
    
    if img.ndim != 3 or txt.ndim != 3:
        raise ValueError("Input img and txt tensors must have 3 dimensions.")

    img = self.img_in(img)
    
    vec = self.time_in(timestep_embedding(timesteps, 256).to(img.dtype))
    
    if self.params.guidance_embed:
        if guidance is None: raise ValueError("Guidance strength not provided for guidance distilled model.")
        vec = vec + self.guidance_in(timestep_embedding(guidance, 256).to(img.dtype))
    
    vec = vec + self.vector_in(y[:, :self.params.vec_in_dim])
    txt = self.txt_in(txt)
    pe = self.pe_embedder(torch.cat((txt_ids, img_ids), dim=1)) if img_ids is not None else None
    
    ori_img = img.clone() 

    skip_forward = False
    if enable_magcache and state['previous_residual'] is not None:
        # The 'cnt' in original MagCache corresponds to the step *after* retention. I guess....
        if state['cnt'] < len(mag_ratios):
            current_mag_ratio = mag_ratios[state['cnt']]
            state['accumulated_ratio'] *= current_mag_ratio
            state['accumulated_err'] += abs(1 - state['accumulated_ratio'])
            state['accumulated_steps'] += 1

            if state['accumulated_err'] <= magcache_thresh and state['accumulated_steps'] <= K:
                skip_forward = True
            else:
                state['accumulated_ratio'] = 1.0
                state['accumulated_err'] = 0.0
                state['accumulated_steps'] = 0
        else: 
            # If we run out of pre-computed ratios, disable caching for safety.
            skip_forward = False

    if skip_forward:
        img = ori_img + state['previous_residual'].to(img.device)
    else:
        for i, block in enumerate(self.double_blocks):
            img, txt = block(img=img, txt=txt, vec=vec, pe=pe, attn_mask=kwargs.get("attn_mask", None))
        
        img = torch.cat((txt, img), 1)
        
        for i, block in enumerate(self.single_blocks):
            img = block(img, vec=vec, pe=pe, attn_mask=kwargs.get("attn_mask", None))
        
        img = img[:, txt.shape[1]:, ...]
        current_residual = img - ori_img
        state['previous_residual'] = current_residual.to(mm.unet_offload_device())
        
    img = self.final_layer(img, vec)
    state['cnt'] += 1
    
    return img

def magcache_wan_forward(self, x, t, context, clip_fea=None, freqs=None, transformer_options={}, **kwargs):
    opts = transformer_options.get("magcache_options", {})
    enable_magcache = opts.get("enable_magcache", False)
    mag_ratios = opts.get("mag_ratios", None)
    magcache_thresh = opts.get("magcache_thresh", 0.0)
    K = opts.get("K", 0)
    cond_or_uncond_key = transformer_options.get("cond_or_uncond", [0])[0]

    if not hasattr(self, 'magcache_state'):
        self.magcache_state = {
            0: {'accumulated_ratio': 1.0, 'accumulated_err': 0.0, 'accumulated_steps': 0, 'previous_residual': None, 'cnt': 0},
            1: {'accumulated_ratio': 1.0, 'accumulated_err': 0.0, 'accumulated_steps': 0, 'previous_residual': None, 'cnt': 0}
        }
    state = self.magcache_state[cond_or_uncond_key]

    from comfy.ldm.wan.model import sinusoidal_embedding_1d
    e = self.time_embedding(sinusoidal_embedding_1d(self.freq_dim, t).to(dtype=x.dtype))
    e0 = self.time_projection(e).unflatten(1, (6, self.dim))
    
    context = self.text_embedding(context)
    context_img_len = None
    if clip_fea is not None and self.img_emb is not None:
        context_clip = self.img_emb(clip_fea)
        context = torch.concat([context_clip, context], dim=1)
        context_img_len = clip_fea.shape[-2]

    ori_x = x.clone()

    skip_forward = False
    if enable_magcache and state['previous_residual'] is not None:
        if state['cnt'] < len(mag_ratios):
            current_mag_ratio = mag_ratios[state['cnt']]
            state['accumulated_ratio'] *= current_mag_ratio
            state['accumulated_err'] += abs(1 - state['accumulated_ratio'])
            state['accumulated_steps'] += 1
            
            if state['accumulated_err'] <= magcache_thresh and state['accumulated_steps'] <= K:
                skip_forward = True
            else:
                state['accumulated_ratio'] = 1.0
                state['accumulated_err'] = 0.0
                state['accumulated_steps'] = 0
        else:
            skip_forward = False

    if skip_forward:
        x = ori_x + state['previous_residual'].to(x.device)
    else:
        for block in self.blocks:
            x = block(x, e=e0, freqs=freqs, context=context, context_img_len=context_img_len)
        current_residual = x - ori_x
        state['previous_residual'] = current_residual.to(mm.unet_offload_device())
        
    x = self.head(x, e)
    state['cnt'] += 1
    return x

# Placeholder for HunyuanVideo. Still WIP
magcache_hunyuan_forward = magcache_flux_forward

class MagCache:
    @classmethod
    def INPUT_TYPES(s):
        model_types = list(MAG_RATIOS.keys())
        return {
            "required": {
                "model": ("MODEL",),
                "model_type": (model_types, {"default": model_types[0]}),
                "magcache_thresh": ("FLOAT", {"default": 0.24, "min": 0.0, "max": 2.0, "step": 0.01, "tooltip": "Accumulated error threshold for caching. Higher allows more skipping."}),
                "magcache_K": ("INT", {"default": 5, "min": 0, "max": 100, "tooltip": "Maximum number of consecutive skipped steps."}),
                "retention_ratio": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Percentage of initial steps to run without caching."}),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "apply_magcache"
    CATEGORY = "MagCache"
    TITLE = "MagCache Accelerator"

    def apply_magcache(self, model, model_type, magcache_thresh, magcache_K, retention_ratio):
        if magcache_thresh == 0 and magcache_K == 0:
            return (model,)

        new_model = model.clone()
        if 'transformer_options' not in new_model.model_options:
            new_model.model_options['transformer_options'] = {}
            
        magcache_options = {
            "magcache_thresh": magcache_thresh,
            "K": magcache_K,
            "retention_ratio": retention_ratio,
            "mag_ratios": MAG_RATIOS[model_type],
        }
        new_model.model_options['transformer_options']["magcache_options"] = magcache_options
        
        diffusion_model = new_model.get_model_object("diffusion_model")
        
        if "flux" in model_type or "hunyuan_video" in model_type:
            patch_func = magcache_flux_forward
            patch_target_name = 'forward_orig'
        elif "wan2.1" in model_type:
            patch_func = magcache_wan_forward
            patch_target_name = 'forward' 
        else:
            raise ValueError(f"Unsupported model_type for MagCache: {model_type}")

        def unet_wrapper_function(model_function, kwargs):
            c = kwargs["c"]
            sigmas = c.get("transformer_options", {}).get("sample_sigmas")
            if sigmas is None:
                print("MagCache Warning: Could not determine total steps from sigmas. Caching will be disabled.")
                c.setdefault('transformer_options', {}).setdefault('magcache_options', {})['enable_magcache'] = False
                return model_function(kwargs["input"], kwargs["timestep"], **c)

            total_steps = len(sigmas)
            timestep = kwargs["timestep"]
            
            current_step_index = 0
            if total_steps > 0:
                time_diffs = torch.abs(sigmas - timestep[0])
                current_step_index = torch.argmin(time_diffs).item()
            
            if current_step_index == 0:
                if hasattr(diffusion_model, 'magcache_state'):
                    delattr(diffusion_model, 'magcache_state')

            retention_ratio = c['transformer_options']['magcache_options']['retention_ratio']
            retention_steps = int(retention_ratio * total_steps)
            c['transformer_options']['magcache_options']['enable_magcache'] = current_step_index >= retention_steps
            
            original_forward = getattr(diffusion_model, patch_target_name)
            setattr(diffusion_model, patch_target_name, patch_func.__get__(diffusion_model, diffusion_model.__class__))
            
            try:
                if "wan2.1" in model_type:

                    output = model_function(kwargs["input"], kwargs["timestep"], **c)

                    grid_sizes = kwargs["c"]["freqs"][1]
                    output = diffusion_model.unpatchify(output, grid_sizes)
                else:
                    output = model_function(kwargs["input"], kwargs["timestep"], **c)
            finally:
                setattr(diffusion_model, patch_target_name, original_forward)
                
            return output

        new_model.set_model_unet_function_wrapper(unet_wrapper_function)
        
        return (new_model,)