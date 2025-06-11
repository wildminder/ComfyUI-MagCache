from .nodes import MagCache

NODE_CLASS_MAPPINGS = {
    "MagCache": MagCache,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MagCache": "MagCache Accelerator",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]