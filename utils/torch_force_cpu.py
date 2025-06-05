"""
Torch CPU-only loading utility.

Monkey-patches torch.load to force CPU loading of all checkpoints,
preventing CUDA device mapping issues when CUDA is compiled but not available.
"""

import os
import torch
import logging

logger = logging.getLogger(__name__)

def force_cpu_load():
    """
    Monkey-patch torch.load so that every checkpoint is
    deserialized onto CPU *unless* a map_location is
    already supplied.
    """
    orig_load = torch.load

    def _cpu_load(*args, **kwargs):
        if "map_location" not in kwargs:
            kwargs["map_location"] = torch.device("cpu")
        return orig_load(*args, **kwargs)

    torch.load = _cpu_load          # ← the one-liner patch
    os.environ["CUDA_VISIBLE_DEVICES"] = ""  # torch will think CUDA is absent
    
    logger.info("Patched torch.load to force CPU loading and disabled CUDA visibility")

def restore_original_load():
    """
    Restore the original torch.load function (if needed for debugging).
    """
    # This would need to store the original function reference
    # For simplicity, we don't implement this as it's rarely needed
    logger.warning("restore_original_load() not implemented - restart Python to restore torch.load") 