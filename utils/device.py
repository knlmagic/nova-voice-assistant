"""
Device utility for automatic device detection with MPS fallback.

Provides reliable device selection that handles MPS compatibility issues gracefully.
"""

import os
import torch
import logging

logger = logging.getLogger(__name__)

def best_device(prefer_mps=True):
    """
    Detect the best available device with proper fallback handling.
    
    Args:
        prefer_mps: Whether to prefer MPS over CPU if available
        
    Returns:
        str: Device string ("cuda", "mps", or "cpu")
    """
    # Set MPS fallback environment variable if not already set
    if "PYTORCH_ENABLE_MPS_FALLBACK" not in os.environ:
        os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
        logger.info("Set PYTORCH_ENABLE_MPS_FALLBACK=1 for better MPS compatibility")
    
    # Check CUDA first (highest priority)
    if torch.cuda.is_available():
        logger.info("CUDA detected and available")
        return "cuda"
    
    # Check MPS (Apple Silicon)
    if prefer_mps and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        try:
            # Test MPS functionality with a simple operation
            test_tensor = torch.empty(1, device="mps")
            del test_tensor
            logger.info("MPS detected and functional")
            return "mps"
        except Exception as e:
            logger.warning(f"MPS detected but not functional: {e}. Falling back to CPU")
    
    # Default to CPU
    logger.info("Using CPU device")
    return "cpu"

def safe_to_device(tensor_or_model, device, fallback_device="cpu"):
    """
    Safely move tensor or model to device with automatic fallback.
    
    Args:
        tensor_or_model: PyTorch tensor or model to move
        device: Target device
        fallback_device: Fallback device if target fails
        
    Returns:
        Tensor or model on the successful device
    """
    try:
        return tensor_or_model.to(device)
    except RuntimeError as e:
        if "Placeholder storage" in str(e) or "MPS" in str(e):
            logger.warning(f"Device {device} failed ({e}), falling back to {fallback_device}")
            return tensor_or_model.to(fallback_device)
        else:
            raise

def mps_safe_forward(model, *args, **kwargs):
    """
    Execute model forward pass with MPS fallback handling.
    
    Args:
        model: PyTorch model
        *args, **kwargs: Arguments to pass to model
        
    Returns:
        Model output with automatic device fallback if needed
    """
    try:
        return model(*args, **kwargs)
    except RuntimeError as e:
        if "Placeholder storage" in str(e):
            logger.warning("[MPS] Kernel unsupported – reverting to CPU")
            
            # Move model to CPU
            model = model.to("cpu")
            
            # Move all tensor arguments to CPU
            cpu_args = []
            for arg in args:
                if isinstance(arg, torch.Tensor):
                    cpu_args.append(arg.to("cpu"))
                else:
                    cpu_args.append(arg)
            
            cpu_kwargs = {}
            for key, value in kwargs.items():
                if isinstance(value, torch.Tensor):
                    cpu_kwargs[key] = value.to("cpu")
                else:
                    cpu_kwargs[key] = value
            
            return model(*cpu_args, **cpu_kwargs)
        else:
            raise 