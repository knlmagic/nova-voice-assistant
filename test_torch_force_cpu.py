#!/usr/bin/env python3
"""
Smoke test for torch_force_cpu utility.
"""

from utils.torch_force_cpu import force_cpu_load
force_cpu_load()

from chatterbox import ChatterboxTTS, models
print('✅ torch.load is now CPU-only')

# quick instantiation to confirm no CUDA warning
t3 = models.t3.T3.from_pretrained('resemble-ai/chatterbox-t3-en')
print('✅ T3 weights loaded on', next(t3.parameters()).device) 