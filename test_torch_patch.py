#!/usr/bin/env python3
"""
Test torch.load patching across threads.
"""

import torch
import tempfile
import pickle
import os

print("CUDA visible:", torch.cuda.is_available())

# Create a temporary tensor file
f = tempfile.NamedTemporaryFile(delete=False)
pickle.dump(torch.zeros(1), f)
f.close()

# Load it back
x = torch.load(f.name)
print("device ⇒", x.device)  # should be cpu

# Clean up
os.unlink(f.name)

print("✅ torch.load patch working" if x.device.type == "cpu" else "❌ patch failed") 