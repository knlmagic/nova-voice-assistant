#!/usr/bin/env python3
from whispercpp import Whisper

try:
    print("Testing with built-in 'small.en' model...")
    w = Whisper.from_pretrained("small.en")
    w.set_threads(4)
    print("✅ loaded", w.context)
    print("✅ No ImportError - wheel problem solved!")
    print("✅ Model loaded successfully!")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc() 