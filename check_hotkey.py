#!/usr/bin/env python3
"""
Quick diagnostic script to test Control-Option-Space detection.
This isolates whether the issue is OS-level permissions/conflicts.
"""

import time
from pynput import keyboard

COMBO = {keyboard.Key.ctrl_l, keyboard.Key.alt_l, keyboard.Key.space}
current = set()

def on_press(key):
    current.add(key)
    if COMBO <= current:
        print("🔥  HOT-KEY DETECTED")
        current.clear()

def on_release(key):
    if key in current:
        current.remove(key)

print("🔧 Testing Control-Option-Space detection...")
print("Press Control-Option-Space to test")
print("(Ctrl-C to quit)")
print("-" * 40)

try:
    with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
        while True:
            time.sleep(1)
except KeyboardInterrupt:
    print("\n✅ Test ended")
except Exception as e:
    print(f"❌ Error: {e}")
    print("This usually means accessibility permission issues") 