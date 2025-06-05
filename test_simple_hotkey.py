#!/usr/bin/env python3
"""
Simple hotkey test with different combinations to isolate issues.
"""

from pynput import keyboard
import sys

def test_f1():
    print("🎯 F1 captured – basic hotkey works!")

def test_ctrl_shift_t():
    print("🎯 Ctrl+Shift+T captured – modifier combo works!")

def on_press(key):
    print(f"Key detected: {key}")

def on_release(key):
    if key == keyboard.Key.esc:
        print("Escape pressed - exiting...")
        return False

print("🔧 Testing simpler hotkey combinations...")
print("Press F1 to test basic hotkey")
print("Press Ctrl+Shift+T to test modifier combo")
print("Press any key to see detection")
print("Press Escape to exit")
print("-" * 50)

try:
    # Test simpler combinations first
    hotkey_listener = keyboard.GlobalHotKeys({
        '<f1>': test_f1,
        '<ctrl>+<shift>+t': test_ctrl_shift_t
    })
    
    key_listener = keyboard.Listener(
        on_press=on_press,
        on_release=on_release
    )
    
    hotkey_listener.start()
    key_listener.start()
    
    print("✅ Listeners started - testing accessibility...")
    key_listener.join()
    
except Exception as e:
    print(f"❌ Error: {e}")
    print("Accessibility permission likely not granted")
    sys.exit(1)
finally:
    print("🛑 Test ended") 