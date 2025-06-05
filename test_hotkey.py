#!/usr/bin/env python3
"""
Quick hotkey test to verify accessibility permissions.
Press Ctrl+Option+Space to test if macOS allows hotkey capture.
"""

from pynput import keyboard
import sys

def on_activate():
    print("⌃⌥␣ captured – hotkey works!")
    print("Accessibility permission is active!")

def on_press(key):
    # Also log any key presses to see if we're getting input at all
    try:
        print(f"Key pressed: {key.char}")
    except AttributeError:
        print(f"Special key pressed: {key}")

def on_release(key):
    if key == keyboard.Key.esc:
        print("Escape pressed - exiting...")
        return False

print("🔧 Testing hotkey capture...")
print("Press ⌃⌥␣ (Ctrl+Option+Space) to test")
print("Press any key to see input detection")
print("Press Escape to exit")
print("-" * 40)

# Test both global hotkey and general key detection
try:
    # Set up hotkey listener
    hotkey_listener = keyboard.GlobalHotKeys({
        '<ctrl>+<alt>+space': on_activate
    })
    
    # Set up general key listener for debugging
    key_listener = keyboard.Listener(
        on_press=on_press,
        on_release=on_release
    )
    
    hotkey_listener.start()
    key_listener.start()
    
    print("✅ Listeners started successfully")
    
    # Keep running until escape is pressed
    key_listener.join()
    
except Exception as e:
    print(f"❌ Error starting listeners: {e}")
    print("This usually means accessibility permission is not granted")
    sys.exit(1)
finally:
    print("🛑 Hotkey test ended") 