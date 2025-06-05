#!/usr/bin/env python3
"""
Fixed hotkey test for Control+Option+Space.
Based on successful Ctrl+Shift+T test, trying different syntax variations.
"""

from pynput import keyboard
import sys

def on_ctrl_alt_space_v1():
    print("🎯 ⌃⌥␣ captured with syntax v1 (ctrl+alt+space)!")

def on_ctrl_alt_space_v2():
    print("🎯 ⌃⌥␣ captured with syntax v2 (ctrl+cmd+space)!")

def on_press(key):
    # Only show modifier keys to reduce spam
    if key in [keyboard.Key.ctrl, keyboard.Key.alt, keyboard.Key.cmd, keyboard.Key.space]:
        print(f"Key: {key}")

def on_release(key):
    if key == keyboard.Key.esc:
        print("Escape pressed - exiting...")
        return False

print("🔧 Testing ⌃⌥␣ hotkey with different syntaxes...")
print("Try pressing Control+Option+Space")
print("Press Escape to exit")
print("-" * 50)

try:
    # macOS: Option = Alt, but let's try both syntaxes
    hotkey_listener = keyboard.GlobalHotKeys({
        '<ctrl>+<alt>+<space>': on_ctrl_alt_space_v1,     # Standard syntax
        '<ctrl>+<cmd>+<space>': on_ctrl_alt_space_v2,     # In case Option maps to Cmd
    })
    
    key_listener = keyboard.Listener(
        on_press=on_press,
        on_release=on_release
    )
    
    hotkey_listener.start()
    key_listener.start()
    
    print("✅ Listeners started successfully!")
    print("Now try Control+Option+Space...")
    key_listener.join()
    
except Exception as e:
    print(f"❌ Error: {e}")
    sys.exit(1)
finally:
    print("🛑 Test ended") 