"""
Global hotkey handler for voice assistant activation.

Implements ⌃⌥␣ (Control-Option-Space) global hotkey for macOS.
Based on PRD specification: "macOS NSEvent global hotkey via pynput"
"""

import asyncio
import threading
from typing import Optional, Callable, Any
from dataclasses import dataclass
from pynput import keyboard
from pynput.keyboard import Key, KeyCode


@dataclass
class HotkeyConfig:
    """Configuration for global hotkey behavior."""
    # PRD specified hotkey: Control-Option-Space
    modifiers: tuple = (Key.ctrl, Key.alt)
    trigger_key: Key = Key.space
    
    # Behavior settings
    hold_duration_ms: int = 50  # Minimum hold time to prevent accidental triggers
    repeat_delay_ms: int = 200  # Delay between repeated triggers while held


class GlobalHotkeyHandler:
    """
    Global hotkey handler for voice assistant activation.
    
    Implements PRD requirement: "⌃⌥␣ Control-Option-Space global hotkey"
    Using pynput for cross-platform compatibility with macOS focus.
    """
    
    def __init__(self, config: Optional[HotkeyConfig] = None):
        self.config = config or HotkeyConfig()
        
        # Callback functions
        self.on_hotkey_press: Optional[Callable] = None
        self.on_hotkey_release: Optional[Callable] = None
        
        # State tracking
        self._pressed_keys = set()
        self._hotkey_active = False
        self._listener: Optional[keyboard.Listener] = None
        self._running = False
        
        # Threading
        self._listener_thread: Optional[threading.Thread] = None
        self._hotkey_lock = threading.Lock()
        
        # Timing for hold detection
        self._press_start_time: Optional[float] = None
        
    def start(self) -> bool:
        """
        Start the global hotkey listener.
        
        Returns:
            True if started successfully, False otherwise
        """
        if self._running:
            return False
        
        try:
            # Create keyboard listener
            self._listener = keyboard.Listener(
                on_press=self._on_key_press,
                on_release=self._on_key_release
            )
            
            # Start listener in separate thread
            self._listener_thread = threading.Thread(
                target=self._run_listener,
                daemon=True
            )
            self._listener_thread.start()
            
            self._running = True
            print("Global hotkey listener started (⌃⌥␣)")
            return True
            
        except Exception as e:
            print(f"Error starting hotkey listener: {e}")
            return False
    
    def stop(self):
        """Stop the global hotkey listener."""
        if not self._running:
            return
        
        self._running = False
        
        if self._listener:
            self._listener.stop()
            self._listener = None
        
        if self._listener_thread:
            self._listener_thread.join(timeout=1.0)
        
        print("Global hotkey listener stopped")
    
    def _run_listener(self):
        """Run the keyboard listener (in separate thread)."""
        try:
            if self._listener:
                self._listener.join()
        except Exception as e:
            print(f"Error in hotkey listener thread: {e}")
    
    def _on_key_press(self, key):
        """Handle key press events."""
        if not self._running:
            return
        
        with self._hotkey_lock:
            self._pressed_keys.add(key)
            
            # Check if hotkey combination is pressed
            if self._is_hotkey_combination_pressed():
                if not self._hotkey_active:
                    self._hotkey_active = True
                    self._press_start_time = asyncio.get_event_loop().time()
                    
                    # Trigger hotkey press callback
                    if self.on_hotkey_press:
                        threading.Thread(
                            target=self.on_hotkey_press,
                            daemon=True
                        ).start()
    
    def _on_key_release(self, key):
        """Handle key release events."""
        if not self._running:
            return
        
        with self._hotkey_lock:
            self._pressed_keys.discard(key)
            
            # Check if hotkey combination is released
            if self._hotkey_active and not self._is_hotkey_combination_pressed():
                self._hotkey_active = False
                
                # Trigger hotkey release callback
                if self.on_hotkey_release:
                    threading.Thread(
                        target=self.on_hotkey_release,
                        daemon=True
                    ).start()
    
    def _is_hotkey_combination_pressed(self) -> bool:
        """Check if the hotkey combination is currently pressed."""
        # Check if all required modifiers are pressed
        modifiers_pressed = all(mod in self._pressed_keys for mod in self.config.modifiers)
        
        # Check if trigger key is pressed
        trigger_pressed = self.config.trigger_key in self._pressed_keys
        
        return modifiers_pressed and trigger_pressed
    
    def is_active(self) -> bool:
        """Check if hotkey is currently being pressed."""
        return self._hotkey_active
    
    def is_running(self) -> bool:
        """Check if hotkey listener is running."""
        return self._running
    
    def set_callbacks(self, on_press: Optional[Callable] = None, on_release: Optional[Callable] = None):
        """
        Set callback functions for hotkey events.
        
        Args:
            on_press: Function to call when hotkey is pressed
            on_release: Function to call when hotkey is released
        """
        self.on_hotkey_press = on_press
        self.on_hotkey_release = on_release


class AsyncHotkeyHandler:
    """Async wrapper for GlobalHotkeyHandler with asyncio integration."""
    
    def __init__(self, config: Optional[HotkeyConfig] = None):
        self.handler = GlobalHotkeyHandler(config)
        self._press_event = asyncio.Event()
        self._release_event = asyncio.Event()
        
        # Set up callbacks to work with asyncio
        self.handler.set_callbacks(
            on_press=self._on_press_callback,
            on_release=self._on_release_callback
        )
    
    def _on_press_callback(self):
        """Internal callback to set asyncio event."""
        try:
            loop = asyncio.get_event_loop()
            loop.call_soon_threadsafe(self._press_event.set)
        except RuntimeError:
            # No event loop running
            pass
    
    def _on_release_callback(self):
        """Internal callback to set asyncio event."""
        try:
            loop = asyncio.get_event_loop()
            loop.call_soon_threadsafe(self._release_event.set)
        except RuntimeError:
            # No event loop running
            pass
    
    async def start(self) -> bool:
        """Start the hotkey handler asynchronously."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.handler.start)
    
    async def stop(self):
        """Stop the hotkey handler asynchronously."""
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self.handler.stop)
    
    async def wait_for_press(self, timeout: Optional[float] = None):
        """
        Wait for hotkey press event.
        
        Args:
            timeout: Maximum time to wait in seconds (None for no timeout)
            
        Raises:
            asyncio.TimeoutError: If timeout is reached
        """
        self._press_event.clear()
        
        if timeout:
            await asyncio.wait_for(self._press_event.wait(), timeout=timeout)
        else:
            await self._press_event.wait()
    
    async def wait_for_release(self, timeout: Optional[float] = None):
        """
        Wait for hotkey release event.
        
        Args:
            timeout: Maximum time to wait in seconds (None for no timeout)
            
        Raises:
            asyncio.TimeoutError: If timeout is reached
        """
        self._release_event.clear()
        
        if timeout:
            await asyncio.wait_for(self._release_event.wait(), timeout=timeout)
        else:
            await self._release_event.wait()
    
    def is_active(self) -> bool:
        """Check if hotkey is currently pressed."""
        return self.handler.is_active()
    
    def is_running(self) -> bool:
        """Check if handler is running."""
        return self.handler.is_running()


# Push-to-talk implementation
class PushToTalkHandler:
    """
    Push-to-talk implementation combining hotkey and audio capture.
    
    Implements PRD requirement: "Push-to-talk via ⌃⌥␣ global hotkey"
    """
    
    def __init__(self, 
                 hotkey_config: Optional[HotkeyConfig] = None,
                 on_start_recording: Optional[Callable] = None,
                 on_stop_recording: Optional[Callable] = None):
        self.hotkey = AsyncHotkeyHandler(hotkey_config)
        self.on_start_recording = on_start_recording
        self.on_stop_recording = on_stop_recording
        
        self._is_recording = False
        self._monitoring_task: Optional[asyncio.Task] = None
    
    async def start(self) -> bool:
        """Start push-to-talk monitoring."""
        if not await self.hotkey.start():
            return False
        
        # Start monitoring task
        self._monitoring_task = asyncio.create_task(self._monitor_hotkey())
        print("Push-to-talk monitoring started")
        return True
    
    async def stop(self):
        """Stop push-to-talk monitoring."""
        if self._monitoring_task:
            self._monitoring_task.cancel()
            try:
                await self._monitoring_task
            except asyncio.CancelledError:
                pass
        
        await self.hotkey.stop()
        print("Push-to-talk monitoring stopped")
    
    async def _monitor_hotkey(self):
        """Monitor hotkey events and trigger recording."""
        try:
            while True:
                # Wait for hotkey press
                await self.hotkey.wait_for_press()
                
                if not self._is_recording:
                    self._is_recording = True
                    print("🎤 Recording started (hotkey pressed)")
                    
                    if self.on_start_recording:
                        if asyncio.iscoroutinefunction(self.on_start_recording):
                            await self.on_start_recording()
                        else:
                            self.on_start_recording()
                
                # Wait for hotkey release
                await self.hotkey.wait_for_release()
                
                if self._is_recording:
                    self._is_recording = False
                    print("🎤 Recording stopped (hotkey released)")
                    
                    if self.on_stop_recording:
                        if asyncio.iscoroutinefunction(self.on_stop_recording):
                            await self.on_stop_recording()
                        else:
                            self.on_stop_recording()
                            
        except asyncio.CancelledError:
            print("Push-to-talk monitoring cancelled")
        except Exception as e:
            print(f"Error in push-to-talk monitoring: {e}")
    
    def is_recording(self) -> bool:
        """Check if currently recording via push-to-talk."""
        return self._is_recording
    
    def is_running(self) -> bool:
        """Check if push-to-talk is running."""
        return self.hotkey.is_running() and self._monitoring_task is not None 