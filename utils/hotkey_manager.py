#!/usr/bin/env python3
"""
Bulletproof hotkey manager that avoids threading issues.
Based on proven pattern that eliminates the self-join race condition.
"""

from pynput import keyboard
import threading
import asyncio
import logging

logger = logging.getLogger(__name__)


class HotKeyManager:
    """
    Bulletproof hotkey manager that avoids threading race conditions.
    
    Key improvements over original implementation:
    - No self-join issues
    - Single threaded listener that blocks properly
    - Clean queue integration
    - Proper shutdown handling
    - Speaking state awareness to prevent audio collisions
    """
    
    def __init__(self, queue: asyncio.Queue, audio_pipeline=None):
        self.queue = queue
        self.audio_pipeline = audio_pipeline  # For speaking state check
        self.listener = None          # pynput listener  
        self.thread = None            # wrapper thread
        self._running = False
        
    def _on_hotkey(self):
        """
        Hotkey callback - executes in listener thread.
        Pushes event to async queue for main loop processing.
        Implements mic lock while TTS is speaking to prevent audio collisions.
        """
        logger.debug("🔥 hot-key callback fired")
        
        # Check if TTS is currently speaking (prevent audio collision)
        if self.audio_pipeline and self.audio_pipeline.is_speaking():
            logger.debug("Hot-key ignored: still speaking")
            return
            
        try:
            # Non-blocking queue put
            self.queue.put_nowait("push_to_talk")
        except asyncio.QueueFull:
            logger.warning("Audio queue full – dropping hot-key")
        except Exception as e:
            logger.error(f"Error in hotkey callback: {e}")
            
    def _run_listener(self):
        """
        Listener thread main function.
        Executes in wrapper thread, blocks until stop() is called.
        """
        try:
            logger.debug("Starting pynput GlobalHotKeys listener")
            with keyboard.GlobalHotKeys({
                '<ctrl>+<alt>+<space>': self._on_hotkey
            }) as hk:
                self._running = True
                hk.join()  # blocks until stop() is called
        except Exception as e:
            logger.error(f"Hotkey listener error: {e}")
        finally:
            self._running = False
            logger.debug("Hotkey listener thread ended")
            
    def start(self):
        """Start the hotkey listener in a background thread."""
        if self.thread and self.thread.is_alive():
            logger.debug("Hotkey listener already running")
            return
            
        self.thread = threading.Thread(
            target=self._run_listener,
            name="HotKeyListener", 
            daemon=True
        )
        self.thread.start()
        logger.info("Global hot-key listener started (⌃⌥␣)")
        
    def stop(self):
        """Stop the hotkey listener gracefully."""
        if self.thread and self.thread.is_alive():
            logger.debug("Stopping hotkey listener")
            try:
                # Gentle poke to exit join() - send harmless key
                keyboard.Controller().press('\0')
                self.thread.join(timeout=1.0)
            except Exception as e:
                logger.debug(f"Hotkey shutdown minor error: {e}")
        
        self._running = False
        logger.info("Hotkey listener stopped")
        
    def is_running(self) -> bool:
        """Check if the hotkey listener is currently running."""
        return self._running and self.thread and self.thread.is_alive() 