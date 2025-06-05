"""
Integrated audio pipeline for voice assistant.

Combines audio capture, playback, hotkey handling, and VAD into a unified async system.
Based on PRD requirements and reference repository patterns.
"""

import asyncio
import logging
from pathlib import Path
from typing import Optional, Callable, Union
from dataclasses import dataclass
import numpy as np

from .capture import AudioConfig, AsyncAudioCapture
from .playback import PlaybackConfig, AudioSystem
from .hotkey import HotkeyConfig, PushToTalkHandler


@dataclass
class AudioPipelineConfig:
    """Configuration for the complete audio pipeline."""
    # Audio settings
    audio_config: AudioConfig = None
    playback_config: PlaybackConfig = None
    hotkey_config: HotkeyConfig = None
    
    # Paths
    sounds_dir: Union[str, Path] = "sounds"
    
    # Pipeline behavior
    enable_push_to_talk: bool = True
    enable_vad: bool = True
    enable_chimes: bool = True
    
    def __post_init__(self):
        """Initialize default configs if not provided."""
        if self.audio_config is None:
            self.audio_config = AudioConfig()
        if self.playback_config is None:
            self.playback_config = PlaybackConfig()
        if self.hotkey_config is None:
            self.hotkey_config = HotkeyConfig()


class AudioPipeline:
    """
    Complete audio pipeline for voice assistant.
    
    Integrates:
    - Push-to-talk via ⌃⌥␣ global hotkey
    - Audio capture with Voice Activity Detection
    - Audio chimes for user feedback
    - Non-blocking async audio architecture
    
    Based on PRD specifications and proven patterns from reference repositories.
    """
    
    def __init__(self, config: Optional[AudioPipelineConfig] = None):
        self.config = config or AudioPipelineConfig()
        
        # Core components
        self.audio_capture = AsyncAudioCapture(self.config.audio_config)
        self.audio_system = AudioSystem(
            self.config.sounds_dir, 
            self.config.playback_config
        )
        
        # Push-to-talk handler
        self.push_to_talk: Optional[PushToTalkHandler] = None
        if self.config.enable_push_to_talk:
            self.push_to_talk = PushToTalkHandler(
                self.config.hotkey_config,
                on_start_recording=self._on_hotkey_press,
                on_stop_recording=self._on_hotkey_release
            )
        
        # State tracking
        self.is_running = False
        self.is_recording = False
        self._current_audio: Optional[np.ndarray] = None
        self._speaking = asyncio.Event()  # Track TTS playback state
        self._speaking.clear()  # Not speaking at start
        
        # Callbacks
        self.on_audio_captured: Optional[Callable[[np.ndarray], None]] = None
        self.on_recording_start: Optional[Callable] = None
        self.on_recording_stop: Optional[Callable] = None
        
        # Setup internal callbacks
        self.audio_capture.set_callbacks(
            on_speech_start=self._on_speech_start,
            on_speech_end=self._on_speech_end,
            on_vad_stop=self._on_vad_stop
        )
        
        # Logging
        self.logger = logging.getLogger(__name__)
    
    async def start(self) -> bool:
        """
        Start the audio pipeline.
        
        Returns:
            True if started successfully, False otherwise
        """
        if self.is_running:
            return False
        
        try:
            # Start push-to-talk if enabled
            if self.push_to_talk:
                if not await self.push_to_talk.start():
                    self.logger.error("Failed to start push-to-talk handler")
                    return False
            
            self.is_running = True
            self.logger.info("Audio pipeline started successfully")
            
            if self.config.enable_push_to_talk:
                print("🎧 Voice assistant ready - Press ⌃⌥␣ (Control-Option-Space) to talk")
            else:
                print("🎧 Voice assistant ready - Manual recording mode")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error starting audio pipeline: {e}")
            return False
    
    async def stop(self):
        """Stop the audio pipeline."""
        if not self.is_running:
            return
        
        try:
            # Stop push-to-talk
            if self.push_to_talk:
                await self.push_to_talk.stop()
            
            # Stop any active recording
            if self.is_recording:
                await self._stop_recording()
            
            # Stop all audio playback
            await self.audio_system.stop_all_audio()
            
            self.is_running = False
            self.logger.info("Audio pipeline stopped")
            print("🎧 Voice assistant stopped")
            
        except Exception as e:
            self.logger.error(f"Error stopping audio pipeline: {e}")
    
    async def start_manual_recording(self) -> bool:
        """
        Start recording manually (without push-to-talk).
        
        Returns:
            True if recording started, False otherwise
        """
        if not self.is_running or self.is_recording:
            return False
        
        return await self._start_recording()
    
    async def stop_manual_recording(self) -> Optional[np.ndarray]:
        """
        Stop manual recording and return captured audio.
        
        Returns:
            Captured audio data or None if no recording active
        """
        if not self.is_recording:
            return None
        
        return await self._stop_recording()
    
    async def play_tts(self, audio: np.ndarray, sample_rate: Optional[int] = None) -> bool:
        """
        Play TTS audio non-blocking (speaking state managed by caller).
        
        Args:
            audio: Audio data to play
            sample_rate: Sample rate (uses config default if None)
            
        Returns:
            True if playback started successfully
        """
        # Speaking state is now managed by the TTS worker
        result = await self.audio_system.play_tts_audio(audio, sample_rate)
        
        # Wait for audio to finish playing
        if result:
            # Estimate playback duration and wait
            duration = len(audio) / (sample_rate or 24000)  # ChatterboxTTS default
            await asyncio.sleep(duration + 0.5)  # Extra 0.5s buffer
            
        return result
    
    # Internal event handlers
    
    async def _on_hotkey_press(self):
        """Handle hotkey press event."""
        if not self.is_recording:
            await self._start_recording()
    
    async def _on_hotkey_release(self):
        """Handle hotkey release event."""
        if self.is_recording:
            await self._stop_recording()
    
    async def _start_recording(self) -> bool:
        """Start audio recording."""
        try:
            # Play start chime if enabled
            if self.config.enable_chimes:
                self.audio_system.play_listen_chime()
            
            # Start recording
            if await self.audio_capture.start_recording():
                self.is_recording = True
                self.logger.info("Recording started")
                
                # Trigger callback
                if self.on_recording_start:
                    if asyncio.iscoroutinefunction(self.on_recording_start):
                        await self.on_recording_start()
                    else:
                        self.on_recording_start()
                
                return True
            else:
                self.logger.error("Failed to start audio recording")
                return False
                
        except Exception as e:
            self.logger.error(f"Error starting recording: {e}")
            return False
    
    async def _stop_recording(self) -> Optional[np.ndarray]:
        """Stop audio recording and process captured audio."""
        try:
            # Stop recording and get audio
            audio_data = await self.audio_capture.stop_recording()
            self.is_recording = False
            
            # Play done chime if enabled
            if self.config.enable_chimes:
                self.audio_system.play_done_chime()
            
            self.logger.info("Recording stopped")
            
            # Store current audio
            self._current_audio = audio_data
            
            # Trigger callbacks
            if self.on_recording_stop:
                if asyncio.iscoroutinefunction(self.on_recording_stop):
                    await self.on_recording_stop()
                else:
                    self.on_recording_stop()
            
            if audio_data is not None and self.on_audio_captured:
                if asyncio.iscoroutinefunction(self.on_audio_captured):
                    await self.on_audio_captured(audio_data)
                else:
                    self.on_audio_captured(audio_data)
            
            return audio_data
            
        except Exception as e:
            self.logger.error(f"Error stopping recording: {e}")
            return None
    
    def _on_speech_start(self):
        """Handle speech start event from VAD."""
        self.logger.debug("Speech detected")
    
    def _on_speech_end(self):
        """Handle speech end event from VAD."""
        self.logger.debug("Speech ended")
    
    def _on_vad_stop(self):
        """Handle VAD auto-stop event."""
        if self.config.enable_vad and self.is_recording:
            self.logger.info("VAD triggered recording stop")
            # Let the existing recording stop naturally
            # This callback is just for logging/notification
    
    # Status and utility methods
    
    def get_status(self) -> dict:
        """Get current pipeline status."""
        return {
            "is_running": self.is_running,
            "is_recording": self.is_recording,
            "is_speaking": self._speaking.is_set(),
            "push_to_talk_active": self.push_to_talk.is_recording() if self.push_to_talk else False,
            "audio_capture_active": self.audio_capture.is_active(),
            "tts_playing": self.audio_system.is_playing_tts(),
            "config": {
                "push_to_talk_enabled": self.config.enable_push_to_talk,
                "vad_enabled": self.config.enable_vad,
                "chimes_enabled": self.config.enable_chimes,
                "vad_threshold": self.config.audio_config.vad_threshold_seconds,
                "sample_rate": self.config.audio_config.sample_rate
            }
        }
    
    def get_last_audio(self) -> Optional[np.ndarray]:
        """Get the last captured audio data."""
        return self._current_audio
    
    def is_speaking(self) -> bool:
        """Check if TTS is currently playing."""
        return self._speaking.is_set()
    
    def set_callbacks(self, 
                     on_audio_captured: Optional[Callable] = None,
                     on_recording_start: Optional[Callable] = None,
                     on_recording_stop: Optional[Callable] = None):
        """
        Set callback functions for pipeline events.
        
        Args:
            on_audio_captured: Called when audio is captured with audio data
            on_recording_start: Called when recording starts
            on_recording_stop: Called when recording stops
        """
        self.on_audio_captured = on_audio_captured
        self.on_recording_start = on_recording_start
        self.on_recording_stop = on_recording_stop
    
    async def __aenter__(self):
        """Async context manager entry."""
        await self.start()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.stop() 