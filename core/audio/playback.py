"""
Audio playback module with chimes system.

Implements non-blocking audio playback via sounddevice for TTS and audio chimes.
Based on PRD requirements and vndee/local-talking-llm patterns.
"""

import asyncio
import threading
from pathlib import Path
from typing import Optional, Union
import numpy as np
import sounddevice as sd
from scipy.io import wavfile
from dataclasses import dataclass


@dataclass
class PlaybackConfig:
    """Audio playback configuration."""
    sample_rate: int = 16000
    channels: int = 1
    dtype: str = 'int16'
    device: Optional[int] = None  # Use default output device


class AudioChimes:
    """Audio chimes system for user feedback."""
    
    def __init__(self, sounds_dir: Union[str, Path]):
        self.sounds_dir = Path(sounds_dir)
        self.config = PlaybackConfig()
        
        # Chime files as specified in PRD
        self.listen_chime_path = self.sounds_dir / "listen.wav"
        self.done_chime_path = self.sounds_dir / "done.wav"
        
        # Pre-load chimes for minimal latency
        self.listen_chime: Optional[np.ndarray] = None
        self.done_chime: Optional[np.ndarray] = None
        
        self._load_chimes()
    
    def _load_chimes(self):
        """Pre-load chime audio files for fast playback."""
        try:
            if self.listen_chime_path.exists():
                sample_rate, audio = wavfile.read(self.listen_chime_path)
                self.listen_chime = self._prepare_audio(audio, sample_rate)
            else:
                print(f"Warning: Listen chime not found at {self.listen_chime_path}")
                self.listen_chime = self._generate_default_chime(frequency=800, duration=0.15)
            
            if self.done_chime_path.exists():
                sample_rate, audio = wavfile.read(self.done_chime_path)
                self.done_chime = self._prepare_audio(audio, sample_rate)
            else:
                print(f"Warning: Done chime not found at {self.done_chime_path}")
                self.done_chime = self._generate_default_chime(frequency=600, duration=0.15)
                
        except Exception as e:
            print(f"Error loading chimes: {e}")
            # Generate default chimes as fallback
            self.listen_chime = self._generate_default_chime(frequency=800, duration=0.15)
            self.done_chime = self._generate_default_chime(frequency=600, duration=0.15)
    
    def _prepare_audio(self, audio: np.ndarray, original_sample_rate: int) -> np.ndarray:
        """Prepare audio for playback (convert format, resample if needed)."""
        # Convert to float32 for processing
        if audio.dtype == np.int16:
            audio = audio.astype(np.float32) / 32768.0
        elif audio.dtype == np.int32:
            audio = audio.astype(np.float32) / 2147483648.0
        
        # Ensure mono
        if len(audio.shape) > 1:
            audio = np.mean(audio, axis=1)
        
        # Resample if needed (simple decimation/interpolation)
        if original_sample_rate != self.config.sample_rate:
            # For simplicity, use basic resampling
            ratio = self.config.sample_rate / original_sample_rate
            new_length = int(len(audio) * ratio)
            audio = np.interp(
                np.linspace(0, len(audio), new_length),
                np.arange(len(audio)),
                audio
            )
        
        # Convert back to int16 for sounddevice
        return (audio * 32767).astype(np.int16)
    
    def _generate_default_chime(self, frequency: float, duration: float) -> np.ndarray:
        """Generate a simple sine wave chime as fallback."""
        samples = int(self.config.sample_rate * duration)
        t = np.linspace(0, duration, samples, False)
        
        # Generate sine wave with fade in/out
        wave = np.sin(2 * np.pi * frequency * t)
        
        # Apply fade in/out to avoid clicks
        fade_samples = int(0.01 * self.config.sample_rate)  # 10ms fade
        if fade_samples > 0:
            fade_in = np.linspace(0, 1, fade_samples)
            fade_out = np.linspace(1, 0, fade_samples)
            
            wave[:fade_samples] *= fade_in
            wave[-fade_samples:] *= fade_out
        
        # Convert to int16
        return (wave * 16383).astype(np.int16)  # Use 50% volume
    
    def play_listen_chime(self):
        """Play the 'listening' chime (start of recording)."""
        if self.listen_chime is not None:
            self._play_audio_async(self.listen_chime)
    
    def play_done_chime(self):
        """Play the 'done' chime (end of recording)."""
        if self.done_chime is not None:
            self._play_audio_async(self.done_chime)
    
    def _play_audio_async(self, audio: np.ndarray):
        """Play audio in a separate thread to avoid blocking."""
        threading.Thread(
            target=self._play_audio_sync,
            args=(audio,),
            daemon=True
        ).start()
    
    def _play_audio_sync(self, audio: np.ndarray):
        """Play audio synchronously (runs in separate thread)."""
        try:
            sd.play(
                audio,
                samplerate=self.config.sample_rate,
                device=self.config.device
            )
            # Note: not calling sd.wait() to allow overlapping playback
        except Exception as e:
            print(f"Error playing chime: {e}")


class AudioPlayback:
    """
    Non-blocking audio playback for TTS and other audio.
    
    Based on PRD requirement: "Non-blocking playback via sounddevice callback thread"
    """
    
    def __init__(self, config: Optional[PlaybackConfig] = None):
        self.config = config or PlaybackConfig()
        self.is_playing = False
        self._current_stream: Optional[sd.OutputStream] = None
        
    def play_audio(self, audio: np.ndarray, sample_rate: Optional[int] = None) -> bool:
        """
        Play audio non-blocking.
        
        Args:
            audio: Audio data as numpy array
            sample_rate: Sample rate of audio (uses config default if None)
            
        Returns:
            True if playback started successfully, False otherwise
        """
        if sample_rate is None:
            sample_rate = self.config.sample_rate
        
        try:
            # Stop any current playback
            self.stop_playback()
            
            # Prepare audio
            prepared_audio = self._prepare_audio_for_playback(audio, sample_rate)
            
            # Start playback in separate thread
            threading.Thread(
                target=self._play_audio_thread,
                args=(prepared_audio, sample_rate),
                daemon=True
            ).start()
            
            return True
            
        except Exception as e:
            print(f"Error starting audio playback: {e}")
            return False
    
    def _prepare_audio_for_playback(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """Prepare audio data for playback."""
        # Ensure correct format
        if audio.dtype != np.float32:
            if audio.dtype == np.int16:
                audio = audio.astype(np.float32) / 32768.0
            elif audio.dtype == np.int32:
                audio = audio.astype(np.float32) / 2147483648.0
            else:
                audio = audio.astype(np.float32)
        
        # Ensure mono for our config
        if len(audio.shape) > 1 and self.config.channels == 1:
            audio = np.mean(audio, axis=1)
        
        # Normalize to prevent clipping
        max_val = np.max(np.abs(audio))
        if max_val > 0.95:
            audio = audio * (0.95 / max_val)
        
        return audio
    
    def _play_audio_thread(self, audio: np.ndarray, sample_rate: int):
        """Play audio in dedicated thread."""
        try:
            self.is_playing = True
            
            # Use sounddevice play with blocking
            sd.play(audio, samplerate=sample_rate, device=self.config.device)
            sd.wait()  # Wait for playback to complete
            
        except Exception as e:
            print(f"Error in audio playback thread: {e}")
        finally:
            self.is_playing = False
    
    def stop_playback(self):
        """Stop current audio playback."""
        if self.is_playing:
            try:
                sd.stop()
                self.is_playing = False
            except Exception as e:
                print(f"Error stopping audio playback: {e}")
    
    def is_active(self) -> bool:
        """Check if audio is currently playing."""
        return self.is_playing


class AsyncAudioPlayback:
    """Async wrapper for AudioPlayback."""
    
    def __init__(self, config: Optional[PlaybackConfig] = None):
        self.playback = AudioPlayback(config)
    
    async def play_audio(self, audio: np.ndarray, sample_rate: Optional[int] = None) -> bool:
        """Play audio asynchronously."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.playback.play_audio, audio, sample_rate)
    
    async def stop_playback(self):
        """Stop playback asynchronously."""
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self.playback.stop_playback)
    
    def is_active(self) -> bool:
        """Check if playback is active."""
        return self.playback.is_active()


# Combined audio system for complete pipeline
class AudioSystem:
    """Combined audio system with chimes and playback capabilities."""
    
    def __init__(self, sounds_dir: Union[str, Path], config: Optional[PlaybackConfig] = None):
        self.chimes = AudioChimes(sounds_dir)
        self.playback = AsyncAudioPlayback(config)
    
    def play_listen_chime(self):
        """Play start recording chime."""
        self.chimes.play_listen_chime()
    
    def play_done_chime(self):
        """Play end recording chime."""
        self.chimes.play_done_chime()
    
    async def play_tts_audio(self, audio: np.ndarray, sample_rate: Optional[int] = None) -> bool:
        """Play TTS audio non-blocking."""
        return await self.playback.play_audio(audio, sample_rate)
    
    async def stop_all_audio(self):
        """Stop all audio playback."""
        await self.playback.stop_playback()
    
    def is_playing_tts(self) -> bool:
        """Check if TTS audio is currently playing."""
        return self.playback.is_active() 