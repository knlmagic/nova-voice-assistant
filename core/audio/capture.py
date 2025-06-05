"""
Audio capture module with Voice Activity Detection (VAD).

Based on research from vndee/local-talking-llm and whisper.cpp examples.
Implements callback-based audio capture with queue system for non-blocking operation.
"""

import asyncio
import threading
import time
from queue import Queue
from typing import Optional, Callable
import numpy as np
import sounddevice as sd
from dataclasses import dataclass


@dataclass
class AudioConfig:
    """Audio configuration based on whisper.cpp requirements."""
    sample_rate: int = 16000  # whisper.cpp native sample rate
    channels: int = 1  # mono for speech recognition
    dtype: str = 'int16'  # PCM 16-bit for whisper.cpp compatibility
    chunk_size_ms: int = 128  # milliseconds per chunk (tested optimal)
    vad_threshold_seconds: float = 0.8  # silence threshold for auto-stop
    
    @property
    def chunk_size_samples(self) -> int:
        """Calculate chunk size in samples."""
        return int(self.sample_rate * self.chunk_size_ms / 1000)


class VoiceActivityDetector:
    """Simple energy-based Voice Activity Detection."""
    
    def __init__(self, config: AudioConfig):
        self.config = config
        self.silence_threshold = 500  # energy threshold for silence detection
        self.silence_duration = 0.0
        self.last_check_time = time.time()
        
    def is_speech(self, audio_chunk: np.ndarray) -> bool:
        """
        Detect if audio chunk contains speech based on energy.
        
        Args:
            audio_chunk: Audio data as numpy array
            
        Returns:
            True if speech detected, False if silence
        """
        # Calculate RMS energy
        energy = np.sqrt(np.mean(audio_chunk.astype(np.float32) ** 2))
        
        current_time = time.time()
        time_delta = current_time - self.last_check_time
        self.last_check_time = current_time
        
        if energy > self.silence_threshold:
            # Speech detected - reset silence counter
            self.silence_duration = 0.0
            return True
        else:
            # Silence detected - accumulate silence time
            self.silence_duration += time_delta
            return False
    
    def should_stop_recording(self) -> bool:
        """Check if recording should stop due to prolonged silence."""
        return self.silence_duration >= self.config.vad_threshold_seconds
    
    def reset(self):
        """Reset VAD state for new recording session."""
        self.silence_duration = 0.0
        self.last_check_time = time.time()


class AudioCapture:
    """
    Real-time audio capture with Voice Activity Detection.
    
    Based on proven patterns from vndee/local-talking-llm:
    - Callback-based capture using sounddevice.RawInputStream
    - Queue system for thread-safe audio data flow
    - Non-blocking architecture with clean start/stop control
    """
    
    def __init__(self, config: Optional[AudioConfig] = None):
        self.config = config or AudioConfig()
        self.vad = VoiceActivityDetector(self.config)
        
        # Audio data queue for thread-safe communication
        self.audio_queue: Queue = Queue()
        
        # Control flags
        self.is_recording = False
        self.stop_event = threading.Event()
        
        # Audio stream
        self.stream: Optional[sd.RawInputStream] = None
        self.recording_thread: Optional[threading.Thread] = None
        
        # Callbacks
        self.on_speech_start: Optional[Callable] = None
        self.on_speech_end: Optional[Callable] = None
        self.on_vad_stop: Optional[Callable] = None
    
    def _audio_callback(self, indata, frames, time, status):
        """
        Sounddevice callback for real-time audio capture.
        
        This runs in a separate audio thread and should be fast.
        """
        if status:
            print(f"Audio callback status: {status}")
        
        if self.is_recording:
            # Convert to numpy array and add to queue
            audio_chunk = np.frombuffer(indata, dtype=np.int16)
            self.audio_queue.put(audio_chunk.copy())
    
    def start_recording(self) -> bool:
        """
        Start audio recording with VAD.
        
        Returns:
            True if recording started successfully, False otherwise
        """
        if self.is_recording:
            return False
        
        try:
            # Reset VAD state
            self.vad.reset()
            self.stop_event.clear()
            
            # Start audio stream
            self.stream = sd.RawInputStream(
                samplerate=self.config.sample_rate,
                channels=self.config.channels,
                dtype=self.config.dtype,
                callback=self._audio_callback,
                blocksize=self.config.chunk_size_samples
            )
            
            self.stream.start()
            self.is_recording = True
            
            # Start VAD processing thread
            self.recording_thread = threading.Thread(
                target=self._process_vad,
                daemon=True
            )
            self.recording_thread.start()
            
            if self.on_speech_start:
                self.on_speech_start()
            
            return True
            
        except Exception as e:
            print(f"Error starting audio recording: {e}")
            return False
    
    def stop_recording(self) -> Optional[np.ndarray]:
        """
        Stop audio recording and return captured audio.
        
        Returns:
            Captured audio as numpy array, or None if no audio captured
        """
        if not self.is_recording:
            return None
        
        # Signal stop
        self.stop_event.set()
        self.is_recording = False
        
        # Stop audio stream
        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.stream = None
        
        # Wait for processing thread to finish
        if self.recording_thread:
            self.recording_thread.join(timeout=1.0)
        
        # Collect all audio data from queue
        audio_chunks = []
        while not self.audio_queue.empty():
            try:
                chunk = self.audio_queue.get_nowait()
                audio_chunks.append(chunk)
            except:
                break
        
        if self.on_speech_end:
            self.on_speech_end()
        
        # Concatenate audio chunks
        if audio_chunks:
            return np.concatenate(audio_chunks)
        else:
            return None
    
    def _process_vad(self):
        """
        Process Voice Activity Detection in separate thread.
        
        Monitors for silence and automatically stops recording
        when VAD threshold is exceeded.
        """
        while self.is_recording and not self.stop_event.is_set():
            try:
                # Get audio chunk with timeout
                chunk = self.audio_queue.get(timeout=0.1)
                
                # Check for speech activity
                has_speech = self.vad.is_speech(chunk)
                
                # Auto-stop on prolonged silence
                if not has_speech and self.vad.should_stop_recording():
                    print(f"VAD: Auto-stopping recording after {self.vad.silence_duration:.1f}s silence")
                    
                    if self.on_vad_stop:
                        self.on_vad_stop()
                    
                    # Trigger stop from main thread
                    threading.Thread(target=self.stop_recording, daemon=True).start()
                    break
                
                # Put chunk back for final audio collection
                self.audio_queue.put(chunk)
                
            except Exception:
                # Timeout or queue empty - continue monitoring
                continue
    
    def is_active(self) -> bool:
        """Check if audio capture is currently active."""
        return self.is_recording
    
    def get_audio_info(self) -> dict:
        """Get current audio configuration info."""
        return {
            "sample_rate": self.config.sample_rate,
            "channels": self.config.channels,
            "dtype": self.config.dtype,
            "chunk_size_ms": self.config.chunk_size_ms,
            "vad_threshold_s": self.config.vad_threshold_seconds,
            "is_recording": self.is_recording
        }


# Async wrapper for use with asyncio
class AsyncAudioCapture:
    """Async wrapper around AudioCapture for integration with async audio pipeline."""
    
    def __init__(self, config: Optional[AudioConfig] = None):
        self.capture = AudioCapture(config)
        self._capture_task: Optional[asyncio.Task] = None
    
    async def start_recording(self) -> bool:
        """Start recording asynchronously."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.capture.start_recording)
    
    async def stop_recording(self) -> Optional[np.ndarray]:
        """Stop recording and get audio data asynchronously."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.capture.stop_recording)
    
    def set_callbacks(self, on_speech_start=None, on_speech_end=None, on_vad_stop=None):
        """Set callback functions for audio events."""
        self.capture.on_speech_start = on_speech_start
        self.capture.on_speech_end = on_speech_end
        self.capture.on_vad_stop = on_vad_stop
    
    def is_active(self) -> bool:
        """Check if recording is active."""
        return self.capture.is_active() 