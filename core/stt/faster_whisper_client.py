# core/stt/faster_whisper_client.py
"""
faster-whisper STT client for voice assistant.

Provides async-compatible interface for real-time speech transcription.
Updated for faster-whisper 0.10.1+ API compatibility.
"""

import asyncio
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, Union

import numpy as np
from faster_whisper import WhisperModel
from utils.device import best_device

logger = logging.getLogger(__name__)


@dataclass
class STTResult:
    """Result from speech transcription."""
    text: str
    confidence: float
    language: str
    language_probability: float
    latency_ms: float
    avg_log_prob: float  # Updated field name for 0.10.1+


@dataclass
class STTConfig:
    """Configuration for faster-whisper STT."""
    model_name: str = "small.en"  # Model size
    device: str = "auto"  # Device: "auto", "cpu", "cuda", "mps"
    compute_type: str = "auto"  # Compute type: auto, float16, float32, int8
    beam_size: int = 1  # Beam search size (1 = fastest decoding)
    max_initial_timestamp: float = 0.0  # Max initial timestamp
    word_timestamps: bool = False  # Enable word-level timestamps
    
    def __post_init__(self):
        """Auto-detect device and compute type if needed."""
        if self.device == "auto":
            self.device = best_device(prefer_mps=False)  # MPS not well supported in faster-whisper
            logger.info(f"Auto-detected device: {self.device}")


class FasterWhisperSTT:
    """
    faster-whisper STT client with async interface.
    
    Provides real-time speech-to-text transcription using faster-whisper
    with automatic device detection and error handling.
    """
    
    def __init__(self, config: STTConfig = None):
        """
        Initialize faster-whisper STT client.
        
        Args:
            config: STTConfig instance (creates default if None)
        """
        self.config = config or STTConfig()
        self.model = None
        self._stats = {
            "total_requests": 0,
            "total_latency": 0,
            "total_audio_duration": 0,
            "errors": 0
        }
        
        logger.info(f"FasterWhisperSTT config: model={self.config.model_name}, device={self.config.device}")

    async def initialize(self) -> bool:
        """
        Initialize the WhisperModel.
        
        Returns:
            True if initialization successful, False otherwise
        """
        if self.model is not None:
            return True
            
        try:
            start_time = time.time()
            
            def _load_model():
                """Load model in thread executor to avoid blocking."""
                return WhisperModel(
                    self.config.model_name,
                    device=self.config.device,
                    compute_type=self.config.compute_type
                )
            
            # Load model in thread pool to avoid blocking event loop
            self.model = await asyncio.get_event_loop().run_in_executor(
                None, _load_model
            )
            
            load_time = time.time() - start_time
            logger.info(f"faster-whisper model '{self.config.model_name}' loaded in {load_time:.2f}s on {self.config.device}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load faster-whisper model: {e}")
            return False

    async def transcribe(self, audio: Union[np.ndarray, str, Path]) -> STTResult:
        """
        Transcribe audio to text with updated 0.10.1+ API.
        
        Args:
            audio: Audio data as numpy array, file path, or Path object
            
        Returns:
            STTResult with transcription and metadata
        """
        if self.model is None:
            await self.initialize()
            
        if self.model is None:
            return STTResult(
                text="", confidence=0.0, language="en", 
                language_probability=0.0, latency_ms=0, avg_log_prob=-10.0
            )
        
        start_time = time.time()
        
        try:
            def _transcribe_sync():
                """Synchronous transcription for thread executor."""
                # Handle numpy array input
                if isinstance(audio, np.ndarray):
                    # Convert to float32 if needed
                    if audio.dtype != np.float32:
                        audio_float = audio.astype(np.float32)
                        if audio.dtype in [np.int16, np.int32]:
                            # Normalize integer audio to [-1, 1] range
                            audio_float = audio_float / np.iinfo(audio.dtype).max
                    else:
                        audio_float = audio
                    
                    # Use optimized settings for speed + quality filtering (Step 9 UX improvements)
                    segments, info = self.model.transcribe(
                        audio_float,
                        beam_size=1,  # Fastest decoding (instead of config.beam_size)
                        word_timestamps=self.config.word_timestamps,
                        initial_prompt=None,
                        condition_on_previous_text=False,  # ~200ms faster
                        temperature=0.0,  # Deterministic output
                        # Prevent garbage noise retry loops
                        compression_ratio_threshold=1.0,
                        no_speech_threshold=0.8  # Higher = treats low-confidence as silence
                    )
                else:
                    # Handle file path input (with same optimizations)
                    segments, info = self.model.transcribe(
                        str(audio),
                        beam_size=1,  # Fastest decoding
                        word_timestamps=self.config.word_timestamps,
                        initial_prompt=None,
                        condition_on_previous_text=False,  # ~200ms faster
                        temperature=0.0,  # Deterministic output
                        # Prevent garbage noise retry loops
                        compression_ratio_threshold=1.0,
                        no_speech_threshold=0.8  # Higher = treats low-confidence as silence
                    )
                
                return segments, info
            
            # Run transcription in thread pool to avoid blocking
            segments, info = await asyncio.get_event_loop().run_in_executor(
                None, _transcribe_sync
            )
            
            # Extract text from segments
            text = " ".join([seg.text for seg in segments]).strip()
            
            # Calculate metrics
            latency_ms = (time.time() - start_time) * 1000
            
            # Use updated field name for 0.10.1+
            avg_log_prob = getattr(info, 'avg_log_prob', getattr(info, 'avg_logprob', -5.0))
            
            # Estimate confidence from avg_log_prob (heuristic)
            confidence = max(0.0, min(1.0, (avg_log_prob + 1.0) / 1.0))
            
            # Update statistics
            self._stats["total_requests"] += 1
            self._stats["total_latency"] += latency_ms
            
            result = STTResult(
                text=text,
                confidence=confidence,
                language=info.language,
                language_probability=info.language_probability,
                latency_ms=latency_ms,
                avg_log_prob=avg_log_prob
            )
            
            logger.debug(f"Transcribed in {latency_ms:.1f}ms: '{text[:50]}...'")
            return result
            
        except Exception as e:
            logger.error(f"Transcription failed: {e}")
            self._stats["errors"] += 1
            
            latency_ms = (time.time() - start_time) * 1000
            return STTResult(
                text="", confidence=0.0, language="en",
                language_probability=0.0, latency_ms=latency_ms, avg_log_prob=-10.0
            )

    async def health_check(self) -> Tuple[bool, str]:
        """
        Check if STT service is healthy.
        
        Returns:
            Tuple of (is_healthy: bool, status_message: str)
        """
        try:
            if self.model is None:
                success = await self.initialize()
                if not success:
                    return False, "Model initialization failed"
            
            # Test with short silence
            test_audio = np.zeros(1600, dtype=np.float32)  # 100ms at 16kHz
            result = await self.transcribe(test_audio)
            
            if result.latency_ms > 0:
                return True, f"Model {self.config.model_name} ready (device: {self.config.device})"
            else:
                return False, "Transcription test failed"
                
        except Exception as e:
            return False, f"Health check failed: {e}"

    def get_stats(self) -> dict:
        """Get performance statistics."""
        stats = self._stats.copy()
        if stats["total_requests"] > 0:
            stats["avg_latency_ms"] = stats["total_latency"] / stats["total_requests"]
            stats["error_rate"] = stats["errors"] / stats["total_requests"]
        else:
            stats["avg_latency_ms"] = 0
            stats["error_rate"] = 0
        return stats


# Convenience functions
async def create_stt_client(model_name: str = "small.en", 
                           device: str = "auto") -> FasterWhisperSTT:
    """Create and health-check STT client."""
    config = STTConfig(model_name=model_name, device=device)
    client = FasterWhisperSTT(config)
    
    is_healthy, status = await client.health_check()
    if not is_healthy:
        logger.warning(f"STT client health check failed: {status}")
    else:
        logger.info(f"STT client ready: {status}")
    
    return client 