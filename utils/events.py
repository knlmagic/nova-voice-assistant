"""
Event system for voice assistant pipeline communication.

This module provides dataclasses for passing data between pipeline stages
and prevents circular dependencies by centralizing queue types.
"""
from dataclasses import dataclass
from typing import Optional, Any
import numpy as np
import asyncio


@dataclass
class AudioEvent:
    """Raw audio data from capture pipeline."""
    audio_data: np.ndarray
    sample_rate: int
    timestamp: float


@dataclass 
class STTEvent:
    """Speech-to-text transcription result."""
    text: str
    confidence: float
    processing_time: float
    timestamp: float


@dataclass
class LLMEvent:
    """LLM response with metadata."""
    text: str
    processing_time: float
    timestamp: float
    token_count: Optional[int] = None


@dataclass
class TTSEvent:
    """Text-to-speech synthesis request."""
    text: str
    priority: str = "normal"  # "high", "normal", "low"
    timestamp: float = 0.0


@dataclass
class ErrorEvent:
    """Error information for diagnostics."""
    stage: str  # "audio", "stt", "llm", "tts"
    error: Exception
    timestamp: float
    recoverable: bool = True


@dataclass
class ControlEvent:
    """Control signals for pipeline management."""
    action: str  # "start", "stop", "pause", "resume"
    component: str = "all"  # "all", "audio", "stt", "llm", "tts"
    timestamp: float = 0.0


class ShutdownEvent(Exception):
    """Signal for graceful shutdown."""
    pass


# Global event queues for pipeline communication
audio_queue: asyncio.Queue[AudioEvent] | None = None
stt_queue: asyncio.Queue[STTEvent] | None = None  
llm_queue: asyncio.Queue[LLMEvent] | None = None
tts_queue: asyncio.Queue[TTSEvent] | None = None
error_queue: asyncio.Queue[ErrorEvent] | None = None
control_queue: asyncio.Queue[ControlEvent] | None = None
hotkey_queue: asyncio.Queue[str] | None = None  # Hotkey events


def initialize_queues(max_size: int = 50) -> None:
    """Initialize all pipeline queues with bounded size."""
    global audio_queue, stt_queue, llm_queue, tts_queue, error_queue, control_queue, hotkey_queue
    
    audio_queue = asyncio.Queue(maxsize=max_size)
    stt_queue = asyncio.Queue(maxsize=max_size)
    llm_queue = asyncio.Queue(maxsize=max_size)
    tts_queue = asyncio.Queue(maxsize=max_size)
    error_queue = asyncio.Queue(maxsize=max_size * 2)  # More room for errors
    control_queue = asyncio.Queue(maxsize=10)
    hotkey_queue = asyncio.Queue(maxsize=10)  # Small queue for hotkey events


def clear_queues() -> None:
    """Clear all queues during shutdown."""
    for queue in [audio_queue, stt_queue, llm_queue, tts_queue, error_queue, control_queue, hotkey_queue]:
        if queue:
            while not queue.empty():
                try:
                    queue.get_nowait()
                except asyncio.QueueEmpty:
                    break 