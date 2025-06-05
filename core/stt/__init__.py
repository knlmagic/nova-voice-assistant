"""
Speech-to-Text (STT) module for voice assistant.

Integrates faster-whisper for real-time speech recognition with high accuracy,
low latency, and rock-solid Python 3.11+ compatibility.
"""

from .faster_whisper_client import FasterWhisperSTT, STTResult

__all__ = ["FasterWhisperSTT", "STTResult"] 