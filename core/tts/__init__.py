"""
Text-to-Speech (TTS) module for voice assistant.

Integrates Chatterbox TTS for high-quality speech synthesis with
default English voice and async non-blocking playback.
"""

from .chatterbox_client import (
    ChatterboxTTSClient,
    TTSResult,
    TTSConfig,
    create_tts_client,
)

__all__ = [
    "ChatterboxTTSClient",
    "TTSResult",
    "TTSConfig",
    "create_tts_client",
] 