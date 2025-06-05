"""
Core audio pipeline for voice assistant.

This module provides the complete audio pipeline including:
- Audio capture with Voice Activity Detection
- Audio playback with chimes system  
- Global hotkey handling
- Non-blocking async audio architecture

Based on research from reference repositories and PRD specifications.
"""

from .capture import (
    AudioConfig,
    VoiceActivityDetector,
    AudioCapture,
    AsyncAudioCapture
)

from .playback import (
    PlaybackConfig,
    AudioChimes,
    AudioPlayback,
    AsyncAudioPlayback,
    AudioSystem
)

from .hotkey import (
    HotkeyConfig,
    GlobalHotkeyHandler,
    AsyncHotkeyHandler,
    PushToTalkHandler
)

from .pipeline import (
    AudioPipeline,
    AudioPipelineConfig
)

__all__ = [
    # Core configurations
    'AudioConfig',
    'PlaybackConfig', 
    'HotkeyConfig',
    'AudioPipelineConfig',
    
    # Audio capture
    'VoiceActivityDetector',
    'AudioCapture',
    'AsyncAudioCapture',
    
    # Audio playback
    'AudioChimes',
    'AudioPlayback',
    'AsyncAudioPlayback',
    'AudioSystem',
    
    # Hotkey handling
    'GlobalHotkeyHandler',
    'AsyncHotkeyHandler',
    'PushToTalkHandler',
    
    # Integrated pipeline
    'AudioPipeline'
] 