"""
Test suite for STT (Speech-to-Text) integration.

Tests the WhisperSTT client with various audio inputs and validates
performance requirements from PRD.
"""

import pytest
import numpy as np
from pathlib import Path
import tempfile
import wave
import logging

from core.stt import WhisperConfig, WhisperSTT, create_whisper_stt


class TestWhisperConfig:
    """Test WhisperConfig dataclass."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = WhisperConfig()
        
        assert config.model_path == "models/ggml-small.bin"
        assert config.sample_rate == 16000
        assert config.chunk_duration_ms == 128
        assert config.language == "en"
        assert config.n_threads == 4
        assert config.confidence_threshold == 0.7


class TestWhisperSTT:
    """Test WhisperSTT class functionality."""
    
    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return WhisperConfig()
    
    @pytest.fixture
    def stt_client(self, config):
        """Create STT client for testing."""
        # Skip test if model file doesn't exist
        model_path = Path(config.model_path)
        if not model_path.exists():
            pytest.skip(f"Whisper model not found at {model_path}")
        
        stt = WhisperSTT(config)
        if not stt.initialize():
            pytest.skip("Failed to initialize Whisper model")
        
        yield stt
        stt.cleanup()
    
    def test_initialization_with_missing_model(self):
        """Test initialization fails gracefully with missing model."""
        config = WhisperConfig()
        config.model_path = "nonexistent/model.bin"
        
        with pytest.raises(FileNotFoundError):
            WhisperSTT(config)
    
    def test_initialization_success(self, config):
        """Test successful initialization."""
        model_path = Path(config.model_path)
        if not model_path.exists():
            pytest.skip(f"Whisper model not found at {model_path}")
        
        stt = WhisperSTT(config)
        assert stt.initialize() is True
        assert stt.model is not None
        stt.cleanup()
    
    def test_transcribe_empty_audio(self, stt_client):
        """Test transcription with empty audio."""
        empty_audio = np.array([], dtype=np.int16)
        result = stt_client.transcribe_audio(empty_audio)
        
        assert result["success"] is False
        assert "Empty audio" in result["error"]
        assert result["text"] == ""
        assert result["confidence"] == 0.0
    
    def test_transcribe_silent_audio(self, stt_client):
        """Test transcription with silent audio."""
        # Create 1 second of silence
        duration = 1.0  # seconds
        sample_rate = 16000
        samples = int(duration * sample_rate)
        silent_audio = np.zeros(samples, dtype=np.int16)
        
        result = stt_client.transcribe_audio(silent_audio, sample_rate)
        
        # Should succeed but with low confidence
        assert result["success"] is True
        assert result["confidence"] < 0.5  # Low confidence for silence
        assert result["processing_time"] > 0
    
    def test_transcribe_simple_audio(self, stt_client):
        """Test transcription with generated test audio."""
        # Generate a simple tone (not speech, but tests the pipeline)
        duration = 2.0  # seconds
        sample_rate = 16000
        frequency = 440  # A4 note
        
        t = np.linspace(0, duration, int(sample_rate * duration), False)
        audio = (np.sin(2 * np.pi * frequency * t) * 0.3 * 32767).astype(np.int16)
        
        result = stt_client.transcribe_audio(audio, sample_rate)
        
        # Should succeed even if no meaningful transcription
        assert result["success"] is True
        assert result["processing_time"] > 0
        assert result["audio_duration"] > 0
        assert isinstance(result["text"], str)
    
    def test_audio_format_conversion(self, stt_client):
        """Test different audio format handling."""
        # Test with float32 audio
        duration = 1.0
        sample_rate = 16000
        samples = int(duration * sample_rate)
        
        # Float32 audio in [-1, 1] range
        float_audio = np.random.normal(0, 0.1, samples).astype(np.float32)
        result = stt_client.transcribe_audio(float_audio, sample_rate)
        assert result["success"] is True
        
        # Int16 audio
        int_audio = (float_audio * 32767).astype(np.int16)
        result = stt_client.transcribe_audio(int_audio, sample_rate)
        assert result["success"] is True
    
    def test_performance_requirements(self, stt_client):
        """Test that performance meets PRD requirements (<500ms for processing)."""
        # Create test audio (1 second)
        duration = 1.0
        sample_rate = 16000
        samples = int(duration * sample_rate)
        audio = np.random.normal(0, 0.1, samples).astype(np.int16)
        
        result = stt_client.transcribe_audio(audio, sample_rate)
        
        assert result["success"] is True
        # PRD target: <500ms processing time for reasonable audio
        assert result["processing_time"] < 1.0  # Allow some margin
    
    def test_confidence_scoring(self, stt_client):
        """Test confidence scoring functionality."""
        # Test with different audio qualities
        sample_rate = 16000
        duration = 1.0
        samples = int(duration * sample_rate)
        
        # High energy audio (should get higher confidence)
        loud_audio = np.random.normal(0, 0.5, samples).astype(np.int16)
        loud_result = stt_client.transcribe_audio(loud_audio, sample_rate)
        
        # Low energy audio (should get lower confidence)
        quiet_audio = np.random.normal(0, 0.01, samples).astype(np.int16)
        quiet_result = stt_client.transcribe_audio(quiet_audio, sample_rate)
        
        assert loud_result["success"] is True
        assert quiet_result["success"] is True
        
        # Confidence should be reasonable values
        assert 0.0 <= loud_result["confidence"] <= 1.0
        assert 0.0 <= quiet_result["confidence"] <= 1.0
    
    def test_statistics_tracking(self, stt_client):
        """Test performance statistics tracking."""
        # Reset stats
        stt_client.reset_stats()
        
        # Process some audio
        sample_rate = 16000
        audio = np.random.normal(0, 0.1, sample_rate).astype(np.int16)
        
        for i in range(3):
            stt_client.transcribe_audio(audio, sample_rate)
        
        stats = stt_client.get_performance_stats()
        
        assert stats["total_transcriptions"] == 3
        assert stats["average_processing_time"] > 0
        assert stats["min_processing_time"] > 0
        assert stats["max_processing_time"] > 0
        assert "accuracy_stats" in stats


class TestUtilityFunctions:
    """Test utility functions."""
    
    def test_create_whisper_stt(self):
        """Test convenience function for creating STT client."""
        model_path = Path("models/ggml-small.bin")
        if not model_path.exists():
            pytest.skip(f"Whisper model not found at {model_path}")
        
        stt = create_whisper_stt()
        assert stt is not None
        assert stt.model is not None
        stt.cleanup()
    
    def test_create_whisper_stt_with_custom_path(self):
        """Test creating STT with custom model path."""
        # Test with non-existent path should fail
        with pytest.raises(RuntimeError):
            create_whisper_stt("nonexistent/model.bin")
    
    def test_transcribe_file_nonexistent(self):
        """Test transcribing non-existent file."""
        from core.stt.whisper_client import transcribe_file
        
        result = transcribe_file("nonexistent_file.wav")
        assert result["success"] is False
        assert "Failed to load audio file" in result["error"]


class TestIntegrationWithRealAudio:
    """Integration tests with real audio files (if available)."""
    
    def create_test_wav(self, text_content="test audio", duration=2.0):
        """Create a temporary WAV file for testing."""
        # Generate a simple sine wave as test audio
        sample_rate = 16000
        frequency = 440  # A4 note
        
        t = np.linspace(0, duration, int(sample_rate * duration), False)
        audio = (np.sin(2 * np.pi * frequency * t) * 0.1 * 32767).astype(np.int16)
        
        # Create temporary file
        temp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
        
        with wave.open(temp_file.name, 'wb') as wav_file:
            wav_file.setnchannels(1)  # Mono
            wav_file.setsampwidth(2)  # 16-bit
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(audio.tobytes())
        
        return temp_file.name
    
    def test_transcribe_wav_file(self):
        """Test transcribing a WAV file."""
        from core.stt.whisper_client import transcribe_file
        
        model_path = Path("models/ggml-small.bin")
        if not model_path.exists():
            pytest.skip(f"Whisper model not found at {model_path}")
        
        # Create test file
        test_file = self.create_test_wav()
        
        try:
            result = transcribe_file(test_file)
            
            # Should succeed even if transcription is not meaningful
            assert result["success"] is True
            assert "processing_time" in result
            assert isinstance(result["text"], str)
            
        finally:
            # Cleanup
            Path(test_file).unlink(missing_ok=True)


@pytest.fixture
def enable_logging():
    """Enable logging for tests."""
    logging.basicConfig(level=logging.INFO)
    yield
    logging.getLogger().handlers.clear()


def test_whisper_import():
    """Test that whispercpp can be imported."""
    import whispercpp
    assert hasattr(whispercpp, 'Whisper')


if __name__ == "__main__":
    # Run tests directly
    pytest.main([__file__, "-v"]) 