"""
Fast unit tests for Nova Voice Assistant (< 3 seconds).

These tests focus on individual components without external dependencies.
Designed for fast CI/CD feedback loops.
"""
import pytest
import asyncio
import numpy as np
import time
from unittest.mock import Mock, patch, AsyncMock
from pathlib import Path

# Import components to test
from utils.events import AudioEvent, STTEvent, LLMEvent, TTSEvent, ErrorEvent
from utils.metrics import TimingStats, MetricsCollector, timer
from config.settings import load_config, NovaConfig, DEFAULT_CONFIG


class TestEvents:
    """Test event system components."""
    
    def test_audio_event_creation(self):
        """Test AudioEvent dataclass creation."""
        audio_data = np.array([1, 2, 3, 4, 5], dtype=np.int16)
        event = AudioEvent(
            audio_data=audio_data,
            sample_rate=16000,
            timestamp=time.time()
        )
        
        assert event.sample_rate == 16000
        assert np.array_equal(event.audio_data, audio_data)
        assert event.timestamp > 0
    
    def test_stt_event_creation(self):
        """Test STTEvent dataclass creation."""
        event = STTEvent(
            text="hello world",
            confidence=0.95,
            processing_time=0.5,
            timestamp=time.time()
        )
        
        assert event.text == "hello world"
        assert event.confidence == 0.95
        assert event.processing_time == 0.5
    
    def test_llm_event_creation(self):
        """Test LLMEvent dataclass creation."""
        event = LLMEvent(
            text="Hello! How can I help you?",
            processing_time=1.2,
            timestamp=time.time(),
            token_count=8
        )
        
        assert event.text == "Hello! How can I help you?"
        assert event.processing_time == 1.2
        assert event.token_count == 8
    
    def test_error_event_creation(self):
        """Test ErrorEvent dataclass creation."""
        test_error = ValueError("Connection failed")
        event = ErrorEvent(
            stage="stt",
            error=test_error,
            recoverable=True,
            timestamp=time.time()
        )
        
        assert event.stage == "stt"
        assert event.error == test_error
        assert event.recoverable is True


class TestMetrics:
    """Test metrics collection components."""
    
    def test_timing_stats_creation(self):
        """Test TimingStats dataclass."""
        stats = TimingStats("test", 0, 0.0, 0.0, 0.0, 0.0, 0.0)
        assert stats.count == 0
        assert stats.total_time == 0.0
        assert stats.min_time == 0.0
        assert stats.max_time == 0.0
    
    def test_timing_stats_recording(self):
        """Test timing stats recording via MetricsCollector."""
        from utils.metrics import _metrics
        _metrics.clear()  # Start fresh
        
        _metrics.record_timing("test", 0.5)
        _metrics.record_timing("test", 1.0)
        _metrics.record_timing("test", 0.2)
        
        stats = _metrics.get_stats("test")
        assert stats.count == 3
        assert stats.total_time == 1.7
        assert stats.min_time == 0.2
        assert stats.max_time == 1.0
        assert abs(stats.avg_time - 0.566667) < 0.001  # 1.7/3
    
    def test_metrics_collector(self):
        """Test MetricsCollector functionality."""
        from utils.metrics import _metrics
        _metrics.clear()  # Start fresh
        
        # Record some timings
        _metrics.record_timing("stt", 0.3)
        _metrics.record_timing("llm", 1.5)
        _metrics.record_timing("stt", 0.4)
        
        # Check stats
        stt_stats = _metrics.get_stats("stt")
        assert stt_stats.count == 2
        assert stt_stats.avg_time == 0.35
        
        llm_stats = _metrics.get_stats("llm")
        assert llm_stats.count == 1
        assert llm_stats.avg_time == 1.5
    
    def test_timer_context_manager(self):
        """Test timer context manager."""
        from utils.metrics import _metrics
        _metrics.clear()  # Start fresh
        
        with timer("test_operation"):
            time.sleep(0.1)  # Sleep for 100ms
        
        stats = _metrics.get_stats("test_operation")
        assert stats.count == 1
        assert 0.08 < stats.avg_time < 0.15  # Should be around 100ms with some tolerance
    
    def test_threshold_monitoring(self):
        """Test threshold monitoring."""
        from utils.metrics import _metrics
        _metrics.clear()  # Start fresh
        
        # Set custom threshold
        _metrics.thresholds["fast_op"] = 0.1  # 100ms threshold
        
        # Record operations
        _metrics.record_timing("fast_op", 0.05)  # Under threshold
        _metrics.record_timing("fast_op", 0.15)  # Over threshold
        
        warnings = _metrics.warnings
        assert len(warnings) > 0
        assert "fast_op" in warnings[0]


class TestConfiguration:
    """Test configuration system."""
    
    def test_default_config_structure(self):
        """Test default configuration has required keys."""
        assert "audio" in DEFAULT_CONFIG
        assert "stt" in DEFAULT_CONFIG
        assert "llm" in DEFAULT_CONFIG
        assert "tts" in DEFAULT_CONFIG
        assert "memory" in DEFAULT_CONFIG
        assert "ui" in DEFAULT_CONFIG
    
    def test_config_loading_defaults(self):
        """Test loading configuration with defaults."""
        with patch('pathlib.Path.exists', return_value=False):
            config = load_config("/nonexistent/path")
        
        assert config.audio["sample_rate"] == 16000
        assert config.llm["model"] == "mistral:7b-instruct-q4_K_M"
        assert config.stt["model_name"] == "small.en"
        assert config.memory["max_turns"] == 8
    
    def test_config_audio_defaults(self):
        """Test audio configuration defaults."""
        config = NovaConfig(**DEFAULT_CONFIG)
        
        assert config.audio["sample_rate"] == 16000
        assert config.audio["vad_threshold_seconds"] == 0.8
        assert config.audio["enable_chimes"] is True
        assert config.audio["enable_push_to_talk"] is True
    
    def test_config_llm_defaults(self):
        """Test LLM configuration defaults."""
        config = NovaConfig(**DEFAULT_CONFIG)
        
        assert config.llm["base_url"] == "http://localhost:11434"
        assert config.llm["temperature"] == 0.7
        assert config.llm["top_p"] == 0.95
        assert config.llm["max_tokens"] == 150


class TestMockIntegrations:
    """Test component interactions with mocked dependencies."""
    
    @pytest.mark.asyncio
    async def test_audio_event_queue_simulation(self):
        """Test audio event processing simulation."""
        from utils.events import initialize_queues, clear_queues
        import utils.events as events
        
        # Initialize queues
        initialize_queues(max_size=10)
        
        # Create mock audio event
        audio_data = np.random.randint(-1000, 1000, 1600, dtype=np.int16)  # 100ms at 16kHz
        event = AudioEvent(
            audio_data=audio_data,
            sample_rate=16000,
            timestamp=time.time()
        )
        
        # Put event in queue
        await events.audio_queue.put(event)
        
        # Get event from queue
        retrieved_event = await events.audio_queue.get()
        
        assert retrieved_event.sample_rate == 16000
        assert len(retrieved_event.audio_data) == 1600
        
        # Cleanup
        clear_queues()
    
    def test_numpy_audio_processing(self):
        """Test basic numpy audio processing."""
        # Generate test audio (sine wave)
        sample_rate = 16000
        duration = 0.1  # 100ms
        frequency = 440  # A4 note
        
        t = np.linspace(0, duration, int(sample_rate * duration), False)
        audio = np.sin(2 * np.pi * frequency * t)
        audio_int16 = (audio * 32767).astype(np.int16)
        
        # Basic checks
        assert len(audio_int16) == 1600  # 100ms at 16kHz
        assert audio_int16.dtype == np.int16
        assert np.max(np.abs(audio_int16)) > 1000  # Should have reasonable amplitude
    
    def test_performance_timing(self):
        """Test performance timing utilities."""
        start_time = time.time()
        
        # Simulate some work
        time.sleep(0.05)  # 50ms
        
        elapsed = time.time() - start_time
        assert 0.04 < elapsed < 0.08  # Should be around 50ms with tolerance


@pytest.mark.performance
class TestPerformanceRequirements:
    """Test that performance requirements are met."""
    
    def test_event_creation_speed(self):
        """Test event creation is fast enough."""
        start_time = time.time()
        
        # Create many events quickly
        for i in range(1000):
            event = STTEvent(
                text=f"test message {i}",
                confidence=0.95,
                processing_time=0.5,
                timestamp=time.time()
            )
        
        elapsed = time.time() - start_time
        assert elapsed < 0.1  # Should create 1000 events in under 100ms
    
    def test_metrics_recording_speed(self):
        """Test metrics recording is fast enough."""
        from utils.metrics import _metrics
        _metrics.clear()  # Start fresh
        start_time = time.time()
        
        # Record many metrics quickly
        for i in range(1000):
            _metrics.record_timing("test", 0.5)
        
        elapsed = time.time() - start_time
        assert elapsed < 0.1  # Should record 1000 metrics in under 100ms


if __name__ == "__main__":
    """Run the fast unit tests."""
    # Run with timing
    start_time = time.time()
    pytest.main([__file__, "-v", "--tb=short"])
    elapsed = time.time() - start_time
    
    print(f"\n⚡ Fast unit tests completed in {elapsed:.2f}s")
    if elapsed > 3.0:
        print("⚠️  WARNING: Tests took longer than 3 seconds!")
    else:
        print("✅ Tests completed within 3-second target") 