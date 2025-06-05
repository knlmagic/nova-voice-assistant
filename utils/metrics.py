"""
Performance metrics and timing utilities for voice assistant pipeline.

Provides context managers for timing operations and latency watchdog functionality.
"""
import time
import logging
from contextlib import contextmanager
from typing import Dict, List, Optional
from dataclasses import dataclass
from collections import defaultdict
import asyncio


@dataclass
class TimingStats:
    """Statistics for a timed operation."""
    stage: str
    count: int
    total_time: float
    avg_time: float
    min_time: float
    max_time: float
    last_time: float


class MetricsCollector:
    """Collects and analyzes performance metrics."""
    
    def __init__(self):
        self.timings: Dict[str, List[float]] = defaultdict(list)
        self.start_times: Dict[str, float] = {}
        self.thresholds: Dict[str, float] = {
            "stt": 0.5,      # 500ms
            "llm": 2.0,      # 2s
            "tts": 3.0,      # 3s
            "audio_capture": 0.1,  # 100ms
            "end_to_end": 5.0,     # 5s total
        }
        self.warnings: List[str] = []
        
    def record_timing(self, stage: str, duration: float) -> None:
        """Record a timing measurement."""
        self.timings[stage].append(duration)
        
        # Check threshold warnings
        threshold = self.thresholds.get(stage)
        if threshold and duration > threshold:
            warning = f"⚠️  {stage} exceeded threshold: {duration:.3f}s > {threshold}s"
            self.warnings.append(warning)
            logging.warning(warning)
    
    def get_stats(self, stage: str) -> Optional[TimingStats]:
        """Get statistics for a specific stage."""
        times = self.timings.get(stage, [])
        if not times:
            return None
            
        return TimingStats(
            stage=stage,
            count=len(times),
            total_time=sum(times),
            avg_time=sum(times) / len(times),
            min_time=min(times),
            max_time=max(times),
            last_time=times[-1]
        )
    
    def log_summary(self) -> None:
        """Print a summary table of all timing stats."""
        print("\n" + "="*70)
        print("📊 PERFORMANCE METRICS SUMMARY")
        print("="*70)
        print(f"{'Stage':<15} {'Count':<6} {'Avg':<8} {'Min':<8} {'Max':<8} {'Last':<8}")
        print("-" * 70)
        
        for stage in sorted(self.timings.keys()):
            stats = self.get_stats(stage)
            if stats:
                print(f"{stage:<15} {stats.count:<6} "
                      f"{stats.avg_time*1000:>6.0f}ms {stats.min_time*1000:>6.0f}ms "
                      f"{stats.max_time*1000:>6.0f}ms {stats.last_time*1000:>6.0f}ms")
        
        if self.warnings:
            print(f"\n⚠️  {len(self.warnings)} threshold violations:")
            for warning in self.warnings[-10:]:  # Show last 10
                print(f"   {warning}")
        
        print("="*70)
    
    def clear(self) -> None:
        """Clear all collected metrics."""
        self.timings.clear()
        self.warnings.clear()


# Global metrics collector
_metrics = MetricsCollector()


@contextmanager
def timer(stage: str):
    """Context manager for timing operations."""
    start_time = time.time()
    try:
        yield
    finally:
        duration = time.time() - start_time
        _metrics.record_timing(stage, duration)


def record_timing(stage: str, duration: float) -> None:
    """Record a timing measurement."""
    _metrics.record_timing(stage, duration)


def get_stats(stage: str) -> Optional[TimingStats]:
    """Get statistics for a specific stage."""
    return _metrics.get_stats(stage)


def log_latency() -> None:
    """Log latency summary table."""
    _metrics.log_summary()


def clear_metrics() -> None:
    """Clear all metrics."""
    _metrics.clear()


def set_threshold(stage: str, threshold_seconds: float) -> None:
    """Set performance threshold for a stage."""
    _metrics.thresholds[stage] = threshold_seconds


class LatencyWatchdog:
    """Monitors pipeline latency and triggers alerts."""
    
    def __init__(self, alert_callback=None):
        self.alert_callback = alert_callback or self._default_alert
        self.active_operations: Dict[str, float] = {}
        
    async def start_operation(self, operation_id: str, stage: str) -> None:
        """Start monitoring an operation."""
        self.active_operations[operation_id] = time.time()
        
    async def end_operation(self, operation_id: str, stage: str) -> float:
        """End monitoring and return duration."""
        if operation_id in self.active_operations:
            duration = time.time() - self.active_operations.pop(operation_id)
            _metrics.record_timing(stage, duration)
            return duration
        return 0.0
    
    def _default_alert(self, stage: str, duration: float, threshold: float):
        """Default alert handler."""
        logging.warning(f"Latency alert: {stage} took {duration:.3f}s (threshold: {threshold:.3f}s)")


# Global watchdog instance
watchdog = LatencyWatchdog()


def get_current_metrics() -> Dict[str, TimingStats]:
    """Get current metrics for all stages."""
    return {stage: _metrics.get_stats(stage) for stage in _metrics.timings.keys()} 