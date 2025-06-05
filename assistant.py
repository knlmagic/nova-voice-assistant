#!/usr/bin/env python3
"""
Nova Voice Assistant - End-to-End Pipeline Orchestrator

This is the main entry point that integrates all pipeline components:
- Audio capture with push-to-talk (⌃⌥␣)
- Speech-to-text via faster-whisper
- LLM processing via Ollama + Mistral 7B
- Text-to-speech via Chatterbox
- SQLite conversation memory
- Performance metrics and error handling

Based on PRD specifications and Step 7 integration playbook.
"""

# Suppress noisy debug logging
import os
import logging
os.environ["NUMBA_LOG_LEVEL"] = "WARNING"
logging.getLogger("numba").setLevel(logging.WARNING)

import asyncio
import logging
import signal
import sys
import time
from pathlib import Path
from typing import Optional
import argparse

# Local imports
from core.audio.pipeline import AudioPipeline, AudioPipelineConfig
from core.stt.faster_whisper_client import FasterWhisperSTT, STTConfig
from agent.llm_client import OllamaLLMClient, LLMConfig
from agent.memory import MemoryIntegratedLLMClient
from core.tts.chatterbox_client import ChatterboxTTSClient, TTSConfig
from utils.events import (
    initialize_queues, clear_queues,
    AudioEvent, STTEvent, LLMEvent, TTSEvent, ErrorEvent, ShutdownEvent
)
import utils.events as events
from utils.metrics import timer, log_latency, clear_metrics, get_current_metrics
from utils.hotkey_manager import HotKeyManager


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('nova_assistant.log')
    ]
)
logger = logging.getLogger(__name__)


class NovaAssistant:
    """
    Main orchestrator for Nova voice assistant.
    
    Coordinates all pipeline components and manages the complete conversation flow:
    Push-to-Talk → Audio Capture → STT → LLM → TTS → Audio Output
    """
    
    def __init__(self, verbose: bool = False, quiet: bool = False):
        """Initialize Nova assistant with all pipeline components."""
        self.verbose = verbose
        self.quiet = quiet
        
        # Configure logging level
        if quiet:
            logging.getLogger().setLevel(logging.WARNING)
        elif verbose:
            logging.getLogger().setLevel(logging.DEBUG)
        
        # Pipeline components
        self.audio_pipeline: Optional[AudioPipeline] = None
        self.stt_client: Optional[FasterWhisperSTT] = None
        self.llm_client: Optional[MemoryIntegratedLLMClient] = None
        self.tts_client: Optional[ChatterboxTTSClient] = None
        self.hotkey_manager: Optional[HotKeyManager] = None
        
        # Task management
        self.tasks = []
        self.shutdown_event = asyncio.Event()
        self.is_running = False
        
        # Performance tracking
        self.conversation_count = 0
        
        logger.info("Nova Assistant initialized")

    async def initialize_components(self) -> bool:
        """Initialize all pipeline components."""
        try:
            logger.info("🔧 Initializing pipeline components...")
            
            # Initialize event queues
            initialize_queues(max_size=50)
            
            # 1. Audio Pipeline (without push-to-talk - handled separately)
            logger.info("📡 Setting up audio pipeline...")
            audio_config = AudioPipelineConfig()
            audio_config.enable_push_to_talk = False  # Disable built-in push-to-talk
            self.audio_pipeline = AudioPipeline(audio_config)
            
            # Set up audio event callback
            self.audio_pipeline.set_callbacks(
                on_audio_captured=self._on_audio_captured,
                on_recording_start=self._on_recording_start,
                on_recording_stop=self._on_recording_stop
            )
            
            # 2. STT Client
            logger.info("🗣️  Setting up STT client...")
            stt_config = STTConfig(model_name="small.en", device="auto")
            self.stt_client = FasterWhisperSTT(stt_config)
            
            if not await self.stt_client.initialize():
                logger.error("Failed to initialize STT client")
                return False
            
            # 3. LLM Client with Memory
            logger.info("🧠 Setting up LLM client with memory...")
            llm_config = LLMConfig()
            base_llm_client = OllamaLLMClient(llm_config)
            self.llm_client = MemoryIntegratedLLMClient(base_llm_client, "ai_memory.db")
            
            # 4. TTS Client  
            logger.info("🎵 Setting up TTS client...")
            tts_config = TTSConfig(device="cpu")
            self.tts_client = ChatterboxTTSClient(tts_config)
            
            if not await self.tts_client.initialize():
                logger.error("Failed to initialize TTS client")
                return False
                
            # 5. LLM Warm-up
            logger.info("🔥 Warming up LLM (eliminating cold start)...")
            try:
                with timer("llm_warmup"):
                    warmup_response = await self.llm_client.generate_response("ping")
                logger.info(f"LLM warmed up successfully in {warmup_response.latency_ms}ms")
            except Exception as e:
                logger.warning(f"LLM warmup failed (continuing anyway): {e}")
            
            # 6. Hotkey Manager (after queues are initialized)
            logger.info("⌨️  Setting up hotkey manager...")
            self.hotkey_manager = HotKeyManager(events.hotkey_queue, self.audio_pipeline)
            self.hotkey_manager.start()
            
            logger.info("✅ All components initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize components: {e}")
            return False

    async def start(self) -> bool:
        """Start the complete voice assistant pipeline."""
        if self.is_running:
            return True
            
        try:
            # Initialize components
            if not await self.initialize_components():
                return False
            
            # Health checks
            logger.info("🔍 Performing health checks...")
            await self._health_check_all()
            
            # Start audio pipeline
            if not await self.audio_pipeline.start():
                logger.error("Failed to start audio pipeline")
                return False
            
            # Start worker tasks
            self.tasks = [
                asyncio.create_task(self._hotkey_worker(), name="hotkey_worker"),
                asyncio.create_task(self._stt_worker(), name="stt_worker"),
                asyncio.create_task(self._llm_worker(), name="llm_worker"), 
                asyncio.create_task(self._tts_worker(), name="tts_worker"),
                asyncio.create_task(self._error_handler(), name="error_handler"),
                asyncio.create_task(self._metrics_reporter(), name="metrics_reporter")
            ]
            
            self.is_running = True
            
            # Show ready message
            if not self.quiet:
                print("\n" + "="*60)
                print("🎧 NOVA VOICE ASSISTANT READY")
                print("="*60)
                print("📱 Press ⌃⌥␣ (Control-Option-Space) to talk")
                print("🛑 Press Ctrl+C to quit")
                print("💬 Say 'hello' to test the pipeline")
                print("="*60 + "\n")
            
            logger.info("Nova Assistant started successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start assistant: {e}")
            return False

    async def stop(self):
        """Stop the voice assistant gracefully."""
        if not self.is_running:
            return
            
        logger.info("🔻 Shutting down Nova Assistant...")
        
        try:
            # Signal shutdown
            self.shutdown_event.set()
            
            # Stop hotkey manager first
            if self.hotkey_manager:
                self.hotkey_manager.stop()
            
            # Cancel all tasks
            for task in self.tasks:
                if not task.done():
                    task.cancel()
            
            # Wait for tasks to complete with timeout
            if self.tasks:
                await asyncio.wait_for(
                    asyncio.gather(*self.tasks, return_exceptions=True),
                    timeout=5.0
                )
            
            # Stop audio pipeline
            if self.audio_pipeline:
                await self.audio_pipeline.stop()
            
            # Periodic database maintenance
            if self.llm_client and hasattr(self.llm_client, 'memory'):
                try:
                    stats = await self.llm_client.get_stats()
                    conversation_count = stats.get('conversation_turns', 0)
                    
                    # Vacuum database periodically (every 100 conversations or daily usage)
                    if conversation_count % 100 == 0 and conversation_count > 0:
                        logger.info("Performing periodic database maintenance...")
                        await self.llm_client.memory.vacuum_database()
                except Exception as e:
                    logger.warning(f"Database maintenance failed: {e}")
            
            # Clear queues
            clear_queues()
            
            # Final metrics
            if not self.quiet:
                print("\n📊 Final Performance Report:")
                log_latency()
            
            self.is_running = False
            logger.info("Nova Assistant stopped gracefully")
            
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")

    # Audio Event Handlers
    
    def _on_audio_captured(self, audio_data):
        """Handle audio captured from push-to-talk."""
        try:
            event = AudioEvent(
                audio_data=audio_data,
                sample_rate=16000,  # Audio pipeline config
                timestamp=time.time()
            )
            
            # Non-blocking queue put (only if queue is initialized)
            if events.audio_queue is not None:
                if events.audio_queue.full():
                    # Drop oldest frame to make room for newest (FIFO policy)
                    try:
                        events.audio_queue.get_nowait()
                        logger.debug("Audio queue full, dropped oldest frame")
                    except:
                        pass
                events.audio_queue.put_nowait(event)
            else:
                logger.debug("Audio queue not initialized yet")
                
        except Exception as e:
            logger.error(f"Error handling audio capture: {e}")

    def _on_recording_start(self):
        """Handle recording start event."""
        if not self.quiet:
            print("🎤 Listening...")

    def _on_recording_stop(self):
        """Handle recording stop event."""
        if not self.quiet:
            print("⏹️  Processing...")

    # Pipeline Workers

    async def _hotkey_worker(self):
        """Hotkey worker: Handle push-to-talk events."""
        logger.info("⌨️  Hotkey worker started")
        
        try:
            while not self.shutdown_event.is_set():
                try:
                    # Wait for hotkey event
                    hotkey_event = await asyncio.wait_for(
                        events.hotkey_queue.get(),
                        timeout=1.0
                    )
                    
                    if hotkey_event == "push_to_talk":
                        logger.debug("🔥 Processing hotkey press")
                        
                        # Start recording
                        if not self.audio_pipeline.is_recording:
                            await self.audio_pipeline.start_manual_recording()
                        else:
                            # Stop recording and get audio
                            audio_data = await self.audio_pipeline.stop_manual_recording()
                            if audio_data is not None:
                                self._on_audio_captured(audio_data)
                    
                except asyncio.TimeoutError:
                    continue
                except Exception as e:
                    await self._report_error("hotkey", e)
                    
        except asyncio.CancelledError:
            logger.info("Hotkey worker cancelled")
        except Exception as e:
            logger.error(f"Hotkey worker failed: {e}")

    async def _stt_worker(self):
        """STT worker: Audio → Text transcription."""
        logger.info("🗣️  STT worker started")
        
        try:
            while not self.shutdown_event.is_set():
                try:
                    # Wait for audio with timeout
                    audio_event = await asyncio.wait_for(
                        events.audio_queue.get(), 
                        timeout=1.0
                    )
                    
                    # Transcribe audio
                    with timer("stt"):
                        stt_result = await self.stt_client.transcribe(audio_event.audio_data)
                    
                    # Only process if we got meaningful text
                    if stt_result.text.strip():
                        event = STTEvent(
                            text=stt_result.text,
                            confidence=stt_result.confidence,
                            processing_time=stt_result.latency_ms / 1000,
                            timestamp=time.time()
                        )
                        
                        if not self.quiet:
                            print(f"🗣️  You said: \"{stt_result.text}\"")
                        
                        await events.stt_queue.put(event)
                    else:
                        if self.verbose:
                            logger.debug("Empty transcription, skipping LLM")
                    
                except asyncio.TimeoutError:
                    continue
                except Exception as e:
                    await self._report_error("stt", e)
                    
        except asyncio.CancelledError:
            logger.info("STT worker cancelled")
        except Exception as e:
            logger.error(f"STT worker failed: {e}")

    async def _llm_worker(self):
        """LLM worker: Text → AI Response."""
        logger.info("🧠 LLM worker started")
        
        try:
            while not self.shutdown_event.is_set():
                try:
                    # Wait for STT result
                    stt_event = await asyncio.wait_for(
                        events.stt_queue.get(),
                        timeout=1.0
                    )
                    
                    # Generate LLM response
                    with timer("llm"):
                        llm_response = await self.llm_client.generate_response(stt_event.text)
                    
                    # Truncate for better TTS performance (Chatterbox sweet spot ≤ 25 words)
                    response_text = llm_response.content
                    words = response_text.split()
                    if len(words) > 25:
                        response_text = " ".join(words[:25]) + "..."
                        logger.debug(f"Truncated LLM response from {len(words)} to 25 words for TTS")
                    
                    event = LLMEvent(
                        text=response_text,
                        processing_time=llm_response.latency_ms / 1000,
                        timestamp=time.time(),
                        token_count=llm_response.completion_tokens
                    )
                    
                    if not self.quiet:
                        print(f"🤖 Nova: \"{response_text}\"")
                    
                    await events.llm_queue.put(event)
                    
                except asyncio.TimeoutError:
                    continue
                except Exception as e:
                    await self._report_error("llm", e)
                    
        except asyncio.CancelledError:
            logger.info("LLM worker cancelled")
        except Exception as e:
            logger.error(f"LLM worker failed: {e}")

    async def _tts_worker(self):
        """TTS worker: Text → Speech synthesis and playback."""
        logger.info("🎵 TTS worker started")
        
        try:
            while not self.shutdown_event.is_set():
                try:
                    # Wait for LLM response
                    llm_event = await asyncio.wait_for(
                        events.llm_queue.get(),
                        timeout=1.0
                    )
                    
                    # Synthesize speech
                    with timer("tts"):
                        tts_result = await self.tts_client.synthesize(llm_event.text)
                    
                    # Play audio via audio pipeline WITH proper speaking state
                    if len(tts_result.audio) > 0:
                        # Set speaking flag BEFORE playback starts
                        self.audio_pipeline._speaking.set()
                        
                        try:
                            await self.audio_pipeline.play_tts(
                                tts_result.audio, 
                                tts_result.sample_rate
                            )
                            
                            self.conversation_count += 1
                            
                            if not self.quiet:
                                print("🔊 Speaking response...")
                        finally:
                            # Clear speaking flag AFTER playback completes
                            self.audio_pipeline._speaking.clear()
                    else:
                        logger.warning("Empty TTS audio generated")
                    
                except asyncio.TimeoutError:
                    continue
                except Exception as e:
                    await self._report_error("tts", e)
                    
        except asyncio.CancelledError:
            logger.info("TTS worker cancelled")
        except Exception as e:
            logger.error(f"TTS worker failed: {e}")

    async def _error_handler(self):
        """Central error handling worker."""
        logger.info("⚠️  Error handler started")
        
        try:
            while not self.shutdown_event.is_set():
                try:
                    error_event = await asyncio.wait_for(
                        events.error_queue.get(),
                        timeout=1.0
                    )
                    
                    logger.error(f"Pipeline error in {error_event.stage}: {error_event.error}")
                    
                    # Decide recovery action based on error type
                    if error_event.recoverable:
                        logger.info(f"Recovering from {error_event.stage} error")
                    else:
                        logger.critical(f"Unrecoverable error in {error_event.stage}, shutting down")
                        self.shutdown_event.set()
                    
                except asyncio.TimeoutError:
                    continue
                    
        except asyncio.CancelledError:
            logger.info("Error handler cancelled")

    async def _metrics_reporter(self):
        """Periodic metrics reporting."""
        if self.quiet:
            return
            
        logger.info("📊 Metrics reporter started")
        
        try:
            while not self.shutdown_event.is_set():
                await asyncio.sleep(30)  # Report every 30 seconds
                
                if self.conversation_count > 0:
                    metrics = get_current_metrics()
                    
                    print(f"\n📊 Performance Summary ({self.conversation_count} conversations):")
                    for stage, stats in metrics.items():
                        if stats and stats.count > 0:
                            print(f"   {stage}: {stats.avg_time*1000:.0f}ms avg ({stats.count} samples)")
                    print()
                    
        except asyncio.CancelledError:
            logger.info("Metrics reporter cancelled")

    # Utility Methods

    async def _health_check_all(self):
        """Perform health checks on all components."""
        checks = []
        
        # STT health check (disabled to avoid double transcription)
        stt_healthy, stt_msg = True, f"Model {self.stt_client.config.model_name} ready (device: {self.stt_client.config.device})"
        checks.append(("STT", stt_healthy, stt_msg))
        
        # LLM health check  
        llm_healthy, llm_msg = await self.llm_client.llm_client.health_check()
        checks.append(("LLM", llm_healthy, llm_msg))
        
        # TTS health check
        tts_healthy, tts_msg = await self.tts_client.health_check()
        checks.append(("TTS", tts_healthy, tts_msg))
        
        # Report results
        all_healthy = True
        for component, healthy, message in checks:
            status = "✅" if healthy else "❌"
            logger.info(f"{status} {component}: {message}")
            if not healthy:
                all_healthy = False
        
        if not all_healthy:
            logger.warning("Some components failed health checks")

    async def _report_error(self, stage: str, error: Exception, recoverable: bool = True):
        """Report error to error queue."""
        error_event = ErrorEvent(
            stage=stage,
            error=error,
            timestamp=time.time(),
            recoverable=recoverable
        )
        
        if not error_queue.full():
            error_queue.put_nowait(error_event)


# CLI and Main Entry Point

def setup_signal_handlers(assistant: NovaAssistant):
    """Setup graceful shutdown on SIGINT/SIGTERM."""
    def signal_handler(signum, frame):
        logger.info(f"Received signal {signum}, initiating shutdown...")
        asyncio.create_task(assistant.stop())
        
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)


async def main():
    """Main entry point for Nova voice assistant."""
    parser = argparse.ArgumentParser(description="Nova Voice Assistant")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")
    parser.add_argument("--quiet", "-q", action="store_true", help="Quiet mode")
    parser.add_argument("--profile", action="store_true", help="Show performance profile and exit")
    
    args = parser.parse_args()
    
    if args.profile:
        # Just show metrics from last run
        print("📊 Performance Profile:")
        log_latency()
        return
    
    # Create and start assistant
    assistant = NovaAssistant(verbose=args.verbose, quiet=args.quiet)
    
    # Setup signal handlers for graceful shutdown
    setup_signal_handlers(assistant)
    
    try:
        # Start the assistant
        if await assistant.start():
            # Keep running until shutdown
            await assistant.shutdown_event.wait()
        else:
            logger.error("Failed to start Nova Assistant")
            return 1
            
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return 1
    finally:
        await assistant.stop()
    
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main())) 