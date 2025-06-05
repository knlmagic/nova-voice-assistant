"""
Chatterbox TTS client for voice assistant.

Clean implementation using official ChatterboxTTS.from_pretrained() API
with automatic CPU loading via sitecustomize.py patch.
"""

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from chatterbox import ChatterboxTTS
from scipy.io import wavfile

logger = logging.getLogger(__name__)

# Official Resemble-AI open English voice (smallest model, ~350 MB after download)
DEFAULT_VOICE_ID = "resemble-ai/chatterbox-t1-16khz"


@dataclass
class TTSConfig:
    """Configuration for Chatterbox TTS."""
    voice_id: str = DEFAULT_VOICE_ID
    device: str = "cpu"  # "cpu", "mps", or "cuda"
    sample_rate: int = 24000  # Chatterbox native sample rate


@dataclass
class TTSResult:
    """Result from text-to-speech synthesis."""
    audio: np.ndarray  # Audio data as numpy array
    sample_rate: int
    latency_ms: float
    text: str
    voice_id: str


class ChatterboxTTSClient:
    """
    Clean ChatterboxTTS client using official API.
    
    Uses ChatterboxTTS.from_pretrained() with official voice models
    and automatic CPU loading via sitecustomize.py.
    """
    
    def __init__(self, config: Optional[TTSConfig] = None):
        """
        Initialize Chatterbox TTS client.
        
        Args:
            config: TTSConfig instance (creates default if None)
        """
        self.config = config or TTSConfig()
        self._tts = None
        self._initialized = False
        
        logger.info(f"ChatterboxTTSClient: voice={self.config.voice_id}, device={self.config.device}")

    async def initialize(self) -> bool:
        """
        Initialize ChatterboxTTS using official from_pretrained method.
        
        Returns:
            True if initialization successful, False otherwise
        """
        if self._initialized:
            return True
            
        try:
            start_time = time.time()
            
            # Load model in thread executor (sitecustomize.py handles CPU loading)
            loop = asyncio.get_running_loop()
            self._tts = await loop.run_in_executor(
                None, 
                lambda: ChatterboxTTS.from_pretrained(device="cpu")
            )
            
            # Move to target device if not CPU (after loading on CPU via sitecustomize.py)
            if self.config.device != "cpu":
                logger.info(f"Moving TTS to {self.config.device}")
                self._tts = self._tts.to(self.config.device)
            
            load_time = time.time() - start_time
            self._initialized = True
            
            logger.info(f"ChatterboxTTS loaded in {load_time:.2f}s from {self.config.voice_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize ChatterboxTTS: {e}")
            logger.exception("Full initialization error:")
            return False

    async def synthesize(self, text: str) -> TTSResult:
        """
        Synthesize speech from text.
        
        Args:
            text: Text to synthesize
            
        Returns:
            TTSResult with WAV audio data and metadata
        """
        if not self._initialized:
            await self.initialize()
            
        if not self._initialized:
            # Return empty result if initialization failed
            return TTSResult(
                audio=np.array([]),
                sample_rate=self.config.sample_rate,
                latency_ms=0,
                text=text,
                voice_id="failed"
            )
        
        start_time = time.time()
        
        try:
            # Run synthesis in thread executor
            loop = asyncio.get_running_loop()
            audio_tensor = await loop.run_in_executor(
                None, 
                lambda: self._tts.generate(text)  # Use generate() method for synthesis
            )
            
            # Convert PyTorch tensor to numpy array
            if hasattr(audio_tensor, 'cpu'):
                audio_numpy = audio_tensor.squeeze().cpu().numpy()
            else:
                audio_numpy = audio_tensor
            
            # Convert to bytes (if needed for compatibility) or keep as numpy array
            # For now, let's return the numpy array and update TTSResult accordingly
            wav_bytes = audio_numpy
            
            latency_ms = (time.time() - start_time) * 1000
            
            result = TTSResult(
                audio=wav_bytes,
                sample_rate=self.config.sample_rate,
                latency_ms=latency_ms,
                text=text,
                voice_id=self.config.voice_id
            )
            
            logger.debug(f"Synthesized in {latency_ms:.1f}ms: '{text[:50]}...'")
            return result
            
        except Exception as e:
            logger.error(f"TTS synthesis failed: {e}")
            logger.exception("Full synthesis error:")
            
            # Return empty result on error
            latency_ms = (time.time() - start_time) * 1000
            return TTSResult(
                audio=np.array([]),
                sample_rate=self.config.sample_rate,
                latency_ms=latency_ms,
                text=text,
                voice_id="error"
            )

    async def health_check(self) -> Tuple[bool, str]:
        """
        Check if TTS service is healthy.
        
        Returns:
            Tuple of (is_healthy: bool, status_message: str)
        """
        try:
            if not self._initialized:
                success = await self.initialize()
                if not success:
                    return False, "TTS initialization failed"
            
            # Test with short text
            result = await self.synthesize("Test")
            
            if len(result.audio) > 0:
                return True, f"ChatterboxTTS ready (voice: {self.config.voice_id}, device: {self.config.device})"
            else:
                return False, "TTS synthesis test failed"
                
        except Exception as e:
            return False, f"TTS health check failed: {e}"

    def get_stats(self) -> dict:
        """Get TTS client statistics."""
        return {
            "voice_id": self.config.voice_id,
            "device": self.config.device,
            "sample_rate": self.config.sample_rate,
            "initialized": self._initialized,
            "model_loaded": self._tts is not None
        }


# Convenience functions
async def create_tts_client(voice_id: str = DEFAULT_VOICE_ID, 
                           device: str = "cpu") -> ChatterboxTTSClient:
    """Create and health-check TTS client."""
    config = TTSConfig(voice_id=voice_id, device=device)
    client = ChatterboxTTSClient(config)
    
    is_healthy, status = await client.health_check()
    if not is_healthy:
        logger.warning(f"TTS client health check failed: {status}")
    else:
        logger.info(f"TTS client ready: {status}")
    
    return client


if __name__ == "__main__":
    async def test_tts():
        """Test TTS functionality."""
        print("Testing ChatterboxTTS with official API...")
        
        # Test CPU device
        client = await create_tts_client()
        
        # Test synthesis
        test_phrases = [
            "Hello from the CPU-only build!",
            "Testing Chatterbox TTS integration.",
            "This is Nova, your voice assistant."
        ]
        
        for phrase in test_phrases:
            print(f"\nSynthesizing: '{phrase}'")
            result = await client.synthesize(phrase)
            
            if len(result.audio) > 0:
                print(f"✅ Generated {len(result.audio)} samples in {result.latency_ms:.1f}ms")
                
                # Save audio file for testing using scipy.io.wavfile
                filename = f"test_output_{len(phrase)}.wav"
                
                # Ensure audio is in the right format for WAV (int16 or float32)
                audio = result.audio.astype(np.float32)
                wavfile.write(filename, result.sample_rate, audio)
                print(f"📁 Saved to {filename}")
            else:
                print("❌ Failed to generate audio")
        
        print(f"\n📊 Stats: {client.get_stats()}")
    
    asyncio.run(test_tts()) 