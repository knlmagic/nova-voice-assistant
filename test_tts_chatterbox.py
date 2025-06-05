#!/usr/bin/env python3
"""
Test script for ChatterboxTTS integration.

Validates the ChatterboxTTSClient with correct API usage based on source code documentation.
"""

import asyncio
import logging
import sys
from pathlib import Path

# Add core modules to path
sys.path.append(str(Path(__file__).parent))

from core.tts.chatterbox_client import (
    ChatterboxTTSClient, 
    TTSConfig, 
    AsyncTTSPlayer,
    create_tts_client
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def test_tts_config():
    """Test TTSConfig device auto-detection."""
    print("\n" + "="*50)
    print("TEST: TTSConfig Device Auto-Detection")
    print("="*50)
    
    # Test auto device detection
    config = TTSConfig()
    print(f"Auto-detected device: {config.device}")
    print(f"Sample rate: {config.sample_rate}")
    print(f"Exaggeration: {config.exaggeration}")
    print(f"CFG weight: {config.cfg_weight}")
    print(f"Temperature: {config.temperature}")
    
    # Test manual device setting
    cpu_config = TTSConfig(device="cpu")
    print(f"Manual CPU device: {cpu_config.device}")
    
    return True


async def test_tts_initialization():
    """Test ChatterboxTTS model initialization."""
    print("\n" + "="*50)
    print("TEST: ChatterboxTTS Initialization")
    print("="*50)
    
    try:
        # Create client with default config
        client = ChatterboxTTSClient()
        
        print("✓ ChatterboxTTSClient created")
        print(f"Stats before init: {client.get_stats()}")
        
        # Test initialization
        print("Initializing ChatterboxTTS model...")
        success = await client.initialize()
        
        if success:
            print("✓ ChatterboxTTS initialized successfully")
            print(f"Stats after init: {client.get_stats()}")
            return client
        else:
            print("✗ ChatterboxTTS initialization failed")
            return None
            
    except Exception as e:
        print(f"✗ Error during initialization: {e}")
        logger.exception("Initialization error details:")
        return None


async def test_tts_health_check(client):
    """Test TTS health check functionality."""
    print("\n" + "="*50)
    print("TEST: ChatterboxTTS Health Check")
    print("="*50)
    
    try:
        is_healthy, status = await client.health_check()
        
        if is_healthy:
            print(f"✓ Health check passed: {status}")
            return True
        else:
            print(f"✗ Health check failed: {status}")
            return False
            
    except Exception as e:
        print(f"✗ Health check error: {e}")
        logger.exception("Health check error details:")
        return False


async def test_tts_synthesis(client):
    """Test text-to-speech synthesis."""
    print("\n" + "="*50)
    print("TEST: ChatterboxTTS Synthesis")
    print("="*50)
    
    test_phrases = [
        "Hello, I'm Nova, your voice assistant.",
        "Testing ChatterboxTTS synthesis.",
        "The weather is nice today.",
        "How can I help you?"
    ]
    
    results = []
    
    for i, phrase in enumerate(test_phrases, 1):
        try:
            print(f"\n{i}. Synthesizing: '{phrase}'")
            
            result = await client.synthesize(phrase)
            
            print(f"   ✓ Generated {len(result.audio)} samples")
            print(f"   ✓ Sample rate: {result.sample_rate} Hz")
            print(f"   ✓ Latency: {result.latency_ms:.1f} ms")
            print(f"   ✓ Voice ID: {result.voice_id}")
            
            results.append(result)
            
        except Exception as e:
            print(f"   ✗ Synthesis failed: {e}")
            logger.exception(f"Synthesis error for phrase {i}:")
    
    if results:
        avg_latency = sum(r.latency_ms for r in results) / len(results)
        total_audio = sum(len(r.audio) for r in results)
        print(f"\nSummary:")
        print(f"✓ {len(results)}/{len(test_phrases)} phrases synthesized")
        print(f"✓ Average latency: {avg_latency:.1f} ms")
        print(f"✓ Total audio samples: {total_audio}")
        
        return results
    else:
        print("\n✗ No successful synthesis")
        return []


async def test_async_player(client):
    """Test AsyncTTSPlayer functionality."""
    print("\n" + "="*50)
    print("TEST: AsyncTTSPlayer")
    print("="*50)
    
    try:
        # Create async player
        player = AsyncTTSPlayer(client)
        print("✓ AsyncTTSPlayer created")
        
        # Test playback queue
        test_phrases = [
            "Testing async playback.",
            "First message.",
            "Second message.",
            "Third message."
        ]
        
        print(f"Queueing {len(test_phrases)} phrases for playback...")
        
        for i, phrase in enumerate(test_phrases, 1):
            print(f"  {i}. Queueing: '{phrase}'")
            result = await player.play_text(phrase)
            print(f"     Synthesized in {result.latency_ms:.1f}ms")
            
            # Brief pause between phrases
            await asyncio.sleep(0.2)
        
        print("✓ All phrases queued for playback")
        
        # Wait for playback to complete
        print("Waiting for playback to complete...")
        await asyncio.sleep(2.0)  # Give time for playback
        
        # Stop playback
        await player.stop_playback()
        print("✓ Playback stopped")
        
        return True
        
    except Exception as e:
        print(f"✗ AsyncTTSPlayer test failed: {e}")
        logger.exception("AsyncTTSPlayer error details:")
        return False


async def test_convenience_function():
    """Test create_tts_client convenience function."""
    print("\n" + "="*50)
    print("TEST: create_tts_client() Convenience Function")
    print("="*50)
    
    try:
        # Test with default config
        print("Creating TTS client with default config...")
        client1 = await create_tts_client()
        print("✓ Default client created")
        
        # Test with custom config
        print("Creating TTS client with custom config...")
        custom_config = TTSConfig(
            device="cpu",
            exaggeration=0.7,
            temperature=0.9
        )
        client2 = await create_tts_client(custom_config)
        print("✓ Custom client created")
        
        # Show stats comparison
        print(f"Default stats: {client1.get_stats()}")
        print(f"Custom stats: {client2.get_stats()}")
        
        return True
        
    except Exception as e:
        print(f"✗ Convenience function test failed: {e}")
        logger.exception("Convenience function error details:")
        return False


async def main():
    """Run comprehensive ChatterboxTTS test suite."""
    print("🎤 ChatterboxTTS Integration Test Suite")
    print("=" * 60)
    
    # Test results tracking
    test_results = []
    
    # Test 1: Configuration
    result = await test_tts_config()
    test_results.append(("Config Test", result))
    
    # Test 2: Initialization 
    client = await test_tts_initialization()
    test_results.append(("Initialization", client is not None))
    
    if client:
        # Test 3: Health Check
        health_ok = await test_tts_health_check(client)
        test_results.append(("Health Check", health_ok))
        
        if health_ok:
            # Test 4: Synthesis
            synthesis_results = await test_tts_synthesis(client)
            test_results.append(("Synthesis", len(synthesis_results) > 0))
            
            # Test 5: Async Player
            player_ok = await test_async_player(client)
            test_results.append(("Async Player", player_ok))
    
    # Test 6: Convenience Function
    convenience_ok = await test_convenience_function()
    test_results.append(("Convenience Function", convenience_ok))
    
    # Print final results
    print("\n" + "="*60)
    print("🎯 TEST RESULTS SUMMARY")
    print("="*60)
    
    passed = 0
    total = len(test_results)
    
    for test_name, result in test_results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {test_name}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("🎉 All tests passed! ChatterboxTTS integration is working correctly.")
        return True
    else:
        print("⚠️  Some tests failed. Check the logs above for details.")
        return False


if __name__ == "__main__":
    # Run the test suite
    success = asyncio.run(main())
    
    # Exit with appropriate code
    sys.exit(0 if success else 1) 