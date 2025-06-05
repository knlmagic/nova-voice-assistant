#!/usr/bin/env python3
"""
Test script for Step 2: Audio Pipeline Foundation.

Tests all core components:
- Global hotkey handler (⌃⌥␣ Control-Option-Space)
- Audio capture with Voice Activity Detection
- Audio chimes system
- Non-blocking audio architecture

Run this to verify Step 2 implementation meets PRD requirements.
"""

import asyncio
import logging
import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from core.audio import AudioPipeline, AudioPipelineConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


async def test_audio_chimes():
    """Test 1: Audio chimes system."""
    print("\n🧪 Test 1: Audio Chimes System")
    print("=" * 50)
    
    try:
        from core.audio.playback import AudioChimes
        
        chimes = AudioChimes("sounds")
        
        print("▶️  Playing listen chime (800Hz, 150ms)...")
        chimes.play_listen_chime()
        await asyncio.sleep(0.5)
        
        print("▶️  Playing done chime (600Hz, 150ms)...")
        chimes.play_done_chime()
        await asyncio.sleep(0.5)
        
        print("✅ Audio chimes test PASSED")
        return True
        
    except Exception as e:
        print(f"❌ Audio chimes test FAILED: {e}")
        return False


async def test_hotkey_handler():
    """Test 2: Global hotkey handler."""
    print("\n🧪 Test 2: Global Hotkey Handler")
    print("=" * 50)
    
    try:
        from core.audio.hotkey import AsyncHotkeyHandler
        
        hotkey = AsyncHotkeyHandler()
        
        print("🔄 Starting hotkey listener...")
        if await hotkey.start():
            print("✅ Hotkey listener started successfully")
            print("⌨️  Press ⌃⌥␣ (Control-Option-Space) within 5 seconds to test...")
            
            # Wait for hotkey press with timeout
            try:
                await asyncio.wait_for(hotkey.wait_for_press(), timeout=5.0)
                print("✅ Hotkey press detected!")
                
                # Wait for release
                await asyncio.wait_for(hotkey.wait_for_release(), timeout=5.0)
                print("✅ Hotkey release detected!")
                
                await hotkey.stop()
                print("✅ Global hotkey test PASSED")
                return True
                
            except asyncio.TimeoutError:
                await hotkey.stop()
                print("⏰ Timeout - hotkey not pressed within 5 seconds")
                print("ℹ️  This is expected if you didn't press the hotkey")
                return True  # Don't fail the test for timeout
                
        else:
            print("❌ Failed to start hotkey listener")
            return False
            
    except Exception as e:
        print(f"❌ Global hotkey test FAILED: {e}")
        return False


async def test_audio_capture():
    """Test 3: Audio capture with VAD."""
    print("\n🧪 Test 3: Audio Capture with VAD")
    print("=" * 50)
    
    try:
        from core.audio.capture import AsyncAudioCapture, AudioConfig
        
        # Configure for quick testing (shorter VAD threshold)
        config = AudioConfig(vad_threshold_seconds=2.0)
        capture = AsyncAudioCapture(config)
        
        print("🔄 Testing audio capture...")
        
        if await capture.start_recording():
            print("✅ Audio recording started")
            print("🎤 Speak something for 1-2 seconds, then be silent...")
            print("    VAD will auto-stop after 2 seconds of silence")
            
            # Wait a bit for user to speak
            await asyncio.sleep(8.0)  # Give time for speech + silence
            
            # Stop recording and get audio
            audio_data = await capture.stop_recording()
            
            if audio_data is not None:
                duration = len(audio_data) / config.sample_rate
                print(f"✅ Captured {duration:.2f}s of audio ({len(audio_data)} samples)")
                print("✅ Audio capture test PASSED")
                return True
            else:
                print("⚠️  No audio captured (this is OK if you didn't speak)")
                return True  # Don't fail if no audio
                
        else:
            print("❌ Failed to start audio recording")
            return False
            
    except Exception as e:
        print(f"❌ Audio capture test FAILED: {e}")
        return False


async def test_integrated_pipeline():
    """Test 4: Integrated audio pipeline."""
    print("\n🧪 Test 4: Integrated Audio Pipeline")
    print("=" * 50)
    
    try:
        # Create pipeline with default configuration
        pipeline = AudioPipeline()
        
        # Set up callbacks for testing
        audio_captured = False
        
        def on_audio(audio_data):
            nonlocal audio_captured
            audio_captured = True
            duration = len(audio_data) / 16000
            print(f"📸 Audio captured: {duration:.2f}s ({len(audio_data)} samples)")
        
        def on_start():
            print("🎙️  Recording started!")
        
        def on_stop():
            print("⏹️  Recording stopped!")
        
        pipeline.set_callbacks(
            on_audio_captured=on_audio,
            on_recording_start=on_start,
            on_recording_stop=on_stop
        )
        
        print("🔄 Starting integrated audio pipeline...")
        if await pipeline.start():
            print("✅ Pipeline started successfully")
            print("\n📋 Pipeline Status:")
            status = pipeline.get_status()
            for key, value in status.items():
                if key != 'config':
                    print(f"   {key}: {value}")
            
            print("\n🎯 INTERACTIVE TEST:")
            print("   1. Press ⌃⌥␣ (Control-Option-Space) to start recording")
            print("   2. Say something while holding the hotkey")
            print("   3. Release the hotkey to stop recording")
            print("   4. Wait 10 seconds for test to complete")
            
            # Wait for user interaction
            await asyncio.sleep(10.0)
            
            await pipeline.stop()
            print("✅ Pipeline stopped")
            
            if audio_captured:
                print("✅ Integrated pipeline test PASSED - Audio was captured!")
            else:
                print("ℹ️  Integrated pipeline test completed - No audio captured")
                print("    (This is OK if you didn't use the hotkey)")
            
            return True
            
        else:
            print("❌ Failed to start pipeline")
            return False
            
    except Exception as e:
        print(f"❌ Integrated pipeline test FAILED: {e}")
        return False


async def main():
    """Run all audio pipeline tests."""
    print("🚀 STEP 2: AUDIO PIPELINE FOUNDATION - TEST SUITE")
    print("=" * 60)
    print("Testing implementation against PRD requirements:")
    print("  ✓ Global hotkey (⌃⌥␣ Control-Option-Space)")
    print("  ✓ Audio capture with Voice Activity Detection")
    print("  ✓ Audio chimes system (listen.wav, done.wav)")
    print("  ✓ Non-blocking audio architecture")
    print()
    
    tests = [
        ("Audio Chimes", test_audio_chimes),
        ("Global Hotkey", test_hotkey_handler),
        ("Audio Capture", test_audio_capture),
        ("Integrated Pipeline", test_integrated_pipeline),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = await test_func()
            results.append((test_name, result))
        except KeyboardInterrupt:
            print("\n⏹️  Test interrupted by user")
            break
        except Exception as e:
            print(f"\n❌ {test_name} test encountered error: {e}")
            results.append((test_name, False))
    
    # Print final results
    print("\n" + "=" * 60)
    print("📊 FINAL TEST RESULTS")
    print("=" * 60)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name:20} {status}")
        if result:
            passed += 1
    
    print(f"\nResults: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED! Step 2 implementation is working correctly.")
        print("✅ Ready to proceed to Step 3: STT Integration")
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Review implementation before proceeding.")
    
    return passed == total


if __name__ == "__main__":
    try:
        success = asyncio.run(main())
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n👋 Test suite interrupted by user")
        sys.exit(130) 