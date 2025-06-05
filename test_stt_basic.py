#!/usr/bin/env python3
"""
Basic test script for STT functionality.
"""

from core.stt import WhisperConfig, WhisperSTT
import numpy as np
import logging

def main():
    # Enable logging
    logging.basicConfig(level=logging.INFO)
    
    print('🎤 Testing WhisperSTT initialization...')
    
    # Create configuration
    config = WhisperConfig()
    print(f'Model path: {config.model_path}')
    print(f'Sample rate: {config.sample_rate}')
    print(f'Confidence threshold: {config.confidence_threshold}')
    
    try:
        # Create STT instance
        stt = WhisperSTT(config)
        print('✅ Created STT instance')
        
        # Initialize model
        print('🔄 Loading Whisper model...')
        if stt.initialize():
            print('✅ Whisper model loaded successfully')
            
            # Test with simple audio
            print('🔄 Testing transcription with test audio...')
            sample_rate = 16000
            duration = 1.0
            samples = int(duration * sample_rate)
            
            # Generate test audio (noise)
            test_audio = np.random.normal(0, 0.1, samples).astype(np.int16)
            
            result = stt.transcribe_audio(test_audio, sample_rate)
            
            print('📊 Transcription result:')
            print(f'  Text: "{result.get("text", "")}"')
            print(f'  Confidence: {result.get("confidence", 0):.3f}')
            print(f'  Processing time: {result.get("processing_time", 0):.3f}s')
            print(f'  Success: {result.get("success", False)}')
            
            if result.get("success"):
                print('✅ STT transcription test passed')
            else:
                print(f'❌ STT transcription failed: {result.get("error", "Unknown error")}')
            
            # Test performance stats
            stats = stt.get_performance_stats()
            print('📈 Performance stats:')
            print(f'  Total transcriptions: {stats["total_transcriptions"]}')
            print(f'  Average processing time: {stats["average_processing_time"]:.3f}s')
            print(f'  Target latency met: {stats["target_latency_met"]}')
            
            stt.cleanup()
            print('✅ STT test completed successfully')
            
        else:
            print('❌ Failed to initialize Whisper model')
            return False
            
    except Exception as e:
        print(f'❌ Error during STT test: {e}')
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1) 