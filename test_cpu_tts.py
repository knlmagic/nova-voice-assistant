#!/usr/bin/env python3
"""Quick CPU-only test for ChatterboxTTS."""

import asyncio
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from core.tts.chatterbox_client import ChatterboxTTSClient, TTSConfig

async def test_cpu():
    print("Testing ChatterboxTTS with CPU device...")
    
    config = TTSConfig(device='cpu')
    client = ChatterboxTTSClient(config)
    
    print("Initializing...")
    success = await client.initialize()
    
    if success:
        print("✅ Initialization successful")
        
        print("Testing synthesis...")
        result = await client.synthesize('Hello world')
        
        if len(result.audio) > 0:
            print(f'✅ Success: Generated {len(result.audio)} samples in {result.latency_ms:.1f}ms')
            print(f'Sample rate: {result.sample_rate} Hz')
        else:
            print('❌ Generated empty audio')
    else:
        print('❌ Failed to initialize')

if __name__ == "__main__":
    asyncio.run(test_cpu()) 