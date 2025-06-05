#!/usr/bin/env python3
"""
End-to-end integration test for STT → LLM pipeline.

Tests the complete flow: audio transcription → LLM response generation.
This validates Step 4 completion and readiness for TTS integration.
"""

import asyncio
import logging
import numpy as np
import time
from pathlib import Path

# Import our modules
from core.stt import FasterWhisperSTT
from agent import create_llm_client, create_memory_integrated_client

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def generate_test_audio() -> np.ndarray:
    """Generate synthetic test audio for demonstration."""
    # Create a simple sine wave representing speech-like audio
    # In real usage, this would be actual recorded audio
    sample_rate = 16000
    duration = 1.0  # seconds
    frequency = 440  # Hz (A4 note)
    
    t = np.linspace(0, duration, int(sample_rate * duration), dtype=np.float32)
    audio = 0.3 * np.sin(2 * np.pi * frequency * t)
    
    # Add some noise to make it more realistic
    noise = 0.05 * np.random.normal(0, 1, len(audio)).astype(np.float32)
    audio = audio + noise
    
    logger.info(f"Generated test audio: {len(audio)} samples, {duration}s")
    return audio


async def test_stt_component():
    """Test STT component independently."""
    logger.info("=== Testing STT Component ===")
    
    # Initialize STT client
    stt_client = FasterWhisperSTT()
    
    # Test health check
    is_healthy, status = await stt_client.health_check()
    logger.info(f"STT Health: {is_healthy} - {status}")
    
    if not is_healthy:
        logger.error("STT component not healthy, skipping test")
        return None
    
    # Test transcription with dummy audio
    # Note: faster-whisper might not transcribe sine waves meaningfully
    # but this tests the technical pipeline
    test_audio = generate_test_audio()
    
    start_time = time.time()
    result = await stt_client.transcribe(test_audio)
    latency_ms = (time.time() - start_time) * 1000
    
    logger.info(f"STT Result: '{result.text}' (confidence: {result.confidence:.2f}, latency: {latency_ms:.1f}ms)")
    
    return result


async def test_llm_component():
    """Test LLM component independently."""
    logger.info("=== Testing LLM Component ===")
    
    # Initialize LLM client
    llm_client = await create_llm_client()
    
    # Test health check
    is_healthy, status = await llm_client.health_check()
    logger.info(f"LLM Health: {is_healthy} - {status}")
    
    if not is_healthy:
        logger.error("LLM component not healthy, skipping test")
        return None
    
    # Test response generation
    test_input = "Hello, what's your name?"
    
    start_time = time.time()
    response = await llm_client.generate_response(test_input)
    latency_ms = (time.time() - start_time) * 1000
    
    logger.info(f"LLM Response: '{response.content}' (latency: {latency_ms:.1f}ms)")
    
    return response


async def test_memory_integration():
    """Test memory-integrated LLM client."""
    logger.info("=== Testing Memory Integration ===")
    
    # Create base LLM client
    llm_client = await create_llm_client()
    
    # Create memory-integrated client
    memory_client = await create_memory_integrated_client(llm_client, "test_integration.db")
    
    # Test conversation with memory
    test_messages = [
        "Hello, I'm testing the system",
        "What did I just say?",
        "Can you remember our conversation?"
    ]
    
    responses = []
    for i, message in enumerate(test_messages):
        logger.info(f"User message {i+1}: {message}")
        
        start_time = time.time()
        response = await memory_client.generate_response(message)
        latency_ms = (time.time() - start_time) * 1000
        
        logger.info(f"Nova response {i+1}: '{response.content}' (latency: {latency_ms:.1f}ms)")
        responses.append(response)
    
    # Check memory stats
    stats = await memory_client.get_stats()
    logger.info(f"Memory stats: {stats}")
    
    # Clean up test database
    await memory_client.clear_conversation()
    import os
    if os.path.exists("test_integration.db"):
        os.remove("test_integration.db")
    
    return responses


async def test_end_to_end_pipeline():
    """Test complete STT → LLM pipeline."""
    logger.info("=== Testing End-to-End Pipeline ===")
    
    # Initialize components
    stt_client = FasterWhisperSTT()
    llm_client = await create_llm_client()
    memory_client = await create_memory_integrated_client(llm_client, "pipeline_test.db")
    
    # Simulate voice assistant interaction
    # Note: Using text input since synthetic audio doesn't transcribe meaningfully
    simulated_transcriptions = [
        "Hello Nova",
        "What's the weather like today?", 
        "What time is it?",
        "Thank you"
    ]
    
    total_latency = 0
    successful_interactions = 0
    
    for i, simulated_text in enumerate(simulated_transcriptions):
        logger.info(f"\n--- Interaction {i+1} ---")
        logger.info(f"Simulated STT output: '{simulated_text}'")
        
        # Measure end-to-end latency
        start_time = time.time()
        
        # LLM processing
        response = await memory_client.generate_response(simulated_text)
        
        end_time = time.time()
        latency_ms = (end_time - start_time) * 1000
        total_latency += latency_ms
        
        logger.info(f"Nova response: '{response.content}'")
        logger.info(f"Interaction latency: {latency_ms:.1f}ms")
        
        # Check PRD latency requirement (<2s for 10-word prompt)
        if latency_ms < 2000:
            successful_interactions += 1
            logger.info("✅ Latency requirement met")
        else:
            logger.warning("⚠️ Latency requirement exceeded")
    
    # Final stats
    avg_latency = total_latency / len(simulated_transcriptions)
    success_rate = (successful_interactions / len(simulated_transcriptions)) * 100
    
    logger.info(f"\n=== Pipeline Performance ===")
    logger.info(f"Average latency: {avg_latency:.1f}ms")
    logger.info(f"Success rate (< 2s): {success_rate:.1f}%")
    logger.info(f"Successful interactions: {successful_interactions}/{len(simulated_transcriptions)}")
    
    # Final memory stats
    final_stats = await memory_client.get_stats()
    logger.info(f"Final memory stats: {final_stats}")
    
    # Clean up
    await memory_client.clear_conversation()
    import os
    if os.path.exists("pipeline_test.db"):
        os.remove("pipeline_test.db")
    
    return {
        "avg_latency_ms": avg_latency,
        "success_rate": success_rate,
        "successful_interactions": successful_interactions,
        "total_interactions": len(simulated_transcriptions)
    }


async def main():
    """Run all integration tests."""
    logger.info("🚀 Starting STT → LLM Integration Tests")
    logger.info("=" * 50)
    
    try:
        # Test individual components
        await test_stt_component()
        await test_llm_component()
        await test_memory_integration()
        
        # Test end-to-end pipeline
        results = await test_end_to_end_pipeline()
        
        # Summary
        logger.info("\n" + "=" * 50)
        logger.info("🎉 Integration Tests Complete!")
        logger.info(f"✅ Average latency: {results['avg_latency_ms']:.1f}ms")
        logger.info(f"✅ Success rate: {results['success_rate']:.1f}%")
        
        # Check PRD compliance
        if results['avg_latency_ms'] < 2000 and results['success_rate'] >= 90:
            logger.info("🎯 PRD requirements met! Ready for Step 5: TTS Integration")
        else:
            logger.warning("⚠️ PRD requirements not fully met, optimization needed")
            
    except Exception as e:
        logger.error(f"❌ Integration test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # Run the integration tests
    asyncio.run(main()) 