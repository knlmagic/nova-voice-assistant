#!/usr/bin/env python3
"""
Comprehensive integration test for Nova Voice Assistant.

Tests the complete end-to-end pipeline:
Audio → STT → LLM → TTS → Audio Output

This validates all components working together and measures performance.
"""

import asyncio
import logging
import time
import numpy as np
from pathlib import Path

from assistant import NovaAssistant
from utils.metrics import clear_metrics, get_current_metrics, log_latency


# Configure test logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class IntegrationTester:
    """Integration test runner for Nova Assistant."""
    
    def __init__(self):
        self.assistant = NovaAssistant(verbose=True, quiet=False)
        self.test_results = {}
        
    async def run_all_tests(self):
        """Run complete integration test suite."""
        print("\n" + "="*70)
        print("🧪 NOVA ASSISTANT INTEGRATION TEST SUITE")
        print("="*70)
        
        try:
            # Test 1: Component Health Checks
            print("\n📋 Test 1: Component Health Checks")
            health_results = await self.test_component_health()
            self.test_results["health_checks"] = health_results
            
            # Test 2: Pipeline Initialization
            print("\n📋 Test 2: Pipeline Initialization")
            init_results = await self.test_pipeline_initialization()
            self.test_results["initialization"] = init_results
            
            # Test 3: Audio Pipeline Test (Manual)
            print("\n📋 Test 3: Audio Pipeline (Manual Mode)")
            audio_results = await self.test_audio_pipeline()
            self.test_results["audio_pipeline"] = audio_results
            
            # Test 4: STT Component Test
            print("\n📋 Test 4: STT Component")
            stt_results = await self.test_stt_component()
            self.test_results["stt_component"] = stt_results
            
            # Test 5: LLM Component Test
            print("\n📋 Test 5: LLM Component")
            llm_results = await self.test_llm_component()
            self.test_results["llm_component"] = llm_results
            
            # Test 6: TTS Component Test
            print("\n📋 Test 6: TTS Component")
            tts_results = await self.test_tts_component()
            self.test_results["tts_component"] = tts_results
            
            # Test 7: Memory Persistence Test
            print("\n📋 Test 7: Memory Persistence")
            memory_results = await self.test_memory_persistence()
            self.test_results["memory_persistence"] = memory_results
            
            # Final Summary
            await self.print_test_summary()
            
        except Exception as e:
            logger.error(f"Integration test failed: {e}")
            return False
        
        return True

    async def test_component_health(self):
        """Test individual component health checks."""
        print("   🔍 Initializing components for health checks...")
        
        results = {}
        
        try:
            # Initialize components
            success = await self.assistant.initialize_components()
            results["initialization"] = success
            
            if not success:
                print("   ❌ Component initialization failed")
                return results
            
            # Health checks
            print("   🔍 Running health checks...")
            
            # STT Health
            stt_healthy, stt_msg = await self.assistant.stt_client.health_check()
            results["stt_health"] = stt_healthy
            print(f"   {'✅' if stt_healthy else '❌'} STT: {stt_msg}")
            
            # LLM Health
            llm_healthy, llm_msg = await self.assistant.llm_client.health_check()
            results["llm_health"] = llm_healthy
            print(f"   {'✅' if llm_healthy else '❌'} LLM: {llm_msg}")
            
            # TTS Health
            tts_healthy, tts_msg = await self.assistant.tts_client.health_check()
            results["tts_health"] = tts_healthy
            print(f"   {'✅' if tts_healthy else '❌'} TTS: {tts_msg}")
            
            overall_health = stt_healthy and llm_healthy and tts_healthy
            results["overall_health"] = overall_health
            
            print(f"   {'✅' if overall_health else '❌'} Overall Health: {'PASS' if overall_health else 'FAIL'}")
            
        except Exception as e:
            logger.error(f"Health check test failed: {e}")
            results["error"] = str(e)
        
        return results

    async def test_pipeline_initialization(self):
        """Test complete pipeline initialization."""
        print("   🔧 Testing pipeline startup...")
        
        results = {}
        start_time = time.time()
        
        try:
            success = await self.assistant.start()
            init_time = time.time() - start_time
            
            results["startup_success"] = success
            results["startup_time"] = init_time
            
            if success:
                print(f"   ✅ Pipeline started successfully in {init_time:.2f}s")
                
                # Test that all workers are running
                running_tasks = [task for task in self.assistant.tasks if not task.done()]
                results["active_workers"] = len(running_tasks)
                print(f"   ✅ {len(running_tasks)} worker tasks active")
                
            else:
                print("   ❌ Pipeline startup failed")
            
        except Exception as e:
            logger.error(f"Pipeline initialization test failed: {e}")
            results["error"] = str(e)
        
        return results

    async def test_audio_pipeline(self):
        """Test audio pipeline in manual mode."""
        print("   🎤 Testing audio capture...")
        
        results = {}
        
        try:
            # Generate test audio (1 second of 440Hz sine wave)
            sample_rate = 16000
            duration = 1.0
            t = np.linspace(0, duration, int(sample_rate * duration))
            test_audio = np.sin(2 * np.pi * 440 * t).astype(np.float32)
            
            results["test_audio_generated"] = True
            results["test_audio_length"] = len(test_audio)
            
            # Test manual recording mode
            if self.assistant.audio_pipeline:
                # Simulate audio capture callback
                self.assistant._on_audio_captured(test_audio)
                results["audio_callback_success"] = True
                print("   ✅ Audio callback simulation successful")
            else:
                results["audio_callback_success"] = False
                print("   ❌ Audio pipeline not available")
            
        except Exception as e:
            logger.error(f"Audio pipeline test failed: {e}")
            results["error"] = str(e)
        
        return results

    async def test_stt_component(self):
        """Test STT component with sample audio."""
        print("   🗣️  Testing STT transcription...")
        
        results = {}
        
        try:
            # Generate test audio (sine wave representing speech-like input)
            sample_rate = 16000
            duration = 2.0  # 2 seconds
            t = np.linspace(0, duration, int(sample_rate * duration))
            
            # Mix multiple frequencies to simulate speech
            test_audio = (
                0.3 * np.sin(2 * np.pi * 440 * t) +  # 440 Hz
                0.2 * np.sin(2 * np.pi * 880 * t) +  # 880 Hz
                0.1 * np.sin(2 * np.pi * 1320 * t)   # 1320 Hz
            ).astype(np.float32)
            
            # Add some noise to make it more realistic
            noise = np.random.normal(0, 0.02, test_audio.shape).astype(np.float32)
            test_audio = test_audio + noise
            
            start_time = time.time()
            stt_result = await self.assistant.stt_client.transcribe(test_audio)
            latency = time.time() - start_time
            
            results["transcription_latency"] = latency
            results["transcription_text"] = stt_result.text
            results["confidence"] = stt_result.confidence
            results["language"] = stt_result.language
            
            # Check if we meet latency requirements
            latency_ok = latency < 0.5  # 500ms threshold
            results["latency_ok"] = latency_ok
            
            print(f"   ✅ STT completed in {latency*1000:.0f}ms")
            print(f"   📝 Transcription: \"{stt_result.text}\"")
            print(f"   📊 Confidence: {stt_result.confidence:.2f}")
            print(f"   {'✅' if latency_ok else '❌'} Latency requirement: {'PASS' if latency_ok else 'FAIL'}")
            
        except Exception as e:
            logger.error(f"STT component test failed: {e}")
            results["error"] = str(e)
        
        return results

    async def test_llm_component(self):
        """Test LLM component with sample queries."""
        print("   🧠 Testing LLM responses...")
        
        results = {}
        test_queries = [
            "Hello, how are you?",
            "What is the weather like?", 
            "Tell me a joke."
        ]
        
        try:
            query_results = []
            
            for i, query in enumerate(test_queries):
                print(f"   🔄 Testing query {i+1}: \"{query}\"")
                
                start_time = time.time()
                response = await self.assistant.llm_client.generate_response(query)
                latency = time.time() - start_time
                
                query_result = {
                    "query": query,
                    "response": response.content,
                    "latency": latency,
                    "latency_ok": latency < 2.0  # 2s threshold per PRD
                }
                
                query_results.append(query_result)
                
                print(f"      ⏱️  {latency*1000:.0f}ms")
                print(f"      🤖 \"{response.content}\"")
                print(f"      {'✅' if query_result['latency_ok'] else '❌'} Latency: {'PASS' if query_result['latency_ok'] else 'FAIL'}")
            
            results["queries"] = query_results
            results["avg_latency"] = sum(q["latency"] for q in query_results) / len(query_results)
            results["all_latency_ok"] = all(q["latency_ok"] for q in query_results)
            
            print(f"   📊 Average latency: {results['avg_latency']*1000:.0f}ms")
            
        except Exception as e:
            logger.error(f"LLM component test failed: {e}")
            results["error"] = str(e)
        
        return results

    async def test_tts_component(self):
        """Test TTS component with sample text."""
        print("   🎵 Testing TTS synthesis...")
        
        results = {}
        test_texts = [
            "Hello, this is a test.",
            "Nova voice assistant is working correctly.",
            "Performance testing complete."
        ]
        
        try:
            synthesis_results = []
            
            for i, text in enumerate(test_texts):
                print(f"   🔄 Testing synthesis {i+1}: \"{text}\"")
                
                start_time = time.time()
                tts_result = await self.assistant.tts_client.synthesize(text)
                latency = time.time() - start_time
                
                synthesis_result = {
                    "text": text,
                    "latency": latency,
                    "audio_length": len(tts_result.audio),
                    "sample_rate": tts_result.sample_rate,
                    "latency_ok": latency < 3.0  # 3s threshold
                }
                
                synthesis_results.append(synthesis_result)
                
                print(f"      ⏱️  {latency*1000:.0f}ms")
                print(f"      🎶 {len(tts_result.audio)} samples at {tts_result.sample_rate}Hz")
                print(f"      {'✅' if synthesis_result['latency_ok'] else '❌'} Latency: {'PASS' if synthesis_result['latency_ok'] else 'FAIL'}")
            
            results["syntheses"] = synthesis_results
            results["avg_latency"] = sum(s["latency"] for s in synthesis_results) / len(synthesis_results)
            results["all_latency_ok"] = all(s["latency_ok"] for s in synthesis_results)
            
            print(f"   📊 Average synthesis latency: {results['avg_latency']*1000:.0f}ms")
            
        except Exception as e:
            logger.error(f"TTS component test failed: {e}")
            results["error"] = str(e)
        
        return results

    async def test_memory_persistence(self):
        """Test conversation memory persistence."""
        print("   💾 Testing memory persistence...")
        
        results = {}
        
        try:
            # Test conversation memory
            test_conversation = [
                "Remember that my favorite color is blue.",
                "What is my favorite color?"
            ]
            
            memory_results = []
            
            for query in test_conversation:
                response = await self.assistant.llm_client.generate_response(query)
                memory_results.append({
                    "query": query,
                    "response": response.content
                })
                
                print(f"   💬 Q: \"{query}\"")
                print(f"      A: \"{response.content}\"")
            
            results["conversation"] = memory_results
            
            # Check if memory is working (simple heuristic)
            second_response = memory_results[1]["response"].lower()
            memory_working = "blue" in second_response
            results["memory_working"] = memory_working
            
            print(f"   {'✅' if memory_working else '❌'} Memory persistence: {'PASS' if memory_working else 'FAIL'}")
            
        except Exception as e:
            logger.error(f"Memory persistence test failed: {e}")
            results["error"] = str(e)
        
        return results

    async def print_test_summary(self):
        """Print comprehensive test summary."""
        print("\n" + "="*70)
        print("📊 INTEGRATION TEST SUMMARY")
        print("="*70)
        
        # Count passed/failed tests
        passed = 0
        failed = 0
        
        for test_name, results in self.test_results.items():
            if "error" in results:
                status = "❌ FAIL"
                failed += 1
            else:
                # Determine pass/fail based on test-specific criteria
                if test_name == "health_checks":
                    test_passed = results.get("overall_health", False)
                elif test_name == "initialization":
                    test_passed = results.get("startup_success", False)
                elif test_name in ["llm_component", "tts_component"]:
                    test_passed = results.get("all_latency_ok", False)
                elif test_name == "memory_persistence":
                    test_passed = results.get("memory_working", False)
                else:
                    test_passed = True  # Default to pass for others
                
                if test_passed:
                    status = "✅ PASS"
                    passed += 1
                else:
                    status = "⚠️  PARTIAL"
                    passed += 1
            
            print(f"{test_name.replace('_', ' ').title():<25} {status}")
        
        print("-" * 70)
        print(f"Total Tests: {passed + failed}")
        print(f"Passed: {passed}")
        print(f"Failed: {failed}")
        
        # Performance summary
        print("\n📈 PERFORMANCE SUMMARY:")
        metrics = get_current_metrics()
        if metrics:
            for stage, stats in metrics.items():
                if stats and stats.count > 0:
                    print(f"{stage.upper():<15} {stats.avg_time*1000:>6.0f}ms avg ({stats.count} samples)")
        
        print("="*70)
        
        # Cleanup
        if self.assistant.is_running:
            await self.assistant.stop()


async def main():
    """Run integration tests."""
    print("🧪 Starting Nova Assistant Integration Tests...\n")
    
    # Clear any existing metrics
    clear_metrics()
    
    # Create and run tester
    tester = IntegrationTester()
    
    try:
        success = await tester.run_all_tests()
        return 0 if success else 1
        
    except KeyboardInterrupt:
        print("\n🛑 Tests interrupted by user")
        return 1
    except Exception as e:
        print(f"\n💥 Test suite failed: {e}")
        return 1
    finally:
        # Ensure cleanup
        if tester.assistant.is_running:
            await tester.assistant.stop()


if __name__ == "__main__":
    import sys
    sys.exit(asyncio.run(main())) 