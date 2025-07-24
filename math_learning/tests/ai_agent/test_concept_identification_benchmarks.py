"""
Performance Benchmark Tests for Concept Identification System

This module contains performance benchmarks and stress tests for the
concept identification system to ensure it meets performance requirements.
"""

import pytest
import asyncio
import time
import statistics
import random
import string
from typing import List, Dict, Any
from unittest.mock import Mock
import re

from math_learning.ai_agent.tools.concept_identification_system import (
    ConceptIdentificationSystem,
    ConceptIdentification
)


class TestPerformanceBenchmarks:
    """Performance benchmark tests."""
    
    @pytest.fixture
    def system(self):
        """Create system for benchmarking."""
        return ConceptIdentificationSystem(llm=None)
    
    @pytest.fixture
    def sample_exercises(self):
        """Generate sample exercises for benchmarking."""
        exercises = [
            # Algebra exercises
            "Solve for x: 2x + 5 = 13",
            "Find the value of y: 3y - 7 = 14",
            "Simplify: 4x + 3x - 2x",
            "Factor: x² - 5x + 6",
            "Graph the line: y = 2x + 3",
            
            # Geometry exercises
            "Find the area of a triangle with base 8 and height 6",
            "Calculate the perimeter of a rectangle with length 10 and width 7",
            "What is the circumference of a circle with radius 5?",
            "Find the volume of a cube with side length 4",
            "In triangle ABC, if angle A = 60° and B = 70°, find C",
            
            # Calculus exercises
            "Find the derivative of f(x) = x³ + 2x² - x + 1",
            "Evaluate the integral: ∫(2x + 3)dx",
            "Find the limit: lim(x→0) sin(x)/x",
            "Use the chain rule to find dy/dx if y = (3x + 1)²",
            "Find the area under y = x² from x = 0 to x = 2",
            
            # Statistics exercises
            "Calculate the mean of: 12, 15, 18, 22, 25",
            "Find the standard deviation of: 2, 4, 6, 8, 10",
            "What is the probability of rolling two 6s with two dice?",
            "In a normal distribution with μ=100, σ=15, find P(85 < X < 115)",
            "Test if μ = 50 with sample mean = 52, n = 30, σ = 8",
            
            # Word problems
            "Sarah has 3 times as many apples as John. If John has 8 apples, how many does Sarah have?",
            "A car travels 240 miles in 4 hours. What is its average speed?",
            "The sum of two consecutive integers is 47. Find the integers.",
            "A rectangle has length 3 more than twice its width. If width is w, find perimeter.",
            "If 30% of a number is 45, what is the number?"
        ]
        return exercises
    
    @pytest.mark.asyncio
    async def test_single_exercise_latency(self, system, sample_exercises):
        """Test latency for analyzing single exercises."""
        latencies = []
        
        for exercise in sample_exercises[:10]:  # Test first 10
            start_time = time.perf_counter()
            result = await system.identify_concepts(exercise, "text", use_llm=False)
            end_time = time.perf_counter()
            
            latency = end_time - start_time
            latencies.append(latency)
            
            # Verify result is valid
            assert isinstance(result, ConceptIdentification)
            assert result.course != "Unknown"
        
        # Calculate statistics
        avg_latency = statistics.mean(latencies)
        median_latency = statistics.median(latencies)
        max_latency = max(latencies)
        min_latency = min(latencies)
        
        print(f"\n📊 Single Exercise Latency Benchmarks:")
        print(f"   Average: {avg_latency*1000:.2f}ms")
        print(f"   Median:  {median_latency*1000:.2f}ms")
        print(f"   Min:     {min_latency*1000:.2f}ms")
        print(f"   Max:     {max_latency*1000:.2f}ms")
        
        # Performance requirements
        assert avg_latency < 0.1, f"Average latency {avg_latency:.3f}s exceeds 100ms limit"
        assert max_latency < 0.2, f"Max latency {max_latency:.3f}s exceeds 200ms limit"
    
    @pytest.mark.asyncio
    async def test_batch_processing_throughput(self, system, sample_exercises):
        """Test throughput for batch processing."""
        batch_sizes = [5, 10, 20, 50]
        
        for batch_size in batch_sizes:
            # Create batch
            batch = [
                {"content": random.choice(sample_exercises), "content_type": "text"}
                for _ in range(batch_size)
            ]
            
            # Measure batch processing time
            start_time = time.perf_counter()
            results = await system.batch_identify_concepts(batch, use_llm=False)
            end_time = time.perf_counter()
            
            total_time = end_time - start_time
            throughput = batch_size / total_time  # exercises per second
            avg_time_per_exercise = total_time / batch_size
            
            print(f"\n📈 Batch Size {batch_size}:")
            print(f"   Total Time: {total_time:.3f}s")
            print(f"   Throughput: {throughput:.1f} exercises/sec")
            print(f"   Avg per exercise: {avg_time_per_exercise*1000:.2f}ms")
            
            # Verify all results are valid
            assert len(results) == batch_size
            assert all(isinstance(r, ConceptIdentification) for r in results)
            
            # Performance requirements
            assert avg_time_per_exercise < 0.05, f"Batch processing too slow: {avg_time_per_exercise:.3f}s per exercise"
            assert throughput > 20, f"Throughput too low: {throughput:.1f} exercises/sec"
    
    @pytest.mark.asyncio
    async def test_memory_usage_under_load(self, system):
        """Test memory usage during sustained load."""
        import gc
        import psutil
        import os
        
        # Get initial memory usage
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Generate many exercises
        exercises = []
        for i in range(100):
            # Generate varied exercises
            templates = [
                f"Solve for x: {random.randint(1,10)}x + {random.randint(1,20)} = {random.randint(1,50)}",
                f"Find the area of a rectangle with length {random.randint(1,20)} and width {random.randint(1,20)}",
                f"Calculate: {random.randint(1,100)} + {random.randint(1,100)} - {random.randint(1,50)}",
                f"Find the derivative of f(x) = {random.randint(1,5)}x² + {random.randint(1,10)}x",
                f"What is the mean of these numbers: {', '.join(str(random.randint(1,20)) for _ in range(5))}"
            ]
            exercises.append(random.choice(templates))
        
        # Process exercises in batches
        batch_size = 10
        memory_readings = []
        
        for i in range(0, len(exercises), batch_size):
            batch = exercises[i:i+batch_size]
            batch_data = [{"content": ex, "content_type": "text"} for ex in batch]
            
            # Process batch
            results = await system.batch_identify_concepts(batch_data, use_llm=False)
            
            # Force garbage collection
            gc.collect()
            
            # Record memory usage
            current_memory = process.memory_info().rss / 1024 / 1024  # MB
            memory_readings.append(current_memory)
            
            # Verify results
            assert len(results) == len(batch)
        
        final_memory = memory_readings[-1]
        max_memory = max(memory_readings)
        memory_growth = final_memory - initial_memory
        
        print(f"\n🧠 Memory Usage Under Load:")
        print(f"   Initial: {initial_memory:.1f} MB")
        print(f"   Final:   {final_memory:.1f} MB")
        print(f"   Max:     {max_memory:.1f} MB")
        print(f"   Growth:  {memory_growth:.1f} MB")
        
        # Memory requirements
        assert memory_growth < 50, f"Memory growth {memory_growth:.1f}MB exceeds 50MB limit"
        assert max_memory - initial_memory < 100, f"Peak memory usage {max_memory - initial_memory:.1f}MB exceeds 100MB limit"
    
    @pytest.mark.asyncio
    async def test_concurrent_processing(self, system, sample_exercises):
        """Test concurrent processing of multiple requests."""
        concurrent_requests = 10
        
        async def process_exercise(exercise_text: str, request_id: int):
            """Process a single exercise with timing."""
            start_time = time.perf_counter()
            result = await system.identify_concepts(exercise_text, "text", use_llm=False)
            end_time = time.perf_counter()
            
            return {
                "request_id": request_id,
                "latency": end_time - start_time,
                "result": result
            }
        
        # Create concurrent tasks
        tasks = []
        for i in range(concurrent_requests):
            exercise = random.choice(sample_exercises)
            task = process_exercise(exercise, i)
            tasks.append(task)
        
        # Execute concurrently
        start_time = time.perf_counter()
        results = await asyncio.gather(*tasks)
        end_time = time.perf_counter()
        
        total_time = end_time - start_time
        latencies = [r["latency"] for r in results]
        avg_latency = statistics.mean(latencies)
        max_latency = max(latencies)
        
        print(f"\n🔄 Concurrent Processing (n={concurrent_requests}):")
        print(f"   Total Time: {total_time:.3f}s")
        print(f"   Avg Latency: {avg_latency*1000:.2f}ms")
        print(f"   Max Latency: {max_latency*1000:.2f}ms")
        print(f"   Throughput: {concurrent_requests/total_time:.1f} req/sec")
        
        # Verify all requests completed successfully
        assert len(results) == concurrent_requests
        assert all(isinstance(r["result"], ConceptIdentification) for r in results)
        
        # Performance requirements for concurrent processing
        assert avg_latency < 0.15, f"Concurrent avg latency {avg_latency:.3f}s too high"
        assert max_latency < 0.3, f"Concurrent max latency {max_latency:.3f}s too high"
    
    def test_pattern_matching_performance(self, system):
        """Test performance of pattern matching operations."""
        # Generate test content with various patterns
        test_content = [
            "solve for x linear equation algebra",
            "find area triangle geometry calculation",
            "derivative calculus function mathematics",
            "probability statistics mean standard deviation",
            "factor polynomial quadratic expression",
            "integral calculus antiderivative function",
            "angle triangle geometry theorem proof",
            "system equations algebra substitution elimination",
            "limit calculus infinity continuous function",
            "hypothesis test statistics significance level"
        ] * 10  # 100 test cases
        
        # Measure pattern matching performance
        start_time = time.perf_counter()
        
        for content in test_content:
            # Test course classification
            course_scores = {}
            for course, patterns in system.course_patterns.items():
                score = 0
                for pattern in patterns:
                    if re.search(pattern, content.lower()):
                        score += 1
                course_scores[course] = score
            
            # Test concept classification
            concept_matches = []
            for concept_id, patterns in system.concept_patterns.items():
                for pattern in patterns:
                    if re.search(pattern, content.lower()):
                        concept_matches.append(concept_id)
                        break
        
        end_time = time.perf_counter()
        
        total_time = end_time - start_time
        avg_time_per_match = total_time / len(test_content)
        
        print(f"\n🔍 Pattern Matching Performance:")
        print(f"   Total Time: {total_time:.3f}s")
        print(f"   Avg per match: {avg_time_per_match*1000:.2f}ms")
        print(f"   Throughput: {len(test_content)/total_time:.1f} matches/sec")
        
        # Pattern matching should be very fast
        assert avg_time_per_match < 0.001, f"Pattern matching too slow: {avg_time_per_match:.4f}s"
    
    @pytest.mark.asyncio
    async def test_large_content_processing(self, system):
        """Test processing of large content blocks."""
        # Generate content of different sizes
        base_exercise = "Solve for x: 2x + 5 = 13. "
        content_sizes = [
            (100, base_exercise * 100),    # ~2KB
            (500, base_exercise * 500),    # ~10KB
            (1000, base_exercise * 1000),  # ~20KB
            (2000, base_exercise * 2000),  # ~40KB
        ]
        
        for size_multiplier, content in content_sizes:
            start_time = time.perf_counter()
            result = await system.identify_concepts(content, "text", use_llm=False)
            end_time = time.perf_counter()
            
            processing_time = end_time - start_time
            
            print(f"\n📄 Large Content (x{size_multiplier}):")
            print(f"   Content Size: ~{len(content)/1024:.1f}KB")
            print(f"   Processing Time: {processing_time:.3f}s")
            print(f"   Rate: {len(content)/1024/processing_time:.1f} KB/sec")
            
            # Verify result
            assert isinstance(result, ConceptIdentification)
            assert result.course in ["Algebra I", "Pre-Algebra"]
            
            # Should handle large content reasonably
            assert processing_time < 0.5, f"Large content processing too slow: {processing_time:.3f}s"
            
            # Source content should be truncated appropriately
            assert len(result.source_content) <= 503  # 500 + "..."


class TestStressTests:
    """Stress tests for extreme conditions."""
    
    @pytest.fixture
    def system(self):
        return ConceptIdentificationSystem(llm=None)
    
    @pytest.mark.asyncio
    async def test_rapid_fire_requests(self, system):
        """Test handling of rapid-fire requests."""
        num_requests = 50
        exercises = [
            f"Solve: {i}x + {i+1} = {i*2+5}"
            for i in range(1, num_requests + 1)
        ]
        
        # Fire requests as fast as possible
        start_time = time.perf_counter()
        
        tasks = [
            system.identify_concepts(exercise, "text", use_llm=False)
            for exercise in exercises
        ]
        
        results = await asyncio.gather(*tasks)
        end_time = time.perf_counter()
        
        total_time = end_time - start_time
        requests_per_second = num_requests / total_time
        
        print(f"\n⚡ Rapid Fire Test ({num_requests} requests):")
        print(f"   Total Time: {total_time:.3f}s")
        print(f"   Throughput: {requests_per_second:.1f} req/sec")
        print(f"   Avg Latency: {total_time/num_requests*1000:.2f}ms")
        
        # Verify all requests succeeded
        assert len(results) == num_requests
        assert all(isinstance(r, ConceptIdentification) for r in results)
        assert all(r.course != "Unknown" for r in results)
        
        # Should handle rapid requests efficiently
        assert requests_per_second > 30, f"Throughput too low: {requests_per_second:.1f} req/sec"
    
    @pytest.mark.asyncio
    async def test_malformed_input_resilience(self, system):
        """Test resilience to malformed or problematic inputs."""
        problematic_inputs = [
            "",  # Empty
            " " * 1000,  # Whitespace only
            "x" * 10000,  # Very long single character
            "solve" + "x" * 5000 + "= 5",  # Malformed with long variable
            "∑∫∂∇⊕⊗⊙⊘" * 100,  # Special characters
            "solve for x: " + "(" * 1000 + "2x + 5" + ")" * 1000,  # Excessive parentheses
            "a" * 100 + "=" + "b" * 100,  # Long variable names
            "\n" * 1000 + "solve for x: 2x + 5 = 13" + "\n" * 1000,  # Excessive whitespace
            "solve for x: 2x + 5 = 13" + "\x00" * 10,  # Null characters
            "𝑥 + 𝑦 = 𝑧" * 500,  # Unicode mathematical symbols
        ]
        
        successful_analyses = 0
        error_count = 0
        processing_times = []
        
        for i, problematic_input in enumerate(problematic_inputs):
            try:
                start_time = time.perf_counter()
                result = await system.identify_concepts(problematic_input, "text", use_llm=False)
                end_time = time.perf_counter()
                
                processing_time = end_time - start_time
                processing_times.append(processing_time)
                
                # Should return a valid result even for problematic input
                assert isinstance(result, ConceptIdentification)
                successful_analyses += 1
                
                # Should not take too long even for problematic input
                assert processing_time < 1.0, f"Input {i} took too long: {processing_time:.3f}s"
                
            except Exception as e:
                error_count += 1
                print(f"Error processing input {i}: {e}")
        
        print(f"\n🛡️  Malformed Input Resilience:")
        print(f"   Total Inputs: {len(problematic_inputs)}")
        print(f"   Successful: {successful_analyses}")
        print(f"   Errors: {error_count}")
        print(f"   Success Rate: {successful_analyses/len(problematic_inputs)*100:.1f}%")
        
        if processing_times:
            print(f"   Avg Processing Time: {statistics.mean(processing_times)*1000:.2f}ms")
            print(f"   Max Processing Time: {max(processing_times)*1000:.2f}ms")
        
        # Should handle most problematic inputs gracefully
        success_rate = successful_analyses / len(problematic_inputs)
        assert success_rate >= 0.8, f"Success rate too low: {success_rate*100:.1f}%"
    
    @pytest.mark.asyncio
    async def test_sustained_load(self, system):
        """Test performance under sustained load."""
        duration_seconds = 10  # 10 second test
        exercises = [
            "Solve for x: 2x + 5 = 13",
            "Find the area of a triangle with base 8 and height 6",
            "Calculate the derivative of f(x) = x²",
            "What is the mean of 1, 2, 3, 4, 5?",
            "Factor: x² - 5x + 6"
        ]
        
        start_time = time.perf_counter()
        end_time = start_time + duration_seconds
        
        request_count = 0
        successful_requests = 0
        latencies = []
        
        while time.perf_counter() < end_time:
            exercise = random.choice(exercises)
            
            request_start = time.perf_counter()
            try:
                result = await system.identify_concepts(exercise, "text", use_llm=False)
                request_end = time.perf_counter()
                
                latency = request_end - request_start
                latencies.append(latency)
                
                assert isinstance(result, ConceptIdentification)
                successful_requests += 1
                
            except Exception as e:
                print(f"Request failed: {e}")
            
            request_count += 1
            
            # Small delay to prevent overwhelming the system
            await asyncio.sleep(0.01)
        
        actual_duration = time.perf_counter() - start_time
        requests_per_second = successful_requests / actual_duration
        
        print(f"\n⏱️  Sustained Load Test ({duration_seconds}s):")
        print(f"   Total Requests: {request_count}")
        print(f"   Successful: {successful_requests}")
        print(f"   Success Rate: {successful_requests/request_count*100:.1f}%")
        print(f"   Throughput: {requests_per_second:.1f} req/sec")
        
        if latencies:
            print(f"   Avg Latency: {statistics.mean(latencies)*1000:.2f}ms")
            print(f"   95th Percentile: {statistics.quantiles(latencies, n=20)[18]*1000:.2f}ms")
        
        # Performance requirements for sustained load
        success_rate = successful_requests / request_count
        assert success_rate >= 0.95, f"Success rate under load too low: {success_rate*100:.1f}%"
        assert requests_per_second >= 10, f"Sustained throughput too low: {requests_per_second:.1f} req/sec"


if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 