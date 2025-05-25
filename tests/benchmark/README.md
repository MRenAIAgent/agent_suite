# Benchmark Tests

This directory contains tests for the benchmark functionality in the codebase.

## Available Tests

- `test_taubench.py` - Tests the minimal TauBench implementation to ensure it loads datasets and evaluates responses correctly

## Running Tests

You can run these tests using pytest:

```bash
pytest tests/benchmark/test_taubench.py -v
```

## Purpose

These tests validate that the benchmark implementations work correctly. They are particularly useful when modifying the benchmark code to ensure that changes don't break existing functionality.

## Relation to Benchmark Directory

These tests are for validating the functionality in the `benchmark/` directory. The actual benchmark code should be found in that directory, not here. 