# 📁 Scripts Reference Guide

All helper scripts have been moved to the `scripts/` directory. Here's a quick reference:

## 🎯 **Main Benchmark Script**
- **`run_benchmark.py`** - Main benchmark runner (stays in root directory)

## 📊 **Dataset Management Scripts**
Located in `scripts/` directory:

### HME100K Dataset Scripts
- **`setup_hme100k.py`** - Setup and download HME100K dataset
- **`download_real_hme100k.py`** - Download full HME100K dataset  
- **`download_hme100k_huggingface.py`** - Download HME100K subset from Hugging Face
- **`hme100k_manager.py`** - Manage HME100K samples (create, list, switch)

### Other Dataset Scripts
- **`dataset_downloader.py`** - General dataset downloader
- **`create_sample_datasets.py`** - Create sample datasets
- **`create_sample_pgdp5k.py`** - Create PGDP5K samples
- **`download_pgdp5k.py`** - Download PGDP5K dataset

## 🔧 **Core Engine Scripts**
- **`improved_benchmark_runner.py`** - Core benchmark execution engine
- **`real_ocr_processor.py`** - OCR processing implementation
- **`ocr_config.py`** - OCR solution configurations
- **`evaluation_metrics.py`** - Accuracy and performance metrics

## 📈 **Analysis & Diagnostic Scripts**
- **`diagnose_results.py`** - Diagnose benchmark results
- **`verify_results.py`** - Verify result accuracy
- **`accuracy_improvement_plan.py`** - Accuracy improvement analysis
- **`ocr_provider_diagnostic.py`** - OCR provider diagnostics

## 🚀 **Usage Examples**

### Setup HME100K Dataset
```bash
# Create demo dataset
python scripts/setup_hme100k.py --create-demo

# Create 50-image sample
python scripts/hme100k_manager.py --new-sample 50 --seed 42

# List available samples
python scripts/hme100k_manager.py --list
```

### Run Benchmarks
```bash
# Main benchmark (from root directory)
python run_benchmark.py -d hme100k_sample -s gpt-5

# Comprehensive comparison
python run_benchmark.py --comprehensive --comparison-table
```

### Analyze Results
```bash
# Diagnose results
python scripts/diagnose_results.py

# Verify accuracy
python scripts/verify_results.py
```

## 📂 **Directory Structure**
```
benchmark/
├── run_benchmark.py          # Main benchmark script
├── README.md                  # Main documentation
├── scripts/                   # All helper scripts
│   ├── setup_hme100k.py
│   ├── hme100k_manager.py
│   ├── improved_benchmark_runner.py
│   └── ... (other scripts)
├── results/                   # Benchmark results
├── real_datasets/            # Dataset files
└── test_images/              # Test images
```

This organization keeps the main directory clean while maintaining all functionality in the scripts subdirectory.
