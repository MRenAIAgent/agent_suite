# 🚀 Math Recognition Benchmark System

## 📋 Overview

This directory contains a comprehensive benchmarking system for evaluating mathematical expression recognition and OCR capabilities. After cleanup, the system provides a streamlined, professional approach to benchmarking with multiple OCR providers and evaluation metrics.

## ⭐ Primary Benchmark Tool

### `improved_benchmark_runner.py` - Main Benchmark Script

The **primary and recommended** tool for all benchmarking activities.

**Features:**
- ✅ OCR configuration system integration
- ✅ Multiple dataset support (ComplexGeometry, PGDP5K, ExpandedDataset, MATH-Vision, MathVerse)
- ✅ Session management with detailed logging
- ✅ Comprehensive evaluation metrics
- ✅ Result export and analysis
- ✅ Command-line interface
- ✅ Async processing for performance
- ✅ Error handling and diagnostics
- ✅ Configuration validation

**Usage:**

```bash
# List available configurations
python3 improved_benchmark_runner.py --list-configs

# Validate a configuration
python3 improved_benchmark_runner.py --validate gpt4v_only

# Run benchmark with specific configuration
python3 improved_benchmark_runner.py --config gpt4v_only --samples 10

# Compare multiple configurations
python3 improved_benchmark_runner.py --compare gpt4v_only geometry_specialist

# Run with custom dataset
python3 improved_benchmark_runner.py --config mathpix_gpt4v_hybrid --dataset /path/to/dataset --samples 50
```

**Available Configurations:**
- `mathpix_gpt4v_hybrid`: Industry standard with Mathpix OCR + GPT-4 Vision fallback
- `gpt4v_only`: Pure GPT-4 Vision processing
- `geometry_specialist`: GOT-OCR2.0 optimized for geometric content
- `math_expression_expert`: UniMERNet specialized for mathematical expressions
- `comprehensive_parallel`: Multiple OCR models in parallel
- `cost_optimized`: Open source models prioritizing cost efficiency

## 🛠️ Supporting Tools

### Configuration & Setup
- **`setup_benchmark.py`** - Interactive setup and configuration utility
- **`ocr_config.py`** - OCR configuration management system

### Dataset Management
- **`scripts/dataset_expander.py`** - Dataset generation and expansion tool
- **`scripts/create_sample_datasets.py`** - Sample data creation utility
- **`scripts/acquire_missing_datasets.py`** - Dataset acquisition tool
- **`scripts/dataset_downloader.py`** - Download benchmark datasets

### Diagnostics & Analysis
- **`scripts/ocr_provider_diagnostic.py`** - OCR provider testing and validation
- **`scripts/accuracy_improvement_plan.py`** - Performance analysis and improvement planning
- **`scripts/evaluation_metrics.py`** - Metrics calculation library
- **`scripts/diagnose_results.py`** - Result analysis and diagnostics

### Data Download & Preparation
- **`scripts/download_pgdp5k.py`** - Download PGDP5K dataset
- **`scripts/create_sample_pgdp5k.py`** - Create PGDP5K samples
- **`scripts/verify_results.py`** - Manual verification tool for results

## 📊 Results & Output

### Results Directory Structure
```
results/
├── geometry_benchmark_results_YYYYMMDD_HHMMSS.json
├── math_expressions_benchmark_YYYYMMDD_HHMMSS.json
├── student_solutions_benchmark_YYYYMMDD_HHMMSS.json
└── benchmark_summary_YYYYMMDD_HHMMSS.json
```

### Result Format
Each benchmark run produces:
- **Detailed JSON results** with per-sample analysis
- **Summary statistics** with overall accuracy metrics
- **Processing time measurements**
- **Error analysis and diagnostics**
- **Configuration metadata**

## 🏗️ Directory Structure

```
benchmark/
├── README.md                          # This file
├── improved_benchmark_runner.py       # ⭐ PRIMARY BENCHMARK TOOL
├── setup_benchmark.py                 # Setup utility
├── ocr_config.py                      # Configuration system
├── BENCHMARK_CLEANUP_PLAN.md          # Cleanup documentation
├── CLEANUP_COMPLETION_SUMMARY.md      # Cleanup completion summary
├── SCRIPTS_CLEANUP_SUMMARY.md         # Scripts cleanup summary
├── configs/                           # Configuration files
├── datasets/                          # Test datasets
├── real_datasets/                     # Production datasets (PGDP5K, etc.)
├── results/                           # Benchmark results
├── sample_data/                       # Sample test data
├── docs/                              # Documentation
├── reports/                           # Analysis reports
└── scripts/                           # Supporting utilities
    ├── dataset_expander.py            # Generate diverse problems
    ├── create_sample_datasets.py      # Create sample data
    ├── acquire_missing_datasets.py    # Download missing datasets
    ├── ocr_provider_diagnostic.py     # OCR provider testing
    ├── accuracy_improvement_plan.py   # Performance analysis
    ├── evaluation_metrics.py          # Metrics library
    ├── download_pgdp5k.py             # PGDP5K downloader
    ├── create_sample_pgdp5k.py        # PGDP5K samples
    ├── dataset_downloader.py          # General downloader
    ├── diagnose_results.py            # Result analysis
    └── verify_results.py              # Manual verification
```

## 📊 Available Datasets

### **Sample Datasets (Ready to Use)**
- **Sample Data**: 13 samples across 3 categories (math expressions, test papers, student solutions)
- **ComplexGeometry**: 5 challenging geometry problems with ground truth
- **ExpandedDataset**: 17 comprehensive math problems across multiple topics

### **Production Datasets (Large Scale)**
- **PGDP5K**: 5,000 real plane geometry diagrams (requires download)
- **MATH-Vision & MathVerse**: Research datasets (acquisition via scripts)

## 🔧 Requirements

### **Environment Setup**
- Python 3.8+
- API keys in `.env` file (OPENAI_API_KEY, MATHPIX_APP_KEY, etc.)
- Dependencies: `pip install -r requirements.txt` (if available)

### **Optional Dependencies**
- `python-dotenv` - For .env file loading (recommended)
- `PIL/Pillow` - For image processing
- `datasets` - For Hugging Face dataset downloads

## 🚀 Quick Start

1. **Setup Environment:**
   ```bash
   python3 setup_benchmark.py
   ```

2. **Configure API Keys:**
   ```bash
   # Create .env file with your API keys
   echo "OPENAI_API_KEY=your_key_here" > .env
   echo "MATHPIX_APP_KEY=your_key_here" >> .env
   ```

3. **List Available Configurations:**
   ```bash
   python3 improved_benchmark_runner.py --list-configs
   ```

4. **Validate Configuration:**
   ```bash
   python3 improved_benchmark_runner.py --validate gpt4v_only
   ```

5. **Run Quick Test:**
   ```bash
   python3 improved_benchmark_runner.py --config gpt4v_only --samples 5
   ```

6. **Full Benchmark:**
   ```bash
   python3 improved_benchmark_runner.py --config mathpix_gpt4v_hybrid --samples 100
   ```

## 📈 Performance Metrics

The benchmark system evaluates:
- **Overall Accuracy**: General recognition performance
- **Shape Detection**: Geometric shape identification
- **Coordinate Extraction**: Mathematical coordinate parsing
- **Text Recognition**: General text OCR accuracy
- **LaTeX Accuracy**: Mathematical expression formatting
- **Processing Time**: Performance measurements
- **Confidence Scores**: Model confidence analysis

### **Expected Performance Ranges**

| Configuration | Sample Data | Complex Geometry | PGDP5K Dataset |
|---------------|-------------|------------------|----------------|
| `gpt4v_only` | 70-85% | 60-80% | 70-85% |
| `mathpix_gpt4v_hybrid` | 75-90% | 70-90% | 75-90% |
| `geometry_specialist` | 65-80% | 80-95% | 85-95% |
| `math_expression_expert` | 80-95% | 50-70% | 60-75% |
| `cost_optimized` | 50-70% | 40-60% | 50-70% |

*Note: Actual performance varies based on problem complexity and image quality.*

## 🔧 Configuration Management

The system uses a centralized configuration approach:
- **OCR Provider Settings**: API keys, endpoints, parameters
- **Processing Strategies**: Different approaches for different content types
- **Evaluation Metrics**: Customizable success criteria
- **Dataset Configurations**: Different test scenarios

## 📝 Recent Cleanup (Completed)

✅ **Major Cleanup Completed:**
- **Removed 29 duplicate/outdated files** from benchmark and scripts directories
- **Fixed environment variable loading** (.env file support added)
- **Eliminated script redundancies** (removed duplicate download scripts)
- **Cleaned cache and temporary files** (__pycache__, verification images)

✅ **Key Improvements:**
- Single source of truth for benchmarking (`improved_benchmark_runner.py`)
- All scripts now serve distinct, non-overlapping purposes
- Environment variables properly loaded from `.env` file
- Clean directory structure with comprehensive documentation
- Reduced maintenance burden and improved clarity

✅ **Documentation Added:**
- `CLEANUP_COMPLETION_SUMMARY.md` - Complete cleanup documentation
- `SCRIPTS_CLEANUP_SUMMARY.md` - Scripts directory cleanup details

## 🔧 Troubleshooting

### **Common Issues**

**"No module named 'math_learning'" Warning:**
- This is a harmless import warning and doesn't affect functionality
- The benchmark system works correctly despite this warning

**"Environment variable OPENAI_API_KEY not set":**
- Ensure your `.env` file is in the benchmark directory
- Check that the API key is correctly formatted: `OPENAI_API_KEY=sk-...`
- Verify the `.env` file has no extra spaces or quotes

**"No data available for dataset_type":**
- Run dataset download scripts: `python3 scripts/download_pgdp5k.py`
- Use sample data for testing: `--dataset ./sample_data`
- Check that dataset paths are correct

### **Diagnostic Tools**
```bash
# Test OCR providers individually
python3 scripts/ocr_provider_diagnostic.py

# Analyze benchmark results
python3 scripts/diagnose_results.py

# Verify results manually
python3 scripts/verify_results.py
```

## 📞 Support

For issues or questions:
1. **Configuration Issues**: Use `--validate` to check setup
2. **API Problems**: Run `scripts/ocr_provider_diagnostic.py`
3. **Dataset Issues**: Check logs in the results directory
4. **Performance Analysis**: Use `scripts/accuracy_improvement_plan.py`
5. **Documentation**: Refer to cleanup summaries and setup guides

### **Useful Commands**
```bash
# Quick health check
python3 improved_benchmark_runner.py --list-configs
python3 improved_benchmark_runner.py --validate gpt4v_only

# Download datasets
python3 scripts/download_pgdp5k.py
python3 scripts/acquire_missing_datasets.py

# Create sample data for testing
python3 scripts/create_sample_datasets.py
python3 scripts/create_sample_pgdp5k.py
```

---

**Ready to benchmark! 🚀** 