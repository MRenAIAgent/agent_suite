# 🧮 Math OCR Benchmark Suite

A comprehensive benchmarking system for testing OCR solutions on mathematical content. This suite provides easy-to-use tools for evaluating different OCR approaches across multiple datasets with detailed performance analytics.

## 🚀 Quick Start

```bash
# See what datasets are available
python run_benchmark.py --list-datasets

# See what OCR solutions are available  
python run_benchmark.py --list-solutions

# Get recommendations for best combinations
python run_benchmark.py --recommendations

# Run a quick test with GPT-5 on custom images
python run_benchmark.py -d custom_images -s gpt-5

# Run comprehensive test with hybrid solution
python run_benchmark.py -d pgdp5k -s mathpix_gpt5_hybrid
```

## 📊 Available Datasets

| Dataset | Samples | Content Type | Difficulty | Best For |
|---------|---------|--------------|------------|----------|
| **custom_images** | 14 | Mixed math problems | Varied | Quick testing & validation |
| **pgdp5k** | 5000+ | Academic documents | Academic | Comprehensive evaluation |
| **expanded_dataset** | 300+ | Complex expressions | Challenging | Stress testing |

### Dataset Details

#### 🗂️ Custom Images (`custom_images`)
- **Content**: Hand-crafted algebra, geometry, calculus, and statistics problems
- **Format**: PNG images with ground truth
- **Use Case**: Quick validation and testing new solutions
- **Location**: `./test_images/`

#### 📚 PGDP5K (`pgdp5k`) 
- **Content**: Large-scale mathematical documents and expressions
- **Format**: Academic paper excerpts with formulas
- **Use Case**: Comprehensive evaluation at scale
- **Location**: `./datasets/pgdp5k/`

#### 📐 Expanded Dataset (`expanded_dataset`)
- **Content**: Diverse mathematical content including complex expressions
- **Format**: Mixed format with challenging edge cases
- **Use Case**: Stress testing and edge case evaluation
- **Location**: `./datasets/expanded/`

## 🤖 Available OCR Solutions

### 🤖 AI Vision Models
| Solution | Primary OCR | Speed | Cost | Best For |
|----------|-------------|-------|------|----------|
| **gpt-5** | GPT-5 Vision | Fast | Medium | General math problems, mixed content |

### 🔄 Hybrid Solutions  
| Solution | Primary OCR | Fallback | Speed | Cost | Best For |
|----------|-------------|----------|-------|------|----------|
| **mathpix_gpt5_hybrid** | Mathpix | GPT-5 | Medium | Medium | Production use, reliable results |

### 🔺 Specialized OCR
| Solution | Primary OCR | Fallback | Speed | Cost | Best For |
|----------|-------------|----------|-------|------|----------|
| **geometry_specialist** | GOT-OCR2.0 | GPT-5 | Slow | Low | Geometric diagrams, coordinate extraction |

### 📐 Math Specialists
| Solution | Primary OCR | Fallback | Speed | Cost | Best For |
|----------|-------------|----------|-------|------|----------|
| **math_expression_expert** | UniMERNet | Mathpix | Slow | Low | Complex mathematical expressions |

### 🔧 Advanced Solutions
| Solution | Description | Best For |
|----------|-------------|----------|
| **comprehensive_parallel** | Multiple OCR models in parallel | Maximum accuracy, research |
| **cost_optimized** | Open source models only | Budget-conscious applications |

## 💡 Recommended Combinations

### For Quick Testing
```bash
# Best balance of speed and accuracy
python run_benchmark.py -d custom_images -s gpt-5
# Expected: 75-85% accuracy in 3-5 minutes
```

### For Geometric Content
```bash
# Specialized for shapes and diagrams
python run_benchmark.py -d custom_images -s geometry_specialist  
# Expected: 85-95% accuracy in 4-6 minutes
```

### For Production Use
```bash
# Reliable hybrid approach
python run_benchmark.py -d pgdp5k -s mathpix_gpt5_hybrid
# Expected: 80-90% accuracy in 2-4 hours
```

### For Maximum Accuracy
```bash
# Comprehensive parallel processing
python run_benchmark.py -d expanded_dataset -s comprehensive_parallel
# Expected: 85-95% accuracy in 30-60 minutes
```

## 📋 Command Reference

### Information Commands
```bash
# List all available datasets with details
python run_benchmark.py --list-datasets

# List all available OCR solutions with details
python run_benchmark.py --list-solutions

# Show recommended dataset-solution combinations
python run_benchmark.py --recommendations
```

### Benchmark Commands
```bash
# Basic benchmark run
python run_benchmark.py -d <dataset> -s <solution>

# Limit number of samples (useful for testing)
python run_benchmark.py -d <dataset> -s <solution> --max-samples 10

# Run on all datasets (use with caution - can take hours)
python run_benchmark.py -d all -s <solution>

# Don't save results to file
python run_benchmark.py -d <dataset> -s <solution> --no-save
```

### Examples
```bash
# Quick test with 5 samples
python run_benchmark.py -d custom_images -s gpt-5 --max-samples 5

# Full evaluation of geometry specialist
python run_benchmark.py -d custom_images -s geometry_specialist

# Test GPT-5 on all available datasets (limited samples)
python run_benchmark.py -d all -s gpt-5 --max-samples 3
```

## 📊 Understanding Results

### Metrics Explained

- **Overall Accuracy**: Percentage of correctly processed problems
- **OCR Success Rate**: Percentage of images successfully processed (no errors)
- **Text Extraction Accuracy**: Accuracy of extracting mathematical text
- **Processing Time**: Total time taken for processing
- **Avg Time/Sample**: Average processing time per image

### Sample Output
```
🎯 BENCHMARK RESULTS SUMMARY
============================================================
🤖 Solution: GPT-5 Vision Only
⏱️  Total Time: 229.0s
📊 Overall Accuracy: 79.5%
📈 Total Samples: 14
⚡ Avg Time/Sample: 16.36s

📊 DATASET BREAKDOWN:
----------------------------------------

🗂️  Custom Test Images
   Samples: 14
   Accuracy: 79.5%
   Time: 229.0s
```

## 🔧 Setup Requirements

### API Keys Required
Create a `.env` file in the benchmark directory:
```bash
# Required for GPT-5 and hybrid solutions
OPENAI_API_KEY=your_openai_api_key_here

# Optional for Mathpix hybrid solutions  
MATHPIX_APP_ID=your_mathpix_app_id_here
MATHPIX_API_KEY=your_mathpix_api_key_here
```

### Python Dependencies
```bash
pip install aiohttp python-dotenv pillow
```

### Dataset Setup
- **custom_images**: Included in repository (`./test_images/`)
- **pgdp5k**: Download separately (see dataset documentation)
- **expanded_dataset**: Generate using provided scripts

## 📁 Results Storage

Results are automatically saved to:
```
./results/unified_benchmarks/benchmark_<dataset>_<solution>_<timestamp>.json
```

Each result file contains:
- Solution configuration details
- Dataset information  
- Performance metrics
- Processing times
- Session IDs for traceability

## 🛠️ Advanced Usage

### Custom Dataset Integration
To add a new dataset, modify the `datasets` dictionary in `run_benchmark.py`:

```python
"my_dataset": {
    "name": "My Custom Dataset",
    "description": "Description of the dataset",
    "path": "./path/to/dataset",
    "samples": "number_of_samples",
    "types": ["content_types"],
    "difficulty": "difficulty_level", 
    "best_for": "recommended_use_case"
}
```

### Custom OCR Solution
To add a new OCR solution, add it to `ocr_config.py` and it will automatically appear in the benchmark suite.

## 🔍 Troubleshooting

### Common Issues

**"Dataset not found" Error**
- Ensure dataset path exists
- Check if you need to download/generate the dataset first

**API Key Errors**
- Verify `.env` file is in the correct location
- Check API key format and validity
- Ensure sufficient API quota

**Low Accuracy Results**
- Check if ground truth data is properly formatted
- Verify image quality and resolution
- Consider using hybrid solutions for better accuracy

**Slow Performance**
- Use `--max-samples` to limit test size
- Choose faster solutions (e.g., `gpt-5` over `geometry_specialist`)
- Ensure good network connectivity for API calls

### Getting Help
- Use `--help` for command-line help
- Check the `./logs/` directory for detailed error logs
- Review saved results in `./results/` for debugging

## 📈 Performance Expectations

| Solution | Speed | Accuracy | Cost | Use Case |
|----------|-------|----------|------|----------|
| gpt-5 | ⚡⚡⚡ | 📊📊📊📊 | 💰💰💰 | General purpose |
| mathpix_gpt5_hybrid | ⚡⚡ | 📊📊📊📊📊 | 💰💰 | Production |
| geometry_specialist | ⚡ | 📊📊📊📊📊 | 💰 | Geometric content |
| math_expression_expert | ⚡ | 📊📊📊📊 | 💰 | Complex expressions |

## 🎯 Best Practices

1. **Start Small**: Use `custom_images` with `--max-samples 5` for initial testing
2. **Choose Right Solution**: Match OCR solution to your content type
3. **Monitor Costs**: Be aware of API usage, especially with large datasets
4. **Save Results**: Keep benchmark results for comparison and analysis
5. **Use Recommendations**: Follow the built-in recommendations for best results

---

**Happy Benchmarking! 🚀**

For issues or questions, check the troubleshooting section or review the saved logs in `./logs/`.