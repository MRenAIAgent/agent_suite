# 🧮 Math OCR Benchmark Suite

A comprehensive benchmarking system for evaluating OCR solutions on mathematical content. This suite provides standardized evaluation across multiple datasets with detailed performance analytics for real-world K-12 mathematical expressions.

## 🚀 Quick Start

```bash
# Setup HME100K dataset (recommended for real evaluation)
python setup_hme100k.py --create-demo
python hme100k_manager.py --new-sample 50 --seed 42

# Run benchmark
python run_benchmark.py -d hme100k_sample -s gpt-5

# Compare all solutions
python run_benchmark.py --comprehensive --comparison-table
```

---

# 1. 📊 Datasets

## Available Datasets Overview

| Dataset | Type | Samples | Content | Ground Truth | Best For |
|---------|------|---------|---------|--------------|----------|
| **HME100K Sample** ⭐ | Real handwritten | 50-500+ | K-12 math expressions | ✅ LaTeX labels | Real-world evaluation |
| **Custom Images** | Synthetic | 14 | Mixed math problems | ⚠️ Basic text | Quick testing |
| **PGDP5K** | Academic | 30 | Research formulas | ⚠️ Basic text | Academic content |
| **Complex Geometry** | Specialized | 5 | Geometric diagrams | ⚠️ Basic text | Geometry testing |

---

## 1.1 🎯 HME100K Sample (Recommended)

### Overview
The HME100K dataset contains real handwritten mathematical expressions from K-12 students. This is the **gold standard** for mathematical OCR evaluation because it provides:

- **Real handwritten math** (not synthetic or typed)
- **LaTeX ground truth** for precise mathematical evaluation
- **K-12 appropriate content** (elementary to high school)
- **Flexible sampling** with reproducible random seeds

### Content Examples
```latex
Elementary:    x + 2 = 5
Fractions:     \frac{1}{2} + \frac{1}{3} = \frac{5}{6}
Algebra:       2x - 3 = 7
Quadratics:    x^2 + 3x + 2 = 0
Geometry:      A = \pi r^2
Trigonometry:  \sin(30°) = \frac{1}{2}
```

### When to Use
- ✅ **Primary choice** for mathematical OCR evaluation
- ✅ Research requiring real handwritten data
- ✅ K-12 educational applications
- ✅ Comparing OCR solutions with mathematical accuracy
- ✅ Publishing benchmark results

### Setup Instructions

#### Quick Demo Setup (10 sample images)
```bash
# Create demo dataset for immediate testing
python setup_hme100k.py --create-demo

# Create sample of desired size
python hme100k_manager.py --new-sample 50 --seed 42

# Verify setup
python hme100k_manager.py --current
```

#### Full Dataset Setup (100K+ images)
```bash
# Show download instructions
python setup_hme100k.py --download

# Manual steps:
# 1. Visit: https://ai.100tal.com/dataset
# 2. Register and download HME100K dataset
# 3. Extract to: ./real_datasets/HME100K/original/

# After download, create samples:
python hme100k_manager.py --new-sample 100 --seed 42
python hme100k_manager.py --new-sample 200 --seed 123
```

#### Sample Management
```bash
# List all available samples
python hme100k_manager.py --list

# Switch between samples
python hme100k_manager.py --use-sample sample_100_train_seed42

# Check current sample
python hme100k_manager.py --current
```

---

## 1.2 🗂️ Custom Images

### Overview
14 hand-crafted mathematical problems covering algebra, geometry, calculus, and statistics. These are synthetic images created for testing purposes.

### Content Examples
- Linear equations: `3x + 5 = 14`
- Quadratic equations: `x² - 4x + 3 = 0`
- Geometry: Triangle area calculations
- Calculus: Basic derivatives like `d/dx(x²) = 2x`
- Statistics: Mean and standard deviation

### When to Use
- ✅ Quick validation of OCR solutions
- ✅ Initial testing before using real data
- ✅ Debugging OCR pipeline issues
- ❌ **Not recommended** for research or production evaluation

### Setup Instructions
```bash
# No setup required - ready to use
python run_benchmark.py -d custom_images -s gpt-5

# Check dataset
python run_benchmark.py --list-datasets
```

**Location**: `./test_images/`

---

## 1.3 📚 PGDP5K

### Overview
30 mathematical expressions and formulas extracted from academic papers and documents. Contains research-level mathematical notation and complex formulas.

### Content Examples
- Research-level mathematical formulas
- Complex equations from academic publications
- Advanced mathematical notation
- Multi-line mathematical expressions

### When to Use
- ✅ Testing on academic/research content
- ✅ Evaluating complex mathematical notation
- ✅ Advanced mathematical expression recognition
- ❌ **Not suitable** for K-12 educational content

### Setup Instructions
```bash
# No setup required - ready to use
python run_benchmark.py -d pgdp5k -s gpt-5 --max-samples 10

# Full dataset
python run_benchmark.py -d pgdp5k -s mathpix_gpt5_hybrid
```

**Location**: `./real_datasets/PGDP5K/images/`

---

## 1.4 🔺 Complex Geometry

### Overview
5 specialized geometric problems with coordinate systems, shapes, and diagrams. Focuses on spatial understanding and geometric relationships.

### Content Examples
- Coordinate geometry problems
- Triangle and circle properties
- Geometric shape relationships
- Diagram-based mathematical problems

### When to Use
- ✅ Testing geometry-specific OCR capabilities
- ✅ Evaluating spatial understanding
- ✅ Diagram recognition testing
- ❌ **Very limited** sample size (only 5 images)

### Setup Instructions
```bash
# No setup required - ready to use
python run_benchmark.py -d complex_geometry -s geometry_specialist
python run_benchmark.py -d complex_geometry -s gpt-5
```

**Location**: `./real_datasets/ComplexGeometry/images/`

---

# 2. 🤖 Benchmark Solutions

## Available Solutions Overview

| Solution | Primary OCR | Strategy | Speed | Cost | Accuracy | Best For |
|----------|-------------|----------|-------|------|----------|----------|
| **gpt-5** | GPT-5 Vision | Vision-only | Medium | High | Excellent | General math OCR |
| **geometry_specialist** | GPT-5 + Custom | Hybrid | Medium | High | Excellent | Geometric content |
| **mathpix_gpt5_hybrid** | Mathpix + GPT-5 | Hybrid | Fast | Medium | Very Good | Production use |
| **got_ocr2_gpt5** | GOT-OCR2.0 + GPT-5 | Hybrid | Fast | Medium | Good | Open source focus |
| **unimer_gpt5** | UniMERNet + GPT-5 | Hybrid | Fast | Low | Good | Research/academic |
| **comprehensive_parallel** | Multiple models | Parallel | Slow | Very High | Excellent | Maximum accuracy |

---

## 2.1 🎯 GPT-5 Vision (`gpt-5`)

### Overview
Pure GPT-5 Vision processing for comprehensive mathematical analysis. Uses OpenAI's latest vision model for direct image-to-text conversion.

### Technical Details
- **Primary OCR**: GPT-5 Vision API
- **Strategy**: Direct vision processing
- **Processing**: Single-pass analysis
- **Strengths**: Excellent mathematical understanding, context awareness
- **Limitations**: Higher cost, API dependency

### Configuration
```python
{
    "name": "GPT-5 Vision Only",
    "description": "Pure GPT-5 Vision processing",
    "strategy": "vision_only",
    "primary_ocr": "gpt5_vision",
    "fallback_ocr": None,
    "post_processing": "gpt5_analysis"
}
```

### When to Use
- ✅ **Best overall choice** for mathematical OCR
- ✅ Complex mathematical expressions
- ✅ Research requiring high accuracy
- ✅ When cost is not a primary concern

### Example Usage
```bash
python run_benchmark.py -d hme100k_sample -s gpt-5
python run_benchmark.py -d all -s gpt-5 --comparison-table
```

---

## 2.2 🔺 Geometry Specialist (`geometry_specialist`)

### Overview
Specialized solution optimized for geometric content, diagrams, and spatial mathematical problems.

### Technical Details
- **Primary OCR**: GPT-5 Vision with geometry prompts
- **Strategy**: Geometry-focused processing
- **Processing**: Enhanced spatial analysis
- **Strengths**: Excellent for diagrams, coordinate systems, geometric shapes
- **Limitations**: May be overkill for simple algebraic expressions

### Configuration
```python
{
    "name": "Geometry Specialist",
    "description": "Optimized for geometric and spatial content",
    "strategy": "geometry_focused",
    "primary_ocr": "gpt5_vision_geometry",
    "post_processing": "geometry_analysis"
}
```

### When to Use
- ✅ **Best choice** for geometric problems
- ✅ Diagrams with coordinate systems
- ✅ Spatial mathematical relationships
- ✅ Complex geometric notation

### Example Usage
```bash
python run_benchmark.py -d complex_geometry -s geometry_specialist
python run_benchmark.py -d hme100k_sample -s geometry_specialist
```

---

## 2.3 ⚡ Mathpix + GPT-5 Hybrid (`mathpix_gpt5_hybrid`)

### Overview
Production-ready hybrid solution combining Mathpix OCR with GPT-5 analysis for balanced speed and accuracy.

### Technical Details
- **Primary OCR**: Mathpix API
- **Fallback OCR**: GPT-5 Vision
- **Strategy**: Fast OCR + intelligent analysis
- **Processing**: Two-stage pipeline
- **Strengths**: Good speed-accuracy balance, production-ready
- **Limitations**: Requires Mathpix API subscription

### Configuration
```python
{
    "name": "Mathpix + GPT-5 Hybrid",
    "description": "Fast Mathpix OCR with GPT-5 analysis",
    "strategy": "hybrid_fast",
    "primary_ocr": "mathpix",
    "fallback_ocr": "gpt5_vision",
    "post_processing": "gpt5_analysis"
}
```

### When to Use
- ✅ **Best choice** for production applications
- ✅ When speed is important
- ✅ Cost-conscious deployments
- ✅ High-volume processing

### Example Usage
```bash
python run_benchmark.py -d pgdp5k -s mathpix_gpt5_hybrid
python run_benchmark.py -d hme100k_sample -s mathpix_gpt5_hybrid --max-samples 100
```

---

## 2.4 🔬 Research Solutions

### GOT-OCR2.0 + GPT-5 (`got_ocr2_gpt5`)
- **Focus**: Open-source OCR with commercial analysis
- **Best For**: Research environments, open-source preference
- **Performance**: Good accuracy, moderate speed

### UniMERNet + GPT-5 (`unimer_gpt5`)
- **Focus**: Academic mathematical expression recognition
- **Best For**: Research papers, academic content
- **Performance**: Good for mathematical expressions

### Comprehensive Parallel (`comprehensive_parallel`)
- **Focus**: Maximum accuracy through multiple models
- **Best For**: Research requiring highest possible accuracy
- **Performance**: Excellent accuracy, slow processing, high cost

---

## 2.5 🛠️ Running Benchmarks

### Single Solution Testing
```bash
# Test specific solution on specific dataset
python run_benchmark.py -d hme100k_sample -s gpt-5

# Limit samples for quick testing
python run_benchmark.py -d hme100k_sample -s gpt-5 --max-samples 20

# Test with verbose output
python run_benchmark.py -d custom_images -s gpt-5 --verbose
```

### Comprehensive Benchmarking
```bash
# Test all solutions on all datasets
python run_benchmark.py --comprehensive --comparison-table

# Quick comprehensive test (limited samples)
python run_benchmark.py --comprehensive --quick --comparison-table

# Test all solutions on specific dataset
python run_benchmark.py -d hme100k_sample -s all --comparison-table

# Test specific solution on all datasets
python run_benchmark.py -d all -s gpt-5 --comparison-table
```

### Information Commands
```bash
# List available datasets
python run_benchmark.py --list-datasets

# List available solutions
python run_benchmark.py --list-solutions

# Get recommendations
python run_benchmark.py --recommendations
```

---

# 3. 📈 Results

## 3.1 📊 Understanding Results

### Accuracy Metrics

#### HME100K Sample (LaTeX Ground Truth)
```python
# Mathematical Equivalence Evaluation
Ground Truth: "x^2 + 3x + 2 = 0"
OCR Output 1: "x² + 3x + 2 = 0"     ✅ 100% (mathematically equivalent)
OCR Output 2: "x^2+3x+2=0"          ✅ 100% (spacing ignored)
OCR Output 3: "x^2 + 2x + 3 = 0"    ❌ 0% (different equation)
OCR Output 4: "x² + 3x + 2 = O"     ❌ 0% (O instead of 0)
```

#### Other Datasets (Text Similarity)
```python
# String Similarity Evaluation
Ground Truth: "triangle area"
OCR Output 1: "triangle area"        ✅ 100% (exact match)
OCR Output 2: "Triangle Area"        ⚠️ 80% (case difference)
OCR Output 3: "triangel area"        ⚠️ 60% (typo)
OCR Output 4: "area of triangle"     ⚠️ 40% (word order)
```

### Performance Metrics
- **Overall Accuracy**: Percentage of correctly recognized expressions
- **Processing Time**: Total time for all samples
- **OCR Success Rate**: Percentage of images successfully processed
- **Average Time/Sample**: Processing time per image
- **Confidence Score**: Model confidence in results (when available)

---

## 3.2 📋 Result Files

### Generated Output Files

#### JSON Results (Detailed)
```bash
# Location: results/unified_benchmarks/
benchmark_hme100k_sample_gpt-5_TIMESTAMP.json
```

**Content Structure**:
```json
{
  "session_id": "gpt-5_1756059912",
  "configuration": {
    "solution": "gpt-5",
    "dataset": "hme100k_sample",
    "strategy": "vision_only"
  },
  "results": {
    "overall_accuracy": 92.5,
    "total_samples": 50,
    "processing_time": 245.3,
    "ocr_success_rate": 100.0
  },
  "detailed_results": [
    {
      "image": "demo_001.png",
      "ground_truth": "x + 2 = 5",
      "ocr_result": "x + 2 = 5",
      "accuracy": 100.0,
      "processing_time": 4.2
    }
  ]
}
```

#### CSV Comparison Tables
```bash
# Location: results/comparison_tables/
comprehensive_comparison_TIMESTAMP.csv
```

**Content**: Tabular comparison of all solution-dataset combinations

#### HTML Reports
```bash
# Location: results/html_reports/
interactive_report_TIMESTAMP.html
```

**Content**: Interactive web-based results with charts and filtering

---

## 3.3 📊 Sample Benchmark Results

### Typical Performance Expectations

#### HME100K Sample (Real Handwritten Math)
```
Solution                 | Accuracy | Time/Sample | Best For
-------------------------|----------|-------------|------------------
gpt-5                   | 85-95%   | 15-25s     | Overall best
geometry_specialist     | 80-90%   | 18-28s     | Geometric content
mathpix_gpt5_hybrid    | 75-85%   | 8-15s      | Production use
got_ocr2_gpt5          | 70-80%   | 10-18s     | Open source
unimer_gpt5            | 65-75%   | 12-20s     | Academic content
```

#### Custom Images (Synthetic)
```
Solution                 | Accuracy | Time/Sample | Notes
-------------------------|----------|-------------|------------------
gpt-5                   | 75-85%   | 12-20s     | Good overall
geometry_specialist     | 85-95%   | 15-25s     | Excellent on geometry
mathpix_gpt5_hybrid    | 70-80%   | 6-12s      | Fast processing
```

---

## 3.4 🎯 Result Interpretation

### What Good Results Look Like

#### Excellent Performance (90%+ accuracy)
- **HME100K Sample**: 90%+ with LaTeX ground truth
- **Processing Time**: <20s per sample
- **OCR Success Rate**: 98%+
- **Use Case**: Production-ready, research-quality

#### Good Performance (75-90% accuracy)
- **HME100K Sample**: 75-90% with LaTeX ground truth
- **Processing Time**: 10-25s per sample
- **OCR Success Rate**: 95%+
- **Use Case**: Suitable for most applications

#### Needs Improvement (<75% accuracy)
- **Possible Issues**: Wrong solution for content type, API problems, poor image quality
- **Recommendations**: Try different solution, check image quality, verify API keys

### Troubleshooting Poor Results

#### Low Accuracy Issues
```bash
# Check if using appropriate solution for content
python run_benchmark.py --recommendations

# Try geometry specialist for geometric content
python run_benchmark.py -d complex_geometry -s geometry_specialist

# Verify API credentials
python run_benchmark.py --list-solutions
```

#### Performance Issues
```bash
# Use faster solutions for large datasets
python run_benchmark.py -d hme100k_sample -s mathpix_gpt5_hybrid

# Limit samples for testing
python run_benchmark.py -d hme100k_sample -s gpt-5 --max-samples 10

# Use quick mode for comprehensive testing
python run_benchmark.py --comprehensive --quick
```

---

## 3.5 📈 Comparative Analysis

### Choosing the Right Solution

#### For Real-World K-12 Math Evaluation
```bash
# Recommended workflow
python setup_hme100k.py --create-demo
python hme100k_manager.py --new-sample 75 --seed 42
python run_benchmark.py -d hme100k_sample -s gpt-5
```

**Expected Results**: 85-95% accuracy with mathematical equivalence

#### For Production Applications
```bash
# Balanced speed and accuracy
python run_benchmark.py -d hme100k_sample -s mathpix_gpt5_hybrid
```

**Expected Results**: 75-85% accuracy, 8-15s per sample

#### For Research & Development
```bash
# Comprehensive evaluation
python run_benchmark.py --comprehensive --comparison-table
```

**Expected Results**: Complete performance matrix across all combinations

---

## 📋 Quick Reference

### Essential Commands
```bash
# Setup and sampling
python setup_hme100k.py --create-demo
python hme100k_manager.py --new-sample 50 --seed 42

# Basic benchmarking
python run_benchmark.py -d hme100k_sample -s gpt-5
python run_benchmark.py --comprehensive --comparison-table

# Information
python run_benchmark.py --list-datasets
python run_benchmark.py --list-solutions
python hme100k_manager.py --current
```

### Best Practices
1. **Start with HME100K**: Use real handwritten math for meaningful evaluation
2. **Use appropriate sample sizes**: 50-100 samples for testing, 200+ for research
3. **Compare multiple solutions**: Use `--comparison-table` for comprehensive analysis
4. **Monitor costs**: Be aware of API usage with large datasets
5. **Save results**: Keep benchmark results for comparison and analysis

---

**🚀 Ready to benchmark mathematical OCR solutions with real-world data!**

For issues or questions, check the troubleshooting section or review the saved logs in `./logs/`.