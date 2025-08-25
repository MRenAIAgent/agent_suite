# 🎲 HME100K Setup & Random Sampling Guide

## 🚀 Quick Start

### 1. **Create Demo Dataset (For Testing)**
```bash
# Create a demo dataset with 10 K-12 math samples
python setup_hme100k.py --create-demo

# This creates LaTeX-labeled samples like:
# - x + 2 = 5
# - 2x - 3 = 7  
# - \frac{1}{2} + \frac{1}{3}
# - x^2 + 3x + 2 = 0
```

### 2. **Create Random Samples (50-100 images)**
```bash
# Create random sample of 50 images
python hme100k_manager.py --new-sample 50

# Create reproducible sample with seed
python hme100k_manager.py --new-sample 100 --seed 42

# Create sample from test set
python hme100k_manager.py --new-sample 75 --seed 123
```

### 3. **Run Benchmarks**
```bash
# Test single solution
python run_benchmark.py -d hme100k_sample -s gpt-5

# Test all solutions with comparison table
python run_benchmark.py -d hme100k_sample -s all --comparison-table

# Quick test (limit samples)
python run_benchmark.py -d hme100k_sample -s gpt-5 --max-samples 10
```

## 📊 **Full HME100K Dataset Setup**

### Option 1: Manual Download (Recommended for Full Dataset)
```bash
# Show download instructions
python setup_hme100k.py --download

# Follow instructions to download from: https://ai.100tal.com/dataset
# Extract to: ./real_datasets/HME100K/original/
```

### Option 2: Demo Dataset (For Testing)
```bash
# Create demo with K-12 appropriate samples
python setup_hme100k.py --create-demo
```

## 🎯 **Random Sampling Commands**

### Basic Sampling
```bash
# Create new random sample of 100 images
python hme100k_manager.py --new-sample 100

# Create smaller sample for quick testing
python hme100k_manager.py --new-sample 50

# Create larger sample for comprehensive testing
python hme100k_manager.py --new-sample 200
```

### Reproducible Sampling (with Seeds)
```bash
# Reproducible sample - same images every time
python hme100k_manager.py --new-sample 100 --seed 42

# Different reproducible samples
python hme100k_manager.py --new-sample 75 --seed 123
python hme100k_manager.py --new-sample 50 --seed 999
```

### Sample Management
```bash
# List all available samples
python hme100k_manager.py --list

# Show current sample info
python hme100k_manager.py --current

# Switch to existing sample
python hme100k_manager.py --use-sample sample_100_train_seed42
```

## 📁 **Dataset Structure**

```
real_datasets/HME100K/
├── original/                          # Full dataset (manual download)
│   ├── train/
│   │   ├── train_images/              # 74,502 training images
│   │   └── train_labels.txt           # LaTeX labels
│   └── test/
│       ├── test_images/               # 24,607 testing images  
│       └── test_labels.txt            # LaTeX labels
└── samples/                           # Random samples
    ├── sample_50_train_seed42/        # Sample with 50 images
    │   ├── images/                    # Selected images
    │   ├── labels.txt                 # Corresponding labels
    │   └── sample_info.json           # Sample metadata
    ├── sample_100_train/              # Another sample
    └── current -> sample_50_train_seed42  # Symlink to current sample
```

## 🎲 **Random Sampling Features**

### **Reproducible Sampling**
- Use `--seed` parameter for consistent results
- Same seed = same images selected
- Perfect for comparing different OCR solutions on identical data

### **Flexible Sample Sizes**
- **Small (10-25)**: Quick testing, development
- **Medium (50-100)**: Standard evaluation  
- **Large (200-500)**: Comprehensive analysis
- **Custom**: Any size up to dataset limit

### **Split Selection**
- `--split train`: Sample from training set (74,502 images)
- `--split test`: Sample from test set (24,607 images)

## 🚀 **Benchmark Integration**

### **Available Commands**
```bash
# Single solution test
python run_benchmark.py -d hme100k_sample -s gpt-5

# All solutions comparison
python run_benchmark.py -d hme100k_sample -s all --comparison-table

# Comprehensive benchmark (all datasets + all solutions)
python run_benchmark.py --comprehensive --comparison-table

# Quick mode (limited samples)
python run_benchmark.py -d hme100k_sample -s gpt-5 --quick
```

### **Expected Results with Real Data**
With HME100K's LaTeX ground truth, you'll get:
- **More accurate evaluation**: LaTeX comparison vs simple text matching
- **Mathematical equivalence**: Recognizes `2x + 3` = `3 + 2x`
- **Real-world performance**: Actual handwritten math, not synthetic
- **K-12 appropriate**: Grade school through high school level

## 📊 **Sample Usage Examples**

### **Development Testing**
```bash
# Quick test with small sample
python hme100k_manager.py --new-sample 10 --seed 1
python run_benchmark.py -d hme100k_sample -s gpt-5 --max-samples 5
```

### **Solution Comparison**
```bash
# Create reproducible sample for fair comparison
python hme100k_manager.py --new-sample 100 --seed 42
python run_benchmark.py -d hme100k_sample -s all --comparison-table
```

### **Performance Analysis**
```bash
# Large sample for statistical significance
python hme100k_manager.py --new-sample 500 --seed 123
python run_benchmark.py -d hme100k_sample -s gpt-5
```

### **Reproducible Research**
```bash
# Document exact sample used
python hme100k_manager.py --new-sample 200 --seed 2024
python hme100k_manager.py --current  # Shows sample details
python run_benchmark.py -d hme100k_sample -s geometry_specialist
```

## 🔧 **Troubleshooting**

### **No Current Sample Set**
```bash
# Error: No current sample for benchmarking
# Solution: Create or select a sample
python hme100k_manager.py --new-sample 50
```

### **Dataset Not Found**
```bash
# Error: Original dataset not found
# Solution: Create demo dataset or download full dataset
python setup_hme100k.py --create-demo
```

### **Sample Too Large**
```bash
# Error: Requested sample larger than available
# Solution: Check available samples and adjust size
python setup_hme100k.py --info
python hme100k_manager.py --new-sample 50  # Smaller size
```

## 🎯 **Best Practices**

1. **Use Seeds for Reproducibility**: Always use `--seed` for research/comparison
2. **Start Small**: Begin with 50-100 samples, scale up as needed
3. **Document Samples**: Note which sample/seed used for each experiment
4. **Regular Sampling**: Create new samples periodically to avoid overfitting
5. **Balanced Testing**: Mix different sample sizes and seeds

## 📈 **Expected Performance Improvements**

| Metric | Current System | With HME100K |
|--------|---------------|--------------|
| **Dataset Size** | 49 samples | 50-500+ samples |
| **Ground Truth** | Basic text | LaTeX expressions |
| **Content Type** | Synthetic | Real handwritten |
| **Grade Level** | Mixed | K-12 appropriate |
| **Reproducibility** | Limited | Full (with seeds) |
| **Evaluation** | Text similarity | Mathematical equivalence |

This setup gives you professional-grade evaluation capabilities with the flexibility to test different sample sizes and maintain reproducibility for research purposes!
