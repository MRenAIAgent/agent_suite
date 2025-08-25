# 🌟 Open Source Math Datasets & Benchmarks Guide

> **Comprehensive guide to open source datasets for handwritten mathematical expression recognition and K-12 homework problems**

## 📊 Dataset Overview

Based on latest research (2024), here are the most important open source datasets for mathematical expression recognition:

### 🏆 **Top Priority Datasets for K-12 Homework**

| Dataset | Size | Type | Best For | Status |
|---------|------|------|----------|--------|
| **MathWriting** | 630K samples | Online Handwritten | Largest collection, diverse expressions | ✅ Available via HuggingFace |
| **HME100K** | 99K samples | Offline Handwritten | Real K-12 expressions | ⚠️ Manual download required |
| **CROHME 2023** | ~10K samples | Online Handwritten | Standard competition benchmark | ✅ Available via Zenodo |

---

## 📚 Detailed Dataset Information

### 1. 🤗 **MathWriting Dataset** (HIGHEST PRIORITY)

- **Size**: 630,000 total samples
  - 230,000 human-written expressions
  - 400,000 synthetic expressions
- **Source**: Hugging Face (`deepcopy/MathWriting-human`)
- **Format**: Image + LaTeX ground truth pairs
- **Difficulty**: Elementary to University level
- **Type**: Online handwritten (stroke-based)
- **Best for**: Training robust OCR models with diverse expressions

**Download:**
```bash
# Install requirements
pip install datasets pillow

# Download with our script (creates benchmark-ready samples)
python scripts/download_mathwriting_dataset.py --download human --sample-size 1000
```

**Advantages:**
- ✅ Largest available dataset
- ✅ High-quality LaTeX ground truth
- ✅ Diverse writing styles
- ✅ Easy programmatic download
- ✅ Both human and synthetic data

---

### 2. 📚 **HME100K Dataset** (REAL K-12 DATA)

- **Size**: 99,109 samples (74K train + 24K test)
- **Source**: 100tal AI + GitHub
- **Format**: Image + LaTeX ground truth pairs
- **Difficulty**: K-12 level (elementary to high school)
- **Type**: Offline handwritten (image-based)
- **Best for**: Real-world K-12 math evaluation

**Download:**
```bash
# Manual download required
python scripts/setup_open_source_datasets.py --info hme100k
```

**Advantages:**
- ✅ Real K-12 student handwriting
- ✅ Large scale (99K samples)
- ✅ Age-appropriate difficulty
- ⚠️ Requires manual registration

---

### 3. 🏛️ **CROHME 2023 Dataset** (STANDARD BENCHMARK)

- **Size**: ~10,000 samples
- **Source**: Zenodo (Competition dataset)
- **Format**: InkML + LaTeX ground truth
- **Difficulty**: Elementary to High School
- **Type**: Online handwritten (stroke-based)
- **Best for**: Standardized evaluation and comparison

**Download:**
```bash
# Visit Zenodo for download links
python scripts/setup_open_source_datasets.py --info crohme2023
```

**Advantages:**
- ✅ Standard competition benchmark
- ✅ High-quality annotations
- ✅ Stroke information available
- ✅ Widely used for comparison

---

### 4. 📝 **MQHS Dataset** (HOMEWORK-LIKE)

- **Size**: Variable (thousands of samples)
- **Source**: GitHub (`Jzliu-dl/MQHSdataset`)
- **Format**: Image + Bounding boxes + LaTeX
- **Difficulty**: Elementary to High School
- **Type**: Offline handwritten
- **Best for**: Homework-style problems with spatial annotations

**Download:**
```bash
git clone https://github.com/Jzliu-dl/MQHSdataset.git real_datasets/MQHS
```

---

### 5. 🔤 **Basic Handwritten Math Symbols Dataset (BHMSDS)**

- **Size**: 27,000 images (18 symbol classes)
- **Source**: GitHub (`wblachowski/bhmsds`)
- **Format**: Image + Symbol labels
- **Difficulty**: Basic symbols (+, -, ×, ÷, etc.)
- **Type**: Symbol recognition
- **Best for**: Symbol-level recognition training

---

### 6. 📰 **MaxTex Dataset** (PRINTED FORMULAS)

- **Size**: 223,000 samples
- **Source**: Figshare
- **Format**: Image + LaTeX pairs
- **Difficulty**: Academic level
- **Type**: Printed formulas
- **Best for**: Printed math formula recognition

---

### 7. 🧮 **Aida Calculus Dataset** (SYNTHETIC)

- **Size**: 100,000 samples
- **Source**: NeurIPS 2020
- **Format**: Image + LaTeX pairs
- **Difficulty**: Calculus level
- **Type**: Synthetic handwritten
- **Best for**: Calculus-specific training

---

## 🚀 Quick Start Guide

### Step 1: Install Requirements
```bash
pip install datasets pillow
```

### Step 2: Download Priority Datasets
```bash
# 1. Download MathWriting (largest, easiest)
python scripts/download_mathwriting_dataset.py --download human --sample-size 1000

# 2. Setup HME100K (real K-12 data) - manual download required
python scripts/setup_open_source_datasets.py --info hme100k

# 3. List all available datasets
python scripts/setup_open_source_datasets.py --list
```

### Step 3: Run Benchmarks
```bash
# Test with MathWriting dataset
python run_benchmark.py -d mathwriting_sample -s gpt-5

# Compare all solutions
python run_benchmark.py -d mathwriting_sample -s all --comparison-table
```

---

## 📈 Integration with Benchmark System

All datasets can be automatically integrated into the benchmark system:

1. **Download** using provided scripts
2. **Samples** are automatically created in benchmark-compatible format
3. **Configuration** is auto-generated for `run_benchmark.py`
4. **Testing** can begin immediately

### Available Benchmark Commands:
```bash
# List datasets (including newly downloaded ones)
python run_benchmark.py --list-datasets

# Test specific dataset
python run_benchmark.py -d [dataset_name] -s gpt-5

# Compare all solutions on dataset
python run_benchmark.py -d [dataset_name] -s all --comparison-table
```

---

## 🎯 Recommendations by Use Case

### **For K-12 Homework Recognition:**
1. **HME100K** - Real student handwriting
2. **MathWriting** - Large-scale training data
3. **MQHS** - Homework-style problems

### **For Research & Benchmarking:**
1. **CROHME 2023** - Standard benchmark
2. **MathWriting** - Comprehensive evaluation
3. **MaxTex** - Printed formula comparison

### **For Symbol Recognition:**
1. **BHMSDS** - Basic math symbols
2. **MathWriting** - Complex expressions
3. **HME100K** - Real-world symbols

### **For Calculus & Advanced Math:**
1. **Aida Calculus** - Calculus-specific
2. **MathWriting** - University-level expressions
3. **MaxTex** - Academic formulas

---

## 🔧 Technical Details

### Dataset Formats Supported:
- ✅ **Image + LaTeX** (most common)
- ✅ **InkML + LaTeX** (stroke data)
- ✅ **Image + Bounding boxes** (spatial annotations)
- ✅ **Symbol classifications**

### Benchmark Integration:
- ✅ Automatic sample creation
- ✅ LaTeX ground truth validation
- ✅ Performance metrics calculation
- ✅ Multi-solution comparison

### Quality Metrics:
- ✅ OCR accuracy (character-level)
- ✅ Mathematical equivalence
- ✅ Processing speed
- ✅ Confidence scores

---

## 📞 Getting Help

### Scripts Available:
- `setup_open_source_datasets.py` - Dataset manager
- `download_mathwriting_dataset.py` - MathWriting downloader
- `run_benchmark.py` - Benchmark runner

### Commands:
```bash
# Get help for any script
python scripts/setup_open_source_datasets.py --help
python scripts/download_mathwriting_dataset.py --help
python run_benchmark.py --help
```

### Priority Order for Download:
1. **MathWriting** (easiest, largest)
2. **CROHME 2023** (standard benchmark)  
3. **HME100K** (requires registration)
4. **MQHS** (homework-style)

---

*Last updated: January 2024*
*Total available samples: 1,000,000+ across all datasets*
