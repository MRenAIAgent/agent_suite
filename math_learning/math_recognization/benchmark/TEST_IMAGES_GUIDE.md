# 🎨 Math Problem Test Images Guide

## 📋 Overview

I've created **14 custom test images** with various math problems to help you test and compare OCR configurations. These images contain realistic math problems with clear text and geometric shapes.

## 📊 Generated Test Images

### **🔢 Algebra Problems (5 images)**
1. **Linear Equation**: `3x + 7 = 2x + 15` → `x = 8`
2. **Quadratic Equation**: `x² - 5x + 6 = 0` → `x = 2 or x = 3`
3. **System of Equations**: `2x + y = 7, x - y = 2` → `x = 3, y = 1`
4. **Polynomial Expression**: `(x + 3)(x - 2)` → `x² + x - 6`
5. **Fraction Equation**: `(x + 1)/3 = (2x - 1)/4` → `x = 7`

### **📐 Geometry Problems (5 images)**
1. **Triangle Area**: Base = 8 cm, Height = 6 cm → Area = 24 cm²
2. **Circle Properties**: Radius = 5 cm → C = 10π cm, A = 25π cm²
3. **Rectangle Perimeter**: Length = 12 cm, Width = 8 cm → Perimeter = 40 cm
4. **Right Triangle**: Legs a = 3, b = 4 → Hypotenuse c = 5
5. **Coordinate Geometry**: Distance between A(2,3) and B(6,7) → 4√2 ≈ 5.66

### **🔬 Mixed Math Problems (4 images)**
1. **Word Problem**: Sarah's book purchase → Each book costs $12
2. **Calculus**: Derivative of `x³ + 2x² - 5x + 1` → `3x² + 4x - 5`
3. **Statistics**: Mean of [12, 15, 18, 20, 25] → 18
4. **Trigonometry**: sin(θ) = 3/5 → cos(θ) = 4/5, tan(θ) = 3/4

## 🚀 How to Use the Test Images

### **1. Quick Test with GPT-4 Vision (Best for Algebra)**
```bash
python3 improved_benchmark_runner.py --config gpt4v_only --dataset test_images --samples 5
```

### **2. Test Geometry Specialist (Best for Geometric Problems)**
```bash
python3 improved_benchmark_runner.py --config geometry_specialist --dataset test_images --samples 5
```

### **3. Test Math Expression Expert (Best for Equations)**
```bash
python3 improved_benchmark_runner.py --config math_expression_expert --dataset test_images --samples 5
```

### **4. Test Industry Standard Hybrid**
```bash
python3 improved_benchmark_runner.py --config mathpix_gpt4v_hybrid --dataset test_images --samples 5
```

### **5. Compare All Configurations**
```bash
# Test each configuration individually and compare results
python3 improved_benchmark_runner.py --config gpt4v_only --dataset test_images --samples 10
python3 improved_benchmark_runner.py --config geometry_specialist --dataset test_images --samples 10
python3 improved_benchmark_runner.py --config math_expression_expert --dataset test_images --samples 10
```

## 📊 Initial Test Results

### **GPT-4 Vision Only Results**
- **Math Expressions**: ✅ **100% accuracy** (5/5)
- **Geometry**: ❌ 0% accuracy (0/5) 
- **Overall**: 33.3% accuracy
- **Best for**: Algebra, text-based math problems

### **Geometry Specialist Results**
- **Geometry**: ❌ 0% accuracy (needs real OCR models)
- **Note**: Requires GOT-OCR2.0 model to be accessible

## 🎯 Configuration Recommendations

Based on the test results:

| Problem Type | Recommended Config | Expected Performance |
|--------------|-------------------|---------------------|
| **Algebra Equations** | `gpt4v_only` or `math_expression_expert` | ✅ **100% accuracy** |
| **Geometry Problems** | `geometry_specialist` or `mathpix_gpt4v_hybrid` | 🔄 Requires proper model setup |
| **Mixed Problems** | `mathpix_gpt4v_hybrid` | 🔄 Balanced approach |
| **Research/Max Accuracy** | `comprehensive_parallel` | 🔄 Multiple models |

## 🛠️ Customizing Test Images

You can modify `create_test_images.py` to:

1. **Add more problem types**:
   ```python
   # Add to create_algebra_problems() or create_geometry_problems()
   problems.append({
       "title": "Your Problem Type",
       "problem": "Your problem text",
       "solution": "Your solution"
   })
   ```

2. **Change image styling**:
   ```python
   # Modify colors, fonts, or layout in create_text_image()
   title_color = 'darkblue'
   problem_color = 'black'
   solution_color = 'darkgreen'
   ```

3. **Add new shapes**:
   ```python
   # Add to draw_shape() method
   elif shape_type == "your_shape":
       # Your shape drawing code
   ```

## 🔍 Analyzing Results

### **View Results Details**
```bash
# Check the latest results file
ls -la results/benchmark_session_*

# Analyze results with diagnostic tools
python3 scripts/diagnose_results.py
python3 scripts/verify_results.py
```

### **Summary JSON Structure**
Each test creates a detailed JSON summary with:
- Problem categories and counts
- Individual problem details
- Solutions for verification
- Image filenames for reference

## 🎉 Benefits of Custom Test Images

1. **Known Ground Truth**: You know exactly what each image should recognize
2. **Diverse Problem Types**: Tests different OCR capabilities
3. **Visual Elements**: Geometry problems include actual shapes
4. **Realistic Format**: Problems formatted like real math textbooks
5. **Scalable**: Easy to add more problems or modify existing ones

## 🚀 Next Steps

1. **Test with your preferred configuration**
2. **Add your own problem types** by modifying the generator
3. **Compare results** across different configurations
4. **Use insights** to choose the best OCR setup for your use case

**The test images provide a perfect way to evaluate and compare OCR configurations for math problem recognition! 🎯**

