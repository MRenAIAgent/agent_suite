#!/usr/bin/env python3
"""
Benchmark Setup Script for Math Recognition System

This script helps users set up and configure the improved benchmarking system
with OCR model switching and comprehensive evaluation capabilities.
"""

import os
import sys
import json
import argparse
import subprocess
from pathlib import Path
from typing import Dict, List, Any, Optional

def print_banner():
    """Print welcome banner."""
    print("\n" + "=" * 80)
    print("🚀 MATH RECOGNITION BENCHMARK SETUP")
    print("=" * 80)
    print("Setting up comprehensive OCR evaluation system with:")
    print("• Multiple OCR provider configurations")
    print("• Automated benchmarking pipeline") 
    print("• Performance comparison tools")
    print("• Industry-standard evaluation metrics")
    print("=" * 80 + "\n")

def check_python_version():
    """Check Python version compatibility."""
    print("🐍 Checking Python version...")
    
    if sys.version_info < (3, 8):
        print("❌ Python 3.8+ required. Current version:", sys.version)
        return False
    
    print(f"✅ Python {sys.version.split()[0]} - Compatible")
    return True

def check_dependencies():
    """Check required dependencies."""
    print("\n📦 Checking dependencies...")
    
    required_packages = [
        "asyncio", "json", "pathlib", "dataclasses", "enum",
        "typing", "argparse", "time", "datetime"
    ]
    
    optional_packages = {
        "openai": "OpenAI GPT-4 Vision support",
        "torch": "PyTorch for local models (GOT-OCR2.0, UniMERNet)",
        "transformers": "HuggingFace transformers for advanced OCR",
        "aiohttp": "Async HTTP requests",
        "numpy": "Numerical computations",
        "matplotlib": "Plotting and visualization",
        "pandas": "Data analysis and reporting"
    }
    
    # Check required packages (all built-in)
    missing_required = []
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_required.append(package)
    
    if missing_required:
        print(f"❌ Missing required packages: {missing_required}")
        return False
    
    print("✅ All required packages available")
    
    # Check optional packages
    print("\n🔧 Optional packages:")
    available_optional = []
    
    for package, description in optional_packages.items():
        try:
            __import__(package)
            print(f"✅ {package} - {description}")
            available_optional.append(package)
        except ImportError:
            print(f"⚠️  {package} - {description} (not installed)")
    
    return True

def setup_directory_structure():
    """Create necessary directory structure."""
    print("\n📁 Setting up directory structure...")
    
    base_dir = Path(__file__).parent
    directories = [
        "configs",
        "results", 
        "datasets",
        "reports",
        "logs"
    ]
    
    for dir_name in directories:
        dir_path = base_dir / dir_name
        dir_path.mkdir(exist_ok=True)
        print(f"✅ Created/verified: {dir_path}")
    
    return base_dir

def create_environment_template():
    """Create environment variable template."""
    print("\n🔐 Creating environment template...")
    
    env_template = """# Math Recognition Benchmark Environment Variables
# Copy this file to .env and fill in your API keys

# OpenAI GPT-4 Vision
OPENAI_API_KEY=your_openai_api_key_here

# Mathpix OCR (optional)
MATHPIX_APP_ID=your_mathpix_app_id_here
MATHPIX_APP_KEY=your_mathpix_app_key_here

# Hugging Face (for downloading models)
HUGGINGFACE_TOKEN=your_hf_token_here

# Optional: Custom model paths
CUSTOM_MODEL_DIR=/path/to/your/models

# Benchmark settings
BENCHMARK_OUTPUT_DIR=./results
BENCHMARK_LOG_LEVEL=INFO
"""
    
    env_file = Path(__file__).parent / ".env.template"
    with open(env_file, 'w') as f:
        f.write(env_template)
    
    print(f"✅ Environment template created: {env_file}")
    print("📝 Copy .env.template to .env and add your API keys")

def install_optional_dependencies():
    """Install optional dependencies with user confirmation."""
    print("\n📦 Optional dependency installation...")
    
    packages_to_install = {
        "openai": "pip install openai>=1.0.0",
        "torch": "pip install torch torchvision torchaudio",
        "transformers": "pip install transformers>=4.20.0",
        "aiohttp": "pip install aiohttp",
        "numpy": "pip install numpy",
        "matplotlib": "pip install matplotlib",
        "pandas": "pip install pandas"
    }
    
    print("The following packages are recommended for full functionality:")
    for package, install_cmd in packages_to_install.items():
        print(f"  • {package}: {install_cmd}")
    
    response = input("\nInstall all recommended packages? (y/n): ").lower().strip()
    
    if response == 'y':
        print("\n🔧 Installing packages...")
        
        for package, install_cmd in packages_to_install.items():
            print(f"\nInstalling {package}...")
            try:
                subprocess.run(install_cmd.split(), check=True, capture_output=True)
                print(f"✅ {package} installed successfully")
            except subprocess.CalledProcessError as e:
                print(f"⚠️  Failed to install {package}: {e}")
                print(f"   You can install manually: {install_cmd}")
    else:
        print("⏭️  Skipping package installation")
        print("   You can install packages later using the commands above")

def create_sample_configs():
    """Create sample configuration files."""
    print("\n⚙️  Creating sample configurations...")
    
    configs_dir = Path(__file__).parent / "configs"
    
    # Sample benchmark configuration
    sample_config = {
        "name": "Quick Test Configuration",
        "description": "Fast configuration for testing the benchmark system",
        "primary_ocr": {
            "provider": "gpt4_vision",
            "model_name": "gpt-4o",
            "api_key_env": "OPENAI_API_KEY",
            "custom_params": {
                "max_tokens": 2000,
                "temperature": 0.1
            }
        },
        "processing_strategy": "vision_only",
        "evaluation_metrics": [
            "ocr_accuracy",
            "processing_time",
            "confidence_score"
        ],
        "dataset_types": ["sample_data"],
        "max_samples": 10,
        "parallel_processing": True
    }
    
    config_file = configs_dir / "quick_test.json"
    with open(config_file, 'w') as f:
        json.dump(sample_config, f, indent=2)
    
    print(f"✅ Sample configuration created: {config_file}")

def run_initial_test():
    """Run initial test to verify setup."""
    print("\n🧪 Running initial system test...")
    
    try:
        # Import our modules
        sys.path.insert(0, str(Path(__file__).parent))
        from ocr_config import get_config_manager
        from improved_benchmark_runner import ImprovedBenchmarkRunner
        
        # Test configuration manager
        config_manager = get_config_manager()
        configs = config_manager.list_configs()
        
        print(f"✅ Configuration manager loaded with {len(configs)} configurations")
        
        # Test benchmark runner initialization
        runner = ImprovedBenchmarkRunner()
        print("✅ Benchmark runner initialized successfully")
        
        print("\n🎯 Available configurations:")
        for name, description in configs.items():
            print(f"  • {name}: {description}")
        
        return True
        
    except Exception as e:
        print(f"❌ System test failed: {e}")
        print("   Check the error above and ensure all files are properly created")
        return False

def print_usage_instructions():
    """Print usage instructions."""
    print("\n" + "=" * 80)
    print("🎉 SETUP COMPLETE!")
    print("=" * 80)
    
    print("\n📋 QUICK START GUIDE:")
    print("-" * 40)
    
    print("\n1️⃣  Configure API Keys:")
    print("   cp .env.template .env")
    print("   # Edit .env and add your OpenAI API key")
    
    print("\n2️⃣  List Available Configurations:")
    print("   python improved_benchmark_runner.py --list-configs")
    
    print("\n3️⃣  Validate Configuration:")
    print("   python improved_benchmark_runner.py --validate gpt4v_only")
    
    print("\n4️⃣  Run Quick Test:")
    print("   python improved_benchmark_runner.py --config gpt4v_only --samples 5")
    
    print("\n5️⃣  Compare Multiple Configurations:")
    print("   python improved_benchmark_runner.py --compare gpt4v_only geometry_specialist")
    
    print("\n🔧 ADVANCED USAGE:")
    print("-" * 40)
    
    print("\n• Custom Configuration:")
    print("   # Edit configs/custom_config.json")
    print("   python improved_benchmark_runner.py --config custom_config")
    
    print("\n• External Dataset:")
    print("   python improved_benchmark_runner.py --config comprehensive_parallel \\")
    print("                                       --dataset /path/to/dataset.json \\")
    print("                                       --samples 1000")
    
    print("\n• Batch Evaluation:")
    print("   for config in gpt4v_only geometry_specialist math_expression_expert; do")
    print("       python improved_benchmark_runner.py --config $config --samples 100")
    print("   done")
    
    print("\n📊 RESULTS LOCATION:")
    print("-" * 40)
    print("   Results are saved in: ./results/")
    print("   Logs are saved in: ./logs/")
    print("   Reports are saved in: ./reports/")
    
    print("\n📚 DOCUMENTATION:")
    print("-" * 40)
    print("   • Evaluation Guide: docs/EVALUATION_RESEARCH_GUIDE.md")
    print("   • Configuration Help: ocr_config.py")
    print("   • Benchmark Runner: improved_benchmark_runner.py")
    
    print("\n🆘 TROUBLESHOOTING:")
    print("-" * 40)
    print("   • Check API keys in .env file")
    print("   • Verify Python 3.8+ is installed")
    print("   • Install optional dependencies if needed")
    print("   • Check logs in ./logs/ for detailed errors")
    
    print("\n" + "=" * 80)
    print("Happy Benchmarking! 🚀")
    print("=" * 80 + "\n")

def main():
    """Main setup function."""
    parser = argparse.ArgumentParser(description="Math Recognition Benchmark Setup")
    parser.add_argument("--skip-deps", action="store_true", 
                       help="Skip optional dependency installation")
    parser.add_argument("--quick", action="store_true",
                       help="Quick setup without interactive prompts")
    
    args = parser.parse_args()
    
    print_banner()
    
    # Step 1: Check Python version
    if not check_python_version():
        return 1
    
    # Step 2: Check dependencies
    if not check_dependencies():
        return 1
    
    # Step 3: Setup directory structure
    base_dir = setup_directory_structure()
    
    # Step 4: Create environment template
    create_environment_template()
    
    # Step 5: Install optional dependencies
    if not args.skip_deps and not args.quick:
        install_optional_dependencies()
    
    # Step 6: Create sample configurations
    create_sample_configs()
    
    # Step 7: Run initial test
    if not run_initial_test():
        print("\n⚠️  Setup completed with warnings. Check errors above.")
        return 1
    
    # Step 8: Print usage instructions
    print_usage_instructions()
    
    return 0

if __name__ == "__main__":
    exit(main()) 