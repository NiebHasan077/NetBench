#!/usr/bin/env python3
"""
System Check Script
Verifies that your system meets the requirements for LLaMA pre-training
"""

import sys
import subprocess
import json

def check_python_version():
    """Check Python version"""
    print("🔍 Checking Python version...")
    version = sys.version_info
    if version.major >= 3 and version.minor >= 8:
        print(f"   ✅ Python {version.major}.{version.minor}.{version.micro}")
        return True
    else:
        print(f"   ❌ Python {version.major}.{version.minor}.{version.micro} (Need 3.8+)")
        return False

def check_cuda():
    """Check CUDA availability"""
    print("\n🔍 Checking CUDA...")
    try:
        import torch
        if torch.cuda.is_available():
            print(f"   ✅ CUDA available: {torch.version.cuda}")
            print(f"   ✅ PyTorch version: {torch.__version__}")
            return True
        else:
            print("   ❌ CUDA not available in PyTorch")
            return False
    except ImportError:
        print("   ⚠️  PyTorch not installed")
        return False

def check_gpus():
    """Check GPU information"""
    print("\n🔍 Checking GPUs...")
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=index,name,memory.total,memory.free', '--format=csv,noheader'],
            capture_output=True,
            text=True,
            check=True
        )
        
        gpus = result.stdout.strip().split('\n')
        print(f"   ✅ Found {len(gpus)} GPU(s):")
        
        total_memory = 0
        for i, gpu in enumerate(gpus):
            parts = [p.strip() for p in gpu.split(',')]
            idx, name, total, free = parts
            total_gb = float(total.split()[0]) / 1024
            free_gb = float(free.split()[0]) / 1024
            total_memory += total_gb
            print(f"      GPU {idx}: {name}")
            print(f"              Total: {total_gb:.1f} GB, Free: {free_gb:.1f} GB")
        
        print(f"\n   📊 Total GPU Memory: {total_memory:.1f} GB")
        
        # Recommendations
        print("\n   💡 Recommendations:")
        if total_memory >= 96:
            print("      • Can train Llama 3.1 8B comfortably with DeepSpeed")
        elif total_memory >= 48:
            print("      • Can train Llama 3.1 8B with ZeRO-3 offloading")
            print("      • Can train Llama 3.2 1B easily")
        elif total_memory >= 24:
            print("      • Can train Llama 3.2 1B")
            print("      • Llama 3.1 8B will need aggressive optimizations")
        else:
            print("      • ⚠️  May struggle with both models")
            print("      • Consider using LoRA or QLoRA for fine-tuning instead")
        
        return True
        
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("   ❌ nvidia-smi not found or failed")
        return False

def check_disk_space():
    """Check available disk space"""
    print("\n🔍 Checking disk space...")
    try:
        result = subprocess.run(
            ['df', '-h', '.'],
            capture_output=True,
            text=True,
            check=True
        )
        lines = result.stdout.strip().split('\n')
        if len(lines) >= 2:
            parts = lines[1].split()
            available = parts[3]
            print(f"   ✅ Available: {available}")
            
            # Parse available space
            size_str = available.rstrip('KMGT')
            unit = available[-1]
            try:
                size_val = float(size_str)
                if unit == 'T' or (unit == 'G' and size_val > 100):
                    print("   💡 Sufficient space for checkpoints and data")
                elif unit == 'G' and size_val > 50:
                    print("   ⚠️  Limited space - monitor checkpoint storage")
                else:
                    print("   ❌ Low disk space - consider cleaning up")
            except ValueError:
                pass
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("   ⚠️  Could not check disk space")
        return False

def check_packages():
    """Check if required packages are installed"""
    print("\n🔍 Checking Python packages...")
    
    required_packages = {
        'torch': 'PyTorch',
        'transformers': 'Transformers',
        'datasets': 'Datasets',
        'accelerate': 'Accelerate',
    }
    
    optional_packages = {
        'deepspeed': 'DeepSpeed',
        'flash_attn': 'Flash Attention',
        'bitsandbytes': 'BitsAndBytes',
    }
    
    all_ok = True
    
    for package, name in required_packages.items():
        try:
            __import__(package)
            print(f"   ✅ {name}")
        except ImportError:
            print(f"   ❌ {name} - REQUIRED")
            all_ok = False
    
    print("\n   Optional packages:")
    for package, name in optional_packages.items():
        try:
            __import__(package)
            print(f"   ✅ {name}")
        except ImportError:
            print(f"   ⚠️  {name} - Recommended")
    
    return all_ok

def check_data():
    """Check if data file exists"""
    print("\n🔍 Checking data...")
    import os
    
    data_file = "data/raw/research_corpus_new.json"
    fallback = "research_corpus_new.json"
    chosen = data_file if os.path.exists(data_file) else fallback
    if os.path.exists(chosen):
        size = os.path.getsize(chosen) / (1024 * 1024)  # MB
        print(f"   ✅ {chosen} found ({size:.1f} MB)")
        
        try:
            with open(chosen, 'r') as f:
                data = json.load(f)
                print(f"   ✅ Contains {len(data)} documents")
        except Exception as e:
            print(f"   ⚠️  Could not parse JSON: {e}")
        
        return True
    else:
        print(f"   ❌ {data_file} not found")
        return False

def main():
    print("="*60)
    print("🚀 LLaMA Pre-training System Check")
    print("="*60)
    
    results = []
    results.append(("Python", check_python_version()))
    results.append(("CUDA", check_cuda()))
    results.append(("GPUs", check_gpus()))
    results.append(("Disk Space", check_disk_space()))
    results.append(("Packages", check_packages()))
    results.append(("Data", check_data()))
    
    print("\n" + "="*60)
    print("📋 Summary")
    print("="*60)
    
    for name, status in results:
        icon = "✅" if status else "❌"
        print(f"{icon} {name}")
    
    all_pass = all(status for _, status in results[:4])  # Only check critical items
    
    if all_pass:
        print("\n✅ System is ready for pre-training!")
        print("\n🎯 Next steps:")
        print("   1. Install missing packages: pip install -r requirements.txt")
        print("   2. Download models: python utils/download_models.py --model 8b")
        print("   3. Prepare data:    python training/prepare_data.py --input_file data/raw/research_corpus_new.json")
        print("   4. Train:           python training/pretrain_transformers.py --model 8b")
    else:
        print("\n⚠️  Some requirements not met. Please address the issues above.")
        print("\n💡 Installation help:")
        print("   pip install torch transformers datasets accelerate")
        print("   pip install deepspeed bitsandbytes")
    
    print("="*60)

if __name__ == "__main__":
    main()
