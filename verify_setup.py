#!/usr/bin/env python3
"""
M1 MacBook Environment Setup Verification Script
Validates PyTorch, PyTorch Geometric, and all research dependencies
"""

import sys
import platform
import subprocess
import importlib
import warnings
warnings.filterwarnings('ignore')

def print_section(title):
    print(f"\n{'='*60}")
    print(f" {title}")
    print('='*60)

def check_system_info():
    print_section("SYSTEM INFORMATION")
    print(f"Python Version: {sys.version}")
    print(f"Platform: {platform.platform()}")
    print(f"Architecture: {platform.machine()}")
    print(f"Processor: {platform.processor()}")
    
    # Check if we're in conda environment
    try:
        conda_env = subprocess.check_output(['conda', 'info', '--envs'], 
                                          text=True, stderr=subprocess.DEVNULL)
        print(f"Conda environments available:")
        for line in conda_env.split('\n'):
            if 'kg-dense' in line or '*' in line:
                print(f"  {line}")
    except:
        print("Conda not available or not in PATH")

def check_package(package_name, import_name=None, version_attr='__version__'):
    """Check if a package is installed and return version info"""
    if import_name is None:
        import_name = package_name
    
    try:
        module = importlib.import_module(import_name)
        version = getattr(module, version_attr, 'Unknown')
        print(f"✓ {package_name}: {version}")
        return True, version
    except ImportError:
        print(f"✗ {package_name}: NOT INSTALLED")
        return False, None

def check_pytorch():
    print_section("PYTORCH VERIFICATION")
    
    # Check PyTorch installation
    success, version = check_package('PyTorch', 'torch')
    if not success:
        return False
    
    import torch
    
    # Check CUDA availability (should be False on M1)
    print(f"CUDA Available: {torch.cuda.is_available()}")
    
    # Check MPS (Metal Performance Shaders) availability
    mps_available = torch.backends.mps.is_available()
    print(f"MPS Available: {mps_available}")
    
    if mps_available:
        print("✓ M1 GPU acceleration (MPS) is available!")
        
        # Test MPS device creation
        try:
            device = torch.device("mps")
            x = torch.randn(10, 10, device=device)
            print("✓ MPS device creation successful")
        except Exception as e:
            print(f"✗ MPS device creation failed: {e}")
            return False
    else:
        print("⚠ MPS not available - will use CPU only")
    
    # Test basic tensor operations
    try:
        x = torch.randn(5, 5)
        y = torch.randn(5, 5)
        z = torch.mm(x, y)
        print("✓ Basic tensor operations working")
    except Exception as e:
        print(f"✗ Basic tensor operations failed: {e}")
        return False
    
    return True

def check_torch_geometric():
    print_section("PYTORCH GEOMETRIC VERIFICATION")
    
    # Check main PyG installation
    success, version = check_package('PyTorch Geometric', 'torch_geometric')
    if not success:
        return False
    
    # Check PyG submodules
    pyg_modules = [
        ('torch_geometric.nn', 'Neural Network Layers'),
        ('torch_geometric.data', 'Data Handling'),
        ('torch_geometric.transforms', 'Data Transforms'),
        ('torch_geometric.utils', 'Utilities'),
        ('torch_geometric.datasets', 'Datasets')
    ]
    
    for module, description in pyg_modules:
        try:
            importlib.import_module(module)
            print(f"✓ {description}: Available")
        except ImportError:
            print(f"✗ {description}: NOT AVAILABLE")
            return False
    
    # Test basic PyG functionality
    try:
        import torch
        from torch_geometric.data import Data
        from torch_geometric.nn import GCNConv
        
        # Create simple graph
        edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long)
        x = torch.randn(3, 16)
        data = Data(x=x, edge_index=edge_index)
        
        # Create simple GCN layer
        conv = GCNConv(16, 32)
        out = conv(data.x, data.edge_index)
        
        print("✓ Basic PyG operations working")
        return True
        
    except Exception as e:
        print(f"✗ Basic PyG operations failed: {e}")
        return False

def check_transformers():
    print_section("TRANSFORMERS VERIFICATION")
    
    success, version = check_package('Transformers', 'transformers')
    if not success:
        return False
    
    # Test basic transformers functionality
    try:
        from transformers import AutoTokenizer, AutoModel
        
        # Test tokenizer loading (lightweight)
        tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
        print("✓ Tokenizer loading working")
        
        # Test basic tokenization
        text = "Hello, this is a test."
        tokens = tokenizer(text, return_tensors='pt')
        print(f"✓ Tokenization working (input_ids shape: {tokens['input_ids'].shape})")
        
        return True
        
    except Exception as e:
        print(f"✗ Transformers functionality failed: {e}")
        return False

def check_additional_packages():
    print_section("ADDITIONAL PACKAGES VERIFICATION")
    
    packages = [
        ('datasets', 'datasets'),
        ('accelerate', 'accelerate'),
        ('faiss-cpu', 'faiss'),
        ('wandb', 'wandb'),
        ('scikit-learn', 'sklearn'),
        ('matplotlib', 'matplotlib'),
        ('seaborn', 'seaborn'),
        ('pandas', 'pandas'),
        ('numpy', 'numpy')
    ]
    
    all_success = True
    for package_name, import_name in packages:
        success, _ = check_package(package_name, import_name)
        if not success:
            all_success = False
    
    return all_success

def run_performance_test():
    print_section("PERFORMANCE TEST")
    
    try:
        import torch
        import time
        
        # CPU test
        print("Running CPU performance test...")
        start_time = time.time()
        x = torch.randn(1000, 1000)
        y = torch.randn(1000, 1000)
        z = torch.mm(x, y)
        cpu_time = time.time() - start_time
        print(f"CPU Matrix Multiplication (1000x1000): {cpu_time:.4f} seconds")
        
        # MPS test (if available)
        if torch.backends.mps.is_available():
            print("Running MPS performance test...")
            device = torch.device("mps")
            start_time = time.time()
            x_mps = torch.randn(1000, 1000, device=device)
            y_mps = torch.randn(1000, 1000, device=device)
            z_mps = torch.mm(x_mps, y_mps)
            torch.mps.synchronize()  # Wait for MPS operations to complete
            mps_time = time.time() - start_time
            print(f"MPS Matrix Multiplication (1000x1000): {mps_time:.4f} seconds")
            
            if mps_time < cpu_time:
                speedup = cpu_time / mps_time
                print(f"✓ MPS Speedup: {speedup:.2f}x faster than CPU")
            else:
                print("⚠ MPS slower than CPU (normal for small operations)")
        
        return True
        
    except Exception as e:
        print(f"✗ Performance test failed: {e}")
        return False

def main():
    print("M1 MacBook Research Environment Verification")
    print("=" * 60)
    
    results = []
    
    # Run all checks
    check_system_info()
    results.append(("PyTorch", check_pytorch()))
    results.append(("PyTorch Geometric", check_torch_geometric()))
    results.append(("Transformers", check_transformers()))
    results.append(("Additional Packages", check_additional_packages()))
    results.append(("Performance Test", run_performance_test()))
    
    # Summary
    print_section("VERIFICATION SUMMARY")
    
    all_passed = True
    for component, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{component:<20}: {status}")
        if not passed:
            all_passed = False
    
    print("\n" + "="*60)
    if all_passed:
        print("🎉 ALL CHECKS PASSED! Environment is ready for research.")
        print("\nNext steps:")
        print("1. Proceed with Dataset Preparation (Task 1.2)")
        print("2. Implement Strong Baselines (Task 1.3)")
        print("3. Begin Phase 2: Core Model Development")
    else:
        print("❌ SOME CHECKS FAILED! Please fix issues before proceeding.")
        print("\nTroubleshooting:")
        print("1. Ensure you're in the 'kg-dense' conda environment")
        print("2. Re-run installation commands if packages are missing")
        print("3. Check conda environment activation")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
