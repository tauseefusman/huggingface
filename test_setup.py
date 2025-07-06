#!/usr/bin/env python3
"""
Quick Test Script for Hugging Face Setup
This script performs basic tests to ensure the Hugging Face environment is working correctly.
"""

import sys
import torch
from transformers import __version__ as transformers_version
from datasets import __version__ as datasets_version

def test_imports():
    """Test if all required libraries can be imported."""
    print("Testing imports...")
    
    try:
        import transformers
        import datasets
        import torch
        import numpy as np
        import pandas as pd
        print("✅ All core libraries imported successfully!")
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def test_versions():
    """Display versions of key libraries."""
    print("\nLibrary versions:")
    print(f"  Python: {sys.version.split()[0]}")
    print(f"  PyTorch: {torch.__version__}")
    print(f"  Transformers: {transformers_version}")
    print(f"  Datasets: {datasets_version}")
    
    # Check if CUDA is available
    if torch.cuda.is_available():
        print(f"  CUDA: Available (GPU: {torch.cuda.get_device_name(0)})")
    else:
        print("  CUDA: Not available (using CPU)")

def test_simple_pipeline():
    """Test a simple pipeline to ensure basic functionality."""
    print("\nTesting simple pipeline...")
    
    try:
        from transformers import pipeline
        
        # Create a simple sentiment analysis pipeline
        classifier = pipeline("sentiment-analysis")
        
        # Test with a simple sentence
        test_text = "This is a test sentence."
        result = classifier(test_text)
        
        print(f"  Input: {test_text}")
        print(f"  Output: {result[0]['label']} (score: {result[0]['score']:.4f})")
        print("✅ Simple pipeline test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Pipeline test failed: {e}")
        return False

def test_model_loading():
    """Test loading a specific model."""
    print("\nTesting model loading...")
    
    try:
        from transformers import AutoTokenizer, AutoModel
        
        model_name = "distilbert-base-uncased"
        print(f"  Loading model: {model_name}")
        
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
        
        print(f"  Model loaded successfully!")
        print(f"  Model size: {sum(p.numel() for p in model.parameters()):,} parameters")
        print("✅ Model loading test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Model loading test failed: {e}")
        return False

def test_basic_inference():
    """Test basic inference with tokenization."""
    print("\nTesting basic inference...")
    
    try:
        from transformers import AutoTokenizer, AutoModel
        import torch
        
        model_name = "distilbert-base-uncased"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
        
        # Test text
        text = "Hello, this is a test!"
        
        # Tokenize
        inputs = tokenizer(text, return_tensors="pt")
        print(f"  Input text: {text}")
        print(f"  Tokenized input shape: {inputs['input_ids'].shape}")
        
        # Run inference
        with torch.no_grad():
            outputs = model(**inputs)
        
        print(f"  Output shape: {outputs.last_hidden_state.shape}")
        print("✅ Basic inference test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Basic inference test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("🤗 Hugging Face Environment Test")
    print("="*50)
    
    # Run tests
    tests = [
        test_imports,
        test_versions,
        test_simple_pipeline,
        test_model_loading,
        test_basic_inference
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"❌ Test failed with exception: {e}")
            results.append(False)
        print()
    
    # Summary
    print("="*50)
    print("TEST SUMMARY")
    print("="*50)
    
    passed = sum(results)
    total = len(results)
    
    print(f"Tests passed: {passed}/{total}")
    
    if passed == total:
        print("🎉 All tests passed! Your Hugging Face environment is ready.")
    else:
        print("⚠️  Some tests failed. Please check the error messages above.")
        print("   This might be due to network connectivity or missing dependencies.")
    
    print("\nNext steps:")
    print("1. Run 'uv run main.py' to see the main example")
    print("2. Run 'uv run advanced_examples.py' for more advanced examples")

if __name__ == "__main__":
    main()
