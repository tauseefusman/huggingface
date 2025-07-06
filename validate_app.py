#!/usr/bin/env python3
"""
Streamlit App Validation
========================

Test script to validate the Streamlit app without running the full server.
This checks imports, basic functionality, and component availability.
"""

import sys
import warnings
warnings.filterwarnings('ignore')

def test_imports():
    """Test all required imports"""
    print("🔍 Testing imports...")
    
    try:
        import streamlit as st
        print("✅ Streamlit imported successfully")
        
        import pandas as pd
        print("✅ Pandas imported successfully")
        
        import plotly.express as px
        import plotly.graph_objects as go
        print("✅ Plotly imported successfully")
        
        from transformers import pipeline
        print("✅ Transformers imported successfully")
        
        from datasets import load_dataset
        print("✅ Datasets imported successfully")
        
        import torch
        print("✅ PyTorch imported successfully")
        
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def test_pipelines():
    """Test basic pipeline creation"""
    print("\n🧪 Testing pipeline creation...")
    
    try:
        from transformers import pipeline
        
        # Test sentiment analysis pipeline
        print("  📊 Testing sentiment analysis pipeline...")
        classifier = pipeline("sentiment-analysis", 
                             model="distilbert-base-uncased-finetuned-sst-2-english")
        result = classifier("This is a test!")
        print(f"    ✅ Sentiment: {result[0]['label']} ({result[0]['score']:.2%})")
        
        # Test text generation pipeline
        print("  💬 Testing text generation pipeline...")
        generator = pipeline("text-generation", 
                           model="gpt2",
                           max_length=30)
        result = generator("Hello world")
        print(f"    ✅ Generated text: {result[0]['generated_text'][:50]}...")
        
        return True
    except Exception as e:
        print(f"❌ Pipeline error: {e}")
        return False

def test_app_components():
    """Test app components without Streamlit server"""
    print("\n🎨 Testing app components...")
    
    try:
        # Test sample data creation
        import pandas as pd
        
        # Sample performance data
        performance_data = pd.DataFrame({
            'Model': ['BERT-base', 'RoBERTa-base', 'DistilBERT'],
            'Accuracy': [0.89, 0.91, 0.85],
            'Speed (ms)': [150, 180, 90],
            'Parameters (M)': [110, 125, 66]
        })
        print(f"    ✅ Sample data created: {len(performance_data)} rows")
        
        # Test plotly visualization
        import plotly.express as px
        fig = px.scatter(performance_data, 
                        x='Speed (ms)', 
                        y='Accuracy',
                        size='Parameters (M)',
                        color='Model')
        print("    ✅ Plotly visualization created")
        
        return True
    except Exception as e:
        print(f"❌ Component error: {e}")
        return False

def test_streamlit_syntax():
    """Test Streamlit app syntax without running"""
    print("\n📝 Testing Streamlit app syntax...")
    
    try:
        # Try to compile the streamlit app
        with open('streamlit_app.py', 'r', encoding='utf-8') as f:
            app_code = f.read()
        
        # Try to compile the code
        compile(app_code, 'streamlit_app.py', 'exec')
        print("    ✅ Streamlit app syntax is valid")
        
        # Check for required components
        required_components = [
            'st.set_page_config',
            'st.sidebar',
            'st.selectbox',
            'st.button',
            'st.markdown',
            'st.text_area',
            'st.plotly_chart'
        ]
        
        missing_components = []
        for component in required_components:
            if component not in app_code:
                missing_components.append(component)
        
        if missing_components:
            print(f"    ⚠️  Missing components: {', '.join(missing_components)}")
        else:
            print("    ✅ All required Streamlit components found")
        
        return len(missing_components) == 0
    except Exception as e:
        print(f"❌ Syntax error: {e}")
        return False

def main():
    """Run all validation tests"""
    print("🤗 Hugging Face Streamlit App Validation")
    print("=" * 50)
    
    tests = [
        ("Imports", test_imports),
        ("Pipelines", test_pipelines),
        ("Components", test_app_components),
        ("Streamlit Syntax", test_streamlit_syntax)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n🔬 Running {test_name} test...")
        result = test_func()
        results.append((test_name, result))
    
    # Summary
    print("\n" + "=" * 50)
    print("📋 Test Summary:")
    
    passed = 0
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {status} {test_name}")
        if result:
            passed += 1
    
    print(f"\n🎯 Results: {passed}/{len(tests)} tests passed")
    
    if passed == len(tests):
        print("\n🎉 All tests passed! Your Streamlit app is ready to run!")
        print("\n🚀 To start the app, run:")
        print("    uv run python run_app.py")
        print("    OR")
        print("    uv run streamlit run streamlit_app.py")
    else:
        print("\n⚠️  Some tests failed. Please check the errors above.")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
