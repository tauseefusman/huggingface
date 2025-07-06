#!/usr/bin/env python3
"""
🤗 Hugging Face Showcase - Usage Guide
=====================================

Quick guide and examples for using the Hugging Face Streamlit showcase app.
"""

def print_header():
    print("🤗 Hugging Face Showcase - Usage Guide")
    print("=" * 50)
    print()

def print_installation():
    print("📦 INSTALLATION")
    print("-" * 20)
    print("1. Make sure UV is installed:")
    print("   curl -LsSf https://astral.sh/uv/install.sh | sh")
    print()
    print("2. Install dependencies:")
    print("   uv sync")
    print()
    print("3. Verify installation:")
    print("   uv run python test_setup.py")
    print()

def print_usage():
    print("🚀 USAGE")
    print("-" * 20)
    print("### Web App (Recommended)")
    print("   uv run python run_app.py")
    print("   OR")
    print("   uv run streamlit run streamlit_app.py")
    print()
    print("### Command Line Examples")
    print("   uv run python main.py              # Basic examples")
    print("   uv run python advanced_examples.py # Advanced NLP tasks")
    print("   uv run python test_setup.py        # Test environment")
    print()

def print_features():
    print("✨ FEATURES")
    print("-" * 20)
    print("📱 Interactive Web App:")
    print("   • Multi-page interface with beautiful UI")
    print("   • Real-time model inference")
    print("   • Interactive visualizations")
    print("   • Model comparison tools")
    print("   • Dataset exploration")
    print("   • Custom pipeline builder")
    print()
    print("🤖 Available Tasks:")
    print("   • Text Classification")
    print("   • Sentiment Analysis")
    print("   • Question Answering")
    print("   • Named Entity Recognition")
    print("   • Text Generation")
    print("   • Custom Pipelines")
    print()

def print_examples():
    print("📚 QUICK EXAMPLES")
    print("-" * 20)
    print("### Sentiment Analysis")
    print('   Text: "I love this product!"')
    print('   Result: POSITIVE (95.2%)')
    print()
    print("### Question Answering")
    print('   Context: "Paris is the capital of France."')
    print('   Question: "What is the capital of France?"')
    print('   Answer: "Paris"')
    print()
    print("### Text Generation")
    print('   Prompt: "Once upon a time..."')
    print('   Generated: "Once upon a time in a magical forest..."')
    print()

def print_troubleshooting():
    print("🔧 TROUBLESHOOTING")
    print("-" * 20)
    print("• Model loading issues:")
    print("  → Check internet connection for model downloads")
    print("  → Some models may require authentication")
    print()
    print("• Memory issues:")
    print("  → Use smaller models (DistilBERT instead of BERT)")
    print("  → Reduce batch size in examples")
    print()
    print("• Port conflicts:")
    print("  → Change port in run_app.py (default: 8501)")
    print()
    print("• Import errors:")
    print("  → Run 'uv sync' to reinstall dependencies")
    print()

def print_urls():
    print("🌐 USEFUL LINKS")
    print("-" * 20)
    print("• Streamlit App: http://localhost:8501")
    print("• Hugging Face Hub: https://huggingface.co")
    print("• Transformers Docs: https://huggingface.co/docs/transformers")
    print("• Datasets Docs: https://huggingface.co/docs/datasets")
    print("• Streamlit Docs: https://docs.streamlit.io")
    print()

def print_tips():
    print("💡 PRO TIPS")
    print("-" * 20)
    print("• Use caching for faster model loading")
    print("• GPU will be used automatically if available")
    print("• Try different models for comparison")
    print("• Upload your own datasets for analysis")
    print("• Combine multiple tasks in custom pipelines")
    print("• Adjust generation parameters for creativity")
    print()

def main():
    print_header()
    print_installation()
    print_usage()
    print_features()
    print_examples()
    print_troubleshooting()
    print_urls()
    print_tips()
    
    print("🎉 Ready to explore Hugging Face models and datasets!")
    print("Start with: uv run python run_app.py")
    print()

if __name__ == "__main__":
    main()
