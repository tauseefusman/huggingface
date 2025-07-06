#!/usr/bin/env python3
"""
Hugging Face Model and Dataset Example
This script demonstrates how to use a pre-trained model and dataset from Hugging Face.
"""

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from datasets import load_dataset
import pandas as pd
import numpy as np
from typing import List, Dict, Any

def setup_device() -> str:
    """Setup and return the appropriate device (CPU/GPU)."""
    if torch.cuda.is_available():
        device = "cuda"
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = "cpu"
        print("Using CPU")
    
    return device

def load_model_and_tokenizer(model_name: str = "distilbert-base-uncased-finetuned-sst-2-english"):
    """
    Load a pre-trained model and tokenizer for sentiment analysis.
    
    Args:
        model_name: Name of the model to load from Hugging Face Hub
        
    Returns:
        tuple: (tokenizer, model)
    """
    print(f"Loading model: {model_name}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    
    print("✅ Model and tokenizer loaded successfully!")
    return tokenizer, model

def load_sample_dataset(dataset_name: str = "imdb", split: str = "test", num_samples: int = 5):
    """
    Load a sample dataset from Hugging Face datasets.
    
    Args:
        dataset_name: Name of the dataset to load
        split: Which split to load (train, test, validation)
        num_samples: Number of samples to load
        
    Returns:
        Dataset subset
    """
    print(f"Loading dataset: {dataset_name} ({split} split)")
    
    # Load the dataset
    dataset = load_dataset(dataset_name, split=split, streaming=True)
    
    # Take only a few samples for demonstration
    samples = []
    for i, sample in enumerate(dataset):
        if i >= num_samples:
            break
        samples.append(sample)
    
    print(f"✅ Loaded {len(samples)} samples from {dataset_name}")
    return samples

def create_sentiment_pipeline(model_name: str = "distilbert-base-uncased-finetuned-sst-2-english"):
    """
    Create a sentiment analysis pipeline using Hugging Face transformers.
    
    Args:
        model_name: Name of the model to use
        
    Returns:
        Hugging Face pipeline object
    """
    print("Creating sentiment analysis pipeline...")
    
    sentiment_pipeline = pipeline(
        "sentiment-analysis",
        model=model_name,
        tokenizer=model_name,
        return_all_scores=True
    )
    
    print("✅ Pipeline created successfully!")
    return sentiment_pipeline

def analyze_text_samples(pipeline_obj, texts: List[str]) -> List[Dict[str, Any]]:
    """
    Analyze sentiment of text samples using the pipeline.
    
    Args:
        pipeline_obj: Hugging Face pipeline object
        texts: List of texts to analyze
        
    Returns:
        List of prediction results
    """
    print("Analyzing text samples...")
    
    results = []
    for i, text in enumerate(texts):
        # Truncate text if too long
        truncated_text = text[:500] + "..." if len(text) > 500 else text
        
        # Get prediction
        prediction = pipeline_obj(truncated_text)
        
        result = {
            "sample_id": i + 1,
            "text": truncated_text,
            "predictions": prediction[0]  # Get first (and only) result
        }
        results.append(result)
        
        print(f"Sample {i + 1}: {prediction[0][0]['label']} ({prediction[0][0]['score']:.4f})")
    
    return results

def demonstrate_manual_inference(tokenizer, model, device: str, text: str):
    """
    Demonstrate manual inference without using pipeline.
    
    Args:
        tokenizer: Tokenizer object
        model: Model object
        device: Device to use for inference
        text: Text to analyze
    """
    print("\n" + "="*50)
    print("MANUAL INFERENCE DEMONSTRATION")
    print("="*50)
    
    # Move model to device
    model.to(device)
    model.eval()
    
    # Tokenize input
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Perform inference
    with torch.no_grad():
        outputs = model(**inputs)
        predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
    
    # Get predicted class
    predicted_class = torch.argmax(predictions, dim=-1).item()
    confidence = predictions[0][predicted_class].item()
    
    # Map class to label (for sentiment analysis models)
    labels = {0: "NEGATIVE", 1: "POSITIVE"}
    predicted_label = labels.get(predicted_class, f"CLASS_{predicted_class}")
    
    print(f"Text: {text}")
    print(f"Predicted: {predicted_label}")
    print(f"Confidence: {confidence:.4f}")
    print(f"All scores: {predictions.cpu().numpy()[0]}")

def main():
    """Main function to demonstrate Hugging Face model and dataset usage."""
    print("🤗 Hugging Face Model and Dataset Example")
    print("="*50)
    
    # Setup device
    device = setup_device()
    
    # Load model and tokenizer
    model_name = "distilbert-base-uncased-finetuned-sst-2-english"
    tokenizer, model = load_model_and_tokenizer(model_name)
    
    # Create pipeline for easy inference
    sentiment_pipeline = create_sentiment_pipeline(model_name)
    
    # Load sample dataset
    print("\n" + "-"*50)
    print("LOADING DATASET")
    print("-"*50)
    
    dataset_samples = load_sample_dataset("imdb", "test", 3)
    
    # Extract texts from dataset samples
    texts = [sample["text"] for sample in dataset_samples]
    
    # Analyze using pipeline
    print("\n" + "-"*50)
    print("PIPELINE ANALYSIS")
    print("-"*50)
    
    results = analyze_text_samples(sentiment_pipeline, texts)
    
    # Display results in a nice format
    print("\n" + "="*50)
    print("RESULTS SUMMARY")
    print("="*50)
    
    for result in results:
        print(f"\nSample {result['sample_id']}:")
        print(f"Text: {result['text'][:100]}...")
        for pred in result['predictions']:
            print(f"  {pred['label']}: {pred['score']:.4f}")
    
    # Demonstrate manual inference
    sample_text = "This movie was absolutely fantastic! I loved every minute of it."
    demonstrate_manual_inference(tokenizer, model, device, sample_text)
    
    # Test with custom examples
    print("\n" + "="*50)
    print("CUSTOM EXAMPLES")
    print("="*50)
    
    custom_texts = [
        "I hate this product, it's terrible!",
        "This is amazing, I love it so much!",
        "The weather is okay today.",
        "Best purchase I've ever made!"
    ]
    
    custom_results = analyze_text_samples(sentiment_pipeline, custom_texts)
    
    print("\n✅ Analysis complete!")
    print("\nModel used:", model_name)
    print("Dataset used: IMDB movie reviews")
    print(f"Device: {device}")

if __name__ == "__main__":
    main()
