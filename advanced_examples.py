#!/usr/bin/env python3
"""
Advanced Hugging Face Examples
This script demonstrates more advanced use cases including:
- Text generation
- Question answering
- Text summarization
- Token classification (NER)
"""

from transformers import (
    pipeline, 
    AutoTokenizer, 
    AutoModelForQuestionAnswering,
    AutoModelForTokenClassification
)
from datasets import load_dataset
import torch

def text_generation_example():
    """Demonstrate text generation using GPT-2."""
    print("\n" + "="*60)
    print("TEXT GENERATION EXAMPLE (GPT-2)")
    print("="*60)
    
    # Create text generation pipeline
    generator = pipeline("text-generation", model="gpt2")
    
    prompts = [
        "The future of artificial intelligence is",
        "In a world where technology",
        "The most important thing about machine learning is"
    ]
    
    for prompt in prompts:
        print(f"\nPrompt: {prompt}")
        print("-" * 40)
        
        # Generate text
        generated = generator(
            prompt, 
            max_length=50, 
            num_return_sequences=1,
            do_sample=True,
            temperature=0.7,
            pad_token_id=generator.tokenizer.eos_token_id
        )
        
        print(f"Generated: {generated[0]['generated_text']}")

def question_answering_example():
    """Demonstrate question answering using BERT."""
    print("\n" + "="*60)
    print("QUESTION ANSWERING EXAMPLE (BERT)")
    print("="*60)
    
    # Create QA pipeline
    qa_pipeline = pipeline("question-answering")
    
    # Sample context and questions
    context = """
    Hugging Face is a company that develops tools for building applications using machine learning. 
    The company was founded in 2016 by Clément Delangue, Julien Chaumond, and Thomas Wolf. 
    Hugging Face is best known for its Transformers library, which provides access to thousands of 
    pre-trained models. The company is headquartered in New York City and has raised significant 
    funding from investors.
    """
    
    questions = [
        "When was Hugging Face founded?",
        "Who are the founders of Hugging Face?",
        "What is Hugging Face best known for?",
        "Where is Hugging Face headquartered?"
    ]
    
    print(f"Context: {context.strip()}")
    print("\n" + "-" * 40)
    
    for question in questions:
        result = qa_pipeline(question=question, context=context)
        print(f"Q: {question}")
        print(f"A: {result['answer']} (confidence: {result['score']:.4f})")
        print()

def summarization_example():
    """Demonstrate text summarization."""
    print("\n" + "="*60)
    print("TEXT SUMMARIZATION EXAMPLE")
    print("="*60)
    
    # Create summarization pipeline
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    
    # Long text to summarize
    text = """
    Artificial intelligence (AI) is intelligence demonstrated by machines, in contrast to the natural 
    intelligence displayed by humans and animals. Leading AI textbooks define the field as the study of 
    "intelligent agents": any device that perceives its environment and takes actions that maximize its 
    chance of successfully achieving its goals. Colloquially, the term "artificial intelligence" is often 
    used to describe machines that mimic "cognitive" functions that humans associate with the human mind, 
    such as "learning" and "problem solving". As machines become increasingly capable, tasks considered to 
    require "intelligence" are often removed from the definition of AI, a phenomenon known as the AI effect. 
    A quip in Tesler's Theorem says "AI is whatever hasn't been done yet." For instance, optical character 
    recognition is frequently excluded from things considered to be AI, having become a routine technology. 
    Modern machine learning techniques are heavy on data and generally produce statistical models that are 
    not easily interpreted by humans but are effective at their intended task.
    """
    
    print("Original text:")
    print(text.strip())
    print("\n" + "-" * 40)
    
    # Summarize the text
    summary = summarizer(text, max_length=100, min_length=30, do_sample=False)
    
    print("Summary:")
    print(summary[0]['summary_text'])

def named_entity_recognition_example():
    """Demonstrate Named Entity Recognition (NER)."""
    print("\n" + "="*60)
    print("NAMED ENTITY RECOGNITION EXAMPLE")
    print("="*60)
    
    # Create NER pipeline
    ner_pipeline = pipeline("ner", model="dbmdz/bert-large-cased-finetuned-conll03-english", aggregation_strategy="simple")
    
    texts = [
        "Apple Inc. was founded by Steve Jobs, Steve Wozniak, and Ronald Wayne in April 1976 in Cupertino, California.",
        "Microsoft Corporation is headquartered in Redmond, Washington and was founded by Bill Gates and Paul Allen.",
        "Elon Musk is the CEO of Tesla and SpaceX, and he was born in Pretoria, South Africa."
    ]
    
    for text in texts:
        print(f"\nText: {text}")
        print("-" * 40)
        
        entities = ner_pipeline(text)
        
        for entity in entities:
            print(f"Entity: {entity['word']}")
            print(f"Label: {entity['entity_group']}")
            print(f"Confidence: {entity['score']:.4f}")
            print()

def dataset_exploration_example():
    """Demonstrate dataset loading and exploration."""
    print("\n" + "="*60)
    print("DATASET EXPLORATION EXAMPLE")
    print("="*60)
    
    # Load a small dataset for exploration
    print("Loading CoLA dataset (Corpus of Linguistic Acceptability)...")
    
    try:
        # Load dataset
        dataset = load_dataset("glue", "cola", split="validation")
        
        print(f"Dataset size: {len(dataset)}")
        print(f"Dataset features: {dataset.features}")
        
        # Show first few examples
        print("\nFirst 3 examples:")
        print("-" * 40)
        
        for i in range(3):
            example = dataset[i]
            label = "Acceptable" if example['label'] == 1 else "Unacceptable"
            print(f"Example {i+1}:")
            print(f"Sentence: {example['sentence']}")
            print(f"Label: {label}")
            print()
            
    except Exception as e:
        print(f"Could not load dataset: {e}")
        print("Using sample data instead...")
        
        sample_data = [
            {"sentence": "The cat sat on the mat.", "label": "Acceptable"},
            {"sentence": "Cat the mat on sat the.", "label": "Unacceptable"},
            {"sentence": "She enjoys reading books.", "label": "Acceptable"}
        ]
        
        for i, example in enumerate(sample_data):
            print(f"Example {i+1}:")
            print(f"Sentence: {example['sentence']}")
            print(f"Label: {example['label']}")
            print()

def model_comparison_example():
    """Compare different models on the same task."""
    print("\n" + "="*60)
    print("MODEL COMPARISON EXAMPLE")
    print("="*60)
    
    # Text to analyze
    text = "I absolutely love this new phone! It's amazing and works perfectly."
    
    # Different sentiment analysis models
    models = [
        "distilbert-base-uncased-finetuned-sst-2-english",
        "cardiffnlp/twitter-roberta-base-sentiment-latest"
    ]
    
    print(f"Text to analyze: {text}")
    print("\n" + "-" * 40)
    
    for model_name in models:
        try:
            print(f"\nUsing model: {model_name}")
            classifier = pipeline("sentiment-analysis", model=model_name)
            result = classifier(text)
            print(f"Result: {result[0]['label']} (confidence: {result[0]['score']:.4f})")
        except Exception as e:
            print(f"Error with model {model_name}: {e}")

def main():
    """Run all advanced examples."""
    print("🤗 Advanced Hugging Face Examples")
    print("This script demonstrates various NLP tasks using Hugging Face models")
    
    try:
        # Text Generation
        text_generation_example()
        
        # Question Answering
        question_answering_example()
        
        # Text Summarization
        summarization_example()
        
        # Named Entity Recognition
        named_entity_recognition_example()
        
        # Dataset Exploration
        dataset_exploration_example()
        
        # Model Comparison
        model_comparison_example()
        
        print("\n" + "="*60)
        print("✅ All examples completed successfully!")
        print("="*60)
        
    except Exception as e:
        print(f"\n❌ Error running examples: {e}")
        print("This might be due to network connectivity or model availability.")

if __name__ == "__main__":
    main()
