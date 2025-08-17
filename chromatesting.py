import chromadb
import json
from datetime import datetime
import uuid

def seed_comprehensive_data():
    """Seed ChromaDB with comprehensive data for Hugging Face project"""
    
    # Create a new ChromaDB client (persistent storage)
    client = chromadb.PersistentClient(path="./chroma_db")
    
    # Create multiple collections for different types of data
    collections = {}
    
    # 1. ML Documentation Collection
    print("🔧 Creating ML Documentation collection...")
    collections['ml_docs'] = client.get_or_create_collection(
        name="ml_documentation",
        metadata={"description": "Machine Learning documentation and guides"}
    )
    
    # 2. Code Examples Collection  
    print("🔧 Creating Code Examples collection...")
    collections['code_examples'] = client.get_or_create_collection(
        name="code_examples",
        metadata={"description": "Code snippets and examples"}
    )
    
    # 3. Research Papers Collection
    print("🔧 Creating Research Papers collection...")
    collections['research'] = client.get_or_create_collection(
        name="research_papers",
        metadata={"description": "AI/ML research papers and abstracts"}
    )
    
    # 4. FAQ Collection
    print("🔧 Creating FAQ collection...")
    collections['faq'] = client.get_or_create_collection(
        name="faq",
        metadata={"description": "Frequently asked questions"}
    )

    # Seed ML Documentation
    ml_docs_data = [
        {
            "id": "transformer_guide",
            "document": """
            Transformers are a type of neural network architecture that relies entirely on attention mechanisms to draw global dependencies between input and output. They were introduced in the paper "Attention Is All You Need" by Vaswani et al. The key innovation is the self-attention mechanism that allows the model to weigh the importance of different parts of the input sequence when processing each element.

            Key components:
            1. Multi-Head Attention: Allows the model to attend to information from different representation subspaces
            2. Position Encoding: Since transformers don't have inherent sequence order, position encodings are added
            3. Feed-Forward Networks: Point-wise fully connected layers
            4. Layer Normalization: Helps with training stability
            5. Residual Connections: Enable training of deeper networks
            """,
            "metadata": {
                "type": "guide",
                "topic": "transformers",
                "difficulty": "intermediate",
                "last_updated": "2024-01-15",
                "author": "ML Team"
            }
        },
        {
            "id": "bert_overview",
            "document": """
            BERT (Bidirectional Encoder Representations from Transformers) is a transformer-based machine learning technique for natural language processing. Unlike previous models that read text sequentially, BERT reads the entire sequence of words at once, allowing it to learn the context of a word based on all of its surroundings.

            Key features:
            - Bidirectional training: Considers context from both left and right sides
            - Masked Language Modeling: Randomly masks words and predicts them
            - Next Sentence Prediction: Learns relationships between sentences
            - Pre-training on large corpora: BookCorpus and English Wikipedia
            - Fine-tuning for downstream tasks: Classification, QA, NER, etc.

            BERT has revolutionized NLP by providing contextualized word representations that significantly improve performance on various tasks.
            """,
            "metadata": {
                "type": "overview",
                "topic": "bert",
                "model_family": "transformer",
                "difficulty": "beginner",
                "last_updated": "2024-01-10"
            }
        },
        {
            "id": "fine_tuning_guide",
            "document": """
            Fine-tuning is the process of taking a pre-trained model and adapting it to a specific task or domain. This approach leverages the knowledge learned during pre-training and applies it to new, often smaller datasets.

            Steps for fine-tuning:
            1. Choose a pre-trained model appropriate for your task
            2. Prepare your dataset in the correct format
            3. Add a task-specific head (e.g., classification layer)
            4. Set appropriate learning rates (usually lower than pre-training)
            5. Train for a small number of epochs
            6. Monitor for overfitting
            7. Evaluate on validation set

            Best practices:
            - Use smaller learning rates for pre-trained layers
            - Gradually unfreeze layers if needed
            - Use data augmentation to prevent overfitting
            - Monitor training metrics closely
            """,
            "metadata": {
                "type": "tutorial",
                "topic": "fine-tuning",
                "difficulty": "intermediate",
                "estimated_time": "30 minutes",
                "prerequisites": "basic ML knowledge, transformer understanding"
            }
        },
        {
            "id": "huggingface_intro",
            "document": """
            Hugging Face is a company and community focused on democratizing machine learning, particularly in natural language processing. They provide:

            🤗 Transformers Library: State-of-the-art pre-trained models for NLP, computer vision, and audio
            📊 Datasets Library: Access to thousands of datasets
            🚀 Hub: Platform for sharing models, datasets, and demos
            ⚡ Accelerate: Library for distributed training
            🎯 Optimum: Hardware optimization tools

            The ecosystem makes it easy to:
            - Download and use pre-trained models
            - Fine-tune models on custom data
            - Share models with the community
            - Deploy models in production
            - Experiment with different architectures

            Popular models include BERT, GPT-2, T5, RoBERTa, ALBERT, and many more across different modalities.
            """,
            "metadata": {
                "type": "introduction",
                "topic": "huggingface",
                "company": "Hugging Face",
                "difficulty": "beginner",
                "categories": "NLP, computer vision, audio"
            }
        }
    ]

    # Add ML documentation
    for doc_data in ml_docs_data:
        collections['ml_docs'].add(
            documents=[doc_data["document"]],
            metadatas=[doc_data["metadata"]],
            ids=[doc_data["id"]]
        )

    # Seed Code Examples
    code_examples_data = [
        {
            "id": "sentiment_analysis_example",
            "document": """
            from transformers import pipeline

            # Load pre-trained sentiment analysis pipeline
            classifier = pipeline("sentiment-analysis")

            # Analyze sentiment of text
            texts = [
                "I love this product!",
                "This is terrible.",
                "It's okay, nothing special."
            ]

            results = classifier(texts)
            for text, result in zip(texts, results):
                print(f"Text: {text}")
                print(f"Sentiment: {result['label']} (confidence: {result['score']:.2f})")
                print("-" * 40)
            """,
            "metadata": {
                "type": "code_example",
                "task": "sentiment_analysis",
                "language": "python",
                "framework": "transformers",
                "difficulty": "beginner",
                "libraries": "transformers"
            }
        },
        {
            "id": "question_answering_example", 
            "document": """
            from transformers import pipeline

            # Initialize question-answering pipeline
            qa_pipeline = pipeline("question-answering")

            # Define context and questions
            context = '''
            The Hugging Face Hub is a platform with over 350,000 models, 75,000 datasets, 
            and 150,000 demo apps (Spaces), all open source and publicly available.
            The Hub works as a central place where anyone can explore, experiment, 
            collaborate, and build ML together.
            '''

            questions = [
                "How many models are on Hugging Face Hub?",
                "What is the Hub used for?",
                "Are the resources open source?"
            ]

            for question in questions:
                result = qa_pipeline(question=question, context=context)
                print(f"Question: {question}")
                print(f"Answer: {result['answer']}")
                print(f"Confidence: {result['score']:.2f}")
                print("-" * 50)
            """,
            "metadata": {
                "type": "code_example",
                "task": "question_answering",
                "language": "python",
                "framework": "transformers",
                "difficulty": "beginner",
                "model_type": "bert"
            }
        },
        {
            "id": "text_generation_example",
            "document": """
            from transformers import pipeline, set_seed

            # Set seed for reproducibility
            set_seed(42)

            # Initialize text generation pipeline
            generator = pipeline("text-generation", model="gpt2")

            # Generate text with different parameters
            prompts = [
                "The future of artificial intelligence",
                "In a world where robots",
                "The secret to happiness"
            ]

            for prompt in prompts:
                print(f"Prompt: {prompt}")
                
                # Generate with different configurations
                outputs = generator(
                    prompt,
                    max_length=100,
                    num_return_sequences=2,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=generator.tokenizer.eos_token_id
                )
                
                for i, output in enumerate(outputs):
                    print(f"Generated {i+1}: {output['generated_text']}")
                print("=" * 60)
            """,
            "metadata": {
                "type": "code_example", 
                "task": "text_generation",
                "language": "python",
                "framework": "transformers",
                "difficulty": "intermediate",
                "model": "gpt2"
            }
        },
        {
            "id": "custom_model_loading",
            "document": """
            from transformers import AutoTokenizer, AutoModelForSequenceClassification
            import torch

            # Load specific model and tokenizer
            model_name = "distilbert-base-uncased-finetuned-sst-2-english"
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForSequenceClassification.from_pretrained(model_name)

            # Prepare text for inference
            text = "This movie is fantastic!"
            inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)

            # Perform inference
            with torch.no_grad():
                outputs = model(**inputs)
                predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)

            # Get predicted class
            predicted_class = torch.argmax(predictions, dim=-1).item()
            confidence = predictions[0][predicted_class].item()

            labels = ["NEGATIVE", "POSITIVE"]
            print(f"Text: {text}")
            print(f"Prediction: {labels[predicted_class]}")
            print(f"Confidence: {confidence:.4f}")

            # Show all class probabilities
            for i, label in enumerate(labels):
                print(f"{label}: {predictions[0][i].item():.4f}")
            """,
            "metadata": {
                "type": "code_example",
                "task": "manual_inference",
                "language": "python", 
                "framework": "transformers",
                "difficulty": "advanced",
                "concepts": "tokenization, manual_inference, torch"
            }
        }
    ]

    # Add code examples
    for code_data in code_examples_data:
        collections['code_examples'].add(
            documents=[code_data["document"]],
            metadatas=[code_data["metadata"]], 
            ids=[code_data["id"]]
        )

    # Seed Research Papers
    research_papers_data = [
        {
            "id": "attention_is_all_you_need",
            "document": """
            Title: Attention Is All You Need
            Authors: Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, Illia Polosukhin
            Published: 2017

            Abstract: The dominant sequence transduction models are based on complex recurrent or convolutional neural networks that include an encoder and a decoder. The best performing models also connect the encoder and decoder through an attention mechanism. We propose a new simple network architecture, the Transformer, based solely on attention mechanisms, dispensing with recurrence and convolutions entirely.

            Key Contributions:
            - Introduced the Transformer architecture
            - Eliminated need for recurrence and convolutions
            - Achieved state-of-the-art results on translation tasks
            - Enabled parallelization during training
            - Foundation for modern NLP models like BERT and GPT

            Impact: This paper revolutionized NLP and became the foundation for most modern language models. The self-attention mechanism it introduced became a core component of subsequent architectures.
            """,
            "metadata": {
                "type": "research_paper",
                "year": 2017,
                "venue": "NIPS",
                "authors": "Vaswani, Shazeer, Parmar",
                "citations": "50000+",
                "field": "NLP",
                "importance": "foundational"
            }
        },
        {
            "id": "bert_paper",
            "document": """
            Title: BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding
            Authors: Jacob Devlin, Ming-Wei Chang, Kenton Lee, Kristina Toutanova
            Published: 2018

            Abstract: We introduce a new language representation model called BERT, which stands for Bidirectional Encoder Representations from Transformers. Unlike recent language representation models, BERT is designed to pre-train deep bidirectional representations from unlabeled text by jointly conditioning on both left and right context in all layers.

            Key Innovations:
            - Bidirectional training of transformers
            - Masked Language Model (MLM) pre-training objective
            - Next Sentence Prediction (NSP) task
            - Transfer learning approach for NLP
            - State-of-the-art results on 11 NLP tasks

            Methodology:
            1. Pre-train on large corpus with MLM and NSP objectives
            2. Fine-tune on downstream tasks with minimal architecture changes
            3. Use bidirectional self-attention in all layers

            Results: BERT achieved significant improvements on GLUE benchmark, SQuAD v1.1/v2.0, and SWAG tasks, establishing new state-of-the-art results.
            """,
            "metadata": {
                "type": "research_paper", 
                "year": 2018,
                "venue": "NAACL",
                "authors": "Devlin, Chang, Lee, Toutanova",
                "model": "BERT",
                "field": "NLP",
                "impact": "high"
            }
        },
        {
            "id": "gpt2_paper",
            "document": """
            Title: Language Models are Unsupervised Multitask Learners
            Authors: Alec Radford, Jeffrey Wu, Rewon Child, David Luan, Dario Amodei, Ilya Sutskever
            Published: 2019

            Abstract: Natural language processing tasks, such as question answering, machine translation, reading comprehension, and summarization, are typically approached with supervised learning on taskspecific datasets. We demonstrate that language models begin to learn these tasks without any explicit supervision when trained on a new dataset of millions of webpages called WebText.

            Key Findings:
            - Large language models can perform tasks without task-specific training
            - Zero-shot task transfer capabilities emerge with scale
            - GPT-2 achieved strong performance on various NLP tasks
            - Demonstrated the importance of dataset quality and scale

            Model Details:
            - 1.5 billion parameters (largest variant)
            - Transformer decoder architecture
            - Trained on WebText dataset (40GB of text)
            - Autoregressive language modeling objective

            Impact: This work demonstrated that scaling up language models leads to emergent capabilities and sparked interest in large language models for general AI.
            """,
            "metadata": {
                "type": "research_paper",
                "year": 2019,
                "authors": "Radford, Wu, Child",
                "model": "GPT-2",
                "parameters": "1.5B",
                "field": "NLP",
                "concept": "zero_shot_learning"
            }
        }
    ]

    # Add research papers
    for research_data in research_papers_data:
        collections['research'].add(
            documents=[research_data["document"]],
            metadatas=[research_data["metadata"]],
            ids=[research_data["id"]]
        )

    # Seed FAQ Data
    faq_data = [
        {
            "id": "what_is_huggingface",
            "document": """
            Q: What is Hugging Face?
            A: Hugging Face is a company and open-source community that focuses on democratizing machine learning, particularly in natural language processing. They provide tools, models, and datasets that make it easy for developers and researchers to work with state-of-the-art ML models. Their main offerings include the Transformers library, Datasets library, and the Hugging Face Hub platform for sharing and discovering models.
            """,
            "metadata": {
                "type": "faq",
                "category": "general",
                "topic": "huggingface_overview",
                "difficulty": "beginner"
            }
        },
        {
            "id": "how_to_use_pretrained_models",
            "document": """
            Q: How do I use pre-trained models from Hugging Face?
            A: You can use pre-trained models in several ways:
            1. Using pipelines (easiest): `pipeline("sentiment-analysis")`
            2. Using Auto classes: `AutoModel.from_pretrained("model-name")`
            3. Loading specific model classes: `BertModel.from_pretrained("bert-base-uncased")`
            
            The pipeline approach is recommended for beginners as it handles preprocessing and postprocessing automatically. For more control, use the Auto classes or specific model classes.
            """,
            "metadata": {
                "type": "faq",
                "category": "usage",
                "topic": "pretrained_models",
                "difficulty": "beginner"
            }
        },
        {
            "id": "difference_bert_gpt",
            "document": """
            Q: What's the difference between BERT and GPT models?
            A: The main differences are:
            
            BERT (Bidirectional Encoder):
            - Reads text in both directions (bidirectional)
            - Good for understanding tasks (classification, Q&A)
            - Uses masked language modeling during training
            - Cannot generate text naturally
            
            GPT (Generative Pre-trained Transformer):
            - Reads text left-to-right (autoregressive)
            - Excellent for text generation
            - Uses next token prediction during training
            - Can perform understanding tasks but less naturally
            
            Choose BERT for understanding tasks and GPT for generation tasks.
            """,
            "metadata": {
                "type": "faq",
                "category": "models",
                "topic": "bert_vs_gpt",
                "difficulty": "intermediate",
                "models": "BERT, GPT"
            }
        },
        {
            "id": "fine_tuning_cost",
            "document": """
            Q: How much does it cost to fine-tune a model?
            A: Fine-tuning costs depend on several factors:
            
            Computational costs:
            - Model size (larger models = more expensive)
            - Dataset size (more data = longer training)
            - Hardware used (GPU type and cloud provider)
            - Training time (epochs and batch size)
            
            Typical costs:
            - Small models (DistilBERT) on Google Colab: Free-$10
            - Medium models (BERT-base) on cloud GPU: $10-100
            - Large models (BERT-large) on high-end GPUs: $100-1000+
            
            Cost-saving tips:
            - Use smaller models when possible
            - Start with fewer epochs
            - Use gradient checkpointing
            - Consider using Hugging Face's AutoTrain for automated fine-tuning
            """,
            "metadata": {
                "type": "faq",
                "category": "training",
                "topic": "fine_tuning_cost",
                "difficulty": "intermediate",
                "considerations": "cost, hardware, optimization"
            }
        },
        {
            "id": "model_deployment",
            "document": """
            Q: How do I deploy Hugging Face models in production?
            A: There are several deployment options:
            
            1. Hugging Face Inference Endpoints (Recommended):
            - Managed service with auto-scaling
            - Easy setup and monitoring
            - Supports various model types
            
            2. Container deployment:
            - Use official Hugging Face containers
            - Deploy on AWS, GCP, Azure
            - Full control over infrastructure
            
            3. Edge deployment:
            - Convert to ONNX for mobile/edge devices
            - Use Optimum library for optimization
            - Consider model quantization
            
            4. API frameworks:
            - FastAPI with transformers
            - Flask applications
            - Streamlit for demos
            
            Consider factors like latency requirements, cost, scalability, and maintenance when choosing.
            """,
            "metadata": {
                "type": "faq",
                "category": "deployment",
                "topic": "production_deployment",
                "difficulty": "advanced",
                "platforms": "cloud, edge, api"
            }
        }
    ]

    # Add FAQ data
    for faq_item in faq_data:
        collections['faq'].add(
            documents=[faq_item["document"]],
            metadatas=[faq_item["metadata"]],
            ids=[faq_item["id"]]
        )

    # Print summary
    print("\n" + "="*60)
    print("📊 CHROMADB SEEDING COMPLETE!")
    print("="*60)
    
    for name, collection in collections.items():
        count = collection.count()
        print(f"📚 {name}: {count} documents")
    
    total_docs = sum(collection.count() for collection in collections.values())
    print(f"\n🎯 Total documents across all collections: {total_docs}")
    
    return collections

def test_search_functionality(collections):
    """Test search functionality across different collections"""
    print("\n" + "="*60)
    print("🔍 TESTING SEARCH FUNCTIONALITY")
    print("="*60)
    
    # Test queries for different collections
    test_queries = [
        ("ml_docs", "What is BERT and how does it work?", 2),
        ("code_examples", "How to analyze sentiment with transformers?", 2),
        ("research", "attention mechanism transformer", 2),
        ("faq", "How to deploy models in production?", 2)
    ]
    
    for collection_name, query, n_results in test_queries:
        print(f"\n🔎 Searching {collection_name} for: '{query}'")
        print("-" * 40)
        
        results = collections[collection_name].query(
            query_texts=[query],
            n_results=n_results
        )
        
        for i, (doc, metadata) in enumerate(zip(results['documents'][0], results['metadatas'][0])):
            print(f"Result {i+1}:")
            print(f"  Type: {metadata.get('type', 'N/A')}")
            print(f"  Topic: {metadata.get('topic', 'N/A')}")
            print(f"  Preview: {doc[:150]}...")
            print()

def main():
    """Main function to seed and test ChromaDB"""
    print("🚀 Starting comprehensive ChromaDB seeding...")
    
    try:
        # Seed the database
        collections = seed_comprehensive_data()
        
        # Test search functionality
        test_search_functionality(collections)
        
        print("\n✅ ChromaDB seeding and testing completed successfully!")
        print("💡 You can now use these collections for various ML/NLP tasks")
        
    except Exception as e:
        print(f"❌ Error during seeding: {e}")
        raise

if __name__ == "__main__":
    main()

