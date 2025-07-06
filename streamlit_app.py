"""
Hugging Face Showcase Streamlit App
===================================

A comprehensive web application showcasing various Hugging Face models and datasets.
This app demonstrates different NLP tasks, models, and datasets in an interactive format.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from transformers import (
    pipeline, 
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    AutoModelForQuestionAnswering,
    AutoModelForCausalLM
)
from datasets import load_dataset
import torch
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="🤗 Hugging Face Showcase",
    page_icon="🤗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #FF6B6B;
        text-align: center;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 2rem;
        color: #4ECDC4;
        border-bottom: 2px solid #4ECDC4;
        padding-bottom: 0.5rem;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .model-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
        border-left: 4px solid #FF6B6B;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 0.25rem;
        padding: 1rem;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 0.25rem;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar for navigation
st.sidebar.title("🤗 Navigation")
page = st.sidebar.selectbox(
    "Choose a demo:",
    [
        "🏠 Home",
        "📝 Text Classification",
        "❓ Question Answering", 
        "🎭 Sentiment Analysis",
        "📊 Dataset Explorer",
        "🔍 Named Entity Recognition",
        "💬 Text Generation",
        "📈 Model Comparison",
        "🔧 Custom Pipeline"
    ]
)

# Utility functions
@st.cache_data
def load_sample_data():
    """Load sample datasets for demonstration"""
    try:
        # Load a small sample of IMDB dataset
        dataset = load_dataset("imdb", split="test[:100]")
        return dataset.to_pandas()
    except Exception as e:
        st.error(f"Error loading dataset: {e}")
        return None

@st.cache_resource
def get_sentiment_pipeline():
    """Initialize sentiment analysis pipeline"""
    return pipeline("sentiment-analysis", 
                   model="cardiffnlp/twitter-roberta-base-sentiment-latest")

@st.cache_resource
def get_qa_pipeline():
    """Initialize question answering pipeline"""
    return pipeline("question-answering", 
                   model="distilbert-base-cased-distilled-squad")

@st.cache_resource
def get_ner_pipeline():
    """Initialize NER pipeline"""
    return pipeline("ner", 
                   model="dbmdz/bert-large-cased-finetuned-conll03-english",
                   aggregation_strategy="simple")

@st.cache_resource
def get_text_generation_pipeline():
    """Initialize text generation pipeline"""
    return pipeline("text-generation", 
                   model="gpt2",
                   max_length=100,
                   temperature=0.7)

# Home Page
if page == "🏠 Home":
    st.markdown('<h1 class="main-header">🤗 Hugging Face Showcase</h1>', unsafe_allow_html=True)
    
    st.markdown("""
    Welcome to our comprehensive Hugging Face demonstration! This application showcases the power and versatility 
    of Hugging Face's transformers library and datasets.
    
    ## 🚀 What you'll find here:
    """)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        ### 🤖 Models
        - Text Classification
        - Question Answering
        - Sentiment Analysis
        - Named Entity Recognition
        - Text Generation
        """)
    
    with col2:
        st.markdown("""
        ### 📊 Datasets
        - IMDB Movie Reviews
        - SQuAD Q&A Dataset
        - CoNLL-2003 NER
        - Custom Dataset Loading
        """)
    
    with col3:
        st.markdown("""
        ### 🔧 Features
        - Interactive Demos
        - Real-time Inference
        - Model Comparisons
        - Performance Metrics
        - Custom Pipelines
        """)
    
    st.markdown("---")
    
    # Quick stats
    st.markdown('<h2 class="section-header">📈 Quick Stats</h2>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Available Models", "5+", "🤖")
    
    with col2:
        st.metric("Datasets", "3+", "📊")
    
    with col3:
        st.metric("Tasks", "8+", "⚡")
    
    with col4:
        st.metric("Languages", "English", "🌍")
    
    # Sample visualization
    st.markdown('<h2 class="section-header">📊 Sample Model Performance</h2>', unsafe_allow_html=True)
    
    # Create sample performance data
    performance_data = pd.DataFrame({
        'Model': ['BERT-base', 'RoBERTa-base', 'DistilBERT', 'GPT-2', 'T5-small'],
        'Accuracy': [0.89, 0.91, 0.85, 0.87, 0.88],
        'Speed (ms)': [150, 180, 90, 200, 160],
        'Parameters (M)': [110, 125, 66, 124, 60]
    })
    
    fig = px.scatter(performance_data, 
                    x='Speed (ms)', 
                    y='Accuracy',
                    size='Parameters (M)',
                    color='Model',
                    title="Model Performance Overview",
                    hover_data=['Parameters (M)'])
    
    fig.update_layout(
        xaxis_title="Inference Speed (ms)",
        yaxis_title="Accuracy",
        showlegend=True
    )
    
    st.plotly_chart(fig, use_container_width=True)

# Text Classification Page
elif page == "📝 Text Classification":
    st.markdown('<h1 class="main-header">📝 Text Classification</h1>', unsafe_allow_html=True)
    
    st.markdown("""
    Text classification is one of the most common NLP tasks. Enter any text below and see how our model classifies it!
    """)
    
    # User input
    user_text = st.text_area(
        "Enter text to classify:",
        value="I love this movie! It's absolutely fantastic and entertaining.",
        height=100
    )
    
    if st.button("🔍 Classify Text", type="primary"):
        if user_text:
            with st.spinner("Analyzing text..."):
                try:
                    # Load sentiment pipeline
                    classifier = get_sentiment_pipeline()
                    
                    # Perform classification
                    result = classifier(user_text)
                    
                    # Display results
                    st.markdown('<div class="success-box">', unsafe_allow_html=True)
                    st.markdown("### 🎯 Classification Results")
                    
                    for i, res in enumerate(result):
                        label = res['label']
                        score = res['score']
                        
                        # Convert label to more readable format
                        if 'POSITIVE' in label.upper():
                            emoji = "😊"
                            color = "green"
                        elif 'NEGATIVE' in label.upper():
                            emoji = "😞"
                            color = "red"
                        else:
                            emoji = "😐"
                            color = "orange"
                        
                        st.markdown(f"""
                        **{emoji} {label}** - Confidence: {score:.2%}
                        """)
                        
                        # Progress bar for confidence
                        st.progress(score)
                    
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Visualization
                    if len(result) > 1:
                        df_results = pd.DataFrame(result)
                        fig = px.bar(df_results, x='label', y='score', 
                                   title="Classification Confidence Scores",
                                   color='score',
                                   color_continuous_scale='viridis')
                        st.plotly_chart(fig, use_container_width=True)
                        
                except Exception as e:
                    st.error(f"Error during classification: {e}")
        else:
            st.warning("Please enter some text to classify!")

# Question Answering Page
elif page == "❓ Question Answering":
    st.markdown('<h1 class="main-header">❓ Question Answering</h1>', unsafe_allow_html=True)
    
    st.markdown("""
    Ask questions about any context text and get AI-powered answers! This demo uses a BERT-based model 
    fine-tuned on the SQuAD dataset.
    """)
    
    # Default context
    default_context = """
    The Hugging Face Hub is a platform with over 350,000 models, 75,000 datasets, and 150,000 demo apps (Spaces), 
    all open source and publicly available, in an online platform where people can easily collaborate and build ML together. 
    The Hub works as a central place where anyone can explore, experiment, collaborate, and build technology with Machine Learning.
    """
    
    # Input fields
    context = st.text_area(
        "Context (provide background information):",
        value=default_context,
        height=150
    )
    
    question = st.text_input(
        "Your question:",
        value="How many models are available on Hugging Face Hub?"
    )
    
    if st.button("🤔 Get Answer", type="primary"):
        if context and question:
            with st.spinner("Finding answer..."):
                try:
                    # Load QA pipeline
                    qa_pipeline = get_qa_pipeline()
                    
                    # Get answer
                    result = qa_pipeline(question=question, context=context)
                    
                    # Display results
                    st.markdown('<div class="success-box">', unsafe_allow_html=True)
                    st.markdown("### 💡 Answer")
                    
                    answer = result['answer']
                    confidence = result['score']
                    start = result['start']
                    end = result['end']
                    
                    st.markdown(f"""
                    **Answer:** {answer}
                    
                    **Confidence:** {confidence:.2%}
                    
                    **Position in text:** Characters {start} to {end}
                    """)
                    
                    # Highlight answer in context
                    highlighted_context = (
                        context[:start] + 
                        f"**{context[start:end]}**" + 
                        context[end:]
                    )
                    
                    st.markdown("### 📝 Context with highlighted answer:")
                    st.markdown(highlighted_context)
                    
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Confidence visualization
                    fig = go.Figure(go.Indicator(
                        mode = "gauge+number",
                        value = confidence * 100,
                        domain = {'x': [0, 1], 'y': [0, 1]},
                        title = {'text': "Answer Confidence (%)"},
                        gauge = {
                            'axis': {'range': [None, 100]},
                            'bar': {'color': "darkblue"},
                            'steps': [
                                {'range': [0, 50], 'color': "lightgray"},
                                {'range': [50, 80], 'color': "yellow"},
                                {'range': [80, 100], 'color': "green"}
                            ],
                            'threshold': {
                                'line': {'color': "red", 'width': 4},
                                'thickness': 0.75,
                                'value': 90
                            }
                        }
                    ))
                    
                    fig.update_layout(height=300)
                    st.plotly_chart(fig, use_container_width=True)
                    
                except Exception as e:
                    st.error(f"Error during question answering: {e}")
        else:
            st.warning("Please provide both context and question!")

# Sentiment Analysis Page
elif page == "🎭 Sentiment Analysis":
    st.markdown('<h1 class="main-header">🎭 Sentiment Analysis</h1>', unsafe_allow_html=True)
    
    st.markdown("""
    Analyze the emotional tone of text using state-of-the-art sentiment analysis models.
    """)
    
    # Text input methods
    input_method = st.radio(
        "Choose input method:",
        ["Single Text", "Multiple Texts", "Upload File"]
    )
    
    if input_method == "Single Text":
        text = st.text_area(
            "Enter text for sentiment analysis:",
            value="I'm really excited about this new project! It's going to be amazing.",
            height=100
        )
        
        if st.button("🎭 Analyze Sentiment"):
            if text:
                with st.spinner("Analyzing sentiment..."):
                    try:
                        classifier = get_sentiment_pipeline()
                        result = classifier(text)
                        
                        # Display results with emoji
                        for res in result:
                            label = res['label']
                            score = res['score']
                            
                            if 'POSITIVE' in label.upper():
                                emoji = "😊"
                                color = "#28a745"
                            elif 'NEGATIVE' in label.upper():
                                emoji = "😞"
                                color = "#dc3545"
                            else:
                                emoji = "😐"
                                color = "#ffc107"
                            
                            st.markdown(f"""
                            <div style="background-color: {color}20; padding: 1rem; border-radius: 0.5rem; border-left: 4px solid {color};">
                                <h3>{emoji} {label}</h3>
                                <p><strong>Confidence:</strong> {score:.2%}</p>
                            </div>
                            """, unsafe_allow_html=True)
                            
                    except Exception as e:
                        st.error(f"Error: {e}")
            else:
                st.warning("Please enter some text!")
    
    elif input_method == "Multiple Texts":
        texts = st.text_area(
            "Enter multiple texts (one per line):",
            value="I love this product!\nThis is terrible.\nIt's okay, nothing special.\nAbsolutely fantastic!",
            height=150
        )
        
        if st.button("🎭 Analyze All"):
            if texts:
                text_list = [text.strip() for text in texts.split('\n') if text.strip()]
                
                if text_list:
                    with st.spinner("Analyzing all texts..."):
                        try:
                            classifier = get_sentiment_pipeline()
                            results = []
                            
                            for i, text in enumerate(text_list):
                                result = classifier(text)
                                results.append({
                                    'Text': text,
                                    'Label': result[0]['label'],
                                    'Confidence': result[0]['score']
                                })
                            
                            # Display results in a dataframe
                            df = pd.DataFrame(results)
                            st.dataframe(df, use_container_width=True)
                            
                            # Visualization
                            fig = px.bar(df, x='Text', y='Confidence', color='Label',
                                       title="Sentiment Analysis Results",
                                       text='Label')
                            fig.update_xaxis(tickangle=45)
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Summary statistics
                            st.markdown("### 📊 Summary")
                            label_counts = df['Label'].value_counts()
                            
                            fig_pie = px.pie(values=label_counts.values, 
                                           names=label_counts.index,
                                           title="Sentiment Distribution")
                            st.plotly_chart(fig_pie, use_container_width=True)
                            
                        except Exception as e:
                            st.error(f"Error: {e}")
                else:
                    st.warning("Please enter at least one text!")
            else:
                st.warning("Please enter some texts!")

# Dataset Explorer Page
elif page == "📊 Dataset Explorer":
    st.markdown('<h1 class="main-header">📊 Dataset Explorer</h1>', unsafe_allow_html=True)
    
    st.markdown("""
    Explore popular datasets from the Hugging Face Hub. Load, analyze, and visualize dataset content.
    """)
    
    # Dataset selection
    dataset_choice = st.selectbox(
        "Choose a dataset:",
        ["IMDB Movie Reviews", "Custom Upload"]
    )
    
    if dataset_choice == "IMDB Movie Reviews":
        if st.button("📥 Load IMDB Dataset Sample"):
            with st.spinner("Loading dataset..."):
                try:
                    df = load_sample_data()
                    
                    if df is not None:
                        st.success(f"Loaded {len(df)} samples from IMDB dataset!")
                        
                        # Dataset overview
                        st.markdown("### 📋 Dataset Overview")
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric("Total Samples", len(df))
                        with col2:
                            st.metric("Features", len(df.columns))
                        with col3:
                            st.metric("Labels", df['label'].nunique())
                        
                        # Display sample data
                        st.markdown("### 🔍 Sample Data")
                        st.dataframe(df.head(10), use_container_width=True)
                        
                        # Label distribution
                        st.markdown("### 📊 Label Distribution")
                        label_counts = df['label'].value_counts()
                        label_names = ['Negative', 'Positive']
                        
                        fig = px.pie(values=label_counts.values, 
                                   names=label_names,
                                   title="Sentiment Distribution in Dataset")
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Text length analysis
                        st.markdown("### 📏 Text Length Analysis")
                        df['text_length'] = df['text'].str.len()
                        
                        fig = px.histogram(df, x='text_length', nbins=50,
                                         title="Distribution of Review Lengths",
                                         labels={'text_length': 'Characters', 'count': 'Frequency'})
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Sample predictions
                        st.markdown("### 🤖 Live Predictions on Sample Data")
                        sample_idx = st.slider("Select sample to analyze:", 0, len(df)-1, 0)
                        
                        sample_text = df.iloc[sample_idx]['text']
                        actual_label = 'Positive' if df.iloc[sample_idx]['label'] == 1 else 'Negative'
                        
                        st.markdown(f"**Actual Label:** {actual_label}")
                        st.text_area("Sample Text:", sample_text, height=150, disabled=True)
                        
                        if st.button("🔮 Predict Sentiment"):
                            with st.spinner("Predicting..."):
                                try:
                                    classifier = get_sentiment_pipeline()
                                    result = classifier(sample_text)
                                    
                                    predicted_label = result[0]['label']
                                    confidence = result[0]['score']
                                    
                                    st.markdown(f"""
                                    **Predicted:** {predicted_label} (Confidence: {confidence:.2%})
                                    
                                    **Actual:** {actual_label}
                                    
                                    **Match:** {'✅ Correct' if predicted_label.upper() in actual_label.upper() else '❌ Incorrect'}
                                    """)
                                    
                                except Exception as e:
                                    st.error(f"Prediction error: {e}")
                        
                    else:
                        st.error("Failed to load dataset!")
                        
                except Exception as e:
                    st.error(f"Error loading dataset: {e}")
    
    elif dataset_choice == "Custom Upload":
        st.markdown("### 📁 Upload Your Own Dataset")
        
        uploaded_file = st.file_uploader(
            "Choose a CSV file",
            type=['csv'],
            help="Upload a CSV file with text data for analysis"
        )
        
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                st.success(f"Loaded {len(df)} rows and {len(df.columns)} columns!")
                
                # Display basic info
                st.markdown("### 📋 Dataset Info")
                st.dataframe(df.head(), use_container_width=True)
                
                # Column selection for text analysis
                text_column = st.selectbox("Select text column:", df.columns)
                
                if text_column:
                    sample_text = df[text_column].iloc[0]
                    st.text_area("Sample text:", sample_text, height=100, disabled=True)
                    
                    if st.button("🔮 Analyze Sample"):
                        with st.spinner("Analyzing..."):
                            try:
                                classifier = get_sentiment_pipeline()
                                result = classifier(sample_text)
                                
                                st.markdown(f"""
                                **Sentiment:** {result[0]['label']}
                                **Confidence:** {result[0]['score']:.2%}
                                """)
                                
                            except Exception as e:
                                st.error(f"Analysis error: {e}")
                
            except Exception as e:
                st.error(f"Error reading file: {e}")

# Named Entity Recognition Page
elif page == "🔍 Named Entity Recognition":
    st.markdown('<h1 class="main-header">🔍 Named Entity Recognition</h1>', unsafe_allow_html=True)
    
    st.markdown("""
    Extract and identify named entities (people, organizations, locations, etc.) from text using advanced NLP models.
    """)
    
    # Sample text
    default_text = """
    Apple Inc. is an American multinational technology company headquartered in Cupertino, California. 
    Tim Cook is the current CEO of Apple. The company was founded by Steve Jobs, Steve Wozniak, and Ronald Wayne in April 1976.
    Apple's products include the iPhone, iPad, Mac, Apple Watch, and Apple TV.
    """
    
    text = st.text_area(
        "Enter text for entity recognition:",
        value=default_text,
        height=150
    )
    
    if st.button("🔍 Extract Entities", type="primary"):
        if text:
            with st.spinner("Extracting entities..."):
                try:
                    ner_pipeline = get_ner_pipeline()
                    entities = ner_pipeline(text)
                    
                    if entities:
                        st.markdown("### 🎯 Extracted Entities")
                        
                        # Create a dataframe for entities
                        entity_data = []
                        for entity in entities:
                            entity_data.append({
                                'Entity': entity['word'],
                                'Label': entity['entity_group'],
                                'Confidence': entity['score'],
                                'Start': entity['start'],
                                'End': entity['end']
                            })
                        
                        df_entities = pd.DataFrame(entity_data)
                        
                        # Display entities table
                        st.dataframe(df_entities, use_container_width=True)
                        
                        # Visualize entity types
                        entity_counts = df_entities['Label'].value_counts()
                        
                        fig = px.bar(x=entity_counts.index, y=entity_counts.values,
                                   title="Entity Types Distribution",
                                   labels={'x': 'Entity Type', 'y': 'Count'},
                                   color=entity_counts.values,
                                   color_continuous_scale='viridis')
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Highlight entities in text
                        st.markdown("### 📝 Text with Highlighted Entities")
                        
                        highlighted_text = text
                        # Sort entities by start position in reverse order to avoid position shifts
                        sorted_entities = sorted(entities, key=lambda x: x['start'], reverse=True)
                        
                        colors = {
                            'PER': '#FF6B6B',  # Person - Red
                            'ORG': '#4ECDC4',  # Organization - Teal
                            'LOC': '#45B7D1',  # Location - Blue
                            'MISC': '#96CEB4'  # Miscellaneous - Green
                        }
                        
                        for entity in sorted_entities:
                            start, end = entity['start'], entity['end']
                            label = entity['entity_group']
                            word = entity['word']
                            color = colors.get(label, '#95A5A6')
                            
                            highlighted_text = (
                                highlighted_text[:start] + 
                                f'<span style="background-color: {color}; padding: 2px 4px; border-radius: 3px; color: white;">{word} ({label})</span>' + 
                                highlighted_text[end:]
                            )
                        
                        st.markdown(highlighted_text, unsafe_allow_html=True)
                        
                        # Entity statistics
                        st.markdown("### 📊 Entity Statistics")
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric("Total Entities", len(entities))
                        with col2:
                            st.metric("Unique Types", len(entity_counts))
                        with col3:
                            avg_confidence = df_entities['Confidence'].mean()
                            st.metric("Avg Confidence", f"{avg_confidence:.2%}")
                        
                    else:
                        st.warning("No entities found in the text!")
                        
                except Exception as e:
                    st.error(f"Error during entity recognition: {e}")
        else:
            st.warning("Please enter some text!")

# Text Generation Page
elif page == "💬 Text Generation":
    st.markdown('<h1 class="main-header">💬 Text Generation</h1>', unsafe_allow_html=True)
    
    st.markdown("""
    Generate creative text using AI language models. Provide a prompt and watch the AI continue the story!
    """)
    
    # Input parameters
    col1, col2 = st.columns([3, 1])
    
    with col1:
        prompt = st.text_area(
            "Enter your prompt:",
            value="Once upon a time in a magical forest,",
            height=100
        )
    
    with col2:
        max_length = st.slider("Max Length", 50, 200, 100)
        temperature = st.slider("Temperature", 0.1, 1.0, 0.7, 0.1)
        num_sequences = st.slider("Number of Variants", 1, 3, 1)
    
    if st.button("✨ Generate Text", type="primary"):
        if prompt:
            with st.spinner("Generating text..."):
                try:
                    generator = get_text_generation_pipeline()
                    
                    # Generate text
                    generated = generator(
                        prompt,
                        max_length=max_length,
                        temperature=temperature,
                        num_return_sequences=num_sequences,
                        pad_token_id=generator.tokenizer.eos_token_id
                    )
                    
                    st.markdown("### 📖 Generated Text")
                    
                    for i, result in enumerate(generated):
                        generated_text = result['generated_text']
                        
                        # Remove the original prompt from display
                        new_text = generated_text[len(prompt):].strip()
                        
                        st.markdown(f"""
                        <div class="model-card">
                            <h4>Variant {i+1}</h4>
                            <p><strong>Prompt:</strong> {prompt}</p>
                            <p><strong>Generated:</strong> {new_text}</p>
                            <p><strong>Total Length:</strong> {len(generated_text)} characters</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Text statistics
                    if len(generated) > 1:
                        st.markdown("### 📊 Generation Statistics")
                        
                        lengths = [len(result['generated_text']) for result in generated]
                        avg_length = sum(lengths) / len(lengths)
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Variants Generated", len(generated))
                        with col2:
                            st.metric("Average Length", f"{avg_length:.0f}")
                        with col3:
                            st.metric("Longest Variant", f"{max(lengths)}")
                        
                        # Length comparison chart
                        df_stats = pd.DataFrame({
                            'Variant': [f"Variant {i+1}" for i in range(len(generated))],
                            'Length': lengths
                        })
                        
                        fig = px.bar(df_stats, x='Variant', y='Length',
                                   title="Generated Text Lengths",
                                   color='Length',
                                   color_continuous_scale='blues')
                        st.plotly_chart(fig, use_container_width=True)
                    
                except Exception as e:
                    st.error(f"Error during text generation: {e}")
        else:
            st.warning("Please enter a prompt!")

# Model Comparison Page
elif page == "📈 Model Comparison":
    st.markdown('<h1 class="main-header">📈 Model Comparison</h1>', unsafe_allow_html=True)
    
    st.markdown("""
    Compare different models' performance on the same task. See how various models handle identical inputs.
    """)
    
    # Task selection
    task = st.selectbox(
        "Select task for comparison:",
        ["Sentiment Analysis", "Text Classification"]
    )
    
    # Sample texts for comparison
    test_texts = [
        "I absolutely love this product! It's amazing!",
        "This is the worst experience I've ever had.",
        "It's okay, nothing special but not bad either.",
        "Fantastic quality and great customer service!",
        "I'm not sure how I feel about this."
    ]
    
    selected_text = st.selectbox("Choose test text:", test_texts)
    custom_text = st.text_input("Or enter custom text:")
    
    test_text = custom_text if custom_text else selected_text
    
    if st.button("⚡ Compare Models", type="primary"):
        if test_text:
            st.markdown(f"### 🧪 Testing with: *{test_text}*")
            
            with st.spinner("Running comparison..."):
                try:
                    # Different sentiment analysis models for comparison
                    models = [
                        "cardiffnlp/twitter-roberta-base-sentiment-latest",
                        "distilbert-base-uncased-finetuned-sst-2-english",
                        "nlptown/bert-base-multilingual-uncased-sentiment"
                    ]
                    
                    results = []
                    
                    for model_name in models:
                        try:
                            classifier = pipeline("sentiment-analysis", model=model_name)
                            result = classifier(test_text)
                            
                            results.append({
                                'Model': model_name.split('/')[-1],  # Short name
                                'Full_Model_Name': model_name,
                                'Label': result[0]['label'],
                                'Confidence': result[0]['score']
                            })
                        except Exception as e:
                            st.warning(f"Could not load model {model_name}: {e}")
                    
                    if results:
                        # Display results table
                        df_results = pd.DataFrame(results)
                        st.dataframe(df_results[['Model', 'Label', 'Confidence']], use_container_width=True)
                        
                        # Confidence comparison chart
                        fig = px.bar(df_results, x='Model', y='Confidence', color='Label',
                                   title="Model Confidence Comparison",
                                   text='Label')
                        fig.update_layout(xaxis_tickangle=45)
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Agreement analysis
                        st.markdown("### 🤝 Model Agreement Analysis")
                        
                        labels = df_results['Label'].tolist()
                        unique_labels = list(set(labels))
                        
                        if len(unique_labels) == 1:
                            st.success(f"✅ All models agree: **{unique_labels[0]}**")
                        else:
                            st.warning(f"⚠️ Models disagree. Predictions: {', '.join(set(labels))}")
                        
                        # Confidence statistics
                        avg_confidence = df_results['Confidence'].mean()
                        max_confidence = df_results['Confidence'].max()
                        min_confidence = df_results['Confidence'].min()
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Average Confidence", f"{avg_confidence:.2%}")
                        with col2:
                            st.metric("Highest Confidence", f"{max_confidence:.2%}")
                        with col3:
                            st.metric("Lowest Confidence", f"{min_confidence:.2%}")
                    
                    else:
                        st.error("No models could be loaded for comparison!")
                        
                except Exception as e:
                    st.error(f"Error during comparison: {e}")
        else:
            st.warning("Please enter text for comparison!")

# Custom Pipeline Page
elif page == "🔧 Custom Pipeline":
    st.markdown('<h1 class="main-header">🔧 Custom Pipeline</h1>', unsafe_allow_html=True)
    
    st.markdown("""
    Create your own custom analysis pipeline by combining multiple NLP tasks!
    """)
    
    # Pipeline configuration
    st.markdown("### ⚙️ Configure Your Pipeline")
    
    col1, col2 = st.columns(2)
    
    with col1:
        include_sentiment = st.checkbox("📊 Sentiment Analysis", value=True)
        include_ner = st.checkbox("🔍 Named Entity Recognition", value=True)
    
    with col2:
        include_qa = st.checkbox("❓ Question Answering", value=False)
        include_generation = st.checkbox("💬 Text Generation", value=False)
    
    # Input text
    input_text = st.text_area(
        "Enter text for analysis:",
        value="Apple Inc. is doing great this year! Tim Cook announced amazing new products in Cupertino.",
        height=150
    )
    
    # Question for QA (if enabled)
    if include_qa:
        question = st.text_input(
            "Question for Q&A:",
            value="Who is the CEO mentioned?"
        )
    
    # Generation prompt (if enabled)
    if include_generation:
        generation_prompt = st.text_input(
            "Additional text to generate:",
            value="Based on this news, the future of Apple looks"
        )
    
    if st.button("🚀 Run Custom Pipeline", type="primary"):
        if input_text:
            st.markdown("### 📊 Pipeline Results")
            
            results = {}
            
            # Sentiment Analysis
            if include_sentiment:
                with st.spinner("Analyzing sentiment..."):
                    try:
                        classifier = get_sentiment_pipeline()
                        sentiment_result = classifier(input_text)
                        results['sentiment'] = sentiment_result
                        
                        st.markdown("#### 🎭 Sentiment Analysis")
                        for result in sentiment_result:
                            st.markdown(f"**{result['label']}** - {result['score']:.2%}")
                    except Exception as e:
                        st.error(f"Sentiment analysis error: {e}")
            
            # Named Entity Recognition
            if include_ner:
                with st.spinner("Extracting entities..."):
                    try:
                        ner_pipeline = get_ner_pipeline()
                        ner_result = ner_pipeline(input_text)
                        results['ner'] = ner_result
                        
                        st.markdown("#### 🔍 Named Entities")
                        if ner_result:
                            for entity in ner_result:
                                st.markdown(f"**{entity['word']}** ({entity['entity_group']}) - {entity['score']:.2%}")
                        else:
                            st.markdown("No entities found.")
                    except Exception as e:
                        st.error(f"NER error: {e}")
            
            # Question Answering
            if include_qa and 'question' in locals():
                with st.spinner("Finding answer..."):
                    try:
                        qa_pipeline = get_qa_pipeline()
                        qa_result = qa_pipeline(question=question, context=input_text)
                        results['qa'] = qa_result
                        
                        st.markdown("#### ❓ Question Answering")
                        st.markdown(f"**Q:** {question}")
                        st.markdown(f"**A:** {qa_result['answer']} (Confidence: {qa_result['score']:.2%})")
                    except Exception as e:
                        st.error(f"Q&A error: {e}")
            
            # Text Generation
            if include_generation and 'generation_prompt' in locals():
                with st.spinner("Generating text..."):
                    try:
                        generator = get_text_generation_pipeline()
                        generation_result = generator(
                            generation_prompt,
                            max_length=100,
                            temperature=0.7,
                            pad_token_id=generator.tokenizer.eos_token_id
                        )
                        results['generation'] = generation_result
                        
                        st.markdown("#### 💬 Text Generation")
                        generated_text = generation_result[0]['generated_text']
                        new_text = generated_text[len(generation_prompt):].strip()
                        st.markdown(f"**Prompt:** {generation_prompt}")
                        st.markdown(f"**Generated:** {new_text}")
                    except Exception as e:
                        st.error(f"Generation error: {e}")
            
            # Summary
            if results:
                st.markdown("### 📋 Pipeline Summary")
                
                summary_data = []
                for task, result in results.items():
                    if task == 'sentiment':
                        summary_data.append({
                            'Task': 'Sentiment Analysis',
                            'Status': '✅ Completed',
                            'Key Result': f"{result[0]['label']} ({result[0]['score']:.2%})"
                        })
                    elif task == 'ner':
                        entity_count = len(result) if result else 0
                        summary_data.append({
                            'Task': 'Named Entity Recognition',
                            'Status': '✅ Completed',
                            'Key Result': f"{entity_count} entities found"
                        })
                    elif task == 'qa':
                        summary_data.append({
                            'Task': 'Question Answering',
                            'Status': '✅ Completed',
                            'Key Result': f"Answer: {result['answer'][:30]}..."
                        })
                    elif task == 'generation':
                        summary_data.append({
                            'Task': 'Text Generation',
                            'Status': '✅ Completed',
                            'Key Result': f"Generated {len(result[0]['generated_text'])} characters"
                        })
                
                df_summary = pd.DataFrame(summary_data)
                st.dataframe(df_summary, use_container_width=True)
        else:
            st.warning("Please enter some text!")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; margin-top: 2rem;">
    <p>🤗 Powered by Hugging Face Transformers | Built with Streamlit</p>
    <p>Explore more models and datasets at <a href="https://huggingface.co" target="_blank">huggingface.co</a></p>
</div>
""", unsafe_allow_html=True)
