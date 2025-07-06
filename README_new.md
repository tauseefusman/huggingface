# 🤗 Hugging Face Showcase

A comprehensive demonstration of Hugging Face's transformers library and datasets, featuring both command-line examples and an interactive Streamlit web application.

## 🚀 Features

### 📱 Interactive Streamlit App
- **Multi-page web interface** with beautiful UI
- **Real-time model inference** with instant results
- **Interactive visualizations** using Plotly
- **Model comparison tools** side-by-side analysis
- **Dataset exploration** with statistics and charts
- **Custom pipeline builder** combine multiple NLP tasks

### 🤖 Available Models & Tasks
- **Text Classification** - Categorize text into different classes
- **Sentiment Analysis** - Analyze emotional tone of text
- **Question Answering** - Extract answers from context
- **Named Entity Recognition** - Identify people, places, organizations
- **Text Generation** - Generate creative text continuations
- **Custom Pipelines** - Combine multiple tasks

### 📊 Dataset Integration
- **IMDB Movie Reviews** - Sentiment analysis dataset
- **SQuAD** - Question answering dataset
- **CoNLL-2003** - Named entity recognition
- **Custom dataset upload** - Analyze your own data

## 🛠️ Installation

This project uses UV for fast dependency management:

```bash
# Clone/navigate to the project
cd huggingface

# Install dependencies (UV will handle everything)
uv sync

# Or manually install additional packages
uv add streamlit plotly pandas numpy
```

## 🎮 Usage

### 🌐 Run the Streamlit Web App (Recommended)

```bash
# Method 1: Using the launcher script
uv run python run_app.py

# Method 2: Direct streamlit command
uv run streamlit run streamlit_app.py

# Method 3: Using UV with streamlit
uv run --with streamlit streamlit run streamlit_app.py
```

The app will open automatically in your browser at `http://localhost:8501`

### 🖥️ Command Line Examples

```bash
# Run basic examples
uv run python main.py

# Run advanced examples
uv run python advanced_examples.py

# Test the setup
uv run python test_setup.py
```

## 📱 Streamlit App Features

### 🏠 Home Page
- Overview of available models and datasets
- Performance metrics and statistics
- Interactive charts showing model capabilities

### 📝 Text Classification
- Real-time text classification
- Confidence scores with visual indicators
- Support for multiple classification models

### ❓ Question Answering
- Interactive Q&A with context highlighting
- Confidence gauges and answer positioning
- Support for custom context and questions

### 🎭 Sentiment Analysis
- Single text, multiple texts, or file upload
- Batch processing with summary statistics
- Distribution charts and sentiment breakdown

### 📊 Dataset Explorer
- Load and explore popular HF datasets
- Statistical analysis and visualizations
- Sample predictions on real data

### 🔍 Named Entity Recognition
- Extract entities with confidence scores
- Color-coded entity highlighting
- Interactive entity type filtering

### 💬 Text Generation
- Creative text generation with GPT-2
- Adjustable parameters (temperature, length)
- Multiple variant generation

### 📈 Model Comparison
- Side-by-side model performance
- Agreement analysis between models
- Confidence comparison charts

### 🔧 Custom Pipeline
- Build custom NLP workflows
- Combine multiple tasks in sequence
- Configurable pipeline components

## 🎯 Example Use Cases

1. **Content Moderation**: Analyze user-generated content for sentiment
2. **Document Processing**: Extract entities and classify documents
3. **Customer Support**: Answer questions from knowledge bases
4. **Content Generation**: Create marketing copy or creative writing
5. **Research Analysis**: Explore model capabilities and compare performance

## 📋 Requirements

### Core Dependencies
- Python 3.11+
- torch >= 2.7.1
- transformers >= 4.52.4
- datasets >= 3.6.0

### Web App Dependencies
- streamlit >= 1.28.0
- plotly >= 5.17.0
- pandas >= 2.1.0
- numpy >= 1.24.0

## 🚀 Quick Start

1. **Install UV** (if not already installed):
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

2. **Clone and setup**:
   ```bash
   git clone <your-repo>
   cd huggingface
   uv sync
   ```

3. **Launch the web app**:
   ```bash
   uv run python run_app.py
   ```

4. **Open your browser** and navigate to `http://localhost:8501`

## 🎨 Customization

### Adding New Models
Edit `streamlit_app.py` and add new models to the pipeline functions:

```python
@st.cache_resource
def get_custom_pipeline():
    return pipeline("your-task", model="your-model-name")
```

### Adding New Pages
Create new page sections in the main `streamlit_app.py` file:

```python
elif page == "🆕 Your New Page":
    st.markdown('<h1 class="main-header">🆕 Your New Page</h1>', unsafe_allow_html=True)
    # Your page content here
```

### Styling
Modify the CSS in the `st.markdown()` section at the top of `streamlit_app.py` to customize colors, fonts, and layout.

## 🐛 Troubleshooting

### Common Issues

1. **Model loading errors**: Some models require authentication or may be temporarily unavailable
2. **Memory issues**: Reduce batch size or use smaller models for limited hardware
3. **Port conflicts**: Change the port in `run_app.py` if 8501 is already in use

### Performance Tips

1. **Use caching**: The app uses `@st.cache_resource` for model loading
2. **GPU acceleration**: Models will automatically use GPU if available
3. **Model selection**: Choose smaller models for faster inference

## 📚 Learning Resources

- [Hugging Face Documentation](https://huggingface.co/docs)
- [Transformers Library](https://huggingface.co/docs/transformers)
- [Datasets Library](https://huggingface.co/docs/datasets)
- [Streamlit Documentation](https://docs.streamlit.io)

## 🤝 Contributing

Feel free to contribute by:
- Adding new models or tasks
- Improving the UI/UX
- Adding more datasets
- Optimizing performance
- Adding new visualizations

## 📄 License

This project is open source and available under the MIT License.
