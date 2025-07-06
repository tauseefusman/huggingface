# Hugging Face Model and Dataset Example Project

This project demonstrates how to use Hugging Face's Transformers library with various pre-trained models and datasets. The project is set up using UV for dependency management.

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- UV package manager

### Installation

1. Clone or navigate to this directory:
```bash
cd d:\tauseefusman\workspacepoc\huggingface
```

2. The dependencies are already installed. If you need to reinstall:
```bash
uv sync
```

### Running the Examples

1. **Test the setup** (recommended first step):
```bash
uv run test_setup.py
```

2. **Run the main example** (sentiment analysis with IMDB dataset):
```bash
uv run main.py
```

3. **Run advanced examples** (text generation, QA, summarization, NER):
```bash
uv run advanced_examples.py
```

## 📁 Project Structure

```
huggingface/
├── main.py                 # Main example with sentiment analysis and IMDB dataset
├── advanced_examples.py    # Advanced NLP tasks demonstration
├── test_setup.py          # Environment testing script
├── requirements.txt       # Python dependencies
├── pyproject.toml         # UV project configuration
├── README.md             # This file
└── .venv/                # Virtual environment (created by UV)
```

## 🤗 What's Included

### Main Example (`main.py`)
- **Model**: DistilBERT for sentiment analysis
- **Dataset**: IMDB movie reviews
- **Features**:
  - Automatic device detection (CPU/GPU)
  - Pipeline-based inference
  - Manual inference demonstration
  - Dataset loading and processing
  - Custom text analysis

### Advanced Examples (`advanced_examples.py`)
- **Text Generation**: Using GPT-2
- **Question Answering**: Using BERT
- **Text Summarization**: Using BART
- **Named Entity Recognition**: Using BERT-NER
- **Dataset Exploration**: CoLA dataset
- **Model Comparison**: Multiple sentiment models

### Test Suite (`test_setup.py`)
- Import verification
- Version checking
- Basic pipeline testing
- Model loading verification
- Inference testing

## 🔧 Dependencies

Core libraries installed:
- `transformers` - Hugging Face transformers library
- `datasets` - Hugging Face datasets library  
- `torch` - PyTorch for deep learning
- `numpy` - Numerical computing
- `pandas` - Data manipulation

## 💡 Key Features Demonstrated

### 1. Model Loading
```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification

tokenizer = AutoTokenizer.from_pretrained("model-name")
model = AutoModelForSequenceClassification.from_pretrained("model-name")
```

### 2. Pipeline Usage
```python
from transformers import pipeline

classifier = pipeline("sentiment-analysis")
result = classifier("I love this!")
```

### 3. Dataset Loading
```python
from datasets import load_dataset

dataset = load_dataset("imdb", split="test")
```

### 4. Manual Inference
```python
inputs = tokenizer(text, return_tensors="pt")
outputs = model(**inputs)
predictions = torch.softmax(outputs.logits, dim=-1)
```

## 🎯 Tasks Demonstrated

1. **Sentiment Analysis** - Classify text as positive/negative
2. **Text Generation** - Generate coherent text continuations
3. **Question Answering** - Answer questions based on context
4. **Text Summarization** - Create concise summaries
5. **Named Entity Recognition** - Extract entities from text
6. **Dataset Processing** - Load and process HF datasets

## 🔍 Models Used

- **DistilBERT**: Lightweight BERT for sentiment analysis
- **GPT-2**: Text generation model
- **BART**: Sequence-to-sequence model for summarization
- **BERT-NER**: Named entity recognition model
- **RoBERTa**: Twitter sentiment analysis

## 📊 Datasets Used

- **IMDB**: Movie review sentiment dataset
- **CoLA**: Corpus of Linguistic Acceptability

## 🖥️ System Requirements

- **CPU**: Any modern CPU (examples work on CPU)
- **GPU**: Optional, CUDA-compatible GPU for faster inference
- **Memory**: 4GB+ RAM recommended
- **Storage**: 2GB+ for model downloads

## 🚨 Troubleshooting

### Common Issues

1. **Network connectivity**: Models download from HuggingFace Hub
2. **Memory issues**: Use smaller models if you have limited RAM
3. **CUDA errors**: Examples fall back to CPU automatically

### Error Solutions

- **ModuleNotFoundError**: Run `uv sync` to install dependencies
- **Connection timeout**: Check internet connection for model downloads
- **Out of memory**: Reduce batch size or use smaller models

## 📚 Learning Resources

- [Hugging Face Documentation](https://huggingface.co/docs/transformers)
- [Transformers Course](https://huggingface.co/course)
- [Model Hub](https://huggingface.co/models)
- [Dataset Hub](https://huggingface.co/datasets)

## 🛠️ Extending the Project

### Add New Models
1. Find a model on [Hugging Face Hub](https://huggingface.co/models)
2. Replace model name in the scripts
3. Adjust preprocessing if needed

### Add New Datasets
1. Browse [Hugging Face Datasets](https://huggingface.co/datasets)
2. Use `load_dataset("dataset-name")` 
3. Process according to dataset format

### Custom Training
Add your own training scripts using the Trainer API:
```python
from transformers import Trainer, TrainingArguments
```

## 📄 License

This project is for educational purposes. Model and dataset licenses apply separately.

## 🤝 Contributing

Feel free to add more examples or improve existing ones!

---

**Happy modeling with Hugging Face! 🤗**