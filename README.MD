# Amazon Reviews Analysis - AI-Powered Sentiment Analysis & Topic Modeling

![Python](https://img.shields.io/badge/Python-3.11.13-blue.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Status](https://img.shields.io/badge/Status-Active-brightgreen.svg)

A comprehensive machine learning project for automated sentiment analysis and topic modeling of Amazon product reviews. This project combines Natural Language Processing (NLP) techniques with multiple machine learning algorithms to extract meaningful insights from customer feedback.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage](#usage)
- [Configuration](#configuration)
- [Model Performance](#model-performance)
- [Results](#results)
- [Documentation](#documentation)
- [Contributing](#contributing)
- [License](#license)

## Overview

This project implements an end-to-end pipeline for analyzing Amazon product reviews with two main objectives:

1. **Sentiment Analysis**: Classify reviews as positive or negative using multiple ML algorithms
2. **Topic Modeling**: Discover hidden topics in reviews using LDA (Latent Dirichlet Allocation)

The system achieved **93.13% accuracy** using Gradient Boosting Classifier for sentiment classification and identified **16 optimal topics** with a coherence score of **0.5141**.

## Features

### Core Functionality

- **Multi-Algorithm Sentiment Analysis**: Compare 9 different ML algorithms
- **Topic Modeling**: Both Gensim and Scikit-learn LDA implementations
- **Automated Text Preprocessing**: Comprehensive NLP pipeline
- **Data Visualization**: Word clouds, topic distributions, performance metrics
- **Modular Architecture**: Clean, maintainable, and extensible codebase

### Advanced Features

- **Kaggle API Integration**: Automatic dataset downloading
- **Configurable Pipeline**: JSON-based configuration management
- **Cross-validation Support**: Robust model evaluation
- **Interactive Visualizations**: pyLDAvis topic exploration
- **Performance Monitoring**: Comprehensive metrics and reporting

## Project Structure

```
Project_Python_For_AI/
├── configs/                           # Configuration files
│   └── balanced_config.json           # Main configuration
├── docs/                              # Documentation
│   ├── overview.md                    # Project overview
│   ├── gensim_lda_report.md          # LDA analysis report
│   └── preprocessing_analyse.md       # Preprocessing analysis
├── notebook/                          # Jupyter notebooks
│   ├── main.ipynb                     # Main analysis notebook
│   ├── sklearn_lda-notebook.ipynb     # LDA analysis
│   └── 0.data_downloader.ipynb        # Data loading
├── src/                               # Source code
│   ├── main.py                        # Main execution script
│   ├── kaggle_data_loader.py          # Kaggle data loading
│   ├── pre_processor.py               # Text preprocessing
│   ├── sklearn_lda.py                 # Scikit-learn LDA
│   ├── gensim_lda.py                  # Gensim LDA
│   ├── lda_utils.py                   # LDA utilities
│   ├── model_trainer.py               # ML model training
│   ├── text_analyzer.py               # Text analysis tools
│   ├── tf_idf_vectorizer.py           # TF-IDF implementation
│   ├── logistic_regression_classifier.py
│   ├── random_forest_classifier.py
│   ├── gradient_boosting_classifier.py
│   └── images/                        # Generated visualizations
├── output/                            # Model outputs
│   └── models/                        # Saved models
├── reports/                           # Analysis reports
├── requirements.txt                   # Python dependencies
├── example.kaggle.json               # Kaggle API template
└── README.md                         # This file
```

## Requirements

### System Requirements

- **Python**: 3.11.13
- **Operating System**: macOS, Linux, or Windows
- **Memory**: Minimum 16GB RAM (24GB recommended for large datasets)
- **Storage**: 5GB free space

### Python Dependencies

See [requirements.txt](requirements.txt) for complete list of dependencies.

Key libraries:

- **pandas** (>=1.5.0) - Data manipulation
- **scikit-learn** (>=1.1.0) - Machine learning algorithms
- **gensim** (>=4.2.0) - Topic modeling
- **nltk** (>=3.7) - Natural language processing
- **matplotlib/seaborn** - Data visualization
- **jupyter** (>=1.0.0) - Notebook environment

## Installation

### Step 1: Clone the Repository

```bash
git clone https://github.com/UIT-June-2025-Nhom-8/Project_Python_For_AI.git
cd Project_Python_For_AI
```

### Step 2: Set Up Python Environment

Ensure you have Python 3.11.13 installed:

```bash
# Check Python version
python --version  # Should show Python 3.11.13

# Create virtual environment
python -m venv .venv

# Activate virtual environment
# On macOS/Linux:
source .venv/bin/activate
# On Windows:
# .venv\Scripts\activate
```

### Step 3: Install Dependencies

```bash
# Upgrade pip
pip install --upgrade pip

# Install requirements
pip install -r requirements.txt

# Download NLTK data
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('omw-1.4')"
```

### Step 4: Set Up Kaggle API (Optional)

For automatic dataset downloading:

```bash
# Create kaggle directory
mkdir -p ~/.kaggle

# Copy your Kaggle API key (rename example.kaggle.json to kaggle.json with your credentials)
cp example.kaggle.json ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json
```

## Quick Start

### Option 1: Run Complete Pipeline

```bash
# Activate virtual environment
source .venv/bin/activate

# Run main analysis
cd src
python main.py
```

### Option 2: Jupyter Notebook Analysis

```bash
# Start Jupyter
jupyter lab

# Open and run notebooks in order:
# 1. notebook/0.data_downloader.ipynb
# 2. notebook/main.ipynb
# 3. notebook/sklearn_lda-notebook.ipynb
```

### Option 3: Custom Configuration

```bash
# Edit configuration
nano configs/balanced_config.json

# Run with custom config
cd src
python main.py
```

## Usage

### Basic Sentiment Analysis

```python
from src.kaggle_data_loader import KaggleDataLoader
from src.pre_processor import PreProcessor
from src.model_trainer import ModelTrainer

# Load data
loader = KaggleDataLoader({"train_size": 1000, "test_size": 200})
train_df, test_df = loader.prepare_dataframes()

# Preprocess
preprocessor = PreProcessor()
train_processed = preprocessor.preprocess_pipeline(train_df)

# Train models
trainer = ModelTrainer()
results = trainer.train_and_evaluate_all(train_processed, test_df)
```

### Topic Modeling

```python
from src.sklearn_lda import SKLearnLDATopicModeler

# Initialize LDA
lda = SKLearnLDATopicModeler(n_topics=16)

# Fit and analyze
doc_topic_matrix = lda.fit_transform(texts)
topics = lda.get_top_words_per_topic(top_n=10)
```

### Visualization

```python
from src.text_analyzer import TextAnalyzer

# Create word clouds
analyzer = TextAnalyzer()
analyzer.create_and_save_wordclouds(data, save_dir="./images/")

# Plot LDA results
from src.lda_utils import plot_coherence_and_perplexity
plot_coherence_and_perplexity(results_df, save_path="./coherence_plot.png")
```

## Configuration

The project uses JSON configuration files in the `configs/` directory:

```json
{
  "dataset_config": {
    "train_size": 100000,
    "test_size": 10000,
    "random_state": 42
  },
  "preprocessing_config": {
    "min_length": 10,
    "max_length": 1000,
    "remove_stopwords": true
  },
  "model_config": {
    "test_size": 0.2,
    "random_state": 42
  }
}
```

## Model Performance

### Sentiment Analysis Results

| Model                 | Train Accuracy | Test Accuracy | F1-Score | Training Time |
| --------------------- | -------------- | ------------- | -------- | ------------- |
| **Gradient Boosting** | 100%           | **93.13%**    | 0.93     | ~98.7s        |
| LightGBM              | 99.61%         | 91.87%        | 0.92     | ~25.2s        |
| Random Forest         | 100%           | 91.87%        | 0.92     | ~15.3s        |
| Logistic Regression   | 91.78%         | 90.62%        | 0.91     | ~3.5s         |

### Topic Modeling Results

| Implementation   | Optimal Topics | Coherence Score | Perplexity |
| ---------------- | -------------- | --------------- | ---------- |
| **Gensim LDA**   | 16             | **0.5141**      | -6.5276    |
| Scikit-learn LDA | 12             | 0.4892          | -7.1234    |

## Results

### Key Achievements

- **93.13%** sentiment classification accuracy
- **16 topics** identified with meaningful interpretation
- **0.5141** coherence score for topic quality
- **Production-ready** modular architecture

### Discovered Topics

1. **Sound Quality & Headphones** - Audio device reviews
2. **Alexa & Echo Devices** - Smart home products
3. **Kindle & E-readers** - Reading devices
4. **Fire Tablets** - Amazon tablet reviews
5. **Prime Video & Streaming** - Entertainment services
6. **Screen Quality & Display** - Visual performance
7. **Battery Life & Performance** - Device performance
8. **Price & Value** - Cost-benefit analysis

### Visualizations

The project generates various visualizations:

- Word clouds for different sentiments
- Topic distribution plots
- Model performance comparisons
- Coherence analysis charts

## Documentation

Detailed documentation is available in the `docs/` directory:

- **[Project Overview](docs/overview.md)** - Comprehensive project analysis
- **[LDA Report](docs/gensim_lda_report.md)** - Topic modeling details
- **[Preprocessing Analysis](docs/preprocessing_analyse.md)** - Text processing insights

## Contributing

We welcome contributions! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Setup

```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
pytest tests/

# Format code
black src/
isort src/
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- **UIT (University of Information Technology)** - Academic support
- **Kaggle** - Dataset provision
- **Amazon** - Review data source
- **Open Source Community** - Library development

## Contact

**Project Team**: Nhóm 8 - June 2025
**Institution**: University of Information Technology (UIT)
**Course**: Python for Machine Learning

For questions or support, please open an issue on GitHub.

---

**Built with ❤️ using Python 3.11.13**
