# Simplified Amazon Reviews Sentiment Analysis

A clean, easy-to-understand implementation of sentiment analysis for Amazon reviews using standard machine learning libraries.

## Features

- **Clean Code**: Simplified implementation using sklearn's built-in functionality
- **Multiple Classifiers**: Random Forest, Logistic Regression, and Gradient Boosting
- **Standardized Interface**: All classifiers follow the same simple API
- **Easy to Use**: One-line training and prediction methods
- **Automatic Hyperparameter Optimization**: Optional GridSearch for better performance
- **Comprehensive Evaluation**: Built-in model comparison and metrics

## Quick Start

### 1. Install Requirements
```bash
pip install -r requirements_simplified.txt
```

### 2. Prepare Your Data
Place your Amazon reviews data in `data/amazon_reviews/`:
- `train.csv` - Training data
- `test.csv` - Test data

Data format: `label,title,text` (where label: 1=negative, 2=positive)

### 3. Run the Pipeline
```bash
cd src
python simplified_main.py
```

## Usage Examples

### Quick Training
```python
from simplified_model_trainer import SimplifiedModelTrainer
import pandas as pd

# Load your data
df = pd.read_csv('your_data.csv')

# Train all models
trainer = SimplifiedModelTrainer()
results = trainer.train_all_models(df, 
                                 text_column='text', 
                                 target_column='sentiment')

# Compare results
trainer.compare_models()
```

### Individual Classifier Usage
```python
from simplified_logistic_regression import SimplifiedLogisticRegressionAnalyzer

# Create and train classifier
classifier = SimplifiedLogisticRegressionAnalyzer()
results = classifier.quick_train(df)

# Make predictions
predictions = classifier.predict(['This product is great!', 'I hate it'])
```

### Custom Training
```python
# For more control over the process
classifier = SimplifiedLogisticRegressionAnalyzer()

# Prepare data manually
X_train, X_test, y_train, y_test = classifier.prepare_data(df)

# Train with hyperparameter optimization
results = classifier.train(X_train, X_test, y_train, y_test, optimize=True)

# Evaluate
classifier.evaluate()
```

## Project Structure

```
src/
├── base_classifier.py              # Base class for all classifiers
├── simplified_logistic_regression.py   # Clean Logistic Regression
├── simplified_random_forest.py         # Clean Random Forest
├── simplified_gradient_boosting.py     # Clean Gradient Boosting
├── simplified_model_trainer.py         # Multi-model trainer
└── simplified_main.py                  # Main pipeline
```

## Key Improvements

### Before (Complex)
- 500+ lines per classifier
- Manual hyperparameter optimization
- Verbose logging everywhere
- Complex data preparation
- Inconsistent interfaces
- Manual feature importance calculation

### After (Simplified)
- ~100 lines per classifier
- Built-in sklearn GridSearchCV
- Clean, essential logging only
- Simple data preparation using sklearn
- Standardized interface for all classifiers
- Leverages sklearn's built-in methods

## Configuration

Edit the CONFIG dictionary in `simplified_main.py`:

```python
CONFIG = {
    "sample_size": 50000,           # Reduce for faster testing
    "test_size": 0.2,               # Train/test split ratio
    "optimize_hyperparameters": False  # Set True for better performance
}
```

## Output

The pipeline generates:
- **Model comparison table** showing accuracy, F1-score, and training time
- **JSON results file** with detailed metrics
- **Saved models** in pickle format for later use
- **Sample predictions** to verify functionality

## Performance Tips

1. **Start Small**: Use `sample_size` parameter for quick testing
2. **Hyperparameter Optimization**: Set `optimize_hyperparameters=True` for better results (slower)
3. **Memory Usage**: The simplified version uses less memory than the original
4. **Parallel Processing**: All classifiers use `n_jobs=-1` for multi-core training

## Dependencies

The simplified version requires only essential packages:
- scikit-learn (core ML functionality)
- pandas (data manipulation)
- numpy (numerical operations)

Optional packages for enhanced features:
- matplotlib/seaborn (visualizations)
- wordcloud (word cloud generation)
- jupyter (notebook support)

## Benefits of Simplified Version

1. **Easier to Understand**: Less code, clearer logic
2. **Easier to Maintain**: Uses standard sklearn patterns
3. **More Reliable**: Leverages well-tested sklearn functionality
4. **Better Performance**: Optimized sklearn implementations
5. **Consistent Interface**: All classifiers work the same way
6. **Easier to Extend**: Add new classifiers by inheriting from base class
