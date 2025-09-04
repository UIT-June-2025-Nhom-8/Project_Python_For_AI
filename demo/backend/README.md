# AI Models Backend API

Flask backend service that serves trained machine learning models for sentiment analysis and topic detection.

## Features

- **Sentiment Analysis API**: Uses trained Logistic Regression model
- **Topic Detection API**: Uses trained Gensim LDA model
- **Model Loading**: Automatically loads models from local files
- **Preprocessing**: Uses original preprocessing pipeline
- **Health Checks**: Endpoint for service monitoring

## Models Included

This backend contains all necessary model files:

```
models/
├── sentiment_analyze_models/
│   └── logistic_regression_20250820_173924.pkl
├── gensim_lda_model/
│   ├── lda_model
│   ├── dictionary.gensim
│   ├── bigram_model
│   └── trigram_model
└── sklearn_lda_model/
    ├── lda12_model.joblib
    ├── lda12_vectorizer.joblib
    └── lda12_labels.json
```

## API Endpoints

### Health Check
```
GET /health
```

### Sentiment Analysis
```
POST /analyze/sentiment
Content-Type: application/json

{
  "text": "This product is amazing!"
}

Response:
{
  "label": "positive",
  "confidence": 0.85,
  "probabilities": {
    "positive": 0.85,
    "negative": 0.10,
    "neutral": 0.05
  }
}
```

### Topic Detection
```
POST /analyze/topics
Content-Type: application/json

{
  "text": "The delivery was fast and product quality is great."
}

Response:
{
  "topics": [
    {
      "id": 0,
      "name": "Product Quality & Features",
      "words": ["quality", "product", "great"],
      "probability": 0.6
    },
    {
      "id": 1,
      "name": "Shipping & Delivery",
      "words": ["delivery", "fast"],
      "probability": 0.4
    }
  ],
  "dominant_topic": {
    "id": 0,
    "name": "Product Quality & Features",
    "probability": 0.6
  }
}
```

### Model Information
```
GET /models/info

Response:
{
  "sentiment_model_loaded": true,
  "lda_model_loaded": true,
  "sentiment_classes": ["1", "2"],
  "lda_num_topics": 12
}
```

## Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Test setup
python test_setup.py

# Start server
python app.py
```

Server runs on `http://localhost:5000`

## Deployment Options

### Option 1: Netlify Functions
1. Deploy this entire backend folder to Netlify
2. Netlify will automatically detect the Python functions
3. Configure environment variables if needed

### Option 2: Heroku
```bash
# Add Procfile
echo "web: python app.py" > Procfile

# Deploy
git add .
git commit -m "Deploy backend"
heroku create your-app-name
git push heroku main
```

### Option 3: Railway/Render
1. Connect repository
2. Set build command: `pip install -r requirements.txt`
3. Set start command: `python app.py`

## Environment Variables

- `PORT`: Server port (default: 5000)
- `FLASK_ENV`: Environment (production/development)

## Performance Notes

- Models are loaded once at startup
- Typical response time: 100-500ms
- Memory usage: ~200-500MB
- Supports concurrent requests

## File Structure

```
backend/
├── app.py              # Main Flask application
├── requirements.txt    # Python dependencies
├── runtime.txt        # Python version for deployment
├── netlify.toml       # Netlify configuration
├── test_setup.py      # Setup verification script
├── models/            # All trained ML models
├── src/               # Preprocessing modules
│   ├── pre_processor.py
│   └── gensim_lda.py
└── README.md          # This file
```

## Self-Contained Design

This backend is completely self-contained:
- ✅ All model files included
- ✅ All preprocessing code included
- ✅ No external file dependencies
- ✅ Ready for independent deployment