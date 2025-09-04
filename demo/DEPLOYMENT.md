# Deployment Guide

This guide covers different deployment options for your AI-powered review analyzer.

## Option 1: Local Development (Recommended for Testing)

### Backend Setup
```bash
cd demo/backend

# Install dependencies
pip install flask flask-cors scikit-learn gensim nltk pandas numpy

# Verify setup
python test_setup.py

# Start backend
python app.py
```

### Frontend Setup
```bash
cd demo

# Install dependencies
npm install

# Start frontend
npm start
```

Visit `http://localhost:3000` to use the full application with real AI models.

## Option 2: Frontend Only (Netlify)

If you only want to deploy the frontend with fallback analysis:

```bash
cd demo

# Build for production
npm run build

# Deploy build/ folder to Netlify
```

The app will work with basic keyword analysis when the backend isn't available.

## Option 3: Full Stack Deployment

### Backend Deployment (Railway/Heroku)

1. **Prepare backend for deployment:**
```bash
cd demo/backend

# Create Procfile
echo "web: python app.py" > Procfile

# Update app.py to use PORT environment variable
# Add this to the end of app.py:
# port = int(os.environ.get('PORT', 5000))
# app.run(host='0.0.0.0', port=port)
```

2. **Deploy to Railway:**
   - Connect your repository to Railway
   - Set build command: `pip install -r requirements.txt`
   - Set start command: `python app.py`
   - Copy your model files to the deployed environment

3. **Deploy to Heroku:**
```bash
heroku create your-ai-backend
git subtree push --prefix=demo/backend heroku main
```

### Frontend Deployment (Netlify/Vercel)

1. **Set environment variables:**
Create `demo/.env.production`:
```
REACT_APP_API_URL=https://your-backend-url.herokuapp.com
```

2. **Build and deploy:**
```bash
cd demo
npm run build
# Deploy build/ folder to Netlify
```

## Option 4: Serverless Deployment

### Backend as Serverless Function

Convert the Flask app to work with serverless platforms:

1. **Vercel Functions:**
   - Create `demo/backend/api/analyze.py`
   - Use Vercel's Python runtime
   - Upload models as static assets

2. **AWS Lambda:**
   - Package models with deployment
   - Use API Gateway for HTTP endpoints
   - Consider model size limits

## Environment Variables

### Frontend (.env)
```
REACT_APP_API_URL=http://localhost:5000  # For development
REACT_APP_API_URL=https://your-api.com   # For production
```

### Backend
```
FLASK_ENV=production
PORT=5000  # Or from hosting provider
```

## Model Files Checklist

Ensure these files are included in your backend deployment:

```
models/
├── sentiment_analyze_models/
│   └── logistic_regression_20250820_173924.pkl
└── gensim_lda_model/
    ├── lda_model
    ├── dictionary.gensim
    ├── bigram_model
    └── trigram_model
```

## Performance Considerations

- **Model Loading**: Models are loaded once at startup
- **Memory Usage**: LDA model requires ~100-500MB RAM
- **Response Time**: Typical analysis takes 100-500ms
- **Concurrency**: Flask handles multiple requests, but consider using Gunicorn for production

## Security Notes

- Models contain no sensitive data
- Input text is processed temporarily
- No user data is stored persistently
- CORS is enabled for frontend access

## Troubleshooting

### Backend Issues
```bash
# Check if models load correctly
python demo/backend/test_setup.py

# Test API endpoints
curl -X POST http://localhost:5000/analyze/sentiment \
  -H "Content-Type: application/json" \
  -d '{"text": "test review"}'
```

### Frontend Issues
- Check browser console for API errors
- Verify REACT_APP_API_URL is set correctly
- Ensure backend is running and accessible

### Common Deployment Issues
1. **Model files missing**: Ensure all model files are uploaded
2. **Import errors**: Check all Python dependencies are installed
3. **CORS errors**: Verify CORS is configured in Flask
4. **Memory limits**: Some hosting providers have RAM restrictions

## Monitoring

Consider adding:
- Health check endpoints
- Request logging
- Error tracking (Sentry)
- Performance monitoring

Your AI models are now ready for production use!