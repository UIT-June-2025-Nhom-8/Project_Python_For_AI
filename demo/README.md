# Product Review Analyzer

A full-stack React TypeScript demo application with Python Flask backend that uses **real trained AI models** for sentiment analysis and topic detection on product reviews.

## 🗂️ Project Structure

This demo is now organized as two independent, self-contained applications:

```
demo/
├── backend/              # 🐍 Complete Python Flask API
│   ├── app.py           # Main Flask application  
│   ├── models/          # All trained ML models (copied)
│   ├── src/             # Preprocessing modules (copied)
│   ├── requirements.txt # Python dependencies
│   ├── netlify.toml     # Netlify Functions config
│   └── README.md        # Backend documentation
│
├── frontend/            # ⚛️ Complete React TypeScript app
│   ├── src/             # React source code
│   ├── public/          # Static assets
│   ├── package.json     # Node dependencies
│   ├── netlify.toml     # Frontend deployment config
│   └── README.md        # Frontend documentation
│
└── README.md            # This overview
```

## ✨ Features

- **Real Sentiment Analysis**: Uses your trained **Logistic Regression model** for accurate classification
- **Real Topic Detection**: Uses your trained **Gensim LDA model** for intelligent topic modeling
- **Self-Contained**: Each directory contains everything needed for independent deployment
- **Netlify Ready**: Both backend and frontend configured for Netlify deployment
- **Fallback System**: Graceful degradation when backend unavailable

## 🚀 Quick Start Options

### Option 1: Full Local Development

**Terminal 1 - Backend:**
```bash
cd demo/backend
pip install -r requirements.txt
python test_setup.py  # Verify setup
python app.py         # Start API server
```

**Terminal 2 - Frontend:**
```bash
cd demo/frontend  
npm install
REACT_APP_API_URL=http://localhost:5000 npm start
```

### Option 2: Deploy to Netlify

**Backend (Netlify Functions):**
```bash
cd demo/backend
# Deploy this folder to Netlify as a Functions site
```

**Frontend (Netlify App):**
```bash
cd demo/frontend
# Deploy this folder to Netlify as a React site
```

## 📋 What's Different Now

### ✅ Self-Contained Design
- **Backend**: Contains all models, preprocessing code, no external dependencies
- **Frontend**: Complete React app with all components, ready to deploy independently
- **No Cross-References**: Neither directory depends on files outside itself

### ✅ Production Ready
- **Netlify Functions**: Backend configured as serverless functions
- **Environment Variables**: Proper API URL configuration
- **Build Optimization**: Both apps optimized for production deployment

### ✅ Flexible Deployment
- **Independent**: Deploy backend and frontend to different services
- **Combined**: Deploy both to Netlify for unified hosting  
- **Local**: Run locally for development and testing

## 🔗 API Endpoints

### Local Development
```bash
POST http://localhost:5000/analyze/sentiment
POST http://localhost:5000/analyze/topics  
GET http://localhost:5000/health
```

### Netlify Functions
```bash
POST /.netlify/functions/analyze/sentiment
POST /.netlify/functions/analyze/topics
GET /.netlify/functions/health
```

## 🧠 AI Models Included

Both directories contain your actual trained models:

- **Logistic Regression**: `models/sentiment_analyze_models/logistic_regression_20250820_173924.pkl`
- **Gensim LDA**: `models/gensim_lda_model/` (complete model + dictionary)
- **Preprocessing**: `src/pre_processor.py` and `src/gensim_lda.py`

## 🌐 Deployment Strategies

### Strategy 1: Netlify All-in-One
Deploy both backend and frontend to Netlify for unified hosting.

### Strategy 2: Split Deployment  
- **Backend**: Heroku, Railway, Google Cloud Functions
- **Frontend**: Netlify, Vercel, GitHub Pages

### Strategy 3: Local Development
Perfect for testing and development with full model capabilities.

## 📚 Documentation

- [`backend/README.md`](backend/README.md) - Complete backend documentation
- [`frontend/README.md`](frontend/README.md) - Complete frontend documentation  
- [`DEPLOYMENT.md`](DEPLOYMENT.md) - Detailed deployment guide

## 🎯 Key Benefits

✅ **Independent Deployments** - Backend and frontend can be deployed separately  
✅ **No External Dependencies** - Each directory is completely self-contained  
✅ **Real AI Models** - Uses your actual trained models, not mock data  
✅ **Production Ready** - Optimized for Netlify and other cloud platforms  
✅ **Development Friendly** - Easy local setup with clear instructions  

## 🚀 Next Steps

1. **Try Local Development**: Follow Quick Start Option 1
2. **Deploy to Netlify**: Upload backend/ and frontend/ as separate sites  
3. **Customize**: Each directory has detailed README for modifications
4. **Scale**: Move to dedicated hosting as your app grows

Each directory is now a complete, deployable application! 🎉
