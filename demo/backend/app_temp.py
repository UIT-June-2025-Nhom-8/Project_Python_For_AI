from flask import Flask, request, jsonify
from flask_cors import CORS
import sys
import os
import pickle
import pandas as pd
import numpy as np
from gensim.models import LdaModel
from gensim import corpora
import logging

# Add local src directory to Python path to import modules
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

try:
    from pre_processor import PreProcessor
    from gensim_lda import GensimLDA
except ImportError as e:
    print(f"Warning: Could not import preprocessing modules: {e}")
    print("Running without preprocessing - using basic text processing")

app = Flask(__name__)
CORS(app)  # Enable CORS for React frontend

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables for loaded models
sentiment_model = None
sentiment_vectorizer = None
sentiment_label_encoder = None
lda_model = None
lda_dictionary = None
lda_preprocessor = None

def load_sentiment_model():
    """Load the trained logistic regression model"""
    global sentiment_model, sentiment_vectorizer, sentiment_label_encoder
    
    try:
        # Path to the logistic regression model
        model_path = os.path.join(
            os.path.dirname(__file__), 
            'models', 'sentiment_analyze_models', 
            'logistic_regression_20250820_173924.pkl'
        )
        
        logger.info(f"Loading sentiment model from: {model_path}")
        
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        
        sentiment_model = model_data['model']
        sentiment_vectorizer = model_data['tfidf_vectorizer']
        sentiment_label_encoder = model_data['label_encoder']
        
        logger.info("Sentiment model loaded successfully")
        logger.info(f"Label classes: {sentiment_label_encoder.classes_}")
        
    except Exception as e:
        logger.error(f"Error loading sentiment model: {e}")
        raise

def load_lda_model():
    """Load the trained Gensim LDA model"""
    global lda_model, lda_dictionary, lda_preprocessor
    
    try:
        # Path to the Gensim LDA model
        model_dir = os.path.join(
            os.path.dirname(__file__), 
            'models', 'gensim_lda_model'
        )
        
        logger.info(f"Loading LDA model from: {model_dir}")
        
        # Load LDA model
        lda_model_path = os.path.join(model_dir, 'lda_model')
        lda_model = LdaModel.load(lda_model_path)
        
        # Load dictionary
        dictionary_path = os.path.join(model_dir, 'dictionary.gensim')
        lda_dictionary = corpora.Dictionary.load(dictionary_path)
        
        # Initialize preprocessor for LDA
        try:
            lda_preprocessor = GensimLDA()
        except:
            # Fallback basic preprocessing
            logger.warning("Using basic preprocessing for LDA")
            lda_preprocessor = None
        
        logger.info("LDA model loaded successfully")
        logger.info(f"Number of topics: {lda_model.num_topics}")
        
    except Exception as e:
        logger.error(f"Error loading LDA model: {e}")
        raise

def preprocess_text_for_sentiment(text):
    """Preprocess text for sentiment analysis"""
    try:
        # Try to use the actual preprocessor
        preprocessor = PreProcessor()
        # Convert tokens back to text for TF-IDF
        cleaned_text = preprocessor.clean_text(text)
        tokens = preprocessor.tokenize_text(cleaned_text)
        tokens_no_stop = preprocessor.remove_stopwords(tokens)
        # Use normalize_token method if available
        if hasattr(preprocessor, 'normalize_token'):
            normalized_tokens = [preprocessor.normalize_token(token) for token in tokens_no_stop]
        else:
            normalized_tokens = tokens_no_stop
        return " ".join(normalized_tokens)
    except:
        # Fallback basic preprocessing
        import re
        text = text.lower()
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        return text

def preprocess_text_for_lda(text):
    """Preprocess text for topic detection"""
    try:
        if lda_preprocessor:
            return lda_preprocessor.preprocess_for_lda(text)
        else:
            # Fallback basic preprocessing
            import re
            text = text.lower()
            text = re.sub(r'[^a-zA-Z\s]', '', text)
            tokens = text.split()
            # Basic stopword removal
            basic_stopwords = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'must', 'shall', 'can', 'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they'}
            tokens = [token for token in tokens if token not in basic_stopwords and len(token) > 2]
            return tokens
    except Exception as e:
        logger.error(f"Error in LDA preprocessing: {e}")
        # Very basic fallback
        return text.lower().split()

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({'status': 'healthy', 'message': 'Backend is running'})

@app.route('/analyze/sentiment', methods=['POST'])
def analyze_sentiment():
    """Analyze sentiment of text using the trained logistic regression model"""
    try:
        data = request.get_json()
        text = data.get('text', '').strip()
        
        if not text:
            return jsonify({'error': 'No text provided'}), 400
        
        if sentiment_model is None:
            return jsonify({'error': 'Sentiment model not loaded'}), 500
        
        # Preprocess text
        processed_text = preprocess_text_for_sentiment(text)
        
        # Transform using TF-IDF vectorizer
        text_vector = sentiment_vectorizer.transform([processed_text])
        
        # Predict
        prediction = sentiment_model.predict(text_vector)[0]
        prediction_proba = sentiment_model.predict_proba(text_vector)[0]
        
        # Convert prediction back to original label
        predicted_label = sentiment_label_encoder.inverse_transform([prediction])[0]
        
        # Get class names and probabilities
        class_names = sentiment_label_encoder.classes_
        probabilities = {}
        
        for i, class_name in enumerate(class_names):
            probabilities[class_name.lower()] = float(prediction_proba[i])
        
        # Determine confidence (max probability)
        confidence = float(max(prediction_proba))
        
        result = {
            'label': predicted_label.lower(),
            'confidence': confidence,
            'probabilities': probabilities
        }
        
        logger.info(f"Sentiment analysis result: {result}")
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error in sentiment analysis: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/analyze/topics', methods=['POST'])
def analyze_topics():
    """Detect topics using the trained Gensim LDA model"""
    try:
        data = request.get_json()
        text = data.get('text', '').strip()
        
        if not text:
            return jsonify({'error': 'No text provided'}), 400
        
        if lda_model is None or lda_dictionary is None:
            return jsonify({'error': 'LDA model not loaded'}), 500
        
        # Preprocess text for LDA
        processed_tokens = preprocess_text_for_lda(text)
        
        # Convert to bag of words
        bow = lda_dictionary.doc2bow(processed_tokens)
        
        # Get topic probabilities
        topic_probs = lda_model.get_document_topics(bow, minimum_probability=0.01)
        
        # Get topic information
        topics = []
        topic_names = {
            0: "Product Quality & Features",
            1: "Shipping & Delivery", 
            2: "Price & Value",
            3: "Customer Service",
            4: "User Experience", 
            5: "Design & Appearance",
            6: "Performance",
            7: "Packaging",
            8: "Durability",
            9: "Comparison & Recommendations",
            10: "Issues & Problems",
            11: "General Experience"
        }
        
        for topic_id, probability in topic_probs:
            # Get top words for this topic
            topic_words = lda_model.show_topic(topic_id, topn=5)
            words = [word for word, _ in topic_words]
            
            topic_name = topic_names.get(topic_id, f"Topic {topic_id}")
            
            topics.append({
                'id': int(topic_id),
                'name': topic_name,
                'words': words,
                'probability': float(probability)
            })
        
        # Sort by probability
        topics.sort(key=lambda x: x['probability'], reverse=True)
        
        # Determine dominant topic
        dominant_topic = topics[0] if topics else {
            'id': 0,
            'name': 'General Review',
            'probability': 1.0
        }
        
        result = {
            'topics': topics,
            'dominant_topic': {
                'id': dominant_topic['id'],
                'name': dominant_topic['name'],
                'probability': dominant_topic['probability']
            }
        }
        
        logger.info(f"Topic analysis result: {len(topics)} topics detected")
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error in topic analysis: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/models/info', methods=['GET'])
def model_info():
    """Get information about loaded models"""
    info = {
        'sentiment_model_loaded': sentiment_model is not None,
        'lda_model_loaded': lda_model is not None,
        'sentiment_classes': list(sentiment_label_encoder.classes_) if sentiment_label_encoder else [],
        'lda_num_topics': lda_model.num_topics if lda_model else 0
    }
    return jsonify(info)

if __name__ == '__main__':
    try:
        logger.info("Starting backend server...")
        
        # Load models
        load_sentiment_model()
        load_lda_model()
        
        logger.info("All models loaded successfully")
        
        # Start Flask app
        app.run(debug=True, host='0.0.0.0', port=5001)
        
    except Exception as e:
        logger.error(f"Failed to start server: {e}")
        sys.exit(1)