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
# Allow all origins, methods, and headers for CORS (including OPTIONS preflight)
CORS(app, resources={r"/*": {"origins": "*"}}, supports_credentials=True, allow_headers="*", methods=["GET", "POST", "OPTIONS", "PUT", "DELETE"])

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables for loaded models
sentiment_models = {}
lda_models = {}
current_sentiment_model = None
current_lda_model = None

def load_sentiment_models():
    """Load all available sentiment analysis models"""
    global sentiment_models, current_sentiment_model
    
    sentiment_dir = os.path.join(
        os.path.dirname(__file__), 
        'models', 'sentiment_analyze_models'
    )
    
    available_models = {
        'logistic_regression': 'logistic_regression_20250905_180831.pkl',
        'gradient_boosting': 'gradient_boosting_20250905_142318.pkl',
        'random_forest': 'random_forest_20250905_011351.pkl'
    }
    
    for model_name, filename in available_models.items():
        try:
            model_path = os.path.join(sentiment_dir, filename)
            logger.info(f"Loading sentiment model {model_name} from: {model_path}")
            
            with open(model_path, 'rb') as f:
                model_data = pickle.load(f)
            
            sentiment_models[model_name] = {
                'model': model_data['model'],
                'vectorizer': model_data['tfidf_vectorizer'],
                'label_encoder': model_data['label_encoder'],
                'name': model_name.replace('_', ' ').title(),
                'filename': filename
            }
            
            logger.info(f"Sentiment model {model_name} loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading sentiment model {model_name}: {e}")
    
    # Set default model (logistic regression)
    if 'logistic_regression' in sentiment_models:
        current_sentiment_model = 'logistic_regression'
        logger.info(f"Default sentiment model set to: {current_sentiment_model}")
    elif sentiment_models:
        current_sentiment_model = list(sentiment_models.keys())[0]
        logger.info(f"Default sentiment model set to: {current_sentiment_model}")
    else:
        logger.error("No sentiment models loaded successfully")

def load_lda_models():
    """Load all available LDA models"""
    global lda_models, current_lda_model
    
    models_dir = os.path.join(os.path.dirname(__file__), 'models')
    
    # Load Gensim LDA model
    try:
        gensim_dir = os.path.join(models_dir, 'gensim_lda_model')
        logger.info(f"Loading Gensim LDA model from: {gensim_dir}")
        
        lda_model_path = os.path.join(gensim_dir, 'lda_model')
        dictionary_path = os.path.join(gensim_dir, 'dictionary.gensim')
        
        gensim_model = LdaModel.load(lda_model_path)
        gensim_dictionary = corpora.Dictionary.load(dictionary_path)
        
        # Initialize preprocessor for LDA
        try:
            gensim_preprocessor = GensimLDA()
        except:
            logger.warning("Using basic preprocessing for Gensim LDA")
            gensim_preprocessor = None
        
        lda_models['gensim_lda'] = {
            'model': gensim_model,
            'dictionary': gensim_dictionary,
            'preprocessor': gensim_preprocessor,
            'name': 'Gensim LDA',
            'type': 'gensim',
            'num_topics': gensim_model.num_topics
        }
        
        logger.info(f"Gensim LDA model loaded successfully with {gensim_model.num_topics} topics")
        
    except Exception as e:
        logger.error(f"Error loading Gensim LDA model: {e}")
    
    # Load sklearn LDA model
    try:
        sklearn_dir = os.path.join(models_dir, 'sklearn_lda_model')
        logger.info(f"Loading sklearn LDA model from: {sklearn_dir}")
        
        import joblib
        import json
        
        model_path = os.path.join(sklearn_dir, 'lda12_model.joblib')
        vectorizer_path = os.path.join(sklearn_dir, 'lda12_vectorizer.joblib')
        labels_path = os.path.join(sklearn_dir, 'lda12_labels.json')
        
        sklearn_model = joblib.load(model_path)
        sklearn_vectorizer = joblib.load(vectorizer_path)
        
        with open(labels_path, 'r') as f:
            sklearn_labels = json.load(f)
        
        lda_models['sklearn_lda'] = {
            'model': sklearn_model,
            'vectorizer': sklearn_vectorizer,
            'labels': sklearn_labels,
            'name': 'Sklearn LDA',
            'type': 'sklearn',
            'num_topics': sklearn_model.n_components
        }
        
        logger.info(f"Sklearn LDA model loaded successfully with {sklearn_model.n_components} topics")
        
    except Exception as e:
        logger.error(f"Error loading sklearn LDA model: {e}")
    
    # Set default model (gensim_lda)
    if 'gensim_lda' in lda_models:
        current_lda_model = 'gensim_lda'
        logger.info(f"Default LDA model set to: {current_lda_model}")
    elif lda_models:
        current_lda_model = list(lda_models.keys())[0]
        logger.info(f"Default LDA model set to: {current_lda_model}")
    else:
        logger.error("No LDA models loaded successfully")

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

def preprocess_text_for_lda(text, model_type='gensim'):
    """Preprocess text for topic detection based on model type"""
    try:
        if model_type == 'gensim':
            # For Gensim LDA
            model_data = lda_models.get('gensim_lda', {})
            preprocessor = model_data.get('preprocessor')
            
            if preprocessor:
                return preprocessor.preprocess_for_lda(text)
            else:
                # Fallback basic preprocessing for Gensim
                import re
                text = text.lower()
                text = re.sub(r'[^a-zA-Z\s]', '', text)
                tokens = text.split()
                # Basic stopword removal
                basic_stopwords = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'must', 'shall', 'can', 'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they'}
                tokens = [token for token in tokens if token not in basic_stopwords and len(token) > 2]
                return tokens
        
        elif model_type == 'sklearn':
            # For sklearn LDA - return as string for TF-IDF
            import re
            text = text.lower()
            text = re.sub(r'[^a-zA-Z\s]', '', text)
            # Basic stopword removal and return as string
            basic_stopwords = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'must', 'shall', 'can', 'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they'}
            tokens = [token for token in text.split() if token not in basic_stopwords and len(token) > 2]
            return ' '.join(tokens)
            
    except Exception as e:
        logger.error(f"Error in LDA preprocessing: {e}")
        # Very basic fallback
        if model_type == 'sklearn':
            return text.lower()
        else:
            return text.lower().split()

@app.route('/models/list', methods=['GET'])
def list_models():
    """Get information about all available models"""
    try:
        available_models = {
            'sentiment_models': [],
            'topic_models': [],
            'current_sentiment_model': current_sentiment_model,
            'current_topic_model': current_lda_model
        }
        
        # Add sentiment models
        for model_key, model_data in sentiment_models.items():
            available_models['sentiment_models'].append({
                'key': model_key,
                'name': model_data['name'],
                'filename': model_data['filename'],
                'type': 'sentiment'
            })
        
        # Add topic models
        for model_key, model_data in lda_models.items():
            available_models['topic_models'].append({
                'key': model_key,
                'name': model_data['name'],
                'type': model_data['type'],
                'num_topics': model_data['num_topics']
            })
        
        return jsonify(available_models)
        
    except Exception as e:
        logger.error(f"Error listing models: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/models/switch', methods=['POST'])
def switch_model():
    """Switch the current active model"""
    global current_sentiment_model, current_lda_model
    
    try:
        data = request.get_json()
        model_type = data.get('model_type')  # 'sentiment' or 'topic'
        model_key = data.get('model_key')
        
        if model_type == 'sentiment':
            if model_key in sentiment_models:
                current_sentiment_model = model_key
                logger.info(f"Switched to sentiment model: {model_key}")
                return jsonify({
                    'status': 'success',
                    'message': f'Switched to sentiment model: {sentiment_models[model_key]["name"]}',
                    'current_model': model_key
                })
            else:
                return jsonify({'error': f'Sentiment model {model_key} not found'}), 404
        
        elif model_type == 'topic':
            if model_key in lda_models:
                current_lda_model = model_key
                logger.info(f"Switched to topic model: {model_key}")
                return jsonify({
                    'status': 'success',
                    'message': f'Switched to topic model: {lda_models[model_key]["name"]}',
                    'current_model': model_key
                })
            else:
                return jsonify({'error': f'Topic model {model_key} not found'}), 404
        
        else:
            return jsonify({'error': 'Invalid model_type. Must be "sentiment" or "topic"'}), 400
        
    except Exception as e:
        logger.error(f"Error switching model: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({'status': 'healthy', 'message': 'Backend is running'})

@app.route('/analyze/sentiment', methods=['POST'])
def analyze_sentiment():
    """Analyze sentiment of text using the currently selected sentiment model"""
    try:
        data = request.get_json()
        text = data.get('text', '').strip()
        model_key = data.get('model', current_sentiment_model)  # Allow model override
        
        if not text:
            return jsonify({'error': 'No text provided'}), 400
        
        if not sentiment_models:
            return jsonify({'error': 'No sentiment models loaded'}), 500
        
        if model_key not in sentiment_models:
            return jsonify({'error': f'Sentiment model {model_key} not found'}), 404
        
        # Get the selected model
        model_data = sentiment_models[model_key]
        sentiment_model = model_data['model']
        sentiment_vectorizer = model_data['vectorizer']
        sentiment_label_encoder = model_data['label_encoder']
        
        # Preprocess text
        processed_text = preprocess_text_for_sentiment(text)
        
        # Transform using TF-IDF vectorizer
        text_vector = sentiment_vectorizer.transform([processed_text])
        
        # Predict
        prediction = sentiment_model.predict(text_vector)[0]
        prediction_proba = sentiment_model.predict_proba(text_vector)[0]
        
        # Convert prediction back to original label
        predicted_label = sentiment_label_encoder.inverse_transform([prediction])[0]
        
        # Map numeric labels to semantic labels
        # Assuming the model was trained with 1=negative, 2=positive (common encoding)
        label_mapping = {
            '1': 'negative',
            '2': 'positive',
            1: 'negative',
            2: 'positive'
        }
        
        # Get class names and probabilities with semantic labels
        class_names = sentiment_label_encoder.classes_
        probabilities = {}
        
        for i, class_name in enumerate(class_names):
            # Map numeric class to semantic label
            semantic_label = label_mapping.get(class_name, label_mapping.get(str(class_name), 'unknown'))
            probabilities[semantic_label] = float(prediction_proba[i])
        
        # Determine confidence (max probability)
        confidence = float(max(prediction_proba))
        
        # Map predicted label to semantic label
        predicted_semantic_label = label_mapping.get(predicted_label, label_mapping.get(str(predicted_label), 'unknown'))
        
        result = {
            'label': predicted_semantic_label,
            'confidence': confidence,
            'probabilities': probabilities,
            'model_used': {
                'key': model_key,
                'name': model_data['name']
            }
        }
        
        logger.info(f"Sentiment analysis result using {model_key}: {result}")
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error in sentiment analysis: {e}")
        return jsonify({'error': str(e)}), 500

def generate_topic_name(words):
    """Generate a meaningful topic name based on the top words"""
    # Define keyword patterns for different topic categories
    topic_patterns = {
        'Music & Audio': ['music', 'song', 'album', 'sound', 'audio', 'track', 'cd', 'listen'],
        'Books & Reading': ['book', 'read', 'reading', 'story', 'character', 'author', 'novel', 'page'],
        'Movies & Entertainment': ['movie', 'film', 'watch', 'funny', 'cinema', 'actor', 'director', 'scene'],
        'Gaming': ['game', 'play', 'player', 'graphic', 'level', 'gaming', 'console', 'controller'],
        'Shopping Experience': ['amazon', 'product', 'received', 'ordered', 'delivery', 'purchase', 'buy', 'seller'],
        'Media & Video': ['dvd', 'video', 'version', 'original', 'disc', 'format', 'quality', 'picture'],
        'Religious & Spiritual': ['god', 'christian', 'john', 'hero', 'powerful', 'faith', 'church', 'bible'],
        'Personal Stories': ['love', 'life', 'world', 'child', 'family', 'personal', 'experience', 'heart'],
        'User Experience': ['great', 'very', 'work', 'use', 'bought', 'good', 'excellent', 'recommend'],
        'General Reviews': ['not', 'but', 'rating', 'one', 'like', 'review', 'opinion', 'think']
    }
    
    # Convert words to lowercase for matching
    words_lower = [word.lower() for word in words]
    
    # Score each topic category based on word matches
    topic_scores = {}
    for topic_name, keywords in topic_patterns.items():
        score = sum(1 for word in words_lower if word in keywords)
        if score > 0:
            topic_scores[topic_name] = score
    
    # Return the highest scoring topic, or generate from top words if no match
    if topic_scores:
        best_topic = max(topic_scores, key=topic_scores.get)
        return best_topic
    else:
        # Fallback: create topic name from top 2-3 meaningful words
        meaningful_words = [word for word in words if len(word) > 3][:3]
        if meaningful_words:
            return f"Topic: {', '.join(meaningful_words).title()}"
        else:
            return f"Topic: {', '.join(words[:2]).title()}"

@app.route('/analyze/topics', methods=['POST'])
def analyze_topics():
    """Detect topics using the currently selected LDA model"""
    try:
        data = request.get_json()
        text = data.get('text', '').strip()
        model_key = data.get('model', current_lda_model)  # Allow model override
        
        if not text:
            return jsonify({'error': 'No text provided'}), 400
        
        if not lda_models:
            return jsonify({'error': 'No LDA models loaded'}), 500
        
        if model_key not in lda_models:
            return jsonify({'error': f'LDA model {model_key} not found'}), 404
        
        # Get the selected model
        model_data = lda_models[model_key]
        model_type = model_data['type']
        
        topics = []
        
        if model_type == 'gensim':
            # Gensim LDA processing
            lda_model = model_data['model']
            lda_dictionary = model_data['dictionary']
            
            # Preprocess text for LDA
            processed_tokens = preprocess_text_for_lda(text, 'gensim')
            
            # Convert to bag of words
            bow = lda_dictionary.doc2bow(processed_tokens)
            
            # Get topic probabilities
            topic_probs = lda_model.get_document_topics(bow, minimum_probability=0.01)
            
            # Get topic information
            for topic_id, probability in topic_probs:
                # Get top words for this topic
                topic_words = lda_model.show_topic(topic_id, topn=10)
                words = [word for word, _ in topic_words]
                top_5_words = words[:5]
                
                # Generate meaningful topic name based on the actual words
                topic_name = generate_topic_name(words)
                
                topics.append({
                    'id': int(topic_id),
                    'name': topic_name,
                    'words': top_5_words,
                    'probability': float(probability),
                    'all_words': words
                })
        
        elif model_type == 'sklearn':
            # Sklearn LDA processing
            lda_model = model_data['model']
            vectorizer = model_data['vectorizer']
            labels = model_data['labels']
            
            # Preprocess text for sklearn
            processed_text = preprocess_text_for_lda(text, 'sklearn')
            
            # Transform text using the vectorizer
            text_vector = vectorizer.transform([processed_text])
            
            # Get topic probabilities
            topic_probs = lda_model.transform(text_vector)[0]
            
            # Get topic information
            feature_names = vectorizer.get_feature_names_out()
            
            for topic_id, probability in enumerate(topic_probs):
                if probability > 0.01:  # Only include topics with significant probability
                    # Get top words for this topic
                    topic_words_indices = lda_model.components_[topic_id].argsort()[-10:][::-1]
                    words = [feature_names[i] for i in topic_words_indices]
                    top_5_words = words[:5]
                    
                    # Use predefined labels if available, otherwise generate
                    if str(topic_id) in labels:
                        topic_name = labels[str(topic_id)]
                    else:
                        topic_name = generate_topic_name(words)
                    
                    topics.append({
                        'id': int(topic_id),
                        'name': topic_name,
                        'words': top_5_words,
                        'probability': float(probability),
                        'all_words': words
                    })
        
        # Sort by probability
        topics.sort(key=lambda x: x['probability'], reverse=True)
        
        # Determine dominant topic
        dominant_topic = topics[0] if topics else {
            'id': 0,
            'name': 'General Review',
            'probability': 1.0,
            'words': []
        }
        
        result = {
            'topics': topics,
            'dominant_topic': {
                'id': dominant_topic['id'],
                'name': dominant_topic['name'],
                'probability': dominant_topic['probability'],
                'words': dominant_topic.get('words', [])
            },
            'model_used': {
                'key': model_key,
                'name': model_data['name'],
                'type': model_type
            }
        }
        
        logger.info(f"Topic analysis result using {model_key}: {len(topics)} topics detected")
        logger.info(f"Dominant topic: {dominant_topic['name']} ({dominant_topic['probability']:.3f})")
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error in topic analysis: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/models/info', methods=['GET'])
def model_info():
    """Get information about loaded models"""
    info = {
        'sentiment_models_loaded': len(sentiment_models),
        'topic_models_loaded': len(lda_models),
        'current_sentiment_model': current_sentiment_model,
        'current_topic_model': current_lda_model,
        'sentiment_models': {key: model['name'] for key, model in sentiment_models.items()},
        'topic_models': {key: {'name': model['name'], 'num_topics': model['num_topics']} for key, model in lda_models.items()}
    }
    return jsonify(info)

if __name__ == '__main__':
    try:
        logger.info("Starting backend server...")
        
        # Load models
        load_sentiment_models()
        load_lda_models()
        
        if sentiment_models:
            logger.info(f"Loaded {len(sentiment_models)} sentiment models: {list(sentiment_models.keys())}")
        else:
            logger.warning("No sentiment models loaded")
        
        if lda_models:
            logger.info(f"Loaded {len(lda_models)} LDA models: {list(lda_models.keys())}")
        else:
            logger.warning("No LDA models loaded")
        
        logger.info("All models loaded successfully")
        
        # Start Flask app
        app.run(debug=True, host='0.0.0.0', port=5001)
        
    except Exception as e:
        logger.error(f"Failed to start server: {e}")
        sys.exit(1)