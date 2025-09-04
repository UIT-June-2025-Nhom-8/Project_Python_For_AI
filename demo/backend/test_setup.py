#!/usr/bin/env python3
"""
Test script to verify the backend setup
"""

import sys
import os
import pickle

def test_imports():
    """Test if all required imports work"""
    try:
        from flask import Flask, jsonify
        from flask_cors import CORS
        print("✅ Flask imports successful")
        return True
    except ImportError as e:
        print(f"❌ Flask import failed: {e}")
        return False

def test_model_files():
    """Test if model files can be loaded"""
    try:
        # Test sentiment model
        sentiment_path = "../../models/sentiment_analyze_models/logistic_regression_20250820_173924.pkl"
        full_path = os.path.join(os.path.dirname(__file__), sentiment_path)
        
        if os.path.exists(full_path):
            with open(full_path, 'rb') as f:
                model_data = pickle.load(f)
            print("✅ Sentiment model loaded successfully")
            print(f"   Model classes: {model_data.get('label_encoder', {}).classes_ if hasattr(model_data.get('label_encoder', {}), 'classes_') else 'N/A'}")
        else:
            print(f"❌ Sentiment model file not found: {full_path}")
            return False
            
        # Test LDA model files
        lda_model_path = "../../models/gensim_lda_model/lda_model"
        lda_dict_path = "../../models/gensim_lda_model/dictionary.gensim"
        
        full_lda_path = os.path.join(os.path.dirname(__file__), lda_model_path)
        full_dict_path = os.path.join(os.path.dirname(__file__), lda_dict_path)
        
        if os.path.exists(full_lda_path) and os.path.exists(full_dict_path):
            print("✅ LDA model files found")
        else:
            print(f"❌ LDA model files not found:")
            print(f"   Model: {full_lda_path}")
            print(f"   Dictionary: {full_dict_path}")
            return False
            
        return True
        
    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        return False

def test_preprocessing():
    """Test preprocessing capabilities"""
    try:
        sys.path.append('../../src')
        from pre_processor import PreProcessor
        
        preprocessor = PreProcessor()
        test_text = "This is a great product! I love it."
        
        cleaned = preprocessor.clean_text(test_text)
        tokens = preprocessor.tokenize_text(cleaned)
        no_stop = preprocessor.remove_stopwords(tokens)
        
        print("✅ Preprocessing works")
        print(f"   Original: {test_text}")
        print(f"   Processed: {no_stop}")
        return True
        
    except Exception as e:
        print(f"⚠️  Preprocessing failed (will use fallback): {e}")
        return False

if __name__ == "__main__":
    print("=== Backend Setup Test ===")
    
    success = True
    
    print("\n1. Testing Flask imports...")
    success &= test_imports()
    
    print("\n2. Testing model files...")
    success &= test_model_files()
    
    print("\n3. Testing preprocessing...")
    test_preprocessing()  # This is optional, won't fail the test
    
    print(f"\n=== Test Results ===")
    if success:
        print("✅ Backend setup is ready!")
        print("You can start the server with: python app.py")
    else:
        print("❌ Backend setup has issues. Please fix the errors above.")
        
    sys.exit(0 if success else 1)