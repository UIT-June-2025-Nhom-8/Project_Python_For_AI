#!/usr/bin/env python3
"""
Test script to verify Kaggle stopwords and preprocessor integration.
"""

import sys
import os

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_kaggle_integration():
    """Test that Kaggle stopwords config integrates correctly with preprocessor."""
    
    print("🧪 Testing Kaggle Integration")
    print("=" * 40)
    
    # Test stopwords config
    try:
        import stopwords_config_kaggle as kaggle_config
        print("✅ Kaggle stopwords config imported successfully")
        
        # Test available functions (without basic stopwords)
        expected_functions = ['get_nltk_stopwords', 'get_extended_stopwords', 'get_sentiment_stopwords', 'initialize_stopwords']
        for func_name in expected_functions:
            if hasattr(kaggle_config, func_name):
                print(f"  ✅ {func_name} available")
            else:
                print(f"  ❌ {func_name} missing")
                return False
        
        # Test that basic stopwords are NOT available (removed)
        if hasattr(kaggle_config, 'get_basic_stopwords'):
            print("  ❌ get_basic_stopwords should be removed")
            return False
        else:
            print("  ✅ get_basic_stopwords correctly removed")
        
        # Test constants (without BASIC_STOPWORDS)
        expected_constants = ['NLTK_STOPWORDS', 'EXTENDED_STOPWORDS', 'SENTIMENT_STOPWORDS', 
                             'DEFAULT_PREPROCESSING_STOPWORDS', 'DEFAULT_WORDCLOUD_STOPWORDS']
        for const_name in expected_constants:
            if hasattr(kaggle_config, const_name):
                print(f"  ✅ {const_name} available")
            else:
                print(f"  ❌ {const_name} missing")
                return False
        
        # Test that BASIC_STOPWORDS is NOT available (removed)
        if hasattr(kaggle_config, 'BASIC_STOPWORDS'):
            print("  ❌ BASIC_STOPWORDS should be removed")
            return False
        else:
            print("  ✅ BASIC_STOPWORDS correctly removed")
            
    except Exception as e:
        print(f"❌ Failed to import or test kaggle stopwords config: {e}")
        return False
    
    # Test preprocessor integration
    try:
        import preprocessor_kaggle
        print("\n✅ Kaggle preprocessor imported successfully")
        
        # Test that it can import get_sentiment_stopwords
        from stopwords_config_kaggle import get_sentiment_stopwords
        print("  ✅ get_sentiment_stopwords import works")
        
        # Test preprocessor creation (should work without basic stopwords)
        preprocessor = preprocessor_kaggle.KagglePreProcessor()
        print("  ✅ Preprocessor created successfully")
        
        # Test that preprocessor can get sentiment stopwords
        stopwords = get_sentiment_stopwords()
        print(f"  ✅ Got {len(stopwords)} sentiment stopwords")
        
        # Verify sentiment words are preserved (not in stopwords)
        sentiment_critical = {"not", "no", "never", "but", "however", "very", "good", "bad", "love", "hate"}
        preserved_count = sum(1 for word in sentiment_critical if word not in stopwords)
        preservation_rate = preserved_count / len(sentiment_critical) * 100
        
        print(f"  ✅ Sentiment preservation: {preserved_count}/{len(sentiment_critical)} ({preservation_rate:.1f}%)")
        
        if preservation_rate < 80:
            print("  ⚠️  Low sentiment word preservation")
            return False
            
    except Exception as e:
        print(f"❌ Failed to test preprocessor integration: {e}")
        return False
    
    print("\n🎉 Integration test passed! Kaggle config works correctly with preprocessor.")
    return True


if __name__ == "__main__":
    success = test_kaggle_integration()
    sys.exit(0 if success else 1)
