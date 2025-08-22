#!/usr/bin/env python3
"""
Test script to compare stopwords coverage between main config and Kaggle config.
This script verifies that the Kaggle version provides equivalent or better stopwords.
"""

import sys
import os

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_stopwords_parity():
    """Test that Kaggle config provides equivalent or better stopwords than main config."""
    
    print("🔍 Testing Stopwords Configuration Parity")
    print("=" * 50)
    
    # Import main config
    try:
        import stopwords_config as main_config
        print("✅ Main config imported successfully")
    except Exception as e:
        print(f"❌ Failed to import main config: {e}")
        return False
    
    # Import kaggle config
    try:
        import stopwords_config_kaggle as kaggle_config
        print("✅ Kaggle config imported successfully")
    except Exception as e:
        print(f"❌ Failed to import Kaggle config: {e}")
        return False
    
    # Test function availability
    print("\n📋 Function Availability Test:")
    main_functions = ['get_nltk_stopwords', 'get_basic_stopwords', 'get_extended_stopwords', 'get_sentiment_stopwords']
    
    for func_name in main_functions:
        main_has = hasattr(main_config, func_name)
        kaggle_has = hasattr(kaggle_config, func_name)
        
        status = "✅" if kaggle_has else "❌"
        print(f"  {status} {func_name}: Main={main_has}, Kaggle={kaggle_has}")
        
        if not kaggle_has:
            print(f"    ⚠️  Missing function in Kaggle config!")
            return False
    
    # Test constants availability
    print("\n📋 Constants Availability Test:")
    main_constants = ['NLTK_STOPWORDS', 'BASIC_STOPWORDS', 'EXTENDED_STOPWORDS', 'SENTIMENT_STOPWORDS',
                      'DEFAULT_PREPROCESSING_STOPWORDS', 'DEFAULT_WORDCLOUD_STOPWORDS']
    
    for const_name in main_constants:
        main_has = hasattr(main_config, const_name)
        kaggle_has = hasattr(kaggle_config, const_name)
        
        status = "✅" if kaggle_has else "❌"
        print(f"  {status} {const_name}: Main={main_has}, Kaggle={kaggle_has}")
        
        if not kaggle_has:
            print(f"    ⚠️  Missing constant in Kaggle config!")
            return False
    
    # Test stopwords coverage (using basic stopwords since NLTK might not be available)
    print("\n📊 Stopwords Coverage Test:")
    
    try:
        main_basic = main_config.get_basic_stopwords()
        kaggle_basic = kaggle_config.get_basic_stopwords()
        
        print(f"  Main basic stopwords: {len(main_basic)}")
        print(f"  Kaggle basic stopwords: {len(kaggle_basic)}")
        
        coverage = len(kaggle_basic & main_basic) / len(main_basic) * 100
        print(f"  Coverage: {coverage:.1f}%")
        
        if coverage < 95:
            print(f"    ⚠️  Coverage is below 95%!")
            return False
        else:
            print(f"    ✅ Good coverage!")
            
    except Exception as e:
        print(f"  ❌ Error testing basic stopwords: {e}")
        return False
    
    # Test sentiment stopwords preservation
    print("\n🎭 Sentiment Words Preservation Test:")
    
    try:
        kaggle_sentiment = kaggle_config.get_sentiment_stopwords()
        
        # These words should NOT be in sentiment stopwords (they should be preserved)
        sentiment_critical = {"not", "no", "never", "but", "however", "very", "good", "bad", "love", "hate"}
        
        preserved_count = 0
        for word in sentiment_critical:
            if word not in kaggle_sentiment:
                preserved_count += 1
            else:
                print(f"    ⚠️  '{word}' was removed from text (should be preserved)")
        
        preservation_rate = preserved_count / len(sentiment_critical) * 100
        print(f"  Sentiment words preserved: {preserved_count}/{len(sentiment_critical)} ({preservation_rate:.1f}%)")
        
        if preservation_rate < 80:
            print(f"    ❌ Preservation rate is too low!")
            return False
        else:
            print(f"    ✅ Good sentiment word preservation!")
            
    except Exception as e:
        print(f"  ❌ Error testing sentiment preservation: {e}")
        return False
    
    print("\n🎉 All tests passed! Kaggle config provides equivalent functionality.")
    return True


if __name__ == "__main__":
    success = test_stopwords_parity()
    sys.exit(0 if success else 1)
