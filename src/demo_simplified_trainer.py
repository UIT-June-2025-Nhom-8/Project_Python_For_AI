"""
Demo script showing how to use the new simplified methods in ModelTrainer
This demonstrates the integration of simplified classifiers into the existing ModelTrainer class
"""

import pandas as pd
import sys
import os

# Add src directory to path if needed
sys.path.append('/Users/ducqhle/Documents/UIT_workspace/Project_Python_For_AI/src')

from model_trainer import ModelTrainer


def create_sample_data():
    """
    Create sample Amazon reviews data for demonstration
    """
    sample_data = pd.DataFrame({
        'text': [
            'I love this product, it is amazing and works perfectly!',
            'This is terrible, worst purchase ever, complete waste of money',
            'Pretty good quality, would recommend to others',
            'Not bad but could be better, average product',
            'Excellent quality and fast delivery, very satisfied',
            'Outstanding service and fantastic product quality',
            'Completely disappointed with this purchase, very poor quality',
            'Good value for money, meets my expectations',
            'Would not recommend this to anyone, terrible experience',
            'Perfect product, exactly what I needed and more',
            'Great customer service and quick shipping',
            'Poor quality materials, broke after one week',
            'Highly recommend this product to everyone',
            'Waste of time and money, very disappointed',
            'Amazing product with excellent features and design'
        ] * 50,  # Repeat to have more samples for better training
        'sentiment': [
            'positive', 'negative', 'positive', 'negative', 'positive',
            'positive', 'negative', 'positive', 'negative', 'positive',
            'positive', 'negative', 'positive', 'negative', 'positive'
        ] * 50
    })
    
    return sample_data


def demo_simplified_methods():
    """
    Demonstrate the new simplified methods in ModelTrainer
    """
    print("="*80)
    print("DEMO: Simplified Classifiers in ModelTrainer")
    print("="*80)
    
    # Create sample data
    print("1. Creating sample data...")
    df = create_sample_data()
    print(f"   Created dataset with {len(df)} samples")
    print(f"   Label distribution: {df['sentiment'].value_counts().to_dict()}")
    
    # Initialize ModelTrainer
    print("\n2. Initializing ModelTrainer...")
    trainer = ModelTrainer(output_dir="reports")
    print("   ✅ ModelTrainer initialized")
    
    # Method 1: Use simplified classifiers with the new method
    print("\n3. Training models using SIMPLIFIED implementation...")
    simplified_results = trainer.run_training_pipeline_with_simplified_classifiers(
        df=df,
        text_column='text',
        target_column='sentiment',
        test_size=0.2,
        optimize_hyperparameters=False,  # Set to True for better performance (slower)
        save_results=True,
        max_features=5000  # Smaller for demo
    )
    
    # Method 2: For comparison, you could also run the original complex implementation
    # (This requires the data to be preprocessed first with TF-IDF)
    
    # Show how to access results
    print("\n4. Accessing results from simplified classifiers...")
    if 'individual_results' in simplified_results:
        for model_name, results in simplified_results['individual_results'].items():
            if 'error' not in results:
                print(f"   {model_name}:")
                print(f"      Accuracy: {results.get('test_accuracy', 0):.4f}")
                print(f"      F1-Score: {results.get('test_f1', 0):.4f}")
                print(f"      Training Time: {results.get('training_time_seconds', 0):.2f}s")
    
    # Show how to make predictions (if you saved the models)
    print("\n5. Making predictions with simplified classifiers...")
    try:
        # You would need to load a saved model or access the trained classifier
        # This is just to show the concept
        test_texts = [
            "This product is absolutely amazing!",
            "Terrible quality, very disappointed",
            "Good value for the price"
        ]
        
        print("   Sample texts for prediction:")
        for i, text in enumerate(test_texts, 1):
            print(f"      {i}. '{text}'")
        
        # Note: To actually make predictions, you'd need to access the trained models
        # from the simplified classifiers or save/load them
        print("   💡 To make predictions, save and load the trained models")
        
    except Exception as e:
        print(f"   Note: Prediction demo skipped - {e}")
    
    print("\n6. Benefits of using simplified methods:")
    print("   ✅ Cleaner, more readable code")
    print("   ✅ Standardized interface across all classifiers") 
    print("   ✅ Built-in data preprocessing")
    print("   ✅ Automatic feature extraction")
    print("   ✅ Easy hyperparameter optimization")
    print("   ✅ Consistent result formatting")
    
    return simplified_results


def demo_comparison():
    """
    Show how to compare different implementations
    """
    print("\n" + "="*80)
    print("DEMO: Comparing Implementation Approaches")
    print("="*80)
    
    df = create_sample_data()
    trainer = ModelTrainer(output_dir="reports")
    
    print("This demo shows how you can compare:")
    print("1. Original complex implementations (run_training_pipeline_with_implemented_classifiers)")
    print("2. Simplified implementations (run_training_pipeline_with_simplified_classifiers)")
    print("3. Direct sklearn implementations (run_training_pipeline_with_tfidf)")
    
    # For a real comparison, you would run both implementations:
    # original_results = trainer.run_training_pipeline_with_implemented_classifiers(...)
    # simplified_results = trainer.run_training_pipeline_with_simplified_classifiers(...)
    # trainer.compare_implementations(original_results, simplified_results)
    
    print("\n💡 Usage recommendation:")
    print("   • Use simplified_classifiers for new projects (cleaner code)")
    print("   • Use implemented_classifiers if you need the existing complex features")
    print("   • Use direct sklearn if you just need basic functionality")


def main():
    """
    Main demo function
    """
    try:
        # Demo the simplified methods
        results = demo_simplified_methods()
        
        # Demo comparison concepts
        demo_comparison()
        
        print("\n" + "="*80)
        print("DEMO COMPLETED SUCCESSFULLY!")
        print("="*80)
        print("Check the 'reports/' directory for saved results")
        print("\nNext steps:")
        print("1. Try with your own data by replacing create_sample_data()")
        print("2. Set optimize_hyperparameters=True for better performance")
        print("3. Experiment with different max_features values")
        print("4. Compare results with the original implementation")
        print("="*80)
        
        return results
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    main()
