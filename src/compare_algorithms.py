"""
Script so sánh trực tiếp giữa thuật toán gốc và thuật toán đã tối ưu
"""

import pandas as pd
import numpy as np
import os
import time
import json
from io import StringIO
import sys


def compare_preprocessing():
    """
    So sánh preprocessing cũ vs mới
    """
    print("🔧 COMPARING PREPROCESSING METHODS")
    print("-" * 50)
    
    # Load sample data
    train_df = pd.read_csv("../data/amazon_reviews/train.csv")
    train_df.columns = ["label", "title", "text"]
    train_df = train_df.head(5000)  # Small sample for comparison
    train_df['sentiment'] = train_df['label'].apply(lambda x: 'Negative' if x == 1 else 'Positive')
    
    print(f"Testing on {len(train_df)} samples")
    
    # ===== ORIGINAL PREPROCESSING =====
    print("\n📊 Original Preprocessing:")
    
    from preprocessor import PreProcessor
    original_preprocessor = PreProcessor()
    
    start_time = time.time()
    
    # Suppress output
    old_stdout = sys.stdout
    sys.stdout = StringIO()
    
    try:
        original_processed = original_preprocessor.process_full_pipeline(train_df.copy(), 'text')
    finally:
        sys.stdout = old_stdout
    
    original_time = time.time() - start_time
    
    original_features = len(original_processed.columns)
    print(f"  ⏱️  Time: {original_time:.2f}s")
    print(f"  📈 Features: {original_features}")
    print(f"  📊 Final samples: {len(original_processed)}")
    
    # ===== ENHANCED PREPROCESSING =====
    print("\n🔧 Enhanced Preprocessing:")
    
    from enhanced_preprocessor import EnhancedPreProcessor
    enhanced_preprocessor = EnhancedPreProcessor()
    
    start_time = time.time()
    
    # Suppress output
    old_stdout = sys.stdout
    sys.stdout = StringIO()
    
    try:
        enhanced_processed = enhanced_preprocessor.enhanced_preprocessing_pipeline(
            train_df.copy(), 
            'text', 
            use_advanced_features=True
        )
    finally:
        sys.stdout = old_stdout
    
    enhanced_time = time.time() - start_time
    
    enhanced_features = len(enhanced_processed.columns)
    print(f"  ⏱️  Time: {enhanced_time:.2f}s")
    print(f"  📈 Features: {enhanced_features}")
    print(f"  📊 Final samples: {len(enhanced_processed)}")
    
    # ===== COMPARISON =====
    print(f"\n📋 Preprocessing Comparison:")
    print(f"  Feature increase: +{enhanced_features - original_features} features")
    print(f"  Time difference: {enhanced_time - original_time:+.2f}s")
    print(f"  Sample retention: Original: {len(original_processed)}, Enhanced: {len(enhanced_processed)}")
    
    return {
        'original': {
            'time': original_time,
            'features': original_features,
            'samples': len(original_processed),
            'data': original_processed
        },
        'enhanced': {
            'time': enhanced_time,
            'features': enhanced_features,
            'samples': len(enhanced_processed),
            'data': enhanced_processed
        }
    }


def compare_models(preprocessing_results):
    """
    So sánh model cũ vs mới
    """
    print(f"\n🤖 COMPARING MODEL PERFORMANCE")
    print("-" * 50)
    
    original_data = preprocessing_results['original']['data']
    enhanced_data = preprocessing_results['enhanced']['data']
    
    results = {}
    
    # ===== ORIGINAL RANDOM FOREST =====
    print(f"\n🌲 Original Random Forest:")
    
    from random_forest_classifier import RandomForestAnalyzer
    
    original_rf = RandomForestAnalyzer()
    
    start_time = time.time()
    
    # Suppress training output
    old_stdout = sys.stdout
    sys.stdout = StringIO()
    
    try:
        original_rf.prepare_data(original_data, 'cleaned_text', test_size=0.2)
        original_rf_results = original_rf.train_model()
    finally:
        sys.stdout = old_stdout
    
    original_rf_time = time.time() - start_time
    
    print(f"  ⏱️  Training time: {original_rf_time:.2f}s")
    print(f"  🎯 Test accuracy: {original_rf_results['test_accuracy']:.4f}")
    print(f"  📈 Overfitting gap: {original_rf_results['train_accuracy'] - original_rf_results['test_accuracy']:.4f}")
    
    results['original_rf'] = {
        'time': original_rf_time,
        'test_accuracy': original_rf_results['test_accuracy'],
        'train_accuracy': original_rf_results['train_accuracy'],
        'overfitting_gap': original_rf_results['train_accuracy'] - original_rf_results['test_accuracy']
    }
    
    # ===== OPTIMIZED RANDOM FOREST =====
    print(f"\n🌲✨ Optimized Random Forest:")
    
    from optimized_random_forest import OptimizedRandomForestAnalyzer
    
    opt_rf = OptimizedRandomForestAnalyzer(optimize_hyperparameters=False)  # Skip HP optimization for speed
    
    start_time = time.time()
    
    # Suppress training output
    old_stdout = sys.stdout
    sys.stdout = StringIO()
    
    try:
        opt_rf.prepare_data(enhanced_data, 'processed_text', test_size=0.2)
        opt_rf_results = opt_rf.train_model()
    finally:
        sys.stdout = old_stdout
    
    opt_rf_time = time.time() - start_time
    
    print(f"  ⏱️  Training time: {opt_rf_time:.2f}s")
    print(f"  🎯 Test accuracy: {opt_rf_results['test_accuracy']:.4f}")
    print(f"  📈 F1-macro: {opt_rf_results['test_f1_macro']:.4f}")
    print(f"  📊 Overfitting gap: {opt_rf_results['overfitting_gap']:.4f}")
    print(f"  🔍 Features used: {opt_rf_results['n_features']}")
    
    results['optimized_rf'] = {
        'time': opt_rf_time,
        'test_accuracy': opt_rf_results['test_accuracy'],
        'train_accuracy': opt_rf_results['train_accuracy'],
        'f1_macro': opt_rf_results['test_f1_macro'],
        'overfitting_gap': opt_rf_results['overfitting_gap'],
        'n_features': opt_rf_results['n_features']
    }
    
    # ===== OPTIMIZED LIGHTGBM (NEW) =====
    print(f"\n⚡ Optimized LightGBM (New):")
    
    from optimized_lgbm import OptimizedLGBMAnalyzer
    
    opt_lgbm = OptimizedLGBMAnalyzer(optimize_hyperparameters=False)
    
    start_time = time.time()
    
    # Suppress training output
    old_stdout = sys.stdout
    sys.stdout = StringIO()
    
    try:
        opt_lgbm.prepare_data(enhanced_data, 'processed_text', test_size=0.2)
        opt_lgbm_results = opt_lgbm.train_with_early_stopping(verbose_eval=False)
    finally:
        sys.stdout = old_stdout
    
    opt_lgbm_time = time.time() - start_time
    
    print(f"  ⏱️  Training time: {opt_lgbm_time:.2f}s")
    print(f"  🎯 Test accuracy: {opt_lgbm_results['test_accuracy']:.4f}")
    print(f"  📈 F1-macro: {opt_lgbm_results['test_f1_macro']:.4f}")
    print(f"  📊 Overfitting gap: {opt_lgbm_results['overfitting_gap']:.4f}")
    print(f"  🔍 Features used: {opt_lgbm_results['n_features']}")
    print(f"  🏆 Best iteration: {opt_lgbm_results['best_iteration']}")
    
    results['optimized_lgbm'] = {
        'time': opt_lgbm_time,
        'test_accuracy': opt_lgbm_results['test_accuracy'],
        'train_accuracy': opt_lgbm_results['train_accuracy'],
        'f1_macro': opt_lgbm_results['test_f1_macro'],
        'overfitting_gap': opt_lgbm_results['overfitting_gap'],
        'n_features': opt_lgbm_results['n_features'],
        'best_iteration': opt_lgbm_results['best_iteration']
    }
    
    return results, original_rf, opt_rf, opt_lgbm


def compare_predictions(original_rf, opt_rf, opt_lgbm):
    """
    So sánh predictions trên cùng test cases
    """
    print(f"\n🔮 COMPARING PREDICTIONS")
    print("-" * 50)
    
    test_cases = [
        "This product is absolutely fantastic! I love everything about it.",
        "Terrible quality, waste of money. Very disappointed.",
        "It's okay, not great but not terrible either.",
        "Amazing quality for the price! Highly recommended.",
        "Poor design and cheap materials. Don't buy this."
    ]
    
    print("Test Cases:")
    for i, case in enumerate(test_cases, 1):
        print(f"{i}. '{case[:60]}{'...' if len(case) > 60 else ''}'")
    
    print(f"\nPrediction Comparison:")
    print(f"{'Case':<4} {'Original RF':<12} {'Optimized RF':<15} {'Optimized LGBM':<15}")
    print("-" * 60)
    
    comparison_results = []
    
    for i, case in enumerate(test_cases, 1):
        try:
            # Original RF prediction
            orig_pred = original_rf.predict([case])[0]
            orig_prob = np.max(original_rf.predict_proba([case])[0])
            
            # Optimized RF prediction
            opt_rf_pred_result = opt_rf.predict_with_confidence([case])[0]
            opt_rf_pred = opt_rf_pred_result['prediction']
            opt_rf_prob = opt_rf_pred_result['confidence']
            
            # Optimized LGBM prediction
            opt_lgbm_pred_result = opt_lgbm.predict_with_probability([case])[0]
            opt_lgbm_pred = opt_lgbm_pred_result['prediction']
            opt_lgbm_prob = opt_lgbm_pred_result['confidence']
            
            print(f"{i:<4} {orig_pred:<12} {opt_rf_pred:<15} {opt_lgbm_pred:<15}")
            
            comparison_results.append({
                'case': i,
                'text': case,
                'original_rf': {'prediction': orig_pred, 'confidence': orig_prob},
                'optimized_rf': {'prediction': opt_rf_pred, 'confidence': opt_rf_prob},
                'optimized_lgbm': {'prediction': opt_lgbm_pred, 'confidence': opt_lgbm_prob}
            })
            
        except Exception as e:
            print(f"{i:<4} ERROR: {str(e)[:40]}")
    
    return comparison_results


def generate_improvement_summary(preprocessing_comparison, model_comparison):
    """
    Tạo tóm tắt về những cải thiện đạt được
    """
    print(f"\n📊 IMPROVEMENT SUMMARY")
    print("=" * 70)
    
    # Preprocessing improvements
    orig_prep = preprocessing_comparison['original']
    enh_prep = preprocessing_comparison['enhanced']
    
    feature_improvement = enh_prep['features'] - orig_prep['features']
    
    print(f"\n🔧 Preprocessing Improvements:")
    print(f"  ✨ Features added: +{feature_improvement}")
    print(f"  📈 Feature engineering techniques:")
    print(f"     • Advanced text cleaning (contractions, negations)")
    print(f"     • Lemmatization with POS tagging")
    print(f"     • Smart stopword removal")
    print(f"     • Numerical feature extraction")
    print(f"     • Enhanced sentiment scoring")
    
    # Model performance improvements
    orig_rf = model_comparison['original_rf']
    opt_rf = model_comparison['optimized_rf']
    opt_lgbm = model_comparison['optimized_lgbm']
    
    print(f"\n🤖 Model Performance Improvements:")
    
    # Random Forest comparison
    rf_acc_improvement = opt_rf['test_accuracy'] - orig_rf['test_accuracy']
    rf_overfit_improvement = orig_rf['overfitting_gap'] - opt_rf['overfitting_gap']
    
    print(f"\n  🌲 Random Forest Optimizations:")
    print(f"     • Accuracy improvement: {rf_acc_improvement:+.4f}")
    print(f"     • Overfitting reduction: {rf_overfit_improvement:+.4f}")
    print(f"     • Enhanced TF-IDF: 10k → 50k features, n-grams(1,3)")
    print(f"     • Class balancing & feature selection")
    print(f"     • Advanced hyperparameters")
    
    # LightGBM advantages
    lgbm_vs_orig_rf = opt_lgbm['test_accuracy'] - orig_rf['test_accuracy']
    
    print(f"\n  ⚡ LightGBM New Algorithm:")
    print(f"     • Accuracy vs Original RF: {lgbm_vs_orig_rf:+.4f}")
    print(f"     • Early stopping prevents overfitting")
    print(f"     • Gradient boosting with optimized params")
    print(f"     • Faster training with better memory usage")
    print(f"     • Built-in feature importance analysis")
    
    # Best improvements
    best_accuracy = max(orig_rf['test_accuracy'], opt_rf['test_accuracy'], opt_lgbm['test_accuracy'])
    best_model = 'Original RF' if best_accuracy == orig_rf['test_accuracy'] else \
                 'Optimized RF' if best_accuracy == opt_rf['test_accuracy'] else 'Optimized LGBM'
    
    improvement_vs_original = best_accuracy - orig_rf['test_accuracy']
    
    print(f"\n🏆 OVERALL BEST RESULTS:")
    print(f"  🥇 Best Model: {best_model}")
    print(f"  📈 Best Accuracy: {best_accuracy:.4f}")
    print(f"  🚀 Total Improvement: {improvement_vs_original:+.4f}")
    
    # Technical improvements
    print(f"\n🔧 Technical Optimizations:")
    print(f"  • Enhanced preprocessing pipeline")
    print(f"  • Hyperparameter optimization support")
    print(f"  • Cross-validation integration")
    print(f"  • Feature importance analysis")
    print(f"  • Memory-optimized for MacBook Air 8GB")
    print(f"  • Early stopping & overfitting prevention")
    print(f"  • Comprehensive evaluation metrics")
    
    return {
        'preprocessing_improvement': {
            'features_added': feature_improvement,
            'techniques': ['advanced_cleaning', 'lemmatization', 'feature_engineering']
        },
        'model_improvements': {
            'random_forest': {
                'accuracy_gain': rf_acc_improvement,
                'overfitting_reduction': rf_overfit_improvement
            },
            'lightgbm_new': {
                'accuracy_vs_original': lgbm_vs_orig_rf,
                'features': ['early_stopping', 'gradient_boosting', 'feature_importance']
            },
            'best_model': best_model,
            'best_accuracy': best_accuracy,
            'total_improvement': improvement_vs_original
        }
    }


def main():
    print("🔬 THUẬT TOÁN CŨ vs MỚI - PHÂN TÍCH SO SÁNH")
    print("=" * 80)
    
    # So sánh preprocessing
    preprocessing_comparison = compare_preprocessing()
    
    # So sánh models
    model_comparison, original_rf, opt_rf, opt_lgbm = compare_models(preprocessing_comparison)
    
    # So sánh predictions
    prediction_comparison = compare_predictions(original_rf, opt_rf, opt_lgbm)
    
    # Tạo summary
    improvement_summary = generate_improvement_summary(preprocessing_comparison, model_comparison)
    
    # Save comprehensive comparison
    comparison_results = {
        'preprocessing_comparison': {
            'original': {
                'time': preprocessing_comparison['original']['time'],
                'features': preprocessing_comparison['original']['features'],
                'samples': preprocessing_comparison['original']['samples']
            },
            'enhanced': {
                'time': preprocessing_comparison['enhanced']['time'],
                'features': preprocessing_comparison['enhanced']['features'],
                'samples': preprocessing_comparison['enhanced']['samples']
            }
        },
        'model_comparison': model_comparison,
        'prediction_comparison': prediction_comparison,
        'improvement_summary': improvement_summary,
        'recommendations': {
            'best_overall_approach': improvement_summary['model_improvements']['best_model'],
            'key_improvements': [
                'Enhanced preprocessing with advanced features',
                'Optimized hyperparameters for better performance',
                'LightGBM with early stopping for efficiency',
                'Cross-validation for model reliability',
                'Memory optimization for MacBook Air 8GB'
            ],
            'next_steps': [
                'Consider ensemble methods combining best models',
                'Implement real-time prediction API',
                'Add more advanced features (word embeddings)',
                'Explore deep learning approaches (BERT)',
                'Create automated model retraining pipeline'
            ]
        }
    }
    
    # Save results
    os.makedirs('models/comparison', exist_ok=True)
    with open('models/comparison/algorithm_comparison.json', 'w', encoding='utf-8') as f:
        json.dump(comparison_results, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 Comparison results saved to: models/comparison/algorithm_comparison.json")
    print(f"\n" + "="*80)
    print(f"✅ COMPARISON COMPLETE!")
    print(f"🏆 Recommended: {improvement_summary['model_improvements']['best_model']}")
    print(f"📈 Total Improvement: {improvement_summary['model_improvements']['total_improvement']:+.4f}")
    print(f"="*80)


if __name__ == "__main__":
    main()
