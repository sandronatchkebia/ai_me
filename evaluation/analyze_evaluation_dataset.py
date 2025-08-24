#!/usr/bin/env python3
"""
AI Me - Evaluation Dataset Analyzer

This script performs comprehensive analysis of the evaluation dataset:
1. Authorship Verification (AUC)
2. Burrow's Delta
3. Character n-gram Cosine Similarity
4. Lexical Richness Analysis
5. Syntactic Complexity
6. Advanced Stylometric Features

Run this after generating the dataset with generate_evaluation_dataset.py
"""

import json
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.util import ngrams
import re
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# Download NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

print("🔍 Starting Comprehensive Evaluation Analysis...")
print("="*60)

class AdvancedStyleAnalyzer:
    """Advanced style analysis with multiple metrics."""
    
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.feature_names = []
        
    def extract_basic_features(self, text: str) -> Dict:
        """Extract basic text features."""
        words = word_tokenize(text.lower())
        sentences = sent_tokenize(text)
        
        # Remove punctuation and get word count
        words = [word for word in words if word.isalnum()]
        
        # Basic statistics
        word_count = len(words)
        sentence_count = len(sentences)
        avg_sentence_length = word_count / sentence_count if sentence_count > 0 else 0
        avg_word_length = np.mean([len(word) for word in words]) if words else 0
        
        # Contraction usage
        contractions = ['don\'t', 'can\'t', 'won\'t', 'i\'m', 'you\'re', 'we\'re', 'they\'re', 'it\'s', 'that\'s']
        contraction_count = sum(1 for word in words if word in contractions)
        contraction_ratio = contraction_count / word_count if word_count > 0 else 0
        
        # Punctuation usage
        exclamation_count = text.count('!')
        exclamation_ratio = exclamation_count / sentence_count if sentence_count > 0 else 0
        question_count = text.count('?')
        question_ratio = question_count / sentence_count if sentence_count > 0 else 0
        
        # Informal indicators
        informal_words = ['hey', 'hi', 'thanks', 'cool', 'awesome', 'great', 'nice', 'good', 'bad', 'stuff', 'thing']
        informal_count = sum(1 for word in words if word in informal_words)
        informal_ratio = informal_count / word_count if word_count > 0 else 0
        
        return {
            'word_count': word_count,
            'sentence_count': sentence_count,
            'avg_sentence_length': avg_sentence_length,
            'avg_word_length': avg_word_length,
            'contraction_ratio': contraction_ratio,
            'exclamation_ratio': exclamation_ratio,
            'question_ratio': question_ratio,
            'informal_ratio': informal_ratio
        }
    
    def extract_lexical_features(self, text: str) -> Dict:
        """Extract lexical richness features."""
        words = word_tokenize(text.lower())
        words = [word for word in words if word.isalnum() and word not in self.stop_words]
        
        if not words:
            return {
                'type_token_ratio': 0,
                'hapax_legomena_ratio': 0,
                'dis_legomena_ratio': 0,
                'yule_k': 0,
                'simpson_d': 0,
                'brunet_w': 0,
                'honore_statistic': 0
            }
        
        word_freq = Counter(words)
        total_words = len(words)
        unique_words = len(word_freq)
        
        # Type-Token Ratio
        type_token_ratio = unique_words / total_words
        
        # Hapax Legomena (words appearing once)
        hapax_count = sum(1 for freq in word_freq.values() if freq == 1)
        hapax_ratio = hapax_count / total_words
        
        # Dis Legomena (words appearing twice)
        dis_count = sum(1 for freq in word_freq.values() if freq == 2)
        dis_ratio = dis_count / total_words
        
        # Yule's K
        yule_k = 10000 * sum(freq * freq for freq in word_freq.values()) / (total_words * total_words)
        
        # Simpson's D
        if total_words > 1:
            simpson_d = sum(freq * (freq - 1) for freq in word_freq.values()) / (total_words * (total_words - 1))
        else:
            simpson_d = 0
        
        # Brunet's W
        try:
            brunet_w = total_words ** (unique_words ** -0.172)
        except (OverflowError, ValueError):
            brunet_w = 0
        
        # Honoré's Statistic
        if hapax_count > 0 and hapax_count < unique_words:
            honore_stat = 100 * np.log(total_words) / (1 - hapax_count / unique_words)
        else:
            honore_stat = 0
        
        return {
            'type_token_ratio': type_token_ratio,
            'hapax_legomena_ratio': hapax_ratio,
            'dis_legomena_ratio': dis_ratio,
            'yule_k': yule_k,
            'simpson_d': simpson_d,
            'brunet_w': brunet_w,
            'honore_statistic': honore_stat
        }
    
    def extract_syntactic_features(self, text: str) -> Dict:
        """Extract syntactic complexity features."""
        sentences = sent_tokenize(text)
        words = word_tokenize(text.lower())
        
        # Function words (common words that don't carry much meaning)
        function_words = ['the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by']
        function_word_count = sum(1 for word in words if word in function_words)
        function_word_ratio = function_word_count / len(words) if words else 0
        
        # Preposition ratio
        prepositions = ['in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'from', 'up', 'about', 'into', 'through', 'during', 'before', 'after', 'above', 'below']
        prep_count = sum(1 for word in words if word in prepositions)
        prep_ratio = prep_count / len(words) if words else 0
        
        # Conjunction ratio
        conjunctions = ['and', 'or', 'but', 'nor', 'yet', 'so', 'because', 'although', 'since', 'unless', 'while', 'where', 'when', 'if', 'unless']
        conj_count = sum(1 for word in words if word in conjunctions)
        conj_ratio = conj_count / len(words) if words else 0
        
        # Average words per sentence
        avg_words_per_sentence = len(words) / len(sentences) if sentences else 0
        
        # Sentence complexity (sentences with multiple clauses)
        complex_sentences = sum(1 for sent in sentences if ',' in sent and len(sent.split()) > 15)
        complexity_ratio = complex_sentences / len(sentences) if sentences else 0
        
        return {
            'function_word_ratio': function_word_ratio,
            'preposition_ratio': prep_ratio,
            'conjunction_ratio': conj_ratio,
            'avg_words_per_sentence': avg_words_per_sentence,
            'complexity_ratio': complexity_ratio
        }
    
    def extract_character_ngram_features(self, text: str, n: int = 3) -> Dict:
        """Extract character n-gram features."""
        # Convert to lowercase and remove extra whitespace
        text = re.sub(r'\s+', ' ', text.lower().strip())
        
        # Generate character n-grams
        char_ngrams = list(ngrams(text, n))
        char_ngram_freq = Counter(char_ngrams)
        
        # Calculate features
        total_ngrams = len(char_ngrams)
        unique_ngrams = len(char_ngram_freq)
        
        if total_ngrams == 0:
            return {
                f'char_{n}gram_ttr': 0,
                f'char_{n}gram_entropy': 0,
                f'char_{n}gram_avg_freq': 0
            }
        
        # Type-Token Ratio for character n-grams
        ttr = unique_ngrams / total_ngrams
        
        # Entropy of character n-gram distribution
        probs = [freq / total_ngrams for freq in char_ngram_freq.values()]
        entropy = -sum(p * np.log2(p) for p in probs if p > 0)
        
        # Average frequency
        avg_freq = total_ngrams / unique_ngrams
        
        return {
            f'char_{n}gram_ttr': ttr,
            f'char_{n}gram_entropy': entropy,
            f'char_{n}gram_avg_freq': avg_freq
        }
    
    def extract_all_features(self, text: str) -> Dict:
        """Extract all available features."""
        features = {}
        
        # Basic features
        features.update(self.extract_basic_features(text))
        
        # Lexical features
        features.update(self.extract_lexical_features(text))
        
        # Syntactic features
        features.update(self.extract_syntactic_features(text))
        
        # Character n-gram features (3-gram and 4-gram)
        features.update(self.extract_character_ngram_features(text, 3))
        features.update(self.extract_character_ngram_features(text, 4))
        
        # Clean up any infinity or NaN values
        for key, value in features.items():
            if isinstance(value, (int, float)) and (np.isinf(value) or np.isnan(value)):
                features[key] = 0.0
        
        # Update feature_names for AUC calculation
        if not self.feature_names:
            self.feature_names = list(features.keys())
        
        return features

def calculate_cosine_similarity(text1: str, text2: str, analyzer: AdvancedStyleAnalyzer) -> float:
    """Calculate cosine similarity between two texts using TF-IDF."""
    try:
        vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words='english',
            ngram_range=(1, 2),
            max_features=1000
        )
        
        # Fit and transform both texts
        tfidf_matrix = vectorizer.fit_transform([text1, text2])
        
        # Calculate cosine similarity
        similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        
        return float(similarity)
    except Exception as e:
        print(f"Warning: Cosine similarity calculation failed: {e}")
        return 0.0

def calculate_burrows_delta(text1: str, text2: str, analyzer: AdvancedStyleAnalyzer) -> float:
    """Calculate Burrow's Delta between two texts."""
    try:
        # Extract features for both texts
        features1 = analyzer.extract_all_features(text1)
        features2 = analyzer.extract_all_features(text2)
        
        # Convert to feature vectors
        feature_names = list(features1.keys())
        vector1 = np.array([features1[name] for name in feature_names])
        vector2 = np.array([features2[name] for name in feature_names])
        
        # Calculate Burrow's Delta
        # Delta = sum of absolute differences in z-scores
        delta = np.sum(np.abs(vector1 - vector2))
        
        return float(delta)
    except Exception as e:
        print(f"Warning: Burrow's Delta calculation failed: {e}")
        return 0.0

def calculate_authorship_verification_auc(original_texts: List[str], 
                                        generated_texts: List[str], 
                                        analyzer: AdvancedStyleAnalyzer) -> Dict:
    """Calculate Authorship Verification AUC score."""
    try:
        # Extract features for all texts
        all_features = []
        labels = []
        
        # Original texts (positive class)
        for text in original_texts:
            features = analyzer.extract_all_features(text)
            all_features.append(list(features.values()))
            labels.append(1)
        
        # Generated texts (negative class)
        for text in generated_texts:
            features = analyzer.extract_all_features(text)
            all_features.append(list(features.values()))
            labels.append(0)
        
        # Convert to numpy arrays
        X = np.array(all_features)
        y = np.array(labels)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Check for infinite or NaN values
        if np.any(np.isinf(X_train_scaled)) or np.any(np.isnan(X_train_scaled)):
            print(f"Warning: Infinite or NaN values detected in training features")
            return {
                'auc_score': 0.0,
                'fpr': [],
                'tpr': [],
                'feature_importance': {}
            }
        
        # Train classifier
        clf = LogisticRegression(random_state=42, max_iter=1000)
        clf.fit(X_train_scaled, y_train)
        
        # Get prediction probabilities
        y_pred_proba = clf.predict_proba(X_test_scaled)[:, 1]
        
        # Calculate AUC
        auc_score = roc_auc_score(y_test, y_pred_proba)
        
        # Calculate ROC curve
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        
        return {
            'auc_score': float(auc_score),
            'fpr': fpr.tolist(),
            'tpr': tpr.tolist(),
            'feature_importance': dict(zip(analyzer.feature_names, clf.coef_[0].tolist()))
        }
    except Exception as e:
        print(f"Warning: Authorship verification AUC calculation failed: {e}")
        return {
            'auc_score': 0.0,
            'fpr': [],
            'tpr': [],
            'feature_importance': {}
        }

def analyze_dataset(dataset: List[Dict]) -> Dict:
    """Perform comprehensive analysis of the evaluation dataset."""
    print("🔍 Performing comprehensive style analysis...")
    
    analyzer = AdvancedStyleAnalyzer()
    results = {
        'sample_analysis': [],
        'overall_metrics': {},
        'similarity_scores': {},
        'authorship_verification': {}
    }
    
    # Analyze each sample
    for item in dataset:
        if item['gpt4_response'].startswith('ERROR') or item['lora_response'].startswith('ERROR'):
            continue
            
        # Extract features for all three versions
        original_features = analyzer.extract_all_features(item['original_response'])
        gpt4_features = analyzer.extract_all_features(item['gpt4_response'])
        lora_features = analyzer.extract_all_features(item['lora_response'])
        
        # Calculate similarities
        gpt4_vs_original_cosine = calculate_cosine_similarity(
            item['original_response'], item['gpt4_response'], analyzer
        )
        lora_vs_original_cosine = calculate_cosine_similarity(
            item['original_response'], item['lora_response'], analyzer
        )
        
        gpt4_vs_original_delta = calculate_burrows_delta(
            item['original_response'], item['gpt4_response'], analyzer
        )
        lora_vs_original_delta = calculate_burrows_delta(
            item['original_response'], item['lora_response'], analyzer
        )
        
        # Store sample analysis
        sample_analysis = {
            'sample_id': item['sample_id'],
            'original_features': original_features,
            'gpt4_features': gpt4_features,
            'lora_features': lora_features,
            'gpt4_vs_original_cosine': gpt4_vs_original_cosine,
            'lora_vs_original_cosine': lora_vs_original_cosine,
            'gpt4_vs_original_delta': gpt4_vs_original_delta,
            'lora_vs_original_delta': lora_vs_original_delta
        }
        
        results['sample_analysis'].append(sample_analysis)
    
    # Calculate overall metrics
    if results['sample_analysis']:
        cosine_scores = {
            'gpt4_vs_original': [s['gpt4_vs_original_cosine'] for s in results['sample_analysis']],
            'lora_vs_original': [s['lora_vs_original_cosine'] for s in results['sample_analysis']]
        }
        
        delta_scores = {
            'gpt4_vs_original': [s['gpt4_vs_original_delta'] for s in results['sample_analysis']],
            'lora_vs_original': [s['lora_vs_original_delta'] for s in results['sample_analysis']]
        }
        
        results['similarity_scores'] = {
            'cosine': {
                'gpt4_vs_original_mean': float(np.mean(cosine_scores['gpt4_vs_original'])),
                'gpt4_vs_original_std': float(np.std(cosine_scores['gpt4_vs_original'])),
                'lora_vs_original_mean': float(np.mean(cosine_scores['lora_vs_original'])),
                'lora_vs_original_std': float(np.std(cosine_scores['lora_vs_original']))
            },
            'delta': {
                'gpt4_vs_original_mean': float(np.mean(delta_scores['gpt4_vs_original'])),
                'gpt4_vs_original_std': float(np.std(delta_scores['gpt4_vs_original'])),
                'lora_vs_original_mean': float(np.mean(delta_scores['lora_vs_original'])),
                'lora_vs_original_std': float(np.std(delta_scores['lora_vs_original']))
            }
        }
        
        # Authorship verification
        original_texts = [item['original_response'] for item in dataset if not item['gpt4_response'].startswith('ERROR')]
        gpt4_texts = [item['gpt4_response'] for item in dataset if not item['gpt4_response'].startswith('ERROR')]
        lora_texts = [item['lora_response'] for item in dataset if not item['lora_response'].startswith('ERROR')]
        
        results['authorship_verification'] = {
            'gpt4_vs_original': calculate_authorship_verification_auc(original_texts, gpt4_texts, analyzer),
            'lora_vs_original': calculate_authorship_verification_auc(original_texts, lora_texts, analyzer)
        }
    
    return results

def create_visualizations(results: Dict, output_dir: str = "evaluation_results"):
    """Create comprehensive visualizations of the analysis results."""
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    plt.style.use('default')
    
    # 1. Cosine Similarity Comparison
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Comprehensive Style Analysis Results', fontsize=16, fontweight='bold')
    
    if results['similarity_scores']:
        # Cosine similarity comparison
        models = ['GPT-4 vs Original', 'LoRA vs Original']
        cosine_means = [
            results['similarity_scores']['cosine']['gpt4_vs_original_mean'],
            results['similarity_scores']['cosine']['lora_vs_original_mean']
        ]
        cosine_stds = [
            results['similarity_scores']['cosine']['gpt4_vs_original_std'],
            results['similarity_scores']['cosine']['lora_vs_original_std']
        ]
        
        bars = axes[0, 0].bar(models, cosine_means, yerr=cosine_stds, capsize=5, alpha=0.8, color=['#FF6B6B', '#4ECDC4'])
        axes[0, 0].set_ylabel('Cosine Similarity Score')
        axes[0, 0].set_title('Cosine Similarity Comparison')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Add value labels
        for bar, value in zip(bars, cosine_means):
            axes[0, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                           f'{value:.3f}', ha='center', va='bottom')
        
        # Burrow's Delta comparison
        delta_means = [
            results['similarity_scores']['delta']['gpt4_vs_original_mean'],
            results['similarity_scores']['delta']['lora_vs_original_mean']
        ]
        delta_stds = [
            results['similarity_scores']['delta']['gpt4_vs_original_std'],
            results['similarity_scores']['delta']['lora_vs_original_std']
        ]
        
        bars = axes[0, 1].bar(models, delta_means, yerr=delta_stds, capsize=5, alpha=0.8, color=['#FF6B6B', '#4ECDC4'])
        axes[0, 1].set_ylabel('Burrow\'s Delta Score')
        axes[0, 1].set_title('Burrow\'s Delta Comparison')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Add value labels
        for bar, value in zip(bars, delta_means):
            axes[0, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                           f'{value:.3f}', ha='center', va='bottom')
    
    # 2. Authorship Verification ROC Curves
    if results['authorship_verification']:
        # GPT-4 ROC curve
        if results['authorship_verification']['gpt4_vs_original']['fpr']:
            axes[1, 0].plot(
                results['authorship_verification']['gpt4_vs_original']['fpr'],
                results['authorship_verification']['gpt4_vs_original']['tpr'],
                label=f"GPT-4 (AUC = {results['authorship_verification']['gpt4_vs_original']['auc_score']:.3f})",
                color='#FF6B6B', linewidth=2
            )
        
        # LoRA ROC curve
        if results['authorship_verification']['lora_vs_original']['fpr']:
            axes[1, 0].plot(
                results['authorship_verification']['lora_vs_original']['fpr'],
                results['authorship_verification']['lora_vs_original']['tpr'],
                label=f"LoRA (AUC = {results['authorship_verification']['lora_vs_original']['auc_score']:.3f})",
                color='#4ECDC4', linewidth=2
            )
        
        # Diagonal line
        axes[1, 0].plot([0, 1], [0, 1], 'k--', alpha=0.5)
        axes[1, 0].set_xlabel('False Positive Rate')
        axes[1, 0].set_ylabel('True Positive Rate')
        axes[1, 0].set_title('Authorship Verification ROC Curves')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
    
    # 3. Improvement Analysis
    if results['similarity_scores']:
        cosine_improvement = ((results['similarity_scores']['cosine']['lora_vs_original_mean'] - 
                              results['similarity_scores']['cosine']['gpt4_vs_original_mean']) / 
                             results['similarity_scores']['cosine']['gpt4_vs_original_mean']) * 100
        
        delta_improvement = ((results['similarity_scores']['delta']['gpt4_vs_original_mean'] - 
                             results['similarity_scores']['delta']['lora_vs_original_mean']) / 
                            results['similarity_scores']['delta']['gpt4_vs_original_mean']) * 100
        
        improvements = ['Cosine Similarity', 'Burrow\'s Delta']
        improvement_values = [cosine_improvement, delta_improvement]
        colors = ['#4ECDC4' if x > 0 else '#FF6B6B' for x in improvement_values]
        
        bars = axes[1, 1].bar(improvements, improvement_values, color=colors, alpha=0.8)
        axes[1, 1].set_ylabel('Improvement (%)')
        axes[1, 1].set_title('LoRA vs GPT-4 Improvement')
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].axhline(y=0, color='black', linestyle='-', alpha=0.3)
        
        # Add value labels
        for bar, value in zip(bars, improvement_values):
            axes[1, 1].text(bar.get_x() + bar.get_width()/2, value + (1 if value > 0 else -1), 
                           f'{value:+.1f}%', ha='center', va='bottom' if value > 0 else 'top')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/comprehensive_analysis_results.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"✅ Visualizations saved to {output_dir}/")

def main():
    """Main analysis function."""
    print("🎯 AI Me - Evaluation Dataset Analyzer")
    print("="*60)
    
    # Load the evaluation dataset
    dataset_file = "evaluation_data/evaluation_dataset.json"
    try:
        with open(dataset_file, 'r', encoding='utf-8') as f:
            dataset = json.load(f)
        print(f"✅ Loaded evaluation dataset with {len(dataset)} samples")
    except FileNotFoundError:
        print(f"❌ Dataset file not found: {dataset_file}")
        print("Please run generate_evaluation_dataset.py first to generate the dataset.")
        return
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        return
    
    # Perform comprehensive analysis
    results = analyze_dataset(dataset)
    
    # Save analysis results
    output_dir = "evaluation_results"
    os.makedirs(output_dir, exist_ok=True)
    
    with open(f'{output_dir}/analysis_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"✅ Analysis results saved to {output_dir}/analysis_results.json")
    
    # Create visualizations
    create_visualizations(results, output_dir)
    
    # Display summary
    print(f"\n📊 ANALYSIS SUMMARY")
    print("="*60)
    
    if results['similarity_scores']:
        print(f"Cosine Similarity:")
        print(f"  GPT-4 vs Original: {results['similarity_scores']['cosine']['gpt4_vs_original_mean']:.3f} ± {results['similarity_scores']['cosine']['gpt4_vs_original_std']:.3f}")
        print(f"  LoRA vs Original:  {results['similarity_scores']['cosine']['lora_vs_original_mean']:.3f} ± {results['similarity_scores']['cosine']['lora_vs_original_std']:.3f}")
        
        cosine_improvement = ((results['similarity_scores']['cosine']['lora_vs_original_mean'] - 
                              results['similarity_scores']['cosine']['gpt4_vs_original_mean']) / 
                             results['similarity_scores']['cosine']['gpt4_vs_original_mean']) * 100
        print(f"  Improvement: {cosine_improvement:+.1f}%")
        
        print(f"\nBurrow's Delta:")
        print(f"  GPT-4 vs Original: {results['similarity_scores']['delta']['gpt4_vs_original_mean']:.3f} ± {results['similarity_scores']['delta']['gpt4_vs_original_std']:.3f}")
        print(f"  LoRA vs Original:  {results['similarity_scores']['delta']['lora_vs_original_mean']:.3f} ± {results['similarity_scores']['delta']['lora_vs_original_std']:.3f}")
        
        delta_improvement = ((results['similarity_scores']['delta']['gpt4_vs_original_mean'] - 
                             results['similarity_scores']['delta']['lora_vs_original_mean']) / 
                            results['similarity_scores']['delta']['gpt4_vs_original_mean']) * 100
        print(f"  Improvement: {delta_improvement:+.1f}%")
    
    if results['authorship_verification']:
        print(f"\nAuthorship Verification (AUC):")
        print(f"  GPT-4 vs Original: {results['authorship_verification']['gpt4_vs_original']['auc_score']:.3f}")
        print(f"  LoRA vs Original:  {results['authorship_verification']['lora_vs_original']['auc_score']:.3f}")
    
    print(f"\n🎉 Analysis complete! Check {output_dir}/ for detailed results and visualizations.")

if __name__ == "__main__":
    main()
