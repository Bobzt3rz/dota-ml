#!/usr/bin/env python3
"""
Fixed Hero Baseline - Use CSV instead of Parquet to avoid dependency issues
"""

import json
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
import os
import joblib

def load_hero_constants():
    """Load hero names for better interpretability"""
    try:
        constants_path = os.path.join(os.path.dirname(__file__), '..', 'libs', 'dotaconstants', 'build')
        
        with open(os.path.join(constants_path, 'heroes.json'), 'r') as f:
            heroes_data = json.load(f)
        
        id_to_name = {}
        for hero_id, hero_info in heroes_data.items():
            hero_id = int(hero_id)
            hero_name = hero_info.get('localized_name', f'Hero {hero_id}')
            id_to_name[hero_id] = hero_name
            
        print(f"✅ Loaded {len(id_to_name)} hero names")
        return id_to_name
        
    except Exception as e:
        print(f"⚠️  Could not load hero names: {e}")
        return {}

def create_hero_onehot_features(df, id_to_name=None):
    """
    Create simple one-hot encoding for hero picks
    Each hero gets 2 features: radiant_has_X and dire_has_X
    """
    print("🎯 Creating hero one-hot features...")
    
    # Get all unique heroes that actually appear in the dataset
    all_heroes = set()
    valid_matches = 0
    
    for _, row in df.iterrows():
        if isinstance(row['radiant_team'], list) and isinstance(row['dire_team'], list):
            if len(row['radiant_team']) == 5 and len(row['dire_team']) == 5:
                all_heroes.update(row['radiant_team'])
                all_heroes.update(row['dire_team'])
                valid_matches += 1
    
    all_heroes = sorted(list(all_heroes))
    print(f"Found {len(all_heroes)} unique heroes in {valid_matches} valid matches")
    
    # Create feature matrix
    feature_data = {}
    
    print("Creating binary features for each hero...")
    for hero_id in all_heroes:
        hero_name = id_to_name.get(hero_id, f'Hero_{hero_id}') if id_to_name else f'Hero_{hero_id}'
        
        # Clean hero name for column names (remove spaces, special chars)
        clean_name = hero_name.replace(' ', '_').replace("'", "").replace('-', '_')
        
        # Radiant has this hero (1 if yes, 0 if no)
        feature_data[f'radiant_has_{clean_name}'] = df['radiant_team'].apply(
            lambda team: 1 if isinstance(team, list) and hero_id in team else 0
        )
        
        # Dire has this hero (1 if yes, 0 if no)
        feature_data[f'dire_has_{clean_name}'] = df['dire_team'].apply(
            lambda team: 1 if isinstance(team, list) and hero_id in team else 0
        )
    
    feature_df = pd.DataFrame(feature_data)
    
    print(f"Created {feature_df.shape[1]} hero features")
    print(f"Feature matrix shape: {feature_df.shape}")
    
    # Verify feature integrity
    print("\n🔍 Feature verification:")
    radiant_totals = feature_df[[c for c in feature_df.columns if 'radiant_has_' in c]].sum(axis=1)
    dire_totals = feature_df[[c for c in feature_df.columns if 'dire_has_' in c]].sum(axis=1)
    
    print(f"Radiant team sizes: min={radiant_totals.min()}, max={radiant_totals.max()}, mean={radiant_totals.mean():.1f}")
    print(f"Dire team sizes: min={dire_totals.min()}, max={dire_totals.max()}, mean={dire_totals.mean():.1f}")
    
    # Show feature distribution
    print(f"\nTop 10 most frequent heroes:")
    hero_frequencies = {}
    for col in feature_df.columns:
        if 'radiant_has_' in col or 'dire_has_' in col:
            hero_name = col.split('_has_')[1]
            if hero_name not in hero_frequencies:
                hero_frequencies[hero_name] = 0
            hero_frequencies[hero_name] += feature_df[col].sum()
    
    for i, (hero, freq) in enumerate(sorted(hero_frequencies.items(), key=lambda x: x[1], reverse=True)[:10], 1):
        print(f"  {i:2d}. {hero:<20} {freq:>6,} picks ({freq/len(df)/10*100:.2f}% pick rate)")
    
    return feature_df

def create_target_variable(df):
    """Create target variable: 1 = Radiant wins, 0 = Dire wins"""
    return df['radiant_win'].astype(int)

def train_baseline_models(X_train, X_val, y_train, y_val):
    """Train simple baseline models"""
    print("\n🤖 Training baseline models...")
    
    models = {}
    results = {}
    
    # Model 1: Logistic Regression
    print("Training Logistic Regression...")
    lr = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced', tol=1e-6)
    lr.fit(X_train, y_train)
    
    lr_pred = lr.predict(X_val)
    lr_prob = lr.predict_proba(X_val)[:, 1]
    
    models['logistic_regression'] = lr
    results['logistic_regression'] = {
        'accuracy': accuracy_score(y_val, lr_pred),
        'auc': roc_auc_score(y_val, lr_prob)
    }
    
    # Model 2: Random Forest (smaller version for speed)
    print("Training Random Forest...")
    rf = RandomForestClassifier(
        n_estimators=100, 
        max_depth=10, 
        random_state=42, 
        class_weight='balanced',
        n_jobs=-1
    )
    rf.fit(X_train, y_train)
    
    rf_pred = rf.predict(X_val)
    rf_prob = rf.predict_proba(X_val)[:, 1]
    
    models['random_forest'] = rf
    results['random_forest'] = {
        'accuracy': accuracy_score(y_val, rf_pred),
        'auc': roc_auc_score(y_val, rf_prob)
    }
    
    return models, results

def analyze_feature_importance(model, feature_names, model_name, top_n=20):
    """Analyze which heroes are most important for predictions"""
    print(f"\n📊 Feature Importance Analysis - {model_name}")
    print("="*60)
    
    if hasattr(model, 'coef_'):
        # Logistic regression coefficients
        importances = abs(model.coef_[0])
    elif hasattr(model, 'feature_importances_'):
        # Tree-based feature importances
        importances = model.feature_importances_
    else:
        print("Model doesn't support feature importance analysis")
        return
    
    # Get top features
    feature_importance = list(zip(feature_names, importances))
    feature_importance.sort(key=lambda x: x[1], reverse=True)
    
    print(f"Top {top_n} most important features:")
    for i, (feature, importance) in enumerate(feature_importance[:top_n], 1):
        team = "Radiant" if "radiant_has_" in feature else "Dire"
        hero = feature.split("_has_")[1].replace('_', ' ')
        print(f"  {i:2d}. {team:<8} has {hero:<20} {importance:.4f}")
    
    # Separate radiant vs dire importance
    radiant_features = [(f, imp) for f, imp in feature_importance if 'radiant_has_' in f]
    dire_features = [(f, imp) for f, imp in feature_importance if 'dire_has_' in f]
    
    print(f"\nTop 10 Radiant heroes (positive impact):")
    for i, (feature, importance) in enumerate(radiant_features[:10], 1):
        hero = feature.split("_has_")[1].replace('_', ' ')
        print(f"  {i:2d}. {hero:<20} {importance:.4f}")
    
    print(f"\nTop 10 Dire heroes (negative impact on Radiant):")
    for i, (feature, importance) in enumerate(dire_features[:10], 1):
        hero = feature.split("_has_")[1].replace('_', ' ')
        print(f"  {i:2d}. {hero:<20} {importance:.4f}")

def quick_model_diagnostics(models):
    """
    Quick diagnostics of your trained models
    """
    print(f"\n🔧 Quick Model Diagnostics:")
    print("="*40)
    
    for model_name, model in models.items():
        print(f"\n{model_name.title()}:")
        
        if hasattr(model, 'n_iter_'):
            print(f"  Iterations: {model.n_iter_[0]}/{model.max_iter}")
            if model.n_iter_[0] < model.max_iter:
                print(f"  Status: ✅ Converged")
            else:
                print(f"  Status: ❌ Hit max_iter limit")
        
        if hasattr(model, 'coef_'):
            weights = model.coef_[0]
            print(f"  Weight stats: min={weights.min():.3f}, max={weights.max():.3f}")
            print(f"  Large weights: {(abs(weights) > 0.1).sum()}/{len(weights)}")
        
        print(f"  Model type: {type(model).__name__}")

def save_prediction_insights(models, X_test, y_test, feature_names, output_dir):
    """Save some prediction insights for analysis"""
    print("\n📋 Saving prediction insights...")
    
    best_model = models['logistic_regression']
    test_probs = best_model.predict_proba(X_test)[:, 1]
    
    # Find most confident predictions
    confident_radiant = np.where(test_probs > 0.8)[0]
    confident_dire = np.where(test_probs < 0.2)[0]
    
    insights = {
        'total_test_samples': len(X_test),
        'confident_radiant_predictions': len(confident_radiant),
        'confident_dire_predictions': len(confident_dire),
        'confident_radiant_accuracy': float(y_test.iloc[confident_radiant].mean()) if len(confident_radiant) > 0 else 0,
        'confident_dire_accuracy': float(1 - y_test.iloc[confident_dire].mean()) if len(confident_dire) > 0 else 0,
        'model_confidence_stats': {
            'mean_prob': float(test_probs.mean()),
            'std_prob': float(test_probs.std()),
            'min_prob': float(test_probs.min()),
            'max_prob': float(test_probs.max())
        }
    }
    
    with open(os.path.join(output_dir, 'prediction_insights.json'), 'w') as f:
        json.dump(insights, f, indent=2)
    
    print(f"High confidence predictions (>80% or <20%):")
    print(f"  Confident Radiant: {len(confident_radiant)} predictions, {insights['confident_radiant_accuracy']:.3f} accuracy")
    print(f"  Confident Dire: {len(confident_dire)} predictions, {insights['confident_dire_accuracy']:.3f} accuracy")

def hero_baseline_pipeline(data_path, output_dir='../data/processed', test_size=0.2):
    """Complete hero baseline pipeline with CSV output"""
    print("🎯 HERO BASELINE PIPELINE")
    print("="*50)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data
    print("1. Loading data...")
    with open(data_path, 'r') as f:
        data = json.load(f)
    
    if isinstance(data, dict) and 'rows' in data:
        matches = data['rows']
    else:
        matches = data
    
    df = pd.DataFrame(matches)
    print(f"Loaded {len(df)} matches")
    
    # Load hero names
    id_to_name = load_hero_constants()
    
    # Create features
    print("\n2. Creating hero one-hot features...")
    X = create_hero_onehot_features(df, id_to_name)
    y = create_target_variable(df)
    
    print(f"Final dataset: {X.shape[0]} matches, {X.shape[1]} features")
    
    # Split data
    print(f"\n3. Splitting data ({int((1-test_size)*100)}% train, {int(test_size*100)}% test)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    
    # Further split training into train/val
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )
    
    print(f"Train: {len(X_train)} samples")
    print(f"Validation: {len(X_val)} samples") 
    print(f"Test: {len(X_test)} samples")
    
    print(f"\nClass balance:")
    print(f"Train - Radiant wins: {y_train.mean():.3f}")
    print(f"Val - Radiant wins: {y_val.mean():.3f}")
    print(f"Test - Radiant wins: {y_test.mean():.3f}")
    
    # Train models
    print("\n4. Training models...")
    models, results = train_baseline_models(X_train, X_val, y_train, y_val)
    
    # Evaluate models
    print("\n5. Model Results on Validation Set:")
    print("="*50)
    baseline_accuracy = max(y_val.mean(), 1 - y_val.mean())
    print(f"Baseline (always predict majority): {baseline_accuracy:.4f}")
    print()
    
    for model_name, metrics in results.items():
        print(f"{model_name.title().replace('_', ' ')}:")
        print(f"  Accuracy: {metrics['accuracy']:.4f} (vs {baseline_accuracy:.4f} baseline)")
        print(f"  AUC: {metrics['auc']:.4f}")
        improvement = (metrics['accuracy'] - baseline_accuracy) / baseline_accuracy * 100
        print(f"  Improvement: {improvement:+.1f}%")
        print()
    
    # Feature importance analysis
    print("\n6. Feature Importance Analysis:")
    for model_name, model in models.items():
        analyze_feature_importance(model, X.columns, model_name)
    
    print("\n6.5. See convergences:")
    quick_model_diagnostics(models)
    
    # Final test evaluation
    print("\n7. Final Test Set Evaluation:")
    print("="*50)
    
    best_model_name = max(results.keys(), key=lambda k: results[k]['accuracy'])
    best_model = models[best_model_name]
    
    test_pred = best_model.predict(X_test)
    test_prob = best_model.predict_proba(X_test)[:, 1]
    test_accuracy = accuracy_score(y_test, test_pred)
    test_auc = roc_auc_score(y_test, test_prob)
    
    print(f"Best model: {best_model_name}")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print(f"Test AUC: {test_auc:.4f}")
    
    # Save models and data
    print(f"\n8. Saving results...")
    
    # Save processed data as CSV (avoid parquet dependency issues)
    train_data = pd.concat([X_train, y_train.rename('target')], axis=1)
    val_data = pd.concat([X_val, y_val.rename('target')], axis=1)
    test_data = pd.concat([X_test, y_test.rename('target')], axis=1)
    
    # Use compression to keep file sizes reasonable
    train_data.to_csv(os.path.join(output_dir, 'hero_baseline_train.csv.gz'), 
                      index=False, compression='gzip')
    val_data.to_csv(os.path.join(output_dir, 'hero_baseline_val.csv.gz'), 
                    index=False, compression='gzip')
    test_data.to_csv(os.path.join(output_dir, 'hero_baseline_test.csv.gz'), 
                     index=False, compression='gzip')
    
    # Save best model
    joblib.dump(best_model, os.path.join(output_dir, f'hero_baseline_{best_model_name}.pkl'))
    
    # Save prediction insights
    save_prediction_insights(models, X_test, y_test, X.columns, output_dir)
    
    # Save metadata
    metadata = {
        'features': list(X.columns),
        'n_features': X.shape[1],
        'n_samples': {
            'train': len(X_train),
            'val': len(X_val),
            'test': len(X_test)
        },
        'results': {k: {kk: float(vv) for kk, vv in v.items()} for k, v in results.items()},
        'best_model': best_model_name,
        'test_metrics': {
            'accuracy': float(test_accuracy),
            'auc': float(test_auc)
        }
    }
    
    with open(os.path.join(output_dir, 'hero_baseline_metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"✅ Hero baseline complete!")
    print(f"📁 Results saved to: {output_dir}")
    print(f"🎯 Best model achieved {test_accuracy:.4f} accuracy on test set")
    
    return models, results, (X_train, X_val, X_test, y_train, y_val, y_test)

if __name__ == "__main__":
    # Run the hero baseline pipeline
    models, results, data_splits = hero_baseline_pipeline(
        '../data/public_matches_combined_2000k.json'
    )
    
    print(f"\n🎯 Next Steps:")
    print(f"1. ✅ Hero baseline established")
    print(f"2. 🔄 Try feature engineering (team composition, game metadata)")
    print(f"3. 🚀 Try advanced models (XGBoost, Neural Networks)")
    print(f"4. 🎮 Build prediction interface")
    print(f"5. 🔍 Analyze prediction confidence patterns")