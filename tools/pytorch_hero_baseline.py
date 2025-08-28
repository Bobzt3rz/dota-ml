#!/usr/bin/env python3
"""
PyTorch Hero Baseline - Logistic Regression with Full Training Visibility
Shows everything sklearn hides: loss curves, gradients, training dynamics
"""

import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
import os
from tqdm import tqdm
import time

# Set style
plt.style.use('default')
sns.set_palette("husl")

class HeroLogisticRegression(nn.Module):
    """
    Simple logistic regression model for hero baseline
    Equivalent to sklearn's LogisticRegression but with full visibility
    """
    def __init__(self, num_features):
        super(HeroLogisticRegression, self).__init__()
        self.linear = nn.Linear(num_features, 1)
        
        # Initialize weights (similar to sklearn's default)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)
    
    def forward(self, x):
        return torch.sigmoid(self.linear(x))
    
    def get_weights(self):
        """Get model weights for analysis"""
        return self.linear.weight.data.cpu().numpy().flatten()
    
    def get_bias(self):
        """Get model bias"""
        return self.linear.bias.data.cpu().item()

class TrainingTracker:
    """Track all training metrics and visualizations"""
    def __init__(self):
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        self.learning_rates = []
        self.gradient_norms = []
        self.epochs = []
        
    def update(self, epoch, train_loss, val_loss, train_acc, val_acc, lr, grad_norm):
        self.epochs.append(epoch)
        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)
        self.train_accuracies.append(train_acc)
        self.val_accuracies.append(val_acc)
        self.learning_rates.append(lr)
        self.gradient_norms.append(grad_norm)
    
    def plot_training_curves(self, save_path=None):
        """Create comprehensive training visualization"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Loss curves
        ax1.plot(self.epochs, self.train_losses, 'b-', label='Training Loss', linewidth=2)
        ax1.plot(self.epochs, self.val_losses, 'r-', label='Validation Loss', linewidth=2)
        ax1.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Binary Cross Entropy Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Accuracy curves
        ax2.plot(self.epochs, self.train_accuracies, 'b-', label='Training Accuracy', linewidth=2)
        ax2.plot(self.epochs, self.val_accuracies, 'r-', label='Validation Accuracy', linewidth=2)
        ax2.set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Learning rate (if it changes)
        ax3.plot(self.epochs, self.learning_rates, 'g-', linewidth=2)
        ax3.set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Learning Rate')
        ax3.grid(True, alpha=0.3)
        
        # Gradient norms
        ax4.plot(self.epochs, self.gradient_norms, 'orange', linewidth=2)
        ax4.set_title('Gradient Norm (Training Stability)', fontsize=14, fontweight='bold')
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('L2 Gradient Norm')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

def load_hero_constants():
    """Load hero names for interpretability"""
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

def create_hero_features(df, id_to_name=None):
    """Create hero one-hot features (same as sklearn version)"""
    print("🎯 Creating hero one-hot features...")
    
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
    feature_names = []
    
    for hero_id in all_heroes:
        hero_name = id_to_name.get(hero_id, f'Hero_{hero_id}') if id_to_name else f'Hero_{hero_id}'
        clean_name = hero_name.replace(' ', '_').replace("'", "").replace('-', '_')
        
        # Radiant has this hero
        radiant_feature = f'radiant_has_{clean_name}'
        feature_data[radiant_feature] = df['radiant_team'].apply(
            lambda team: 1 if isinstance(team, list) and hero_id in team else 0
        )
        feature_names.append(radiant_feature)
        
        # Dire has this hero
        dire_feature = f'dire_has_{clean_name}'
        feature_data[dire_feature] = df['dire_team'].apply(
            lambda team: 1 if isinstance(team, list) and hero_id in team else 0
        )
        feature_names.append(dire_feature)
    
    feature_df = pd.DataFrame(feature_data)
    
    print(f"Created {feature_df.shape[1]} hero features")
    print(f"Feature matrix shape: {feature_df.shape}")
    
    return feature_df, feature_names

def create_data_loaders(X_train, X_val, X_test, y_train, y_val, y_test, batch_size=512):
    """Create PyTorch DataLoaders for efficient training"""
    print(f"📦 Creating DataLoaders with batch_size={batch_size}")
    
    # Convert to PyTorch tensors
    X_train_tensor = torch.FloatTensor(X_train.values)
    X_val_tensor = torch.FloatTensor(X_val.values)
    X_test_tensor = torch.FloatTensor(X_test.values)
    y_train_tensor = torch.FloatTensor(y_train.values)
    y_val_tensor = torch.FloatTensor(y_val.values)
    y_test_tensor = torch.FloatTensor(y_test.values)
    
    # Create datasets
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    
    # Create data loaders - simple synchronous version
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False
    )
    
    print(f"✅ DataLoaders created:")
    print(f"   Train: {len(train_loader)} batches")
    print(f"   Val: {len(val_loader)} batches")
    print(f"   Test: {len(test_loader)} batches")
    
    return train_loader, val_loader, test_loader

def compute_accuracy(model, data_loader, device):
    """Compute accuracy on a dataset"""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for X_batch, y_batch in data_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch).squeeze()
            predicted = (outputs > 0.5).float()
            total += y_batch.size(0)
            correct += (predicted == y_batch).sum().item()
    
    return correct / total

def compute_loss(model, data_loader, criterion, device):
    """Compute average loss on a dataset"""
    model.eval()
    total_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for X_batch, y_batch in data_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch).squeeze()
            loss = criterion(outputs, y_batch)
            total_loss += loss.item()
            num_batches += 1
    
    return total_loss / num_batches

def train_pytorch_model(train_loader, val_loader, num_features, 
                       learning_rate=0.001, num_epochs=100, device='cuda'):
    """
    Train PyTorch logistic regression with maximum GPU utilization
    """
    print(f"\n🚀 Training PyTorch Logistic Regression (GPU Optimized)")
    print(f"   Device: {device}")
    print(f"   Learning rate: {learning_rate}")
    print(f"   Epochs: {num_epochs}")
    print(f"   Features: {num_features}")
    
    # Enable optimizations
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True  # Optimize for consistent input sizes
        print(f"   🔧 cuDNN benchmark mode enabled")
    
    print("="*60)
    
    # Initialize model with diagnostic tracking
    model = HeroLogisticRegression(num_features).to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Store initial weights for comparison
    initial_weights = model.get_weights().copy()
    initial_bias = model.get_bias()
    
    print(f"🔧 Initial model state:")
    print(f"   Weights range: [{initial_weights.min():.6f}, {initial_weights.max():.6f}]")
    print(f"   Initial bias: {initial_bias:.6f}")
    print(f"   Weight std: {initial_weights.std():.6f}")
    
    # Debug: Verify model is actually on GPU
    print(f"🔧 Model device verification:")
    print(f"   Model weights device: {next(model.parameters()).device}")
    print(f"   Target device: {device}")
    
    # Test a small batch to ensure GPU pipeline works
    if device.type == 'cuda':
        try:
            # Create a small test batch
            test_batch = torch.randn(32, num_features).to(device)
            test_output = model(test_batch)
            print(f"   ✅ GPU pipeline test successful: {test_output.shape}")
            print(f"   GPU memory after test: {torch.cuda.memory_allocated(device) / 1024**2:.1f} MB")
        except Exception as e:
            print(f"   ❌ GPU pipeline test failed: {e}")
            print(f"   Falling back to CPU...")
            device = torch.device('cpu')
            model = model.cpu()
    
    # Training tracker with early stopping detection
    tracker = TrainingTracker()
    
    # Early stopping parameters
    best_val_loss = float('inf')
    patience = 10
    patience_counter = 0
    converged_epoch = None
    
    # Training loop with full visibility
    model.train()
    start_time = time.time()
    
    print(f"🚀 Starting training loop...")
    print(f"   Total batches per epoch: {len(train_loader)}")
    print(f"   Expected GPU memory usage: ~1-3GB")
    
    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        epoch_train_loss = 0.0
        num_batches = 0
        
        # Training phase - simple synchronous version
        model.train()
        for batch_idx, (X_batch, y_batch) in enumerate(train_loader):
            # Simple blocking transfer
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            
            # Debug: Check if data is actually on GPU (first batch only)
            if epoch == 0 and batch_idx == 0:
                print(f"🔧 Data device verification:")
                print(f"   X_batch device: {X_batch.device}")
                print(f"   y_batch device: {y_batch.device}")
                print(f"   Batch shape: {X_batch.shape}")
                if device.type == 'cuda':
                    print(f"   GPU memory after first batch: {torch.cuda.memory_allocated(device) / 1024**2:.1f} MB")
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(X_batch).squeeze()
            loss = criterion(outputs, y_batch)
            
            # Backward pass
            loss.backward()
            
            # Compute gradient norm (for monitoring training stability)
            grad_norm = 0.0
            for param in model.parameters():
                if param.grad is not None:
                    grad_norm += param.grad.data.norm(2).item() ** 2
            grad_norm = grad_norm ** 0.5
            
            # Update parameters
            optimizer.step()
            
            epoch_train_loss += loss.item()
            num_batches += 1
        
        avg_train_loss = epoch_train_loss / num_batches
        
        # Validation phase
        val_loss = compute_loss(model, val_loader, criterion, device)
        train_acc = compute_accuracy(model, train_loader, device)
        val_acc = compute_accuracy(model, val_loader, device)
        
        # Track metrics and check for convergence
        current_lr = optimizer.param_groups[0]['lr']
        tracker.update(epoch + 1, avg_train_loss, val_loss, train_acc, val_acc, current_lr, grad_norm)
        
        epoch_time = time.time() - epoch_start_time
        
        # Early stopping check
        if val_loss < best_val_loss - 1e-6:  # Improvement threshold
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            
        # Detect convergence
        if patience_counter >= patience and converged_epoch is None:
            converged_epoch = epoch + 1
            print(f"🎯 Model converged at epoch {converged_epoch} (no improvement for {patience} epochs)")
            print(f"   Continuing training to demonstrate plateau...")
        
        # Print progress with GPU info
        gpu_info = ""
        timing_info = f" | Epoch Time: {epoch_time:.1f}s"
        if device.type == 'cuda':
            gpu_memory = torch.cuda.memory_allocated(device) / 1024**3  # GB
            gpu_cached = torch.cuda.memory_reserved(device) / 1024**3   # GB
            gpu_info = f" | GPU Mem: {gpu_memory:.1f}GB ({gpu_cached:.1f}GB cached)"
        
        print(f"Epoch {epoch+1:3d}/{num_epochs} | "
                f"Train Loss: {avg_train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f} | "
                f"Train Acc: {train_acc:.4f} | "
                f"Val Acc: {val_acc:.4f} | "
                f"Grad Norm: {grad_norm:.4f}{gpu_info}{timing_info}")
    
    training_time = time.time() - start_time
    print(f"\n✅ Training completed in {training_time:.2f} seconds")
    if converged_epoch:
        print(f"🎯 Model effectively converged at epoch {converged_epoch}/{num_epochs}")
        print(f"   Wasted compute time: {(num_epochs - converged_epoch) / num_epochs * 100:.1f}%")
    else:
        print(f"🔄 Model may not have fully converged - consider more epochs")
    
    return model, tracker

def analyze_feature_importance(model, feature_names, top_n=20):
    """Analyze feature importance using model weights"""
    print(f"\n📊 Feature Importance Analysis")
    print("="*60)
    
    weights = model.get_weights()
    bias = model.get_bias()
    
    print(f"Model bias: {bias:.4f}")
    print(f"Weight statistics:")
    print(f"  Min: {weights.min():.4f}")
    print(f"  Max: {weights.max():.4f}")
    print(f"  Mean: {weights.mean():.4f}")
    print(f"  Std: {weights.std():.4f}")
    
    # Get top features by absolute weight
    feature_importance = list(zip(feature_names, weights))
    feature_importance.sort(key=lambda x: abs(x[1]), reverse=True)
    
    print(f"\nTop {top_n} most important features:")
    for i, (feature, weight) in enumerate(feature_importance[:top_n], 1):
        team = "Radiant" if "radiant_has_" in feature else "Dire"
        hero = feature.split("_has_")[1].replace('_', ' ')
        direction = "↑" if weight > 0 else "↓"
        print(f"  {i:2d}. {team:<8} {hero:<20} {weight:+.4f} {direction}")
    
    return feature_importance

def evaluate_model(model, test_loader, device):
    """Final model evaluation"""
    print(f"\n🎯 Final Test Evaluation")
    print("="*60)
    
    model.eval()
    all_predictions = []
    all_probabilities = []
    all_targets = []
    
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch).squeeze()
            predictions = (outputs > 0.5).float()
            
            all_predictions.extend(predictions.cpu().numpy())
            all_probabilities.extend(outputs.cpu().numpy())
            all_targets.extend(y_batch.cpu().numpy())
    
    # Compute metrics
    accuracy = accuracy_score(all_targets, all_predictions)
    auc = roc_auc_score(all_targets, all_probabilities)
    
    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"Test AUC: {auc:.4f}")
    
    # Detailed classification report
    print(f"\nDetailed Classification Report:")
    print(classification_report(all_targets, all_predictions, 
                              target_names=['Dire Win', 'Radiant Win']))
    
    return accuracy, auc

def pytorch_hero_baseline_pipeline(data_path, 
                                 learning_rate=0.001,
                                 num_epochs=100,
                                 batch_size=512,
                                 test_size=0.2,
                                 output_dir='../results'):
    """Complete PyTorch hero baseline pipeline"""
    
    print("🎮 PYTORCH HERO BASELINE PIPELINE")
    print("="*60)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        print(f"CUDA Version: {torch.version.cuda}")
    else:
        print("❌ CUDA not available - training will be slow!")
        print("   Check PyTorch CUDA installation")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data
    print(f"\n1. Loading data from {data_path}...")
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
    print(f"\n2. Creating features...")
    X, feature_names = create_hero_features(df, id_to_name)
    y = df['radiant_win'].astype(int)
    
    print(f"Final dataset: {X.shape[0]} matches, {X.shape[1]} features")
    
    # Split data
    print(f"\n3. Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )
    
    print(f"Train: {len(X_train)} samples")
    print(f"Validation: {len(X_val)} samples")
    print(f"Test: {len(X_test)} samples")
    
    # Create data loaders
    print(f"\n4. Creating data loaders...")
    train_loader, val_loader, test_loader = create_data_loaders(
        X_train, X_val, X_test, y_train, y_val, y_test, batch_size
    )
    
    # Train model
    print(f"\n5. Training model...")
    model, tracker = train_pytorch_model(
        train_loader, val_loader, X.shape[1], 
        learning_rate, num_epochs, device
    )
    
    # Visualize training
    print(f"\n6. Creating training visualizations...")
    tracker.plot_training_curves(save_path=os.path.join(output_dir, 'pytorch_training_curves.png'))
    
    # Analyze features
    print(f"\n7. Analyzing feature importance...")
    feature_importance = analyze_feature_importance(model, feature_names)
    
    # Final evaluation
    print(f"\n8. Final evaluation...")
    test_accuracy, test_auc = evaluate_model(model, test_loader, device)
    
    # Save model and results
    print(f"\n9. Saving results...")
    torch.save(model.state_dict(), os.path.join(output_dir, 'pytorch_hero_baseline.pth'))
    
    # Save training history
    training_history = {
        'epochs': tracker.epochs,
        'train_losses': tracker.train_losses,
        'val_losses': tracker.val_losses,
        'train_accuracies': tracker.train_accuracies,
        'val_accuracies': tracker.val_accuracies,
        'learning_rates': tracker.learning_rates,
        'gradient_norms': tracker.gradient_norms
    }
    
    with open(os.path.join(output_dir, 'pytorch_training_history.json'), 'w') as f:
        json.dump(training_history, f, indent=2)
    
    print(f"\n✅ PyTorch Hero Baseline Complete!")
    print(f"📁 Results saved to: {output_dir}")
    print(f"🎯 Final accuracy: {test_accuracy:.4f}")
    print(f"🚀 GPU acceleration utilized: {device.type == 'cuda'}")
    
    return model, tracker, test_accuracy, test_auc

if __name__ == "__main__":
    # Extreme optimization for simple logistic regression
    CONFIG = {
        'data_path': '../data/public_matches_combined_2000k.json',
        'learning_rate': 0.001,
        'num_epochs': 100,
        'batch_size': 32768,     # Huge batches to saturate GPU
        'test_size': 0.2,
        'output_dir': '../results'
    }
    
    print(f"🎮 PyTorch Hero Baseline Configuration:")
    for key, value in CONFIG.items():
        print(f"   {key}: {value}")
    
    # Run pipeline
    model, tracker, accuracy, auc = pytorch_hero_baseline_pipeline(**CONFIG)
    
    print(f"\n🎯 Comparison with sklearn baseline:")
    print(f"   sklearn accuracy: 55.76% (your previous result)")
    print(f"   PyTorch accuracy: {accuracy*100:.2f}%")
    print(f"   Difference: {(accuracy - 0.5576)*100:+.2f}%")
    
    print(f"\n🎉 Benefits gained:")
    print(f"   ✅ Full training visibility (loss curves, gradients)")
    print(f"   ✅ GPU acceleration")
    print(f"   ✅ Real-time training monitoring")
    print(f"   ✅ Learning rate experimentation capability")
    print(f"   ✅ Advanced optimization techniques available")