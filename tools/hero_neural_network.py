#!/usr/bin/env python3
"""
Hero Neural Network - Standalone Implementation
Simple neural network for Dota 2 hero prediction with non-linear capabilities
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
import time

plt.style.use('default')
sns.set_palette("husl")

class HeroNeuralNetwork(nn.Module):
    """
    Simple neural network for hero prediction
    Architecture: 252 -> 64 -> 32 -> 1
    """
    def __init__(self, num_features=252):
        super(HeroNeuralNetwork, self).__init__()
        
        self.network = nn.Sequential(
            # Layer 1: Hero feature extraction
            nn.Linear(num_features, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            # Layer 2: Feature combination
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            # Output layer
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
    
    def forward(self, x):
        return self.network(x).squeeze()
    
    def get_parameter_count(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

class TrainingTracker:
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
        
        # Learning rate
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
            
        print(f"Loaded {len(id_to_name)} hero names")
        return id_to_name
        
    except Exception as e:
        print(f"Could not load hero names: {e}")
        return {}

def create_hero_features(df, id_to_name=None):
    """Create hero one-hot features"""
    print("Creating hero one-hot features...")
    
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
    
    feature_data = {}
    feature_names = []
    
    for hero_id in all_heroes:
        hero_name = id_to_name.get(hero_id, f'Hero_{hero_id}') if id_to_name else f'Hero_{hero_id}'
        clean_name = hero_name.replace(' ', '_').replace("'", "").replace('-', '_')
        
        radiant_feature = f'radiant_has_{clean_name}'
        feature_data[radiant_feature] = df['radiant_team'].apply(
            lambda team: 1 if isinstance(team, list) and hero_id in team else 0
        )
        feature_names.append(radiant_feature)
        
        dire_feature = f'dire_has_{clean_name}'
        feature_data[dire_feature] = df['dire_team'].apply(
            lambda team: 1 if isinstance(team, list) and hero_id in team else 0
        )
        feature_names.append(dire_feature)
    
    feature_df = pd.DataFrame(feature_data)
    
    print(f"Created {feature_df.shape[1]} hero features")
    return feature_df, feature_names

def create_data_loaders(X_train, X_val, X_test, y_train, y_val, y_test, batch_size=1024):
    """Create PyTorch DataLoaders"""
    print(f"Creating DataLoaders with batch_size={batch_size}")
    
    X_train_tensor = torch.FloatTensor(X_train.values)
    X_val_tensor = torch.FloatTensor(X_val.values)
    X_test_tensor = torch.FloatTensor(X_test.values)
    y_train_tensor = torch.FloatTensor(y_train.values)
    y_val_tensor = torch.FloatTensor(y_val.values)
    y_test_tensor = torch.FloatTensor(y_test.values)
    
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"DataLoaders created:")
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
            outputs = model(X_batch)
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
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            total_loss += loss.item()
            num_batches += 1
    
    return total_loss / num_batches

def train_neural_network(train_loader, val_loader, num_features, 
                        learning_rate=0.001, num_epochs=50, device='cuda'):
    """Train the neural network"""
    
    print(f"Training Hero Neural Network")
    print(f"   Device: {device}")
    print(f"   Learning rate: {learning_rate}")
    print(f"   Epochs: {num_epochs}")
    print(f"   Features: {num_features}")
    print("="*60)
    
    # Initialize model
    model = HeroNeuralNetwork(num_features).to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-3)
    
    print(f"Model parameters: {model.get_parameter_count():,}")
    
    # Training tracker
    tracker = TrainingTracker()
    
    # Early stopping
    best_val_loss = float('inf')
    patience = 10
    patience_counter = 0
    
    start_time = time.time()
    
    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        
        # Training phase
        model.train()
        epoch_train_loss = 0.0
        num_batches = 0
        
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            
            # Compute gradient norm
            grad_norm = 0.0
            for param in model.parameters():
                if param.grad is not None:
                    grad_norm += param.grad.data.norm(2).item() ** 2
            grad_norm = grad_norm ** 0.5
            
            optimizer.step()
            
            epoch_train_loss += loss.item()
            num_batches += 1
        
        avg_train_loss = epoch_train_loss / num_batches
        
        # Validation phase
        val_loss = compute_loss(model, val_loader, criterion, device)
        train_acc = compute_accuracy(model, train_loader, device)
        val_acc = compute_accuracy(model, val_loader, device)
        
        # Track metrics
        current_lr = optimizer.param_groups[0]['lr']
        tracker.update(epoch + 1, avg_train_loss, val_loss, train_acc, val_acc, current_lr, grad_norm)
        
        # Early stopping check
        if val_loss < best_val_loss - 1e-6:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
        
        epoch_time = time.time() - epoch_start_time
        
        # Print progress
        gpu_info = ""
        if device.type == 'cuda':
            gpu_memory = torch.cuda.memory_allocated(device) / 1024**3
            gpu_info = f" | GPU Mem: {gpu_memory:.1f}GB"
        
        print(f"Epoch {epoch+1:3d}/{num_epochs} | "
                f"Train Loss: {avg_train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f} | "
                f"Train Acc: {train_acc:.4f} | "
                f"Val Acc: {val_acc:.4f} | "
                f"Grad Norm: {grad_norm:.4f}{gpu_info} | "
                f"Time: {epoch_time:.1f}s")
        
        # Early stopping
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1} (no improvement for {patience} epochs)")
            break
    
    training_time = time.time() - start_time
    print(f"\nTraining completed in {training_time:.2f} seconds")
    
    return model, tracker

def evaluate_model(model, test_loader, device):
    """Final model evaluation"""
    print(f"\nFinal Test Evaluation")
    print("="*40)
    
    model.eval()
    all_predictions = []
    all_probabilities = []
    all_targets = []
    
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)
            predictions = (outputs > 0.5).float()
            
            all_predictions.extend(predictions.cpu().numpy())
            all_probabilities.extend(outputs.cpu().numpy())
            all_targets.extend(y_batch.cpu().numpy())
    
    accuracy = accuracy_score(all_targets, all_predictions)
    auc = roc_auc_score(all_targets, all_probabilities)
    
    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"Test AUC: {auc:.4f}")
    
    return accuracy, auc

def main():
    """Main training pipeline"""
    
    print("Hero Neural Network Pipeline")
    print("="*50)
    
    # Configuration
    config = {
        'data_path': '../data/public_matches_combined_2000k.json',
        'learning_rate': 0.001,
        'num_epochs': 50,
        'batch_size': 2048,
        'test_size': 0.2,
        'output_dir': '../results'
    }
    
    print("Configuration:")
    for key, value in config.items():
        print(f"   {key}: {value}")
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    # Load data
    print(f"\n1. Loading data...")
    with open(config['data_path'], 'r') as f:
        data = json.load(f)
    
    if isinstance(data, dict) and 'rows' in data:
        matches = data['rows']
    else:
        matches = data
    
    df = pd.DataFrame(matches)
    print(f"Loaded {len(df)} matches")
    
    # Create features
    print(f"\n2. Creating features...")
    id_to_name = load_hero_constants()
    X, feature_names = create_hero_features(df, id_to_name)
    y = df['radiant_win'].astype(int)
    
    print(f"Dataset: {X.shape[0]} matches, {X.shape[1]} features")
    
    # Split data
    print(f"\n3. Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config['test_size'], random_state=42, stratify=y
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
        X_train, X_val, X_test, y_train, y_val, y_test, config['batch_size']
    )
    
    # Train model
    print(f"\n5. Training neural network...")
    model, tracker = train_neural_network(
        train_loader, val_loader, X.shape[1], 
        config['learning_rate'], config['num_epochs'], device
    )
    
    # Visualize training
    print(f"\n6. Creating visualizations...")
    os.makedirs(config['output_dir'], exist_ok=True)
    tracker.plot_training_curves(save_path=os.path.join(config['output_dir'], 'nn_training_curves.png'))
    
    # Final evaluation
    print(f"\n7. Final evaluation...")
    test_accuracy, test_auc = evaluate_model(model, test_loader, device)
    
    # Save model
    torch.save(model.state_dict(), os.path.join(config['output_dir'], 'hero_neural_network.pth'))
    
    print(f"\nResults Summary:")
    print(f"Final test accuracy: {test_accuracy:.4f}")
    print(f"Expected improvement over logistic regression: {(test_accuracy - 0.5629)*100:+.2f}%")
    print(f"Model saved to: {config['output_dir']}/hero_neural_network.pth")

if __name__ == "__main__":
    main()