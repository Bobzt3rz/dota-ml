# Quick GPU test
import torch

device = torch.device('cuda')
print(f"Testing GPU with device: {device}")

# Create test data
X_test = torch.randn(1000, 252).to(device)  # Same size as your features
y_test = torch.randint(0, 2, (1000,)).float().to(device)

print(f"Data created on: {X_test.device}")
print(f"GPU memory used: {torch.cuda.memory_allocated(device) / 1024**2:.1f} MB")

# Simple computation
result = torch.mm(X_test, X_test.t())
print(f"✅ GPU computation successful: {result.shape}")