import torch
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, TensorDataset

def get_synthetic_data(input_dim, n_samples=2000, noise=0.1):
    """
    Generates Rank-1 Global Interaction data.
    Formula: y = (w^T x) * (v^T x)
    Crucial for proving that SR fails on global interactions.
    """
    torch.manual_seed(42)
    np.random.seed(42)
    
    X = torch.rand(n_samples, input_dim) * 2 - 1 # [-1, 1]
    
    # Dense Projection Vectors (All features matter)
    w = torch.randn(input_dim)
    w = w / torch.norm(w)
    
    v = torch.randn(input_dim)
    v = v / torch.norm(v)
    
    # Ground Truth: Global Interaction
    y = (X @ w) * (X @ v)
    
    # Add noise
    y_rms = y.std()
    y_noisy = y + torch.randn_like(y) * noise * y_rms
    
    return X.numpy(), y_noisy.numpy()

def get_mnist_data(root='./data', classes=[0, 1], n_samples=None):
    """
    Loads MNIST, filters specific classes (binary classification), and flattens.
    Returns: X [N, 784], y [N, 1] (0 or 1)
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    dataset = datasets.MNIST(root, train=True, download=True, transform=transform)
    
    # Filter classes
    idx = torch.zeros(len(dataset.targets), dtype=torch.bool)
    for c in classes:
        idx |= (dataset.targets == c)
    
    X = dataset.data[idx].float() / 255.0
    y = dataset.targets[idx]
    
    # Flatten: [N, 28, 28] -> [N, 784]
    X = X.view(X.size(0), -1)
    
    # Map labels to 0, 1, 2...
    y_new = torch.zeros_like(y)
    for i, c in enumerate(classes):
        y_new[y == c] = i
    y = y_new.float().unsqueeze(1)
    
    if n_samples:
        X = X[:n_samples]
        y = y[:n_samples]
        
    return X.numpy(), y.numpy()