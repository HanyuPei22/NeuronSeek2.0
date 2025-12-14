import torch
from torch.utils.data import TensorDataset

class SyntheticGenerator:
    """
    Generates synthetic datasets to verify Dual-Stream STRidge disentanglement.
    Supports both regression (continuous) and classification (discrete) targets.
    """
    def __init__(self, n_samples=2000, input_dim=10, task_type='regression', num_classes=1, noise_level=0.01):
        self.n = n_samples
        self.d = input_dim
        self.task = task_type
        self.c = num_classes
        self.noise = noise_level

    def get_data(self, mode: str) -> TensorDataset:
        """
        Main entry point to generate data based on the specific test scenario.
        
        Args:
            mode: 'pure_only', 'interact_only', 'hybrid_sparse', or 'linear_vs_quadratic'
        """
        # 1. Generate standard Gaussian features
        X = torch.randn(self.n, self.d)
        
        # 2. Generate signal based on task
        if self.task == 'regression':
            y_logits = self._generate_regression_signal(X, mode)
            # Add observation noise
            y = y_logits + self.noise * torch.randn_like(y_logits)
            
        elif self.task == 'classification':
            y_logits = self._generate_classification_logits(X, mode)
            # Convert logits to hard labels
            y = torch.argmax(y_logits, dim=1)
            
        else:
            raise ValueError(f"Unknown task_type: {self.task}")
            
        return TensorDataset(X, y)

    def _generate_regression_signal(self, X, mode):
        """Generates continuous target y [N, 1]."""
        
        if mode == 'pure_only':
            # Ground Truth: y = 3x^2 - 1.5x^3 (Summed over all features)
            # Objective: Test if Interaction Stream is correctly pruned.
            term_sq = 3.0 * (X**2).sum(dim=1, keepdim=True)
            term_cub = -1.5 * (X**3).sum(dim=1, keepdim=True)
            return term_sq + term_cub
            
        elif mode == 'interact_only':
            # Ground Truth: y = 5(x0*x1) + 3(x2*x3*x4)
            # Objective: Test if Pure Stream is correctly pruned.
            # 2nd order interaction
            term2 = 5.0 * (X[:, 0] * X[:, 1]).view(-1, 1)
            # 3rd order interaction
            term3 = 3.0 * (X[:, 2] * X[:, 3] * X[:, 4]).view(-1, 1)
            return term2 + term3
            
        elif mode == 'hybrid_sparse':
            # Ground Truth: y = 2*x0^2 + 3*(x1*x2)
            # Objective: Test selective identification amidst noise features (x3...xD).
            term_pure = 2.0 * (X[:, 0]**2).view(-1, 1)
            term_int  = 3.0 * (X[:, 1] * X[:, 2]).view(-1, 1)
            return term_pure + term_int
            
        else:
            raise ValueError(f"Unknown mode {mode} for regression.")

    def _generate_classification_logits(self, X, mode):
        """Generates class logits [N, C]."""
        logits = torch.zeros(self.n, self.c)
        
        if mode == 'linear_vs_quadratic':
            # Class 0: Driven by Linear sum
            logits[:, 0] = X.sum(dim=1)
            # Class 1: Driven by Quadratic sum (Pure x^2)
            logits[:, 1] = (X**2).sum(dim=1)
            
            # Class 2 (Optional): Driven by Interaction (x0*x1)
            if self.c > 2:
                logits[:, 2] = (X[:, 0] * X[:, 1]) * 5.0
                
        else:
            raise ValueError(f"Unknown mode {mode} for classification.")
            
        return logits