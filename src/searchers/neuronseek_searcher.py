import torch
import numpy as np
from .base import BaseStructureSearcher
from src.models.sparse_search_agent import SparseSearchAgent

class NeuronSeekSearcher(BaseStructureSearcher):
    """Wrapper for NeuronSeek-TD (Gradient-based Sparse Search)."""

    def __init__(self, input_dim: int, epochs: int = 150, batch_size: int = 64):
        super().__init__(input_dim)
        self.epochs = epochs
        self.batch_size = batch_size
        self.agent = SparseSearchAgent(input_dim=input_dim, num_classes=1)
        
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        # Convert to Tensor
        X_t = torch.tensor(X, dtype=torch.float32)
        y_t = torch.tensor(y, dtype=torch.float32).reshape(-1, 1)
        
        # Simple training loop for structure discovery
        optimizer = torch.optim.Adam([
            {'params': [p for n, p in self.agent.named_parameters() if 'gates' not in n], 'lr': 0.01, 'weight_decay': 1e-3},
            {'params': [p for n, p in self.agent.named_parameters() if 'gates' in n], 'lr': 0.05}
        ])
        
        self.agent.train()
        dataset = torch.utils.data.TensorDataset(X_t, y_t)
        loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        for epoch in range(self.epochs):
            # Warmup logic: Unfreeze gates after 40% of epochs
            warmup_end = int(self.epochs * 0.4)
            if epoch == warmup_end:
                 for n, p in self.agent.named_parameters():
                     if 'gates' in n: p.requires_grad = True

            for bx, by in loader:
                optimizer.zero_grad()
                pred = self.agent(bx)
                mse = torch.nn.functional.mse_loss(pred, by)
                
                reg = 0.0
                if epoch >= warmup_end:
                    reg = 2.0 * self.agent.calculate_regularization() # Lambda=2.0
                
                loss = mse + reg
                loss.backward()
                optimizer.step()

    def get_structure_info(self) -> dict:
        """Returns the discovered sparsity masks."""
        pure_idx, int_idx = self.agent.get_structure(threshold=0.5)
        return {
            'pure_indices': pure_idx,
            'interact_indices': int_idx
        }