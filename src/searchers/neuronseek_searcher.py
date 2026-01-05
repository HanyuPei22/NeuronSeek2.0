import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any, Union

# Assuming base structure exists
from .base import BaseStructureSearcher 
from src.models.sparse_search_agent import SparseSearchAgent

class NeuronSeekSearcher(BaseStructureSearcher):
    """
    Wrapper for NeuronSeek-TD (Gradient-based Sparse Search).
    
    Updates:
    - Supports dynamic 'num_classes' for parallel neuron instantiation.
    - Accepts 'rank' as a fixed hyperparameter.
    - Automatically switches between MSE (Regression) and CrossEntropy (Classification).
    """

    def __init__(self, input_dim: int, num_classes: int = 1, rank: int = 8, 
                 epochs: int = 150, batch_size: int = 64):
        super().__init__(input_dim)
        self.epochs = epochs
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.rank = rank
        
        # Instantiate the Agent with specific Rank and Class count
        self.agent = SparseSearchAgent(
            input_dim=input_dim, 
            num_classes=num_classes, 
            rank=rank
        )
        
        # Determine device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.agent.to(self.device)
        
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Fits the structure search agent.
        Adapts data formatting and loss function based on 'num_classes'.
        """
        # 1. Data Preparation
        X_t = torch.tensor(X, dtype=torch.float32).to(self.device)
        
        if self.num_classes > 1:
            # Classification: y should be class indices [Batch], dtype=Long
            # Assumes y is passed as 1D array of labels or [Batch, 1]
            y_t = torch.tensor(y, dtype=torch.long).reshape(-1).to(self.device)
            loss_fn = nn.CrossEntropyLoss()
        else:
            # Regression: y should be values [Batch, 1], dtype=Float
            y_t = torch.tensor(y, dtype=torch.float32).reshape(-1, 1).to(self.device)
            loss_fn = nn.MSELoss()
        
        # 2. Optimizer Setup
        # Different learning rates for structural gates vs. weights
        optimizer = torch.optim.Adam([
            {'params': [p for n, p in self.agent.named_parameters() if 'gates' not in n], 'lr': 0.01, 'weight_decay': 1e-3},
            {'params': [p for n, p in self.agent.named_parameters() if 'gates' in n], 'lr': 0.05}
        ])
        
        # 3. Training Loop
        self.agent.train()
        dataset = torch.utils.data.TensorDataset(X_t, y_t)
        loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        # Freeze gates initially? 
        # Strategy: Initialize with requires_grad=True but maybe suppress updates or rely on warmup logic.
        # User's logic: "Unfreeze after warmup".
        # Better impl: Set requires_grad=False initially if strictly following warmup.
        warmup_end = int(self.epochs * 0.4)
        
        # Initially freeze gates to let weights stabilize
        for n, p in self.agent.named_parameters():
            if 'gates' in n: 
                p.requires_grad = False

        print(f"Starting Search (Task: {'Classification' if self.num_classes > 1 else 'Regression'}, Rank: {self.rank})...")

        for epoch in range(self.epochs):
            # Warmup Logic: Unfreeze gates
            if epoch == warmup_end:
                 print(f"Warmup done (Epoch {epoch}). Unfreezing gates for structure pruning.")
                 for n, p in self.agent.named_parameters():
                     if 'gates' in n: 
                         p.requires_grad = True

            total_loss = 0
            for bx, by in loader:
                optimizer.zero_grad()
                
                # Forward: [Batch, Num_Classes]
                pred = self.agent(bx)
                
                # Task Loss
                # Regression: MSE(pred[B,1], y[B,1])
                # Classification: CE(pred[B,C], y[B])
                task_loss = loss_fn(pred, by)
                
                # Regularization (L0 on gates)
                reg_loss = 0.0
                if epoch >= warmup_end:
                    reg_loss = 0.7 * self.agent.calculate_regularization() # Lambda hyperparam
                
                loss = task_loss + reg_loss
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()

            if (epoch + 1) % 10 == 0:
                # Simple logging
                pass # print(f"Epoch {epoch+1}, Loss: {total_loss / len(loader):.4f}")

    def get_structure_info(self) -> Dict[str, Any]:
        """
        Returns the discovered sparsity masks and essential hyperparameters.
        Includes 'rank' so the evaluator knows how to reconstruct the model.
        """
        pure_idx, int_idx = self.agent.get_structure(threshold=0.5)
        
        # Debug print
        print(f"Structure Found - Pure: {pure_idx}, Interact: {int_idx}")
        
        return {
            'type': 'neuronseek',      # Tag for Evaluator
            'pure_indices': pure_idx,
            'interact_indices': int_idx,
            'rank': self.rank          # Crucial: Must pass rank to evaluator
        }