import torch
import torch.nn as nn
import torch.optim as optim
from src.core.tensor_interaction import TensorInteractionLayer

class SearchAgent(nn.Module):
    def __init__(self, input_dim=512, num_classes=10, rank=32, poly_order=3, reg_lambda=0.01):
        super().__init__()
        self.core = TensorInteractionLayer(input_dim, num_classes, rank, poly_order)
        self.reg_lambda = reg_lambda
        self.criterion = nn.CrossEntropyLoss()

    def fit_stridge(self, dataset, epochs=50, batch_size=64, device='cuda'):
        self.to(device)
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        
        # STRidge Schedule: Prune halfway through training
        prune_epoch = epochs // 2
        
        print(f"--- Starting Search (Order={self.core.poly_order}, Rank={self.core.rank}) ---")
        
        for epoch in range(epochs):
            self.train()
            
            # Execute Pruning
            if epoch == prune_epoch:
                self._prune_insignificant_terms(threshold=0.05)

            total_loss = 0
            for X, y in loader:
                X, y = X.to(device), y.to(device)
                optimizer.zero_grad()
                
                logits, reg_loss = self.core(X)
                
                # Main Loss + L1 Regularization (Group Lasso)
                loss = self.criterion(logits, y) + self.reg_lambda * reg_loss
                
                loss.backward()
                
                # Zero out gradients for pruned terms to prevent revival
                for i in range(self.core.poly_order):
                    if self.core.mask_interact[i] == 0:
                        self.core.coeffs_interact[i].grad.fill_(0.0)
                    if self.core.mask_pure[i] == 0:
                        self.core.coeffs_pure[i].grad.fill_(0.0)
                        
                optimizer.step()
                total_loss += loss.item()
            
            if (epoch+1) % 10 == 0:
                avg_loss = total_loss / len(loader)
                print(f"Epoch {epoch+1}: Loss {avg_loss:.4f}")

    def _prune_insignificant_terms(self, threshold):
        """
        Prunes terms based on the Frobenius norm (energy) of their coefficient matrices.
        Checks Pure terms and Interaction terms independently.
        """
        print("\n[STRidge] Pruning Check...")
        with torch.no_grad():
            # 1. Check Interaction Terms
            for i in range(self.core.poly_order):
                energy = torch.norm(self.core.coeffs_interact[i], p='fro').item()
                if energy < threshold:
                    print(f" -> Pruning Interaction Order {i+1} (Energy: {energy:.4f})")
                    self.core.mask_interact[i] = 0.0
                    self.core.coeffs_interact[i].fill_(0.0)
                else:
                    print(f" -> Keeping Interaction Order {i+1} (Energy: {energy:.4f})")

            # 2. Check Pure Power Terms
            for i in range(self.core.poly_order):
                energy = torch.norm(self.core.coeffs_pure[i], p='fro').item()
                if energy < threshold:
                    print(f" -> Pruning Pure x^{i+1} (Energy: {energy:.4f})")
                    self.core.mask_pure[i] = 0.0
                    self.core.coeffs_pure[i].fill_(0.0)
                else:
                    print(f" -> Keeping Pure x^{i+1} (Energy: {energy:.4f})")
        print("--------------------------\n")

    def get_discovered_orders(self):
            """
            Refactored to return a dictionary structure consistent with Dual-Stream logic.
            Used by the discovery script.
            """
            structure = {
                'pure': [],
                'interact': []
            }
            # Check Pure Masks
            for i in range(self.core.poly_order):
                if self.core.mask_pure[i] == 1.0:
                    structure['pure'].append(i+1)
            
            # Check Interaction Masks
            for i in range(self.core.poly_order):
                if self.core.mask_interact[i] == 1.0:
                    structure['interact'].append(i+1)
                    
            return structure