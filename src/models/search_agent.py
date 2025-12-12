import torch
import torch.nn as nn
import torch.optim as optim
from src.core.tensor_interaction import TensorInteractionLayer

class SearchAgent(nn.Module):
    # Increased default reg_lambda to 0.01 to encourage sparsity
    def __init__(self, input_dim=512, num_classes=10, rank=32, poly_order=3, reg_lambda=0.1, task_type='classification'):
        super().__init__()
        self.core = TensorInteractionLayer(input_dim, num_classes, rank, poly_order)
        self.reg_lambda = reg_lambda
        
        if task_type == 'classification':
            self.criterion = nn.CrossEntropyLoss()
        else:
            self.criterion = nn.MSELoss()

    def fit_stridge(self, dataset, epochs=50, batch_size=64, device='cuda'):
        self.to(device)
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        
        prune_epoch = epochs // 2
        print(f"--- Starting Search (Order={self.core.poly_order}, Reg={self.reg_lambda}) ---")
        
        for epoch in range(epochs):
            self.train()
            
            # Pruning Step
            if epoch == prune_epoch:
                # Increased threshold to 0.10 (10% of max energy)
                self._prune_insignificant_terms(threshold=0.10)

            total_loss = 0
            for X, y in loader:
                X, y = X.to(device), y.to(device)
                optimizer.zero_grad()
                
                logits, reg_loss = self.core(X)
                
                loss = self.criterion(logits, y) + self.reg_lambda * reg_loss
                loss.backward()
                
                # Zero out gradients for pruned terms
                for i in range(self.core.poly_order):
                    if self.core.mask_interact[i] == 0 and self.core.coeffs_interact[i].grad is not None:
                        self.core.coeffs_interact[i].grad.fill_(0.0)
                    if self.core.mask_pure[i] == 0 and self.core.coeffs_pure[i].grad is not None:
                        self.core.coeffs_pure[i].grad.fill_(0.0)
                        
                optimizer.step()
                total_loss += loss.item()
            
            if (epoch+1) % 10 == 0:
                print(f"Epoch {epoch+1}: Loss {total_loss/len(loader):.4f}")

    def _prune_insignificant_terms(self, threshold):
            """
            Prunes based on the 'Effective Contribution' to the output logits.
            This normalizes for the parameter size differences between Pure (D->C) and Interaction (R->C).
            """
            print("\n[STRidge] Analyzing Effective Contribution (Data-Driven Estimation)...")
            
            with torch.no_grad():
                stats = [] # (type, order, effective_magnitude)
                
                for i in range(self.core.poly_order):
                    # --- 1. Pure Stream Effective Mag ---
                    # W_pure is [Input_Dim, Classes]
                    # Effective Strength ~ Average per-parameter contribution
                    w_pure = self.core.coeffs_pure[i]
                    # Divide by sqrt(numel) to make it independent of input dimension size
                    mag_pure = w_pure.norm(p='fro') / (w_pure.numel() ** 0.5)
                    
                    stats.append(('pure', i, mag_pure.item()))

                    # --- 2. Interaction Stream Effective Mag ---
                    # This simulates the reconstruction: ||W_eff|| <= ||U|| * ||V|| * ||K||
                    
                    # A. Product of Factor Norms (Average norm per rank column)
                    factor_product = 1.0
                    for f in self.core.factors[i]: # List of U matrices
                        # f shape: [Input, Rank]. We want avg column norm.
                        f_norm = f.norm(p='fro') / (f.shape[0] ** 0.5) 
                        factor_product *= f_norm
                    
                    # B. Coefficient Norm (K)
                    k = self.core.coeffs_interact[i] # [Rank, Classes]
                    k_norm = k.norm(p='fro') / (k.shape[0] ** 0.5)
                    
                    # C. Total Effective Strength
                    mag_int = factor_product * k_norm
                    
                    stats.append(('interact', i, mag_int.item()))

                # --- Compare and Prune ---
                # Find the strongest term across BOTH streams
                max_mag = max([s[2] for s in stats]) + 1e-9
                
                for (type_str, i, mag) in stats:
                    ratio = mag / max_mag
                    
                    if type_str == 'pure':
                        if ratio < threshold:
                            print(f" -> Pruning Pure x^{i+1} (Eff.Mag: {mag:.4f}, Ratio: {ratio:.3f})")
                            self.core.mask_pure[i] = 0.0
                            self.core.coeffs_pure[i].fill_(0.0)
                        else:
                            print(f" -> Keeping Pure x^{i+1} (Eff.Mag: {mag:.4f}, Ratio: {ratio:.3f})")
                    else:
                        if ratio < threshold:
                            print(f" -> Pruning Interaction Order {i+1} (Eff.Mag: {mag:.4f}, Ratio: {ratio:.3f})")
                            self.core.mask_interact[i] = 0.0
                            self.core.coeffs_interact[i].fill_(0.0)
                        else:
                            print(f" -> Keeping Interaction Order {i+1} (Eff.Mag: {mag:.4f}, Ratio: {ratio:.3f})")
            print("--------------------------\n")

    def get_discovered_orders(self):
        """
        Returns dictionary of active terms.
        """
        structure = {'pure': [], 'interact': []}
        for i in range(self.core.poly_order):
            if self.core.mask_pure[i] == 1.0:
                structure['pure'].append(i+1)
            if self.core.mask_interact[i] == 1.0:
                structure['interact'].append(i+1)
        return structure