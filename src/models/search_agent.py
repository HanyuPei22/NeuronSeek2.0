import torch
import torch.nn as nn
import torch.optim as optim
from src.core.tensor_interaction import TensorInteractionLayer

class SearchAgent(nn.Module):
    # Increased default reg_lambda to 0.01 to encourage sparsity
    def __init__(self, input_dim=512, num_classes=10, rank=32, poly_order=3, reg_lambda=0.01):
        super().__init__()
        self.core = TensorInteractionLayer(input_dim, num_classes, rank, poly_order)
        self.reg_lambda = reg_lambda
        self.criterion = nn.CrossEntropyLoss()

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
        print("\n[STRidge] Pruning Check...")
        with torch.no_grad():
            # 1. Collect all energies to find global max
            energies = []
            term_info = [] # (type, index, energy)

            for i in range(self.core.poly_order):
                # Pure Energy
                e_pure = torch.norm(self.core.coeffs_pure[i], p='fro').item()
                energies.append(e_pure)
                term_info.append(('pure', i, e_pure))
                
                # Interaction Energy
                e_int = torch.norm(self.core.coeffs_interact[i], p='fro').item()
                energies.append(e_int)
                term_info.append(('interact', i, e_int))
            
            max_e = max(energies) + 1e-9
            
            # 2. Prune based on relative ratio
            for (type_str, i, e) in term_info:
                ratio = e / max_e
                
                if type_str == 'pure':
                    if ratio < threshold:
                        print(f" -> Pruning Pure x^{i+1} (Ratio: {ratio:.3f})")
                        self.core.mask_pure[i] = 0.0
                        self.core.coeffs_pure[i].fill_(0.0)
                    else:
                        print(f" -> Keeping Pure x^{i+1} (Ratio: {ratio:.3f})")
                else:
                    if ratio < threshold:
                        print(f" -> Pruning Interaction Order {i+1} (Ratio: {ratio:.3f})")
                        self.core.mask_interact[i] = 0.0
                        self.core.coeffs_interact[i].fill_(0.0)
                    else:
                        print(f" -> Keeping Interaction Order {i+1} (Ratio: {ratio:.3f})")
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