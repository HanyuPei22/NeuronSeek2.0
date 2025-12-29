import torch
import torch.nn as nn
import numpy as np
from src.core.dual_interaction_layer import DualStreamInteractionLayer

class L0Gate(nn.Module):
    """
    Implements a differentiable L0 Gate using the Hard Concrete distribution.
    Allows for gradient-based pruning of neural network components.
    """
    def __init__(self, temperature=0.66, limit_l=-0.1, limit_r=1.1, init_prob=0.9):
        super().__init__()
        self.temp = temperature
        self.limit_l = limit_l
        self.limit_r = limit_r
        
        # Initialize log_alpha to achieve the desired initial probability
        init_val = np.log(init_prob / (1 - init_prob))
        self.log_alpha = nn.Parameter(torch.Tensor([init_val]))

    def forward(self, x, training=True):
        if training:
            # Sample from Hard Concrete during training
            u = torch.rand_like(self.log_alpha)
            s = torch.sigmoid((torch.log(u + 1e-8) - torch.log(1 - u + 1e-8) + self.log_alpha) / self.temp)
            s = s * (self.limit_r - self.limit_l) + self.limit_l
        else:
            # Deterministic gate during inference
            s = torch.sigmoid(self.log_alpha) * (self.limit_r - self.limit_l) + self.limit_l
            
        # Clamp to [0, 1] range to act as a gate
        z = torch.clamp(s, min=0.0, max=1.0)
        return x * z

    def regularization_term(self):
        """Returns the expected L0 cost (probability of the gate being non-zero)."""
        return torch.sigmoid(self.log_alpha - self.temp * np.log(-self.limit_l / self.limit_r))


class SparseSearchAgent(nn.Module):
    """
    The Orchestrator for Differentiable Search.
    
    Responsibilities:
    1. Instantiates the Math Core (DualStreamInteractionLayer).
    2. Manages control flow components (Gates, BatchNorm, Bias).
    3. Aggregates outputs from all streams.
    4. Calculates sparsity regularization costs.
    """
    def __init__(self, input_dim=10, num_classes=1, rank=8, max_order=5):
        super().__init__()
        self.input_dim = input_dim
        self.core = DualStreamInteractionLayer(input_dim, num_classes, rank, max_order)

        self.bias = nn.Parameter(torch.zeros(num_classes))
        self.gates_pure = nn.ModuleList([L0Gate() for _ in range(max_order)])
        self.gates_int  = nn.ModuleList([L0Gate() for _ in range(max_order)])

        self.bn_pure = nn.ModuleList(nn.BatchNorm1d(num_classes, affine=False) for _ in range(max_order))
        self.bn_int = nn.ModuleList(nn.BatchNorm1d(num_classes, affine=False) for _ in range(max_order))

    def forward(self, x, training=True):
        # Start with global bias
        output = self.bias
        
        # --- Stream 1: Pure Power Terms ---
        for i, gate in enumerate(self.gates_pure):

            term = self.core.get_pure_term(x, i)
            term_norm = self.bn_pure[i](term)
            output = output + gate(term_norm, training=training)

        # --- Stream 2: Interaction Terms ---
        for i, gate in enumerate(self.gates_int):

            term = self.core.get_interaction_term(x, i)
            term_norm = self.bn_int[i](term)
            output = output + gate(term_norm, training=training)
            
        return output
    
    def get_structure(self, threshold=0.5):
        """
        Extracts the discovered structure based on gate probabilities.
        Returns indices of active terms (1-based index).
        """
        pure_active = []
        int_active = []
        
        with torch.no_grad():
            for i, gate in enumerate(self.gates_pure):
                # regularization_term() returns the probability P
                if gate.regularization_term() > threshold:
                    pure_active.append(i + 1)
            
            # Check Interaction Stream Gates
            for i, gate in enumerate(self.gates_int):
                if gate.regularization_term() > threshold:
                    int_active.append(i + 1)
                    
        return pure_active, int_active

    def calculate_regularization(self):
        """
        Computes the total L0 regularization loss.
        Costs are balanced based on input dimensions (Sqrt Penalty).
        """
        reg_loss = 0.0
        
        # Cost factor: Sqrt(Input_Dim) ensures fair competition between simple and complex terms
        cost_base = np.sqrt(self.input_dim)
        
        for gate in self.gates_pure:
            reg_loss += gate.regularization_term() #* cost_base
            
        for gate in self.gates_int:
            reg_loss += gate.regularization_term() #* cost_base
            
        return reg_loss

    def inspect_gates(self, threshold=0.5):
        """Utility to visualize gate status and weight magnitudes."""
        print(f"\n>>> Gate Inspection (Threshold={threshold}) <<<")
        
        def _fmt(name, gates, coeffs):
            info = []
            for i, (gate, coeff) in enumerate(zip(gates, coeffs)):
                prob = gate.regularization_term().item()
                weight = coeff.detach().abs().mean().item()
                status = "[ON]" if prob > threshold else " .  "
                info.append(f"Ord{i+1}:{status} P={prob:.4f} W={weight:.4f}")
            return f"{name}:\n  " + " | ".join(info)

        print(_fmt("Pure Stream", self.gates_pure, self.core.coeffs_pure))
        print(_fmt("Int  Stream", self.gates_int, self.core.coeffs_interact))
        print("-" * 60)