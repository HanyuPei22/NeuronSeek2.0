import torch
import torch.nn as nn
import numpy as np
from src.core.tensor_interaction import TensorInteractionLayer

class L0Gate(nn.Module):
    """
    Hard Concrete Gate for differentiable L0 regularization.
    """
    def __init__(self, temperature=0.66, limit_l=-0.1, limit_r=1.1, init_prob=0.9):
        super().__init__()
        self.temp = temperature
        self.limit_l = limit_l
        self.limit_r = limit_r
        init_val = np.log(init_prob / (1 - init_prob))
        self.log_alpha = nn.Parameter(torch.Tensor([init_val]))

    def forward(self, x, training=True):
        if training:
            u = torch.rand_like(self.log_alpha)
            s = torch.sigmoid((torch.log(u + 1e-8) - torch.log(1 - u + 1e-8) + self.log_alpha) / self.temp)
            s = s * (self.limit_r - self.limit_l) + self.limit_l
        else:
            s = torch.sigmoid(self.log_alpha) * (self.limit_r - self.limit_l) + self.limit_l
        z = torch.clamp(s, min=0.0, max=1.0)
        return x * z

    def regularization_term(self):
        return torch.sigmoid(self.log_alpha - self.temp * np.log(-self.limit_l / self.limit_r))

class SparseSearchAgent(nn.Module):
    def __init__(self, input_dim=10, num_classes=1, rank=8, max_order=5):
        super().__init__()
        self.core = TensorInteractionLayer(input_dim, num_classes, rank, max_order)
        self.input_dim = input_dim
        self.rank = rank
        
        # [Fix] Define bias explicitly
        # Bias is necessary to handle non-zero mean of polynomial terms (e.g., x^2)
        self.bias = nn.Parameter(torch.zeros(num_classes))
        
        self.gates_pure = nn.ModuleList([L0Gate() for _ in range(max_order)])
        self.gates_int  = nn.ModuleList([L0Gate() for _ in range(max_order)])

        self.bn_pure = nn.ModuleList(nn.BatchNorm1d(num_classes, affine= False) for _ in range(max_order))
        self.bn_int = nn.ModuleList(nn.BatchNorm1d(num_classes, affine= False) for _ in range(max_order))
        
        self._init_weights()

    def _init_weights(self):
        # Standard Xavier Init
        for p in self.core.coeffs_pure:
            nn.init.xavier_normal_(p, gain=1.0)
        for i in range(len(self.core.coeffs_interact)):
            nn.init.xavier_normal_(self.core.coeffs_interact[i], gain=1.0)
            for f in self.core.factors[i]:
                nn.init.xavier_normal_(f, gain=1.0)

    def forward(self, x, training=True):
        # [Fix] Use self.bias as starting point
        output = self.bias
        
        # Pure Stream
        for i, gate in enumerate(self.gates_pure):
            order = i + 1
            term = (x ** order) @ self.core.coeffs_pure[i]
            term_norm = self.bn_pure[i](term)
            output = output + gate(term_norm, training=training)

        # Interaction Stream
        for i, gate in enumerate(self.gates_int):
            comp_prod = 1.0
            for factor in self.core.factors[i]:
                comp_prod = comp_prod * (x @ factor)
            
            term = comp_prod @ self.core.coeffs_interact[i]
            term_norm = self.bn_int[i](term)
            output = output + gate(term_norm, training=training)
            
        return output

    def calculate_regularization(self):
        reg_loss = 0.0
        # Sqrt Complexity Penalty
        cost_p = np.sqrt(self.input_dim)
        for gate in self.gates_pure:
            reg_loss += gate.regularization_term() * cost_p
            
        cost_i = np.sqrt(2 * self.input_dim * self.rank + self.rank)
        for gate in self.gates_int:
            reg_loss += gate.regularization_term() * cost_i
            
        return reg_loss

    def get_structure(self, threshold=0.5):
        pure_active = []
        int_active = []
        with torch.no_grad():
            for i, gate in enumerate(self.gates_pure):
                if gate.regularization_term() > threshold:
                    pure_active.append(i + 1)
            for i, gate in enumerate(self.gates_int):
                if gate.regularization_term() > threshold:
                    int_active.append(i + 1)
        return pure_active, int_active
    
    def inspect_gates(self):
            print(f"\n>>> Gate Status Inspection (Threshold=0.5) <<<")
            
            def format_stream(name, gates):
                info = []
                for i, gate in enumerate(gates):
                    prob = gate.regularization_term().item()
                    status = "[ON]" if prob > 0.5 else " .  "
                    p_str = f"{prob:.4f}"
                    info.append(f"Ord{i+1}:{status} {p_str}")
                return f"{name}:\n  " + " | ".join(info)

            print(format_stream("Pure Stream", self.gates_pure))
            print(format_stream("Int  Stream", self.gates_int))
            print(f"{'-'*60}")