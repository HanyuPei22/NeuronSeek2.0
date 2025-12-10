import torch
import torch.nn as nn


class CPPolynomialTerm(nn.Module):
    def __init__(self, in_dim, rank, order):
        super().__init__()
        self.in_dim = in_dim
        self.rank = rank
        self.order = order
        
        self.factors = nn.ParameterList([
            nn.Parameter(torch.randn(in_dim, rank) * 0.01) 
            for _ in range(order)
        ])
    
    def forward(self, x):
        projections = [x @ U for U in self.factors]
        result = projections[0]
        for p in projections[1:]:
            result = result * p
        return result


class ClassificationHead(nn.Module):
    def __init__(self, rank, num_classes):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(rank, num_classes) * 0.01)
        self.bias = nn.Parameter(torch.zeros(num_classes))
    
    def forward(self, H):
        return H @ self.weight + self.bias
    
    def importance(self):
        return torch.norm(self.weight, p='fro').item()


class RegressionHead(nn.Module):
    def __init__(self, rank):
        super().__init__()
        self.coef = nn.Parameter(torch.randn(1) * 0.01)
        self.bias = nn.Parameter(torch.zeros(1))
    
    def forward(self, H):
        t = H.sum(dim=1, keepdim=True)
        return t * self.coef + self.bias
    
    def importance(self):
        return abs(self.coef.item())
