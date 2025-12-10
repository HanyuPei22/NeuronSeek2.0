import torch
import torch.nn as nn


class PolynomialConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, 
                 bias=False, active_orders=[1], rank=16):
        super().__init__()
        self.active_orders = active_orders
        self.stride = stride
        self.rank = rank
        
        if 1 in active_orders:
            self.conv_linear = nn.Conv2d(in_channels, out_channels, kernel_size, 
                                         stride, padding, bias=bias)
            self.bn_linear = nn.BatchNorm2d(out_channels)
        
        self.poly_terms = nn.ModuleDict()
        for order in active_orders:
            if order == 1: 
                continue
            
            # CP分解: order个独立投影
            projections = nn.ModuleList([
                nn.Conv2d(in_channels, rank, kernel_size=1, stride=1, bias=False)
                for _ in range(order)
            ])
            self.poly_terms[f'ord_{order}_projs'] = projections
            
            self.poly_terms[f'ord_{order}_out'] = nn.Conv2d(rank, out_channels, 
                                                             kernel_size=kernel_size, 
                                                             stride=stride, padding=padding)
            self.poly_terms[f'ord_{order}_bn'] = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        out = 0.0
        
        if 1 in self.active_orders:
            out = out + self.bn_linear(self.conv_linear(x))
            
        for order in self.active_orders:
            if order == 1: 
                continue
            
            projs = self.poly_terms[f'ord_{order}_projs']
            map_out = self.poly_terms[f'ord_{order}_out']
            bn = self.poly_terms[f'ord_{order}_bn']
            
            # CP交叉项: (x @ U1) ⊙ (x @ U2) ⊙ ... ⊙ (x @ U_order)
            features = [proj(x) for proj in projs]
            interaction = torch.stack(features, dim=0).prod(dim=0)
            
            term_out = bn(map_out(interaction))
            out = out + term_out
        
        return out