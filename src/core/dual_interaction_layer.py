import torch
import torch.nn as nn

class DualStreamInteractionLayer(nn.Module):
    """
    Implements the mathematical core of the Dual-Stream architecture.
    
    Responsibilities:
    1. Allocate parameters for Pure (Power) and Interaction (CP) streams.
    2. Perform atomic feature computations (Project -> Product).
    3. Manage parameter initialization logic.
    
    Note: This layer is agnostic to pruning or gating. It strictly computes features.
    """
    def __init__(self, input_dim: int, num_classes: int, rank: int, poly_order: int):
        super().__init__()
        self.poly_order = poly_order
        self.rank = rank
        
        # --- Parameter Allocation ---
        # 1. Interaction Stream: CP Decomposition Factors
        # Structure: List[Order] -> List[Factor_Matrices]
        self.factors = nn.ModuleList()
        for i in range(1, poly_order + 1):
            # Create a list of 'i' matrices for order 'i'
            # Using torch.empty for efficient memory allocation
            order_factors = nn.ParameterList([
                nn.Parameter(torch.empty(input_dim, rank)) 
                for _ in range(i)
            ])
            self.factors.append(order_factors)
            
        # 2. Interaction Coefficients: Rank -> Output
        self.coeffs_interact = nn.ParameterList([
            nn.Parameter(torch.empty(rank, num_classes)) 
            for _ in range(poly_order)
        ])
        
        # 3. Pure Stream Coefficients: Input -> Output
        self.coeffs_pure = nn.ParameterList([
            nn.Parameter(torch.empty(input_dim, num_classes))
            for _ in range(poly_order)
        ])
        
        # Initialize all parameters immediately
        self.reset_parameters()

    def reset_parameters(self):
        """
        Applies Xavier (Glorot) Uniform initialization to all parameters.
        Crucial for maintaining variance stability in high-order tensor products.
        """
        # Initialize Factors
        for order_factors in self.factors:
            for factor in order_factors:
                nn.init.xavier_uniform_(factor)

        # Initialize Coefficients
        for coeff in self.coeffs_interact:
            nn.init.xavier_uniform_(coeff)
        for coeff in self.coeffs_pure:
            nn.init.xavier_uniform_(coeff)

    def _compute_cp_features(self, x: torch.Tensor, order_idx: int) -> torch.Tensor:
        """
        Computes the latent interaction features via Implicit CP Decomposition.
        
        Mechanism:
        1. Project input 'x' onto 'order' different rank-spaces.
        2. Compute element-wise product across the order dimension.
        """
        order_factors = self.factors[order_idx]
        
        # Projection: [Batch, D] @ [D, R] -> List of [Batch, R]
        projections = [x @ u for u in order_factors]
        
        # Interaction: Element-wise product (Hadamard product)
        # Stack -> [Order, Batch, R] -> Prod(dim=0) -> [Batch, R]
        combined = torch.stack(projections, dim=0).prod(dim=0)
        
        return combined

    def get_interaction_term(self, x: torch.Tensor, order_idx: int) -> torch.Tensor:
        """
        Computes the final term for the Interaction Stream at a specific order.
        Returns: [Batch, Num_Classes]
        """
        feats = self._compute_cp_features(x, order_idx)
        return feats @ self.coeffs_interact[order_idx]

    def get_pure_term(self, x: torch.Tensor, order_idx: int) -> torch.Tensor:
        """
        Computes the final term for the Pure Stream at a specific order.
        Returns: [Batch, Num_Classes]
        """
        order = order_idx + 1
        term = x if order == 1 else x.pow(order)
        return term @ self.coeffs_pure[order_idx]