import torch
from torch.utils.data import TensorDataset

class SyntheticGenerator:
    """
    Holistic Synthetic Generator with ROBUST COEFFICIENT STRATEGIES.
    
    Now supports:
    1. Randomized Coefficients: To prove generic applicability.
    2. Dynamic Range (Strong vs Weak): To test sensitivity.
    3. Sign Flipping: To prevent magnitude bias.
    """
    def __init__(self, n_samples=2500, input_dim=10, noise_level=0.01):
        self.n = n_samples
        self.d = input_dim
        self.noise = noise_level

    def get_data(self, mode: str, variant: int = 0):
        X = torch.randn(self.n, self.d)
        
        if mode == 'pure':
            y, truth = self._formula_pure(X, variant)
        elif mode == 'interact':
            y, truth = self._formula_interact(X, variant)
        elif mode == 'hybrid':
            y, truth = self._formula_hybrid(X, variant)
        else:
            raise ValueError(f"Unknown mode: {mode}")

        y = y + self.noise * torch.randn_like(y)
        return TensorDataset(X, y), truth

    def _get_random_coeff(self, low=0.5, high=3.0):
        """Helper: Returns a random coefficient with random sign."""
        mag = torch.FloatTensor(1).uniform_(low, high)
        sign = torch.randint(0, 2, (1,)).float() * 2 - 1 # -1 or 1
        return (mag * sign).item()

    def _generate_global_interaction(self, X, order):
        """Generates global interaction signal (Product of projections)."""
        signal = torch.ones(X.size(0), 1)
        for _ in range(order):
            w = torch.randn(self.d, 1) / (self.d ** 0.5) 
            signal = signal * (X @ w)
        return signal

    # ------------------------------------------------------------------
    # Revised Formulas with Randomization & Dynamic Range
    # ------------------------------------------------------------------

    def _formula_pure(self, X, v):
        truth = {'pure': [], 'interact': []}
        
        c1 = self._get_random_coeff(1.0, 3.0)
        c2 = self._get_random_coeff(1.0, 3.0)

        if v == 0: # Randomized Quadratic
            # y = c * x^2
            y = c1 * (X**2).sum(dim=1, keepdim=True)
            truth['pure'] = [2]
            
        elif v == 1: # Randomized Cubic
            # y = c * x^3
            y = c1 * (X**3).sum(dim=1, keepdim=True)
            truth['pure'] = [3]
            
        elif v == 2: # [Dynamic Range] Strong Linear + Weak Quadratic
            # y = 10.0*x + 0.2*x^2 (Can we find the weak x^2?)
            y = 10.0 * X.sum(dim=1, keepdim=True) + 0.2 * (X**2).sum(dim=1, keepdim=True)
            truth['pure'] = [1, 2]
            
        elif v == 3: # [Dynamic Range] Strong Quadratic + Weak Quartic
            # y = 5.0*x^2 + 0.5*x^4
            y = 5.0 * (X**2).sum(dim=1, keepdim=True) + 0.5 * (X**4).sum(dim=1, keepdim=True)
            truth['pure'] = [2, 4]
            
        elif v == 4: # High Order
            # y = c * x^5
            y = c1 * (X**5).sum(dim=1, keepdim=True)
            truth['pure'] = [5]
            
        return y, truth

    def _formula_interact(self, X, v):
        truth = {'pure': [], 'interact': []}
        
        c1 = self._get_random_coeff(1.0, 3.0)
        c2 = self._get_random_coeff(1.0, 3.0)

        if v == 0: # Random 2nd Order
            y = c1 * self._generate_global_interaction(X, 2)
            truth['interact'] = [2]
            
        elif v == 1: # Mixed (2 + 3)
            y = c1 * self._generate_global_interaction(X, 2) + c2 * self._generate_global_interaction(X, 3)
            truth['interact'] = [2, 3]
            
        elif v == 2: # [Dynamic Range] Strong 2nd + Weak 3rd
            # y = 8.0 * I(2) + 0.3 * I(3)
            y = 8.0 * self._generate_global_interaction(X, 2) + 0.3 * self._generate_global_interaction(X, 3)
            truth['interact'] = [2, 3]
            
        elif v == 3: # Mixed Even (2 + 4)
            y = c1 * self._generate_global_interaction(X, 2) + c2 * self._generate_global_interaction(X, 4)
            truth['interact'] = [2, 4]
            
        elif v == 4: # High Order
            y = c1 * self._generate_global_interaction(X, 4)
            truth['interact'] = [4]
            
        return y, truth

    def _formula_hybrid(self, X, v):
        truth = {'pure': [], 'interact': []}
        
        c1 = self._get_random_coeff(1.0, 3.0)
        c2 = self._get_random_coeff(1.0, 3.0)

        if v == 0: # Standard Hybrid
            y = c1*(X**2).sum(dim=1, keepdim=True) + c2*self._generate_global_interaction(X, 2)
            truth['pure'] = [2]; truth['interact'] = [2]
            
        elif v == 1: # Linear + Bilinear
            y = c1*X.sum(dim=1, keepdim=True) + c2*self._generate_global_interaction(X, 2)
            truth['pure'] = [1]; truth['interact'] = [2]
            
        elif v == 2: # [Dynamic Range] Strong Pure + Weak Interaction
            # y = 5.0 * x^3 + 0.2 * Interact(2)
            # This tests if 'Pure dominance' hides the weak interaction
            y = 5.0 * (X**3).sum(dim=1, keepdim=True) + 0.2 * self._generate_global_interaction(X, 2)
            truth['pure'] = [3]; truth['interact'] = [2]
            
        elif v == 3: # [Dynamic Range] Weak Pure + Strong Interaction
            # y = 0.2 * x^2 + 8.0 * Interact(3)
            # This tests if 'Interaction dominance' hides the weak pure term
            y = 0.2 * (X**2).sum(dim=1, keepdim=True) + 8.0 * self._generate_global_interaction(X, 3)
            truth['pure'] = [2]; truth['interact'] = [3]
            
        elif v == 4: # Complex High Order
            y = c1*(X**2).sum(dim=1, keepdim=True) + c2*self._generate_global_interaction(X, 3)
            truth['pure'] = [2]; truth['interact'] = [3]
            
        return y, truth