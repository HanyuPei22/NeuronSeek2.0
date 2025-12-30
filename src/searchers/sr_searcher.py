import numpy as np
from gplearn.genetic import SymbolicRegressor
from .base import BaseStructureSearcher

class SRSearcher(BaseStructureSearcher):
    """Wrapper for Genetic Programming (gplearn)."""

    def __init__(self, input_dim: int, population_size: int = 2000, generations: int = 20):
        super().__init__(input_dim)
        # Exclude sin/cos to maintain fair comparison with polynomial methods
        # Function set can be expanded if comparing with EQL
        self.est = SymbolicRegressor(
            population_size=population_size,
            generations=generations,
            function_set=('add', 'sub', 'mul', 'div'), 
            metric='mse',
            n_jobs=1,
            random_state=42
        )
        self.best_program = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        # gplearn expects 1D target array
        y_flat = y.ravel() if y.ndim > 1 else y
        self.est.fit(X, y_flat)
        self.best_program = self.est._program

    def get_structure_info(self) -> str:
        """Returns the Lisp-style S-expression string (e.g., 'add(mul(X0, X1), X2)')."""
        if self.best_program is None:
            raise RuntimeError("Model not fitted yet.")
        return str(self.best_program)