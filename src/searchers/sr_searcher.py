import numpy as np
import re
from gplearn.genetic import SymbolicRegressor
from .base import BaseStructureSearcher

class SRSearcher(BaseStructureSearcher):
    """
    Wrapper for Standard Genetic Programming (gplearn).
    Discovers sparse, index-based features (x0 * x1).
    """

    def __init__(self, input_dim: int, population_size: int = 2000, generations: int = 20):
        super().__init__(input_dim)
        # Standard function set for polynomial discovery
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
        y_flat = y.ravel() if y.ndim > 1 else y
        self.est.fit(X, y_flat)
        self.best_program = self.est._program

    def get_structure_info(self) -> dict:
        if self.best_program is None:
            return {'type': 'explicit_terms', 'terms': []}
        
        raw_str = str(self.best_program)
        print(f"[Standard SR] Found: {raw_str}")
        
        # Parse gplearn string to our terms format
        # gplearn outputs: sub(add(X0, X1), mul(X2, X2))
        terms = self._parse_gplearn_formula(raw_str)
        
        return {
            'type': 'explicit_terms',
            'raw_formula': raw_str,
            'terms': terms
        }

    def _parse_gplearn_formula(self, raw_str):
        """
        Parses gplearn string into terms.
        Since converting arbitrary trees to sum-of-products is hard, 
        we classify the WHOLE formula as a single 'complex' term 
        that operates on the specific indices involved.
        """
        # Extract all used indices (e.g. X0, X10)
        indices = [int(x) for x in re.findall(r'X(\d+)', raw_str)]
        indices = sorted(list(set(indices)))
        
        if not indices: return [] # Constant
        
        # In Stage 2, we will treat this as a "Blackbox Feature"
        # that takes [x0, x1...] and computes the gplearn formula.
        # But for comparison, we mainly care IF it found the right indices.
        
        # Simple Heuristic Classification for Metrics:
        term_type = 'linear'
        if 'mul' in raw_str: term_type = 'interact'
        
        return [{
            'type': 'gplearn_tree', # Special type for Stage 2 to handle via eval/tree execution
            'indices': indices,
            'raw': raw_str,
            'gplearn_obj': self.best_program # Pass object for execution if possible
        }]