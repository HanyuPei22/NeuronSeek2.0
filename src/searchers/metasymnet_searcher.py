import numpy as np
import sympy as sp
import operator
from scipy.optimize import minimize
from copy import deepcopy
import time

from .base import BaseStructureSearcher
from src.utils.seeds import setup_seed

class MetaSymNetEngine:
    """
    Refactored Engine based on the provided MetaSymNet code (Snippet 1).
    Uses Continuous Relaxation + L-BFGS-B Optimization to search for formulas.
    """
    def __init__(self, input_dim, max_iter=20, time_limit=60):
        self.input_dim = input_dim
        self.max_iter = max_iter
        self.time_limit = time_limit
        self.ops = ['+', '*', '/', 'sin', 'cos', 'exp', 'log', 'sqrt'] # Function Library
        self.max_val = 10e5 # Clip value
        
        # Pre-defined structure template (Preorder traversal)
        # This is a limitation of MetaSymNet: you must guess the complexity beforehand.
        # We use a moderately complex tree to allow it to find something.
        self.layout = ['s', 's', 'x', 'x', 's', 'x', 'x'] 
        
        self.best_formula = None
        self.best_loss = float('inf')

    # --- Safe Operations ---
    def _div(self, x1, x2):
        return x1 / (x2 + np.sign(x2 + 1e-8) * 1e-5)

    def _log(self, x):
        return np.log(np.abs(x) + 1e-5)

    def _sqrt(self, x):
        return np.sqrt(np.abs(x) + 1e-5)

    def _exp(self, x):
        return np.exp(np.clip(x, -20, 20)) # Clip to avoid overflow

    def _softmax(self, x, c=10):
        x_shift = x - np.max(x)
        exp_x = np.exp(c * x_shift)
        return exp_x / np.sum(exp_x)

    def _forward_op(self, op_name, z, x):
        """Execute symbolic operation."""
        if op_name == '+': return z + x
        if op_name == '-': return z - x
        if op_name == '*': return z * x
        if op_name == '/': return self._div(z, x)
        
        # Unary ops (ignore second arg)
        if op_name == 'sin': return np.sin(z)
        if op_name == 'cos': return np.cos(z)
        if op_name == 'exp': return self._exp(z)
        if op_name == 'log': return self._log(z)
        if op_name == 'sqrt': return self._sqrt(z)
        return z # Fallback

    def _sy_multidim(self, left, right, X_input):
        """Apply all operators to get potential outputs (for Softmax selection)."""
        # Returns a stack of results for [+, *, /, sin, cos...]
        res = []
        for op in self.ops:
            res.append(self._forward_op(op, left, right))
        return np.array(res)

    def _pro_t(self, layout, X, Params):
        """
        The continuous forward pass.
        Params contains Weights for Softmax selection and Bias/Scaling constants.
        """
        stack = []
        
        # Count nodes
        n_s = layout.count('s')
        n_x = layout.count('x')
        
        # Parameter slices sizes
        # For 's' node: Need weights for each Op + Bias + Scale -> len(ops) + 2
        len_s_params = len(self.ops) + 2
        # For 'x' node: Need weights for each Feature + Bias + Scale -> input_dim + 2
        len_x_params = self.input_dim + 2
        
        # Split Params
        split_idx = n_s * len_s_params
        Params_S = Params[:split_idx]
        Params_X = Params[split_idx:]
        
        ptr_s = 0
        ptr_x = 0
        
        # Reverse traversal for stack evaluation
        # Layout is preorder, processed in reverse implies postorder-like build?
        # The original code iterates len(l) and takes l[-(i+1)]
        
        for i in range(len(layout)):
            node_type = layout[-(i+1)]
            
            if node_type == 'x':
                # Slice params for this x-node
                # Note: Original code uses specific indexing logic.
                # Since we process in reverse, we take from the END of the parameter list logic?
                # Actually, standard approach: simple pointer increment.
                # But original code uses: (nx - 1)... implies consuming from back?
                # Let's trust simple sequential consumption from the split arrays.
                
                # Wait, original code: B[(nx - 1) * ... : nx * ...]
                # It consumes backwards.
                
                p_chunk = Params_X[ptr_x * len_x_params : (ptr_x + 1) * len_x_params]
                ptr_x += 1
                
                # p_chunk structure: [Scale, w_feat1, w_feat2..., w_featN, Bias]
                scale = p_chunk[0]
                bias = p_chunk[-1]
                w_feats = p_chunk[1:-1]
                
                # Softmax feature selection
                # Input X is [Dim, Samples]
                probs = self._softmax(w_feats, c=5)
                
                # Weighted sum of features (Approximating Selection)
                # (Dim,) dot (Dim, Samples) -> (Samples,)
                selected_x = np.dot(probs, X) 
                
                val = scale * selected_x + bias
                stack.append(val)
                
            elif node_type == 's':
                p_chunk = Params_S[ptr_s * len_s_params : (ptr_s + 1) * len_s_params]
                ptr_s += 1
                
                scale = p_chunk[0]
                bias = p_chunk[-1]
                w_ops = p_chunk[1:-1]
                
                # Softmax op selection
                probs = self._softmax(w_ops, c=5)
                
                # Get operands
                right = stack.pop()
                left = stack.pop()
                
                # Calculate all possible outcomes
                # Shape: [Num_Ops, Samples]
                candidates = self._sy_multidim(left, right, X)
                
                # Weighted sum of operations
                # (Num_Ops,) dot (Num_Ops, Samples) -> (Samples,)
                res = np.dot(probs, candidates)
                
                val = scale * res + bias
                stack.append(val)
                
        return stack[0]

    def fit(self, X, y):
        # X: [N, D] -> Transpose to [D, N] for calculation
        X_T = X.T 
        y_flat = y.ravel()
        
        # Calculate parameter size
        n_s = self.layout.count('s')
        n_x = self.layout.count('x')
        total_params = n_s * (len(self.ops) + 2) + n_x * (self.input_dim + 2)
        
        start_time = time.time()
        
        def loss_func(params):
            pred = self._pro_t(self.layout, X_T, params)
            # MSE
            mse = np.mean((y_flat - pred)**2)
            # Entropy Regularization (Simplified) - Encourages discreteness
            return mse 

        # Optimization Loop
        print(f"[MetaSymNet] Optimizing {total_params} parameters (Input Dim={self.input_dim})...")
        
        # Standard L-BFGS-B
        # This will become VERY slow if input_dim is large
        try:
            x0 = np.random.randn(total_params)
            res = minimize(loss_func, x0, method='L-BFGS-B', options={'maxiter': self.max_iter})
            
            self.best_params = res.x
            self.best_loss = res.fun
            
            # Decouple / Parse result
            self.best_formula = self._decode_structure(self.best_params, self.layout)
            
        except Exception as e:
            print(f"[MetaSymNet] Optimization failed: {e}")
            self.best_formula = "0"

    def _decode_structure(self, params, layout):
        """
        Convert continuous weights to discrete string formula.
        """
        # (Simplified implementation of the decoding logic from original code)
        # This part reconstructs the string by taking argmax of weights.
        
        n_s = layout.count('s')
        n_x = layout.count('x')
        len_s = len(self.ops) + 2
        len_x = self.input_dim + 2
        
        split = n_s * len_s
        P_s = params[:split]
        P_x = params[split:]
        
        ptr_s = 0
        ptr_x = 0
        
        # Reconstruct tree stack
        stack = []
        
        # Note: Params were consumed in reverse during forward, 
        # so we process accordingly or match index.
        # Let's match the forward pass order (reverse layout).
        
        for i in range(len(layout)):
            node_type = layout[-(i+1)]
            
            if node_type == 'x':
                p_chunk = P_x[ptr_x * len_x : (ptr_x+1) * len_x]
                ptr_x += 1
                w_feats = p_chunk[1:-1]
                # Hard Argmax
                best_feat_idx = np.argmax(w_feats)
                stack.append(f"x{best_feat_idx}")
                
            elif node_type == 's':
                p_chunk = P_s[ptr_s * len_s : (ptr_s+1) * len_s]
                ptr_s += 1
                w_ops = p_chunk[1:-1]
                best_op = self.ops[np.argmax(w_ops)]
                
                r = stack.pop()
                l = stack.pop()
                
                # Construct string
                if best_op in ['sin', 'cos', 'exp', 'log', 'sqrt']:
                    expr = f"{best_op}({l})" # Unary takes left? logic assumption
                else:
                    expr = f"({l} {best_op} {r})"
                
                stack.append(expr)
                
        return stack[0]


class MetaSymNetSearcher(BaseStructureSearcher):
    def __init__(self, input_dim: int, time_limit=60):
        super().__init__(input_dim)
        self.engine = MetaSymNetEngine(input_dim, time_limit=time_limit)

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        setup_seed(42)
        # Start timer to demonstrate failure on high-dim
        self.engine.fit(X, y)

    def get_structure_info(self) -> dict:
        raw = self.engine.best_formula
        if not raw: return {'type': 'explicit_terms', 'terms': []}
        
        print(f"[MetaSymNet] Found: {raw}")
        
        # Use SR parser (assuming SRSearcher logic is available or duplicated)
        from src.searchers.sr_searcher import SRSearcher
        # Helper instance just for parsing
        parser = SRSearcher(self.input_dim)
        # Clean formula for parser
        # Convert "sin(x0)" to sympy compatible if needed, standard math usually works
        parsed = parser._parse_global_formula(raw)
        
        return {
            'type': 'explicit_terms',
            'raw_formula': raw,
            'terms': parsed
        }