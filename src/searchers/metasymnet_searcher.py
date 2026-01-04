import numpy as np
import sympy as sp
import operator
import re
from scipy.optimize import minimize
from copy import deepcopy
import time

from .base import BaseStructureSearcher
from src.utils.seeds import setup_seed

class MetaSymNetEngine:
    """
    MetaSymNet Engine (Continuous Relaxation).
    Based on original paper logic with full function library.
    """
    def __init__(self, input_dim, max_iter=20, time_limit=60):
        self.input_dim = input_dim
        self.max_iter = max_iter
        self.time_limit = time_limit
        self.ops = ['+', '*', '/', 'sin', 'cos', 'exp', 'log', 'sqrt'] 
        self.layout = ['s', 's', 'x', 'x', 's', 'x', 'x'] 
        self.best_formula = None
        self.best_loss = float('inf')

    # --- Operations ---
    def _div(self, x1, x2): return x1 / (x2 + np.sign(x2 + 1e-8) * 1e-5)
    def _log(self, x): return np.log(np.abs(x) + 1e-5)
    def _sqrt(self, x): return np.sqrt(np.abs(x) + 1e-5)
    def _exp(self, x): return np.exp(np.clip(x, -20, 20))
    
    def _softmax(self, x, c=10):
        x_shift = x - np.max(x)
        exp_x = np.exp(c * x_shift)
        return exp_x / np.sum(exp_x)

    def _forward_op(self, op, z, x):
        if op == '+': return z + x
        if op == '-': return z - x
        if op == '*': return z * x
        if op == '/': return self._div(z, x)
        if op == 'sin': return np.sin(z)
        if op == 'cos': return np.cos(z)
        if op == 'exp': return self._exp(z)
        if op == 'log': return self._log(z)
        if op == 'sqrt': return self._sqrt(z)
        return z 

    def _sy_multidim(self, left, right, X):
        res = []
        for op in self.ops:
            res.append(self._forward_op(op, left, right))
        return np.array(res)

    def _pro_t(self, layout, X, Params):
        stack = []
        n_s = layout.count('s')
        n_x = layout.count('x')
        len_s = len(self.ops) + 2
        len_x = self.input_dim + 2
        
        split_idx = n_s * len_s
        Params_S = Params[:split_idx]
        Params_X = Params[split_idx:]
        
        ptr_s, ptr_x = 0, 0
        
        for i in range(len(layout)):
            node_type = layout[-(i+1)]
            if node_type == 'x':
                p_chunk = Params_X[ptr_x * len_x : (ptr_x + 1) * len_x]
                ptr_x += 1
                scale, bias = p_chunk[0], p_chunk[-1]
                w_feats = p_chunk[1:-1]
                
                probs = self._softmax(w_feats, c=5)
                selected_x = np.dot(probs, X)
                stack.append(scale * selected_x + bias)
                
            elif node_type == 's':
                p_chunk = Params_S[ptr_s * len_s : (ptr_s + 1) * len_s]
                ptr_s += 1
                scale, bias = p_chunk[0], p_chunk[-1]
                w_ops = p_chunk[1:-1]
                
                probs = self._softmax(w_ops, c=5)
                right = stack.pop()
                left = stack.pop()
                candidates = self._sy_multidim(left, right, X)
                res = np.dot(probs, candidates)
                stack.append(scale * res + bias)
                
        return stack[0]

    def fit(self, X, y):
        X_T = X.T 
        y_flat = y.ravel()
        n_s = self.layout.count('s')
        n_x = self.layout.count('x')
        total_params = n_s * (len(self.ops) + 2) + n_x * (self.input_dim + 2)
        
        def loss_func(params):
            pred = self._pro_t(self.layout, X_T, params)
            return np.mean((y_flat - pred)**2)

        print(f"[MetaSymNet] Optimizing {total_params} params...")
        try:
            x0 = np.random.randn(total_params)
            res = minimize(loss_func, x0, method='L-BFGS-B', options={'maxiter': self.max_iter})
            self.best_formula = self._decode_structure(res.x, self.layout)
        except Exception as e:
            print(f"[MetaSymNet] Failed: {e}")
            self.best_formula = "0"

    def _decode_structure(self, params, layout):
        """Reconstructs the discrete formula from weights."""
        n_s = layout.count('s')
        n_x = layout.count('x')
        len_s = len(self.ops) + 2
        len_x = self.input_dim + 2
        
        split = n_s * len_s
        P_s = params[:split]
        P_x = params[split:]
        
        ptr_s, ptr_x = 0, 0
        stack = []
        
        for i in range(len(layout)):
            node_type = layout[-(i+1)]
            if node_type == 'x':
                p_chunk = P_x[ptr_x * len_x : (ptr_x+1) * len_x]
                ptr_x += 1
                w_feats = p_chunk[1:-1]
                best_feat_idx = np.argmax(w_feats)
                stack.append(f"x{best_feat_idx}")
                
            elif node_type == 's':
                p_chunk = P_s[ptr_s * len_s : (ptr_s+1) * len_s]
                ptr_s += 1
                w_ops = p_chunk[1:-1]
                best_op = self.ops[np.argmax(w_ops)]
                
                r = stack.pop()
                l = stack.pop()
                
                if best_op in ['sin', 'cos', 'exp', 'log', 'sqrt']:
                    expr = f"{best_op}({l})" # Unary simplified to apply to Left
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
        self.engine.fit(X, y)

    def get_structure_info(self) -> dict:
        raw = self.engine.best_formula
        if not raw: return {'type': 'explicit_terms', 'terms': []}
        
        print(f"[MetaSymNet] Found: {raw}")
        
        # Custom parser for MetaSymNet format (x0, x1...)
        terms = self._parse_indexed_formula(raw)
        
        return {
            'type': 'explicit_terms',
            'raw_formula': raw,
            'terms': terms
        }

    def _parse_indexed_formula(self, formula_str):
        """
        Parses formulas with indexed variables like 'x0 + sin(x1)'.
        """
        try:
            # Create symbols for all possible inputs to be safe
            local_dict = {f'x{i}': sp.Symbol(f'x{i}') for i in range(self.input_dim)}
            # Standard math functions
            local_dict.update({'sin': sp.sin, 'cos': sp.cos, 'exp': sp.exp, 'log': sp.log, 'sqrt': sp.sqrt})
            
            expr = sp.sympify(formula_str, locals=local_dict)
            expr = sp.expand(expr)
        except Exception as e:
            print(f"[MetaSymNet] Parse Warning: {e}")
            return []

        parsed_terms = []
        args = expr.args if expr.func == sp.core.add.Add else [expr]
        
        for arg in args:
            s_term = str(arg)
            # Find all indices used in this term
            indices = [int(x) for x in re.findall(r'x(\d+)', s_term)]
            indices = sorted(list(set(indices))) # Unique indices
            
            if not indices: continue
            
            term_info = {'raw': s_term, 'indices': indices}
            
            # Simple classification for Stage 2
            if 'sin' in s_term or 'cos' in s_term:
                term_info['type'] = 'transcendental' # Mark as complex
            elif '**' in s_term:
                term_info['type'] = 'power'
                term_info['power'] = 2 # Simplified extraction
            elif len(indices) > 1:
                term_info['type'] = 'interact'
            else:
                term_info['type'] = 'linear'
                
            parsed_terms.append(term_info)
            
        return parsed_terms
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Evaluates the best symbolic formula found by MetaSymNet on input X.
        """
        # 1. Retrieve the formula string
        # MetaSymNetEngine stores it in best_formula
        formula = None
        if hasattr(self, 'engine') and self.engine.best_formula:
            formula = self.engine.best_formula

        if not formula:
            return np.zeros(X.shape[0])

        # 2. Prepare execution context
        # Map variable names x0, x1... to columns of X
        local_dict = {}
        for i in range(X.shape[1]):
            local_dict[f'x{i}'] = X[:, i]
        
        # Map math functions to numpy equivalents
        # Note: MetaSymNet ops include +, -, *, /, sin, cos, exp, log, sqrt
        local_dict.update({
            'sin': np.sin,
            'cos': np.cos,
            'exp': np.exp,
            'log': np.log,
            'sqrt': np.sqrt,
            'abs': np.abs
        })

        # 3. Evaluate
        try:
            # Use Python's eval with numpy context
            # We explicitly allow numpy broadcasting for scalars
            pred = eval(formula, {"__builtins__": None}, local_dict)
            
            # Handle case where formula is a constant (e.g. "0.5")
            if np.isscalar(pred):
                pred = np.full(X.shape[0], pred)
                
            return pred.astype(np.float32)
            
        except Exception as e:
            # print(f"[MetaSymNet Predict Error] Formula: {formula}, Error: {e}")
            return np.zeros(X.shape[0])