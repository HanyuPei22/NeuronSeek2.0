import numpy as np
import operator
import re
from copy import deepcopy
from random import randint, random, shuffle
from sympy import sympify, expand, Symbol
import sympy
from tqdm import tqdm
from typing import Dict, List, Any

from src.utils.seeds import setup_seed
from .base import BaseStructureSearcher

# ==============================================================================
# PART 1: The Original VecSymRegressor Engine (Unmodified Logic)
# ==============================================================================

class VecSymRegressor:
    def __init__(self,
                 random_state=100,
                 pop_size=5000,
                 max_generations=20,
                 tournament_size=10,
                 coefficient_range=None,
                 x_pct=0.7,
                 xover_pct=0.3,
                 save=False,
                 operations=None):

        setup_seed(random_state)
        self.random_state = random_state
        self.pop_size = pop_size
        self.max_generations = max_generations
        self.tournament_size = tournament_size or round(pop_size * 0.03)

        self.x_pct = x_pct
        self.xover_pct = xover_pct
        self.save = save

        self.global_best = float("inf")
        self.best_prog = None
        self.neuron = None

        if coefficient_range is None:
            self.coefficient_range = [-1, 1]
        else:
            self.coefficient_range = coefficient_range

        self.operations = operations or (
            {"func": operator.add, "arg_count": 2, "format_str": "({} + {})"},
            {"func": operator.sub, "arg_count": 2, "format_str": "({} - {})"},
            {"func": operator.mul, "arg_count": 2, "format_str": "({} * {})"},
            {"func": operator.neg, "arg_count": 1, "format_str": "-({})"},
        )

    def render_prog(self, node):
        if "children" not in node:
            return node["feature_name"]
        return node["format_str"].format(*[self.render_prog(c) for c in node["children"]])

    def simp(self, tree):
        # Uses sympy to simplify, then replaces * with @ for vector eval in 'evaluate'
        return str(expand(sympify(self.render_prog(tree)))).replace("*", "@").replace('@@', '**')

    def evaluate(self, expr, x_data):
        # Original Vectorized Evaluation Logic with String Injection
        x = x_data
        temp = re.split(' ', expr)
        for n, i_exp in enumerate(temp):
            if '@' in i_exp:
                index = i_exp.find('@')
                tem = list(i_exp)
                # Inject vectorized ones: 3 -> [3, 3, 3...]
                tem[index - 1] = str((eval(''.join(i_exp[0:index])) * np.ones((1, x_data.shape[0]))).tolist())
                del tem[0:index - 1]
                temp[n] = ''.join(tem)
        ex = ''.join(temp)
        # Note: 'x' variable is assumed to be in local scope for eval
        return expr, eval(ex)

    def rand_w(self):
        return str(np.random.randint(low=self.coefficient_range[0], high=self.coefficient_range[1]))

    def random_prog(self, depth=0):
        n_d = depth
        op = self.operations[randint(0, len(self.operations) - 1)]
        if randint(0, 10) >= depth and n_d <= 6:
            n_d += 1
            return {
                "func": op["func"],
                "children": [self.random_prog(depth + 1) for _ in range(op["arg_count"])],
                "format_str": op["format_str"],
            }
        else:
            return {"feature_name": 'x'} if random() < self.x_pct else {"feature_name": self.rand_w()}

    def select_random_node(self, selected, parent, depth):
        if "children" not in selected:
            return parent
        if randint(0, 10) < 2 * depth:
            return selected
        child_count = len(selected["children"])
        return self.select_random_node(
            selected["children"][randint(0, child_count - 1)],
            selected, depth + 1)

    def do_mutate(self, selected):
        offspring = deepcopy(selected)
        mutate_point = self.select_random_node(offspring, None, 0)
        if mutate_point: # Check if None
            child_count = len(mutate_point.get("children", []))
            if child_count > 0:
                mutate_point["children"][randint(0, child_count - 1)] = self.random_prog(0)
        return offspring

    def do_xover(self, selected1, selected2):
        offspring = deepcopy(selected1)
        xover_point1 = self.select_random_node(offspring, None, 0)
        xover_point2 = self.select_random_node(selected2, None, 0)
        
        # Safety check for nodes with children
        if xover_point1 and xover_point2 and "children" in xover_point1:
            child_count = len(xover_point1["children"])
            if child_count > 0:
                xover_point1["children"][randint(0, child_count - 1)] = deepcopy(xover_point2)
        return offspring

    def get_random_parent(self, popu, fitne):
        tournament_members = [
            randint(0, self.pop_size - 1) for _ in range(self.tournament_size)]
        member_fitness = [(fitne[i], popu[i]) for i in tournament_members]
        return min(member_fitness, key=lambda x: x[0])[1]

    def get_offspring(self, popula, ftns):
        tempt = random()
        parent1 = self.get_random_parent(popula, ftns)
        if tempt < self.xover_pct:
            parent2 = self.get_random_parent(popula, ftns)
            return self.do_xover(parent1, parent2)
        elif self.xover_pct <= tempt < 0.9:
            return self.do_mutate(parent1)
        else:
            return parent1

    def node_count(self, x):
        if "children" not in x:
            return 1
        return sum([self.node_count(c) for c in x["children"]])

    def compute_fitness(self, func, pred, label):
        m = func.count('x')
        if m == 0 or m == 1:
            return float("inf")
        else:
            try:
                # Transpose handling: pred is [1, N], label is [1, N] inside this class logic
                mse = np.mean(np.square(pred - label))
                return mse
            except:
                return float("inf")

    def fit(self, X, y):
        # Original logic expects [Features, Samples] -> Transpose input
        X = X.T
        y = y.T
        
        self.population = [self.random_prog() for _ in range(self.pop_size)]
        self.box = {}

        # Removed file logging for clean library usage
        
        for gen in range(self.max_generations):
            fitness = []
            for prog in self.population:
                try:
                    s_prog = self.simp(prog)
                    # evaluate returns (expr_string, numpy_result)
                    func_str, prediction = self.evaluate(s_prog, X)
                    score = self.compute_fitness(func_str, prediction, y)
                except Exception:
                    score = float("inf")
                    func_str = ""
                
                fitness.append(score)

                if score < self.global_best:
                    self.global_best = score
                    self.best_prog = func_str

                # Box logic (Pareto-ish storage)
                if len(self.box) < self.pop_size * 0.05:
                    self.box[score] = prog
                else:
                    key_sort = sorted(self.box)
                    if score < key_sort[-1]:
                        self.box.pop(key_sort[-1])
                        self.box[score] = prog

            lst = list(self.box.values())
            self.population += lst
            shuffle(self.population)
            
            # Offspring generation
            population_new = [self.get_offspring(self.population, fitness) for _ in range(self.pop_size)]
            self.population = population_new + lst

        self.neuron = self.best_prog


# ==============================================================================
# PART 2: The Wrapper (Adapter Pattern)
# ==============================================================================

class SRSearcher(BaseStructureSearcher):
    """
    Wrapper for the original TN-SR VecSymRegressor.
    Translates input data and parses output strings into 'global_power' terms.
    """
    def __init__(self, input_dim: int, population_size=2000, generations=20):
        super().__init__(input_dim)
        # Using the original engine
        self.engine = VecSymRegressor(
            pop_size=population_size, 
            max_generations=generations,
            random_state=42
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        X: [N_samples, Input_Dim]
        y: [N_samples, 1]
        """
        # The engine internally does Transpose, so we pass as is or check docstring.
        # Docstring says: "Args: X: Input data." -> Inside: "X = X.T"
        # So we pass standard [N, D] numpy arrays.
        self.engine.fit(X, y)

    def get_structure_info(self) -> dict:
        """
        Parses the found formula (e.g. 'x**2 + 2@x') into a structure config.
        Target format: [{'type': 'global_power', 'p': 2}, ...]
        """
        raw_formula = self.engine.neuron
        if not raw_formula:
            print("[SR] Warning: No formula found.")
            return {'type': 'explicit_terms', 'terms': []}
            
        # Clean the specific format from the engine:
        # e.g., "x**2 + 0.5@x" -> "x**2 + 0.5*x"
        clean_formula = raw_formula.replace('@', '*')
        print(f"[SR] Best Formula: {clean_formula}")
        
        parsed_terms = self._parse_global_formula(clean_formula)
        
        return {
            'type': 'explicit_terms',
            'raw_formula': clean_formula,
            'terms': parsed_terms
        }

    def _parse_global_formula(self, formula_str: str) -> List[Dict]:
        """
        Parses a polynomial of a single variable 'x' (representing the whole tensor).
        """
        try:
            x = Symbol('x')
            # 1. Expand: (x+1)**2 -> x**2 + 2*x + 1
            expr = expand(sympify(formula_str))
        except Exception as e:
            print(f"[SR] Parse Error: {e}")
            return []

        parsed_terms = []
        
        # Handle single term vs sum
        if expr.func == sympy.core.add.Add:
            args = expr.args
        else:
            args = [expr]
            
        for arg in args:
            # Separate coeff and term: 2*x**2 -> coeff=2, term=x**2
            coeff, term = arg.as_coeff_Mul()
            
            # Constant check
            if term == sympy.S.One:
                # This is a bias term (e.g. + 5)
                # We can ignore it as the Layer has a bias parameter, 
                # or add a 'constant' type. Let's ignore for structure.
                continue
                
            # Get degree of x
            degree = sympy.degree(term, gen=x)
            
            # We only care about the power. The coefficient will be retrained.
            if degree > 0:
                parsed_terms.append({
                    'type': 'global_power', 
                    'p': int(degree),
                    'raw': str(term)
                })
                
        return parsed_terms