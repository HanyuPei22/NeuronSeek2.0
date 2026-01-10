import sys
import os
import time
import argparse
import pandas as pd
import numpy as np
import torch

# --- Path Setup ---
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from src.utils.data_utils import SyntheticGenerator
from src.searchers.diagnostic_searcher import DiagnosticNeuronSeekSearcher
from src.searchers.tnsr_searcher import TNSRSearcher
from src.searchers.metasymnet_searcher import MetaSymNetSearcher

# ==============================================================================
# 1. Logger
# ==============================================================================

class ConvergenceLogger:
    def __init__(self, method_name, dim, trial):
        self.method = method_name
        self.dim = dim
        self.trial = trial
        self.start_time = time.time()
        self.history = [] 

    def log(self, mse):
        elapsed = time.time() - self.start_time
        # Clip for visualization safety
        if np.isnan(mse) or np.isinf(mse) or mse > 1e5:
            mse = 1e5
        self.history.append({
            'Method': self.method,
            'Dim': self.dim,
            'Trial': self.trial,
            'Time': elapsed,
            'MSE': mse
        })

# ==============================================================================
# 2. Logged NeuronSeek Wrapper
# ==============================================================================

class LoggedNeuronSeek(DiagnosticNeuronSeekSearcher):
    """
    Overrides fit loop to log MSE per epoch and respect timeout.
    """
    def fit(self, X, y, logger, timeout=60):
        X_t = torch.tensor(X, dtype=torch.float32).to(self.agent.device)
        y_t = torch.tensor(y, dtype=torch.float32).reshape(-1, 1).to(self.agent.device)
        dataset = torch.utils.data.TensorDataset(X_t, y_t)
        loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        optimizer = torch.optim.Adam([
            {'params': self.agent.core.coeffs_pure.parameters(), 'lr': 0.005},
            {'params': [self.agent.core.beta], 'lr': 0.005},
            {'params': self.agent.core.factors.parameters(), 'lr': 0.01}, 
            {'params': list(self.agent.bn_pure.parameters()) + list(self.agent.bn_int.parameters()), 'lr': 0.01},
            {'params': list(self.agent.gates_pure.parameters()) + list(self.agent.gates_int.parameters()), 'lr': 0.01}
        ])
        
        self.agent.train()
        start_time = time.time()
        
        # Continuous loop until timeout
        while time.time() - start_time < timeout:
            total_loss = 0
            for bx, by in loader:
                optimizer.zero_grad()
                pred = self.agent(bx)
                task_loss = torch.nn.MSELoss()(pred, by)
                loss = task_loss + 0.05 * self.agent.calculate_regularization()
                loss.backward()
                optimizer.step()
                total_loss += task_loss.item()
            
            avg_mse = total_loss / len(loader)
            logger.log(avg_mse)

# ==============================================================================
# 3. Main Experiment
# ==============================================================================

def run_convergence_experiment():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dim', type=int, default=100, help='Input dimension')
    parser.add_argument('--timeout', type=int, default=30, help='Time limit (seconds)')
    parser.add_argument('--trials', type=int, default=3, help='Number of trials')
    parser.add_argument('--output', type=str, default='convergence_data.csv')
    args = parser.parse_args()

    print(f"=== Convergence Analysis | Dim={args.dim} | Timeout={args.timeout}s ===")
    
    # Use 'interact' mode to highlight structure search differences
    gen = SyntheticGenerator(n_samples=2500, input_dim=args.dim)
    dataset, _ = gen.get_data('interact', 0) 
    X, y = dataset.tensors
    X = X.numpy()
    y = y.numpy()
    
    split = 2000
    X_train, y_train = X[:split], y[:split]
    
    all_logs = []

    for trial in range(args.trials):
        print(f"\n>>> Trial {trial+1}/{args.trials}")
        
        # --- 1. NeuronSeek ---
        print("Running NeuronSeek...")
        logger = ConvergenceLogger("NeuronSeek", args.dim, trial)
        ns = LoggedNeuronSeek(input_dim=args.dim, batch_size=64, reg_lambda=0.05)
        try:
            ns.fit(X_train, y_train, logger, timeout=args.timeout)
        except Exception as e: print(f"NS Error: {e}")
        all_logs.extend(logger.history)
        
        # --- 2. MetaSymNet (Now clean, utilizing callback) ---
        print("Running MetaSymNet...")
        logger = ConvergenceLogger("MetaSymNet", args.dim, trial)
        msn = MetaSymNetSearcher(input_dim=args.dim, time_limit=args.timeout)
        try:
            # Pass logger.log as the callback function
            msn.fit(X_train, y_train, callback=logger.log)
        except Exception as e: print(f"MSN Error: {e}")
        all_logs.extend(logger.history)
        
        # --- 3. TN-SR (Monkey Patch for Logging only) ---
        print("Running TN-SR...")
        logger = ConvergenceLogger("TN-SR", args.dim, trial)
        tnsr = TNSRSearcher(input_dim=args.dim, population_size=1000)
        
        # Dynamic replacement to extract generation logs
        engine = tnsr.engine
        def logged_fit_logic(X_in, y_in):
            X_T = X_in.T
            y_T = y_in.T
            engine.population = [engine.random_prog() for _ in range(engine.pop_size)]
            engine.box = {}
            start_time = time.time()
            
            while time.time() - start_time < args.timeout:
                fitness = []
                for prog in engine.population:
                    try:
                        s_prog = engine.simp(prog)
                        fstr, pred = engine.evaluate(s_prog, X_T)
                        score = engine.compute_fitness(fstr, pred, y_T)
                    except: score = float('inf')
                    fitness.append(score)
                    if score < engine.global_best: engine.global_best = score
                
                # Log current generation best
                logger.log(engine.global_best)
                
                # Simple Evolution
                new_pop = []
                for _ in range(engine.pop_size):
                    parent = engine.get_random_parent(engine.population, fitness)
                    child = engine.do_mutate(parent)
                    new_pop.append(child)
                engine.population = new_pop

        tnsr.engine.fit = logged_fit_logic
        try:
            tnsr.fit(X_train, y_train)
        except Exception as e: print(f"TN-SR Error: {e}")
        all_logs.extend(logger.history)

    # Save Results
    df = pd.DataFrame(all_logs)
    df.to_csv(args.output, index=False)
    print(f"\nSaved convergence data to {args.output}")
    print("\nSummary (Min MSE per method):")
    print(df.groupby('Method')['MSE'].min())

if __name__ == "__main__":
    run_convergence_experiment()