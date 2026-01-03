import sys
import os
import time
import pandas as pd
import numpy as np
import torch
from func_timeout import func_timeout, FunctionTimedOut

# Adjust path to project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

# Imports (保持不变)
from src.searchers.neuronseek_searcher import NeuronSeekSearcher
from src.searchers.tnsr_searcher import TNSRSearcher
from src.searchers.sr_searcher import SRSearcher
from src.searchers.eql_searcher import EQLSearcher
from src.searchers.metasymnet_searcher import MetaSymNetSearcher
from src.utils.data_utils import get_synthetic_data
from experiments.common.structure_evaluator import retrain_and_evaluate

def run_experiment(searcher_cls, params, X_train, y_train, X_test, y_test, timeout=60, seed=42):
    """
    Executes one experiment run with explicit seed setting if applicable.
    """
    # Set seed for reproducibility frameworks (numpy/torch)
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # --- 1. Initialization ---
    try:
        # Some searchers might accept 'seed' or 'random_state' in params
        # We can inject it if the class supports it, or rely on global seed
        model = searcher_cls(**params)
    except Exception as e:
        print(f"    [Init Error] {searcher_cls.__name__}: {e}")
        return 0.0, "Init_Fail", 999.0

    # --- 2. Search Phase (Fit) ---
    start_time = time.time()
    status = "Success"
    try:
        func_timeout(timeout, model.fit, args=(X_train, y_train))
    except FunctionTimedOut:
        status = "Timeout"
    except Exception as e:
        print(f"    [Fit Error] {e}")
        status = "Failed"
    
    search_time = time.time() - start_time

    # --- 3. Structure Extraction ---
    try:
        structure = model.get_structure_info()
    except Exception as e:
        print(f"    [Structure Extraction Error] {e}")
        structure = {'type': 'failed'}

    # --- 4. Retrain Phase (Strict Probe) ---
    mse = 999.0
    
    if status != "Init_Fail":
        try:
            mse = retrain_and_evaluate(
                model, 
                structure, 
                X_train, y_train, 
                X_test, y_test, 
                epochs=50
            )
            # Clip extreme values for stability
            if np.isnan(mse) or np.isinf(mse) or mse > 1e6:
                mse = 1e6
        except Exception as e:
            print(f"    [Retrain Error] {e}")

    return search_time, status, mse

def main():
    # --- Experiment Config ---
    dims = [10, 30, 50, 100]
    timeout_sec = 60
    n_trials = 5  # [CRITICAL] Run 5 independent trials per setting
    
    results = []
    
    print(f"Starting Scalability Experiment: {n_trials} trials per setting, Timeout={timeout_sec}s")

    for D in dims:
        print(f"\n{'='*60}\nTesting Dimension: {D}\n{'='*60}")
        
        # Generate Data (Fix data for all methods to ensure fair comparison on this Dimension)
        X, y = get_synthetic_data(D, n_samples=2500)
        split = 2000
        X_train, y_train = X[:split], y[:split]
        X_test, y_test = X[split:], y[split:]
        
        # Protocol Definition
        # Note: Rank=8 is explicit for NeuronSeek
        protocol = [
            ("NeuronSeek", NeuronSeekSearcher, {'input_dim': D, 'epochs': 80, 'rank': 8}),
            ("TN-SR", TNSRSearcher, {'input_dim': D, 'population_size': 1000, 'generations': 10}),
            ("EQL", EQLSearcher, {'input_dim': D, 'epochs': 500}),
            ("MetaSymNet", MetaSymNetSearcher, {'input_dim': D, 'time_limit': timeout_sec})
        ]
        
        try:
            protocol.insert(2, ("Standard-SR", SRSearcher, {'input_dim': D, 'population_size': 1000, 'generations': 10}))
        except: pass

        # Iterate over methods
        for name, cls, params in protocol:
            print(f">>> Method: {name}")
            
            trial_mses = []
            trial_times = []
            success_count = 0
            
            # [CRITICAL] Loop for repeated trials
            for i in range(n_trials):
                # Use different seed for each trial
                seed = 42 + i 
                
                t, stat, mse = run_experiment(cls, params, X_train, y_train, X_test, y_test, timeout=timeout_sec, seed=seed)
                
                trial_mses.append(mse)
                trial_times.append(t)
                
                if stat == "Success" or (stat == "Timeout" and mse < 1e5):
                    success_count += 1
                
                print(f"    Trial {i+1}/{n_trials}: Time={t:.1f}s | MSE={mse:.4f} | ({stat})")
            
            # Calculate Statistics
            mean_mse = np.mean(trial_mses)
            std_mse = np.std(trial_mses)
            mean_time = np.mean(trial_times)
            
            print(f"  [AVG] {name}: MSE = {mean_mse:.4f} ± {std_mse:.4f}")
            
            # Record Result Row
            row = {
                'Dim': D,
                'Method': name,
                'MSE_Mean': mean_mse,
                'MSE_Std': std_mse,
                'Time_Mean': mean_time,
                'Success_Rate': success_count / n_trials
            }
            results.append(row)

    # Save Final Aggregated Report
    df = pd.DataFrame(results)
    print("\nFINAL SCALABILITY REPORT (Aggregated)")
    print(df)
    df.to_csv("scalability_strict_aggregated.csv", index=False)

if __name__ == "__main__":
    main()