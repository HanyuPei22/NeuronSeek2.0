import sys
import os
import time
import argparse
import pandas as pd
import numpy as np
import torch
from func_timeout import func_timeout, FunctionTimedOut
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from src.searchers.neuronseek_searcher import NeuronSeekSearcher
from src.searchers.tnsr_searcher import TNSRSearcher
from src.searchers.sr_searcher import SRSearcher
from src.searchers.eql_searcher import EQLSearcher
from src.searchers.metasymnet_searcher import MetaSymNetSearcher
from src.utils.data_utils import get_synthetic_data
from experiments.common.structure_evaluator import retrain_and_evaluate

def run_experiment(searcher_cls, params, X_train, y_train, X_test, y_test, timeout=60, seed=42):
    """
    Executes one experiment run with explicit seed setting.
    """
    # Set seed for reproducibility frameworks (numpy/torch)
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # --- 1. Initialization ---
    try:
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

def parse_arguments():
    parser = argparse.ArgumentParser(description="Run Scalability Experiments")
    
    # Dimensions: Accept single int or list of ints
    parser.add_argument('--dims', type=int, nargs='+', default=[10, 30, 50, 100],
                        help='List of input dimensions to test (e.g., --dims 10 30)')
    
    # Methods: Accept specific method names
    parser.add_argument('--methods', type=str, nargs='+', 
                        default=['NeuronSeek', 'TN-SR', 'Standard-SR', 'EQL', 'MetaSymNet'],
                        choices=['NeuronSeek', 'TN-SR', 'Standard-SR', 'EQL', 'MetaSymNet'],
                        help='List of methods to run')
    
    # Trials and Seeds
    parser.add_argument('--n_trials', type=int, default=3, help='Number of trials per setting')
    parser.add_argument('--start_seed', type=int, default=42, help='Starting random seed')
    
    # Other settings
    parser.add_argument('--timeout', type=int, default=60, help='Timeout in seconds')
    parser.add_argument('--output', type=str, default='scalability_strict_results.csv', 
                        help='Output CSV filename')
    
    return parser.parse_args()

def main():
    args = parse_arguments()
    
    results = []
    
    print(f"Starting Experiment: Trials={args.n_trials}, Dims={args.dims}, Methods={args.methods}")

    for D in args.dims:
        print(f"\n{'='*60}\nTesting Dimension: {D}\n{'='*60}")
        
        # Generate Data (Fix data for all methods to ensure fair comparison on this Dimension)
        X, y = get_synthetic_data(D, n_samples=2500)
        split = 2000
        X_train, y_train = X[:split], y[:split]
        X_test, y_test = X[split:], y[split:]
        
        # Protocol Definition
        # Define all available methods here
        all_methods = {
            "NeuronSeek": (NeuronSeekSearcher, {'input_dim': D, 'epochs': 80, 'rank': 8}),
            "TN-SR": (TNSRSearcher, {'input_dim': D, 'population_size': 1000, 'generations': 10}),
            "Standard-SR": (SRSearcher, {'input_dim': D, 'population_size': 1000, 'generations': 10}),
            "EQL": (EQLSearcher, {'input_dim': D, 'epochs': 500}),
            "MetaSymNet": (MetaSymNetSearcher, {'input_dim': D, 'time_limit': args.timeout})
        }
        
        # Filter methods based on arguments
        protocol = [(name, all_methods[name][0], all_methods[name][1]) 
                    for name in args.methods if name in all_methods]

        # Iterate over methods
        for name, cls, params in protocol:
            print(f">>> Method: {name}")
            
            trial_mses = []
            trial_times = []
            success_count = 0
            
            for i in range(args.n_trials):
                # Dynamic Seed: Base + Trial Index
                current_seed = args.start_seed + i
                
                t, stat, mse = run_experiment(
                    cls, params, 
                    X_train, y_train, 
                    X_test, y_test, 
                    timeout=args.timeout, 
                    seed=current_seed
                )
                
                trial_mses.append(mse)
                trial_times.append(t)
                
                if stat == "Success" or (stat == "Timeout" and mse < 1e5):
                    success_count += 1
                
                print(f"    Trial {i+1}/{args.n_trials} (Seed {current_seed}): Time={t:.1f}s | MSE={mse:.4f} | ({stat})")
            
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
                'Success_Rate': success_count / args.n_trials
            }
            results.append(row)

    # Save Final Aggregated Report
    df = pd.DataFrame(results)
    
    # If file exists, append (header only if new)
    file_exists = os.path.isfile(args.output)
    #df.to_csv(args.output, mode='a', header=not file_exists, index=False)
    
    #print(f"\nResults saved to {args.output}")
    print(df)

if __name__ == "__main__":
    main()