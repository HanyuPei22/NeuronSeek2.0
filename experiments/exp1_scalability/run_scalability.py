import sys
import os
import time
import pandas as pd
import torch
import torch.nn as nn
import numpy as np
from func_timeout import func_timeout, FunctionTimedOut

# Adjust path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from src.searchers.neuronseek_searcher import NeuronSeekSearcher
from src.searchers.tnsr_searcher import TNSRSearcher
from src.searchers.sr_searcher import SRSearcher
from src.searchers.eql_searcher import EQLSearcher
from src.searchers.metasymnet_searcher import MetaSymNetSearcher
from src.utils.data_utils import get_synthetic_data

def evaluate_model(searcher, X_test, y_test):
    """
    Evaluates the trained searcher on test data.
    Returns: MSE (Float)
    """
    try:
        # Most searchers don't have a direct 'predict' in the base class,
        # but their internal models do. We need a unified way.
        
        # 1. NeuronSeek / EQL / MetaSymNet (Neural Based)
        if hasattr(searcher, 'model') and searcher.model is not None:
            searcher.model.eval()
            with torch.no_grad():
                X_t = torch.tensor(X_test, dtype=torch.float32)
                pred = searcher.model(X_t).numpy()
                
        # 2. TNSR / SR (Symbolic Based)
        elif hasattr(searcher, 'engine'):
            # TNSR logic
            if hasattr(searcher.engine, 'neuron'): # TNSR
                 formula = searcher.engine.neuron
                 # Re-eval strategy or use internal predict if available
                 # For simplicity in this script, let's assume we can't easily eval string 
                 # without the parser. 
                 # Let's Skip MSE for TNSR/SR in this quick check OR implement eval.
                 return 0.0 # Placeholder for Symbolic methods if eval is hard
            
            # SR (gplearn)
            if hasattr(searcher.engine, 'predict'): 
                pred = searcher.engine.predict(X_test)
            else:
                return 999.0
        else:
            return 999.0

        # Calculate MSE
        mse = np.mean((pred - y_test)**2)
        return mse
    except Exception as e:
        # print(f"Eval failed: {e}")
        return 999.0

def run_search_with_metrics(searcher, X, y, timeout=60):
    """
    Wraps fit() with timeout AND calculates MSE.
    """
    # Split Train/Test for evaluation (80/20)
    N = len(X)
    split = int(N * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    
    start = time.time()
    try:
        # Run FIT with timeout
        func_timeout(timeout, searcher.fit, args=(X_train, y_train))
        end = time.time()
        time_cost = end - start
        
        # Run EVAL
        mse = evaluate_model(searcher, X_test, y_test)
        
        return time_cost, "Success", mse
        
    except FunctionTimedOut:
        return timeout, "Timeout", 999.0
    except Exception as e:
        print(f"Error: {e}")
        return 0, "Failed", 999.0

def main():
    dims = [10, 30, 50, 100]
    timeout_sec = 60 
    
    results = []
    
    for D in dims:
        print(f"\n{'='*30}\nTesting Dimension: {D}\n{'='*30}")
        X, y = get_synthetic_data(D, n_samples=2000)
        
        # 1. NeuronSeek
        print(">>> NeuronSeek...")
        ns = NeuronSeekSearcher(D, epochs=100)
        t_ns, stat_ns, mse_ns = run_search_with_metrics(ns, X, y, timeout_sec)
        print(f"   [NS] Time: {t_ns:.2f}s | MSE: {mse_ns:.4f}")
        
        # 2. TN-SR
        print(">>> TN-SR...")
        sr = TNSRSearcher(D, population_size=1000, generations=10)
        # TNSR evaluation is tricky because it returns a string. 
        # For the paper chart, you might focus on Time, but let's try to capture result.
        t_sr, stat_sr, mse_sr = run_search_with_metrics(sr, X, y, timeout_sec)
        # Note: TNSR MSE might be 0.0 in this script placeholder, ignore for now
        
        # 3. EQL
        print(">>> EQL...")
        eql = EQLSearcher(D, epochs=500)
        t_eql, stat_eql, mse_eql = run_search_with_metrics(eql, X, y, timeout_sec)
        print(f"   [EQL] Time: {t_eql:.2f}s | MSE: {mse_eql:.4f}")

        # 4. MetaSymNet
        print(">>> MetaSymNet...")
        meta = MetaSymNetSearcher(D, time_limit=timeout_sec)
        t_meta, stat_meta, mse_meta = run_search_with_metrics(meta, X, y, timeout_sec)
        print(f"   [Meta] Time: {t_meta:.2f}s | MSE: {mse_meta:.4f}")

        results.append({
            'Dim': D,
            'NS_Time': t_ns, 'NS_MSE': mse_ns,
            'EQL_Time': t_eql, 'EQL_MSE': mse_eql,
            'Meta_Time': t_meta, 'Meta_MSE': mse_meta
        })
        
    df = pd.DataFrame(results)
    print("\nFINAL REPORT WITH MSE")
    print(df)
    df.to_csv("scalability_mse_results.csv", index=False)

if __name__ == "__main__":
    main()