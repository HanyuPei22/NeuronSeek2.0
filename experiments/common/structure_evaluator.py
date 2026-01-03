import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np

class StructuralProbe(nn.Module):
    """
    Strict Structural Probe (Fixed Rank, Active Orders Only).
    
    CORRECTED LOGIC:
    1. Rank is a HYPERPARAMETER (read from structure_info['rank']).
    2. No Projection Matrix (W_int). Aggregation is done via SUM.
    3. Parallel Instantiation: Parameters are shaped [D, R, C].
    """
    def __init__(self, input_dim, structure_info, num_classes=1):
        super().__init__()
        self.structure = structure_info
        self.num_classes = num_classes
        self.mode = 'linear_subset' # Default fallback
        
        type_ = structure_info.get('type', 'explicit_terms')
        
        # ======================================================================
        # Mode A: NeuronSeek (Strict CP, No Head)
        # ======================================================================
        if type_ == 'neuronseek':
            self.mode = 'neuronseek'

            self.rank = structure_info.get('rank', 8) 
            self.p_ords = sorted(structure_info.get('pure_indices', []))
            self.i_ords = sorted(structure_info.get('interact_indices', []))
            
            # Instantiate Interaction Stream 
            # Structure: Map order -> List[Tensor[D, R, C]]
            self.interact_modules = nn.ModuleDict()
            for order in self.i_ords:

                factors = nn.ParameterList([
                    nn.Parameter(torch.randn(input_dim, self.rank, num_classes) * 0.02) 
                    for _ in range(order)
                ])
                self.interact_modules[str(order)] = factors

            # Instantiate Pure Stream 
            # Structure: Map order -> Tensor[D, C]
            self.pure_modules = nn.ParameterDict()
            for order in self.p_ords:
                self.pure_modules[str(order)] = nn.Parameter(
                    torch.randn(input_dim, num_classes) * 0.02
                )
            
            # Global bias [C]
            self.bias = nn.Parameter(torch.zeros(num_classes))

        # ======================================================================
        # Mode B: Explicit Terms (EQL / Baseline)
        # ======================================================================
        elif type_ == 'explicit_terms':

            terms = structure_info.get('terms', [])
            if terms and ('gplearn_raw' in terms[0].get('type', '') or 'raw_formula' in structure_info):
                 self.mode = 'symbolic_scaling'
                 self.scale = nn.Linear(1, 1) 
                 return

            self.active_indices = []
            for term in terms:
                self.active_indices.extend(term.get('indices', []))
            
            self.active_indices = sorted(list(set(self.active_indices)))
            if not self.active_indices: self.active_indices = [0]
            
            self.register_buffer('selected_idxs', torch.tensor(self.active_indices, dtype=torch.long))
            # Baseline uses Linear Regression
            self.linear = nn.Linear(len(self.active_indices), num_classes) 
            self.mode = 'linear_subset'

    def forward(self, x, formula_pred=None):
        # 1. NeuronSeek Logic
        if self.mode == 'neuronseek':
            batch_size = x.size(0)
            # Initialize with bias: [Batch, C]
            output = self.bias.unsqueeze(0).expand(batch_size, -1).clone()
            
            # --- Active Pure Orders ---
            for order_str, weight in self.pure_modules.items():
                order = int(order_str)
                term = x if order == 1 else x.pow(order)
                # [B, D] @ [D, C] -> [B, C]
                output = output + (term @ weight)

            # --- Active Interaction Orders (Strict CP) ---
            for order_str, factors in self.interact_modules.items():
                # Step 1: Parallel Projection [Batch, Rank, Class]
                # Same einsum logic as the Searcher
                projections = [torch.einsum('bd, drc -> brc', x, u) for u in factors]
                
                # Step 2: Product
                combined = projections[0]
                for p in projections[1:]:
                    combined = combined * p
                
                # Step 3: Direct Summation (NO Projection Matrix)
                # [Batch, Rank, Class] -> [Batch, Class]
                output = output + torch.sum(combined, dim=1)
            
            return output
            
        # ... (Baseline logic) ...
        elif self.mode == 'linear_subset':
            x_sel = torch.index_select(x, 1, self.selected_idxs)
            return self.linear(x_sel)
        elif self.mode == 'symbolic_scaling':
            if formula_pred is None: return torch.zeros(x.shape[0], 1).to(x.device)
            return self.scale(formula_pred)
            
        return torch.zeros(x.shape[0], 1).to(x.device)

def retrain_and_evaluate(searcher, structure_info, X_train, y_train, X_test, y_test, epochs=50):
    """
    Retrains the discovered structure.
    Correctly handles 'num_classes' for parallel instantiation.
    """
    input_dim = X_train.shape[1]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. Determine Num Classes (Parallel Neurons)
    # If y is [Batch, 1] or [Batch], it's 1 class (Regression).
    # If y is [Batch, C] (One-hot), it's C classes.
    # For standard classification using Indices, we usually assume output dim matches unique labels.
    # Here we assume y_train is pre-processed to be consistent.
    
    # Simple Heuristic:
    if y_train.ndim > 1 and y_train.shape[1] > 1:
         num_classes = y_train.shape[1]
         # Classification task logic
         loss_fn = nn.MSELoss() # Or BCEWithLogitsLoss if binary multi-label
    else:
         num_classes = 1
         # Regression task logic
         yt_train = torch.tensor(y_train, dtype=torch.float32).reshape(-1, 1).to(device)
         yt_test = torch.tensor(y_test, dtype=torch.float32).reshape(-1, 1).to(device)
         loss_fn = nn.MSELoss()

    Xt_train = torch.tensor(X_train, dtype=torch.float32).to(device)
    Xt_test = torch.tensor(X_test, dtype=torch.float32).to(device)
    
    # 2. Build Probe with Hyperparameter Rank & Num Classes
    try:
        probe = StructuralProbe(
            input_dim, 
            structure_info, 
            num_classes=num_classes
        ).to(device)
    except Exception as e:
        print(f"    [Probe Error] {e}")
        return 999.0

    # ... (Symbolic Pre-calc logic same as before) ...
    sym_train = None
    sym_test = None
    if probe.mode == 'symbolic_scaling':
         # ... (保持之前的符号计算逻辑) ...
         pass

    # 3. Training Loop
    optimizer = optim.Adam(probe.parameters(), lr=0.01)
    
    probe.train()
    # Ensure dataset y aligns with num_classes logic
    if num_classes == 1:
        dataset = TensorDataset(Xt_train, yt_train)
    else:
        # If classification, ensure y is float for MSE or Long for CE. 
        # For this generic probe, we stick to MSE/Float to verify 'fitting' capability.
        target_t = torch.tensor(y_train, dtype=torch.float32).to(device)
        dataset = TensorDataset(Xt_train, target_t)
        
    loader = DataLoader(dataset, batch_size=128, shuffle=True)

    for _ in range(epochs):
        for bx, by in loader:
            optimizer.zero_grad()
            if probe.mode == 'symbolic_scaling':
                pass # ...
            else:
                pred = probe(bx)
                loss = loss_fn(pred, by)
                loss.backward()
                optimizer.step()
                
        # Symbolic full batch update...

    # 4. Final Evaluation
    probe.eval()
    with torch.no_grad():
        if probe.mode == 'symbolic_scaling':
            # ...
            pass
        else:
            pred = probe(Xt_test)
        
        # Calculate Metric
        if num_classes == 1:
            mse = loss_fn(pred, yt_test).item()
        else:
            target_test = torch.tensor(y_test, dtype=torch.float32).to(device)
            mse = loss_fn(pred, target_test).item()
            
    return mse