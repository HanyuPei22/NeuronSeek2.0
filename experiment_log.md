## Experiment Log

### Synthetic Data Experiment

#### V1.0

The first version of using L0gate to control and select the polynomial significant term. Not that good on searching power term(add too many terms), need stricter condition/stronger penalization on norm. Disaster on interaction term, coeffs are too small, maybe due to L0 regularization is unaffair between power and interaction term.

The hyperparameters are set as:
    LAMBDA_L0 = 1       # Sparsity penalty strength
    EPOCHS = 150          # Total training epochs
    WARMUP_EPOCHS = 70    # Epochs before enabling L0 penalty
    LR = 0.01             # Learning rate