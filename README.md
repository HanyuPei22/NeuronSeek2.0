# NeuronSeek-TD 2.0

Data-driven polynomial neuron discovery framework.

## Structure

```
NeuronSeek/
├── src/
│   ├── core/              # Core mathematical operators
│   │   ├── tensor_interaction.py    # CP decomposition for Stage 1
│   │   └── task_neuron_layers.py    # Polynomial layers for Stage 2
│   ├── models/            # Model implementations
│   │   ├── cp_polynomial.py         # Basic CP modules
│   │   ├── proxy_model.py           # Stage 1 proxy model
│   │   ├── search_agent.py          # STRidge search agent
│   │   ├── task_driven_layers.py    # Task-driven neurons
│   │   └── custom_resnet.py         # ResNet with polynomial neurons
│   └── utils/             # Utilities
│       └── feature_extractor.py     # CIFAR feature extraction
├── scripts/               # Experiment scripts
│   ├── discover_formula.py          # Stage 1: Formula discovery
│   ├── train_resnet.py              # Stage 2: Train ResNet-TN
│   └── regression_demo.py           # Regression task demo
└── experiments/           # Results and logs
```

## Usage

**Stage 1: Discover Formula**
```bash
python scripts/discover_formula.py
```

**Stage 2: Train Network**
```bash
python scripts/train_resnet.py
```

**Regression Demo**
```bash
python scripts/regression_demo.py
```
