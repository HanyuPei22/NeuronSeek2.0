# Mathematical Derivation & Workflow: Tensor Interaction Layer

## Part 1: Equivalence Proof of Implicit CP Decomposition

**Theorem:**
Calculating the interaction term via **Implicit CP Decomposition** (product of projections) is mathematically equivalent to contracting input vector $x$ with a **reconstructed high-order weight tensor** $\mathcal{W}$ derived from standard CP decomposition.

### 1. Definitions
* **Input:** $x \in \mathbb{R}^D$.
* **Target Interaction:** A $K$-th order polynomial term $y \in \mathbb{R}$.
* **Weight Tensor:** $\mathcal{W} \in \mathbb{R}^{D \times D \times \dots \times D}$ ($K$ times).
* **CP Rank:** $R$.
* **Factor Matrices:** $\{U^{(1)}, \dots, U^{(K)}\}$, where each $U^{(k)} \in \mathbb{R}^{D \times R}$. Let $\mathbf{u}^{(k)}_r$ denote the $r$-th column of matrix $U^{(k)}$.

### 2. Standard Static Form (Tensor Reconstruction)
In standard CP decomposition, the weight tensor $\mathcal{W}$ is approximated as the sum of $R$ rank-1 tensors (outer products):

$$
\mathcal{W} \approx \sum_{r=1}^{R} \mathbf{u}^{(1)}_r \circ \mathbf{u}^{(2)}_r \circ \dots \circ \mathbf{u}^{(K)}_r
$$

### 3. Functional Form (Tensor Contraction)
The output $y$ is the result of contracting the tensor $\mathcal{W}$ with the input vector $x$ along all $K$ modes:

$$
y = \mathcal{W} \times_1 x \times_2 x \dots \times_K x
$$

### 4. Derivation
Substitute the CP approximation into the contraction equation:

$$
\begin{aligned}
y &= \left( \sum_{r=1}^{R} \mathbf{u}^{(1)}_r \circ \mathbf{u}^{(2)}_r \circ \dots \circ \mathbf{u}^{(K)}_r \right) \times_1 x \dots \times_K x \\
\end{aligned}
$$

By the **distributive property** of tensor contraction over addition, and the property that contracting a rank-1 outer product results in the product of scalar dot products ($(u \circ v) \times_1 x \times_2 x = (u^Tx)(v^Tx)$):

$$
\begin{aligned}
y &= \sum_{r=1}^{R} \left( (\mathbf{u}^{(1)}_r \cdot x) \times (\mathbf{u}^{(2)}_r \cdot x) \times \dots \times (\mathbf{u}^{(K)}_r \cdot x) \right) \\
\end{aligned}
$$

This can be rewritten in matrix notation. Let $P^{(k)} = x^T U^{(k)} \in \mathbb{R}^{1 \times R}$ be the projection of $x$ onto the $k$-th factor space. The term inside the summation corresponds to the element-wise product (Hadamard product) of these projections at index $r$:

$$
y = \sum_{r=1}^{R} \left( \prod_{k=1}^{K} P^{(k)}_r \right)
$$

**Conclusion:**
This final equation matches exactly the code implementation: **Project** input to latent space $\rightarrow$ **Element-wise Product** across orders $\rightarrow$ **Sum** (via linear layer).

***

## Part 2: Mathematical Workflow (Order $K=5$)

This section details the forward pass and training dynamics for a **5th-Order** interaction term.

### 1. Hypothesis & Setup
* **Objective:** Model a global 5th-order interaction $\mathcal{I}_5(x)$.
* **Input Batch:** $X \in \mathbb{R}^{B \times D}$ ($B$: Batch size, $D$: Input dim).
* **Hyperparameters:** Rank $R$, Order $K=5$, Output Classes $C_{out}$.
* **Learnable Parameters:**
    * **Factors:** 5 matrices $\{U^{(1)}, U^{(2)}, U^{(3)}, U^{(4)}, U^{(5)}\}$, each $U^{(k)} \in \mathbb{R}^{D \times R}$.
    * **Coefficients:** $W_{int} \in \mathbb{R}^{R \times C_{out}}$.

### 2. Forward Propagation

#### Step A: Latent Projection (Dimension Reduction)
For each order $k \in \{1, \dots, 5\}$, project the input $X$ into the shared rank space using the specific factor matrix $U^{(k)}$.

$$
P^{(k)} = X U^{(k)}
$$
* **Dim Change:** $[B, D] \times [D, R] \rightarrow [B, R]$
* *Result:* We obtain a list of 5 matrices: $[P^{(1)}, P^{(2)}, P^{(3)}, P^{(4)}, P^{(5)}]$.

#### Step B: Interaction (Hadamard Product)
Compute the non-linear interaction by taking the element-wise product of all projected features along the order dimension.

$$
Z = P^{(1)} \odot P^{(2)} \odot P^{(3)} \odot P^{(4)} \odot P^{(5)}
$$
* **Operation:** $\odot$ denotes Element-wise Hadamard product.
* **Dim Change:** Collapse 5 tensors of $[B, R] \rightarrow [B, R]$.
* *Meaning:* $Z_{b,r}$ represents the strength of the $r$-th interaction pattern for sample $b$.

#### Step C: Output Mapping (Linear Combination)
Map the latent interaction features $Z$ to the final class logits.

$$
\hat{Y}_{int} = Z W_{int}
$$
* **Dim Change:** $[B, R] \times [R, C_{out}] \rightarrow [B, C_{out}]$.

### 3. Training Process

#### Loss Function
The total objective function $\mathcal{L}$ combines prediction error and regularization.

$$
\mathcal{L} = \underbrace{\text{MSE}(\hat{Y}, Y_{GT})}_{\text{Fidelity}} + \lambda \underbrace{\mathcal{R}(Gates)}_{\text{Sparsity}} + \gamma \sum_{k=1}^5 \|U^{(k)}\|_F^2
$$

#### Backpropagation (Gradient Flow)
Gradients flow backward from $\mathcal{L}$ through the chain rule.

1.  **Update Coefficients ($W_{int}$):**
    $$\frac{\partial \mathcal{L}}{\partial W_{int}} = Z^T (\hat{Y} - Y)$$
    * *Update:* Simple linear regression update.

2.  **Update Factors ($U^{(k)}$):**
    The gradient for the $k$-th factor depends on the product of **all other factors** ($j \neq k$).
    $$\frac{\partial \mathcal{L}}{\partial U^{(k)}} = X^T \left[ \left( (\hat{Y} - Y) W_{int}^T \right) \odot \left( \bigodot_{j \neq k} P^{(j)} \right) \right]$$
    * *Insight:* For $U^{(k)}$ to learn, the other projections $P^{(j)}$ must be non-zero. This creates a coupled optimization landscape (non-convex), requiring good initialization (Xavier) and warm-up to establish a direction.
