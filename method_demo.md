# NeuronSeek-TD 2.0: Methodological Framework

## 1. Overview

NeuronSeek-TD 2.0 is a framework designed to discover the optimal mathematical structure (aggregation function) for artificial neurons in a data-driven manner. Unlike traditional Neural Architecture Search (NAS) that searches for network topology, NeuronSeek focuses on the **microscopic level**: determining whether a neuron should perform a linear combination ($\sum w_i x_i$) or a high-order polynomial interaction (e.g., $\sum w_{ij} x_i x_j$).

The framework operates in two distinct stages:

1.  **Stage 1 (Neuronal Formula Discovery):** Uses a shallow, interpretable proxy model based on Tensor Decomposition to identify significant polynomial terms from data.
2.  **Stage 2 (Network Construction):** Constructs a deep neural network (e.g., ResNet) using the discovered aggregation functions to replace standard linear/convolutional layers.



## 2. Stage 1: Neuronal Formula Discovery

In this stage, we aim to approximate the optimal target mapping $F(\mathbf{x})$ using a high-order polynomial expansion. To avoid the combinatorial explosion of parameters in high-order tensors, we employ **CP Decomposition (Canonical Polyadic Decomposition)** with **Implicit Contraction**.

### 2.1 The Mathematical Core: Implicit Tensor Contraction

For a polynomial of order $N$, instead of maintaining a dense weight tensor $\mathcal{W} \in \mathbb{R}^{D \times \dots \times D}$, we decompose it into $N$ factor matrices $\mathbf{U}^{(1)}, \dots, \mathbf{U}^{(N)}$, where each $\mathbf{U} \in \mathbb{R}^{D \times R}$. Here, $D$ is the input dimension and $R$ is the Rank.

#### 2.1.1 The "Curse of Dimensionality" in Full Tensors
Consider an input vector $\mathbf{x} \in \mathbb{R}^D$. A full $N$-th order polynomial interaction term is defined as the contraction between the input $\mathbf{x}$ and a weight tensor $\mathcal{W}^{[N]}$:

$$
y = \mathcal{W}^{[N]} \times_1 \mathbf{x} \times_2 \mathbf{x} \dots \times_N \mathbf{x} = \sum_{i_1=1}^{D} \dots \sum_{i_N=1}^{D} \mathcal{W}_{i_1, \dots, i_N} \cdot x_{i_1} \dots x_{i_N}
$$

The number of parameters scales as $O(D^N)$. For $D=512, N=3$, this requires $\approx 1.34 \times 10^8$ parameters, which is computationally intractable.

#### 2.1.2 CP Decomposition and Implicit Calculation
We approximate the weight tensor as a sum of $R$ rank-one tensors:
$$
\mathcal{W}^{[N]} \approx \sum_{r=1}^{R} \mathbf{u}_1^{(r)} \circ \mathbf{u}_2^{(r)} \circ \dots \circ \mathbf{u}_N^{(r)}
$$

By substituting this into the polynomial equation and utilizing the **distributive property** of tensor contraction, we derive the implicit calculation form:

$$
y \approx \sum_{r=1}^{R} \underbrace{\left( (\mathbf{u}_1^{(r)} \circ \dots \circ \mathbf{u}_N^{(r)}) \times_1 \mathbf{x} \dots \times_N \mathbf{x} \right)}_{\text{Term}_r}
$$

$$
\text{Term}_r = (\mathbf{x}^\top \mathbf{u}_1^{(r)}) \cdot (\mathbf{x}^\top \mathbf{u}_2^{(r)}) \cdots (\mathbf{x}^\top \mathbf{u}_N^{(r)})
$$

This derivation proves that the high-order interaction can be computed by **projecting** the input onto factor vectors and then **multiplying** the scalar results.

### 2.2 Algorithm Implementation: Projection & Interaction

Based on the derivation above, we formulate the forward pass using matrix operations. This process consists of two steps:

1.  **Linear Projection (Dimensionality Reduction):**
    We project the input $\mathbf{x} \in \mathbb{R}^{B \times D}$ onto the factor matrices $\mathbf{U}^{(n)} \in \mathbb{R}^{D \times R}$ for each order $n$.
    $$
    \mathbf{P}_n = \mathbf{x} \mathbf{U}^{(n)} \quad \in \mathbb{R}^{B \times R}
    $$

2.  **Interaction via Hadamard Product (Non-linearity):**
    We compute the element-wise product of the projected features across all orders. This step generates the polynomial cross-terms implicitly within the latent rank space.
    $$
    \mathbf{H} = \mathbf{P}_1 \odot \mathbf{P}_2 \odot \dots \odot \mathbf{P}_n \quad \in \mathbb{R}^{B \times R}
    $$
    *Result:* The matrix $\mathbf{H}$ contains $R$ independent high-order feature channels.

---

### 2.3 Task-Specific Workflows

The handling of the extracted interaction features $\mathbf{H}$ differs fundamentally between classification and regression tasks to balance **Information Capacity** vs. **Parsimony**.

#### Path A: Classification (e.g., CIFAR-100)
* **Goal:** Discriminative Feature Extraction.
* **Strategy:** **Bottleneck Representation**. We treat the Rank $R$ as the "width" of a hidden layer. We must preserve the rank dimension to maintain sufficient information bandwidth for separating classes.
* **Hyperparameters:** High Rank ($R \approx 32 \sim 128$).

**Data Flow:**
1.  **Input:** $\mathbf{x} \in \mathbb{R}^{B \times D}$.
2.  **Interaction:** Compute $\mathbf{H} \in \mathbb{R}^{B \times R}$ via Hadamard product. **Do not sum over $R$.**
3.  **Mapping to Logits:** Use a learnable coefficient matrix $\mathbf{K} \in \mathbb{R}^{R \times C}$ (where $C$ is num\_classes).
    $$
    \mathbf{Y}_{logits} = \mathbf{H} \cdot \mathbf{K} + \mathbf{b}
    $$
    *(Dimensions: $[B, R] \times [R, C] \to [B, C]$)*
4.  **Significance Metric:** The Frobenius norm of the matrix $\mathbf{K}$ ($\|\mathbf{K}\|_F$) determines the importance of this polynomial order.

#### Path B: Regression (e.g., Equation Discovery)
* **Goal:** Scalar Function Approximation / Physical Law Discovery.
* **Strategy:** **Parsimony & Reduction**. We assume physical laws are sparse. We aggregate the rank information into a single scalar intensity.
* **Hyperparameters:** Low Rank ($R \approx 3 \sim 5$).

**Data Flow:**
1.  **Input:** $\mathbf{x} \in \mathbb{R}^{B \times D}$.
2.  **Interaction:** Compute $\mathbf{H} \in \mathbb{R}^{B \times R}$.
3.  **Reduction (Summation):** Collapse the rank dimension to obtain scalar intensity.
    $$
    \mathbf{t} = \sum_{r=1}^{R} \mathbf{H}_{:, r}
    $$
    *(Dimensions: $[B, R] \to [B, 1]$)*
4.  **Scalar Mapping:** Use a scalar coefficient $c \in \mathbb{R}$.
    $$
    \mathbf{y} = \mathbf{t} \cdot c + b
    $$
5.  **Significance Metric:** The absolute value of the scalar $|c|$.

---

### 2.4 Automatic Term Selection (STRidge)

To identify the optimal formula (e.g., deciding whether to keep $x^2$ or $x^3$), we employ **Sequential Threshold Ridge regression (STRidge)**.

**Algorithm:**
1.  **Train** the proxy model with $L_1$ regularization on coefficients ($\mathbf{K}$ or $c$).
2.  **Evaluate** the Importance Score ($\text{Score}_n$) for each polynomial order $n$.
3.  **Hard Pruning:** If $\text{Score}_n < \tau$ (dynamic threshold), set coefficients and their gradients to 0.
4.  **Fine-tune** the remaining active terms to recover accuracy.

---

## 3. Stage 2: Task-Driven Network Construction

Once the optimal polynomial structure is discovered (e.g., "The task requires $x$ and $x^2$ terms"), we instantiate a deep neural network where standard neurons are replaced by **Task-Driven Neurons**.

### 3.1 The Task-Driven Neuron Module

If Stage 1 identifies that orders $\mathcal{S} = \{1, 2\}$ are significant, we replace standard layers (Linear or Conv2d) with a composite layer defined as:

$$
\mathbf{y} = \text{Norm}(\text{Layer}_1(\mathbf{x})) + \text{Norm}(\text{Layer}_2(\Phi(\mathbf{x})))
$$

* **Explicit Calculation ($\Phi(\mathbf{x})$):**
    * For **Regression-like simple features**: We use element-wise power $\mathbf{x}^2$.
    * For **Classification-like complex features**: We use a low-rank bottleneck layer (instantiating the CP structure found in Stage 1) to capture cross-terms $x_i x_j$.
* **Normalization:** High-order terms (e.g., $x^2, x^3$) can cause gradient instability. We explicitly apply **Batch Normalization (BN)** after each polynomial projection to unify feature scales.

### 3.2 Integration with Backbones

* **ResNet:** Replace the $3 \times 3$ Conv in the residual block.
    * *Original:* $y = \text{Conv}(\mathbf{x})$
    * *Task-Driven:* $y = \text{Conv}_1(\mathbf{x}) + \text{Conv}_2(\mathbf{x}^2)$
* **MLP-Mixer / ViT:** Replace the Feed-Forward Network (FFN) linear layers to introduce non-linear inductive bias.