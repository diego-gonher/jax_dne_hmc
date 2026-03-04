# Linear Model with parameter dependent covariance matrices

## Toy Model for Mean and $\theta$-Dependent Covariance Emulation

### Parameter Space

We define a 2D parameter vector:

$$
\theta = (m, b)
$$

with uniform grids:

$$
m \in [3, 8], \quad b \in [1, 4]
$$

using 10 equally spaced values per dimension (total of 100 parameter pairs).

---

### Deterministic Model (Mean Emulator Target)

We define 11 data points:

$$
x_i = \text{linspace}(0, 5, 11)
$$

The noiseless model is the linear function:

$$
\mu_i(\theta) = m x_i + b
$$

This is the target of the **mean emulator**.

---

### $\theta$-Dependent Covariance Model

We construct a covariance matrix:

$$
\Sigma(\theta) \in \mathbb{R}^{11 \times 11}
$$

designed to:

* Be symmetric positive definite (SPD)
* Depend smoothly on $\theta$
* Have stronger off-diagonal correlations for small $x$ values

---

#### 1. Global Amplitude

$$
A(\theta) = 0.10 \left( 1 + 0.20 |m| + 0.10 |b| \right)
$$

---

#### 2. Base Correlation Length

$$
\ell(\theta) = 0.60 \left( 1 + 0.30 \tanh\left(\frac{m}{5}\right) \right)
$$

---

#### 3. Low-$x$ Correlation Taper

To emphasize correlations for small $x$:

$$
w_i(\theta) = \exp\left( -\frac{x_i}{\lambda(\theta)} \right)
$$

where

$$
\lambda(\theta) = 1.0 + 0.4 \left( 1 - \tanh\left(\frac{b}{3}\right) \right)
$$

This ensures off-diagonal structure is strongest in the upper-left (small $x$ region).

---

#### 4. Stationary Kernel

$$
K_{ij}(\theta) = \exp\left(-\frac{(x_i - x_j)^2}{2 \ell(\theta)^2}\right)
$$

---

#### 5. Tapered Correlation Structure

$$
\Sigma^{\text{corr}}_{ij}(\theta)=w_i(\theta) w_j(\theta) K_{ij}(\theta)
$$

---

#### 6. Heteroscedastic Diagonal Noise

Per-bin standard deviation:

$$
\sigma_i(\theta)=\sigma_0(\theta)\left(1 + 0.25 \frac{x_i}{\max(x)}\right)
$$

with

$$
\sigma_0(\theta)
=
0.03
\left(
1 + 0.15 |m| + 0.10 |b|
\right)
$$

Diagonal contribution:

$$
\Sigma^{\text{diag}}(\theta)
=
\mathrm{diag}(\sigma_i^2)
$$

---

### Final Covariance Matrix

$$
\Sigma(\theta)
=
A(\theta)^2 \Sigma^{\text{corr}}(\theta)
+
\Sigma^{\text{diag}}(\theta)
+
\epsilon I
$$

where $\epsilon \sim 10^{-6}$ ensures numerical positive definiteness.

---

### Correlation Matrix (as in the Paper)

For visualization, we compute the correlation matrix:

$$
C_{ij}(\theta)
=
\frac{\Sigma_{ij}(\theta)}
{\sqrt{\Sigma_{ii}(\theta)\Sigma_{jj}(\theta)}}
$$

This isolates correlation structure independent of variance scaling.

---

### Data Generation

For each $\theta$:

$$
y^{(k)} \sim \mathcal{N}\left(
\mu(\theta),
\Sigma(\theta)
\right)
\quad k=1,\dots,N_{\text{MOCK}}
$$

---

### Covariance Emulator Target Representation

Instead of emulating $\Sigma$ directly, we:

1. Compute Cholesky factor:
   $$
   \Sigma = L L^\top
   $$

2. Store:

   * Lower-triangular entries
   * Log of diagonal entries

This guarantees positive definiteness when reconstructing:

$$
L_{ii} = \exp(\hat{d}_i)
$$

and

$$
\hat{\Sigma} = L L^\top
$$

---

### Purpose of This Toy Model

This construction mimics the structure of the cosmology case:

* $\theta$-dependent covariance
* Structured off-diagonal correlations
* Heteroscedastic variance
* Smooth mapping $\theta \to \Sigma(\theta)$
* Guaranteed SPD matrices

while remaining analytically controlled and computationally lightweight.
