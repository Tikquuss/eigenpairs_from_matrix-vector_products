# Eigenpairs from matrix–vector products

Code for the blog post
[**Eigenpairs from Matrix–Vector Products: Power Iteration, Deflation, and Lanczos**](https://tikquuss.github.io/blog/eigenpairs-from-matrix-vector-products/).

The post studies what can be learned about the eigenvalues and eigenvectors of a real symmetric matrix
**A** that is known only through the map **x** ↦ **Ax**. This repository contains small, readable
implementations of the procedures of the post and the scripts that produce every figure in it.

## Contents

| Path | What it contains |
| --- | --- |
| `mvp/power.py` | Power iteration (Proposition 1), plain and shifted deflated power iteration (Corollary 1, Lemma 1), and Algorithm 1 (deflated squared power iteration with sign separation), with the stage loop of Corollary 2 |
| `mvp/lanczos.py` | The Lanczos recurrence with full reorthogonalization, Ritz pairs (Lemma 4), Algorithm 2 for the pairs of largest absolute value, and continuation with new starting vectors (Proposition 4) |
| `mvp/operators.py` | Matrices known through their products (Section 2 of the post): the adjacency matrix of a bipartite graph, a sample covariance matrix, and matrices with prescribed spectra |
| `mvp/hessian.py` | Hessian–vector products of the training loss of a small tanh network, by automatic differentiation (PyTorch) |
| `figures/` | One script per figure of the post |
| `figures/output/` | The published figures (SVG) and, for each, a JSON file with the numbers quoted in its caption |

Every solver accesses the matrix only through a function `apply_a(x)` that returns `A @ x`.

## Figures

| Figure | Statement of the post | Matrix (Section 2 of the post) | Script |
| --- | --- | --- | --- |
| `power-iteration.svg` | Proposition 1, Remarks 2, 3 and 5, Proposition 2 | prescribed spectra, size 60 | `figures/power_iteration.py` |
| `complete-eigenbasis.svg` | Corollary 2 | adjacency matrix of a bipartite graph | `figures/complete_eigenbasis.py` |
| `residual-certificate.svg` | Lemma 2 and Remark 11 | sample covariance matrix (principal components) | `figures/residual_certificate.py` |
| `hessian-deflation.svg` | Algorithm 1 | Hessian of a network training loss | `figures/hessian_deflation.py` |
| `lanczos.svg` | Lemma 5 and Proposition 3 | prescribed spectrum with repeated eigenvalues | `figures/lanczos_ritz.py` |
| `lanczos-vs-deflation.svg` | Algorithm 2 against Algorithm 1, Proposition 4 | network Hessian and bipartite graph | `figures/lanczos_vs_deflation.py` |

## Reproducing the figures

```bash
pip install -r requirements.txt
python make_figures.py                       # all figures, written to figures/output/
python make_figures.py --only lanczos_ritz   # a single figure
python make_figures.py --preview previews    # PNG copies as well
```

All random draws are seeded, so the outputs are deterministic for a given software stack. The six
figures take about three minutes on a laptop CPU, most of it for the network Hessian.

## Using the solvers

```python
import numpy as np
from mvp import bipartite_graph, certify_top_k, deflated_eigenpairs

rng = np.random.default_rng(0)
# Adjacency matrix of a bipartite graph on 30 + 20 nodes, applied through its edge list;
# the dense copy is only used here to set the tolerance.
apply_w, dense, edges = bipartite_graph((15, 15), (10, 10), 0.45, 0.08, rng)
tolerance = dict(eps_abs=1e-10 * np.abs(np.linalg.eigvalsh(dense)).max(), eps_rel=0.0)

# Algorithm 1: the 6 eigenpairs of largest absolute value, one deflation stage each.
stages = deflated_eigenpairs(apply_w, 50, 6, rng, **tolerance)
print([round(stage.eigenvalue, 4) for stage in stages], [stage.cost for stage in stages])
# [7.0658, -7.0658, -5.207, 5.207, 3.5323, -3.5323] [74, 78, 64, 60, 226, 254]

# Algorithm 2: the same pairs from one Lanczos run.
first_step, values, residuals, vectors = certify_top_k(apply_w, rng.standard_normal(50), 6, 50, **tolerance)
print(np.round(values, 4), first_step[-1] + 6, residuals.max())
# [-7.0658  7.0658  5.207  -5.207   3.5323 -3.5323] 44 3.7e-10
```

Algorithm 1 uses 756 products for the six pairs, Algorithm 2 uses 44 (38 Lanczos steps and one
explicit check per pair). The tie between each eigenvalue and its opposite, which stalls ordinary power
iteration, is resolved by the sign separation of Algorithm 1.

## A note on tolerances in Algorithm 1

For a candidate **q** orthogonal to the stored vectors (the columns of **Q**), the residual for **A**
splits into two orthogonal parts:

‖**Aq** − λ̂**q**‖² = ‖**Bq** − λ̂**q**‖² + ‖**Q**ᵀ**Aq**‖², with **B** = **PAP**.

The inner iteration drives the first part to zero, but the second part is inherited from the residuals
of the stored vectors, and nothing in the inner iteration makes it small. If the threshold of a later
stage is smaller than this inherited part, the stage never stops. With the relative test ε_abs + ε_rel‖**Aq**‖, the threshold
shrinks with |λ|, and this happened in most runs of `hessian-deflation.svg`. The scripts therefore use a
constant tolerance (ε_rel = 0, ε_abs = 10⁻¹⁰‖**A**‖), or, for a complete eigenbasis, run each stage until
the residual for **B** is below 10⁻¹³ before checking the residual for **A**.
