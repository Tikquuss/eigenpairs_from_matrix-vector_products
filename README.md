# Eigenpairs from matrix–vector products

Code for the blog post
[**Eigenpairs from Matrix–Vector Products: Power Iteration, Deflation, Lanczos, and Arnoldi**](https://tikquuss.github.io/blog/eigenpairs-from-matrix-vector-products/).

The post studies what can be learned about the eigenvalues and eigenvectors of a real symmetric matrix
**A** that is known only through the map **x** ↦ **Ax**, and its last section treats nonsymmetric matrices
with the Arnoldi method. This repository contains small, readable implementations of the procedures of the
post and the scripts that produce every figure in it.

The same code also lives in the repository of the site, in
[`agent/scripts/eigenpairs-from-matrix-vector-products`](https://github.com/Tikquuss/tikquuss.github.io/tree/master/agent/scripts/eigenpairs-from-matrix-vector-products),
where it writes the figures directly into the images of the post. That copy is the one used to build the
post; this repository mirrors it as a standalone package.

## Contents

| Path | What it contains |
| --- | --- |
| `mvp/power.py` | Power iteration (Proposition 1), plain and shifted deflated power iteration (Corollary 1, Lemma 1), and Algorithm 1 (deflated squared power iteration with sign separation), with the stage loop of Corollary 2 |
| `mvp/lanczos.py` | The Lanczos recurrence with full reorthogonalization, Ritz pairs (Lemma 4), Algorithm 2 for the pairs of largest absolute value, and continuation with new starting vectors (Proposition 4) |
| `mvp/arnoldi.py` | The Arnoldi recurrence with full reorthogonalization (Definition 3), Ritz pairs with their inexpensive residuals (Proposition 5), and the boundary of the field of values (Remark 18) |
| `mvp/operators.py` | Matrices known through their products (Sections 2 and 5 of the post): the adjacency matrix of a bipartite graph, a sample covariance matrix, matrices with prescribed spectra, the transition matrix of a random walk on a directed graph, and a discretized convection–diffusion operator |
| `mvp/hessian.py` | Hessian–vector products of the training loss of a small tanh network, by automatic differentiation (PyTorch) |
| `figures/` | One script per figure of the post |
| `figures/output/` | The published figures (SVG) and, for each, a JSON file with the numbers quoted in its caption |

Every solver accesses the matrix only through a function `apply_a(x)` that returns `A @ x`.

## Figures

| In the post | File | Statement of the post | Matrix (Sections 2 and 5 of the post) | Script |
| --- | --- | --- | --- | --- |
| Figure 1 | `power-iteration.svg` | Proposition 1, Theorem 1, Remarks 2 and 5, Proposition 2 | prescribed spectra, size 60 | `figures/power_iteration.py` |
| Figure 2 | `bipartite-graph.svg` | the graph that Figure 3 decomposes (Corollary 2): drawing, adjacency matrix, symmetric spectrum | adjacency matrix of a bipartite graph | `figures/bipartite_graph.py` |
| Figure 3 | `complete-eigenbasis.svg` | Corollary 2 | adjacency matrix of a bipartite graph | `figures/complete_eigenbasis.py` |
| Figure 4 | `residual-certificate.svg` | Lemma 2 and Remark 11 | sample covariance matrix (principal components) | `figures/residual_certificate.py` |
| Figure 5 | `hessian-deflation.svg` | Algorithm 1, with the training of the network and its Hessian spectrum during training and at the last step | Hessian of a network training loss | `figures/hessian_deflation.py` |
| Figure 6 | `lanczos.svg` | Lemma 5 and Proposition 3 | prescribed spectrum with repeated eigenvalues | `figures/lanczos_ritz.py` |
| Figure 7 | `lanczos-vs-deflation.svg` | Algorithm 2 against Algorithm 1, Proposition 4 | network Hessian and bipartite graph | `figures/lanczos_vs_deflation.py` |
| Figure 8 | `directed-graph.svg` | the random walk of Section 5 that Figure 9 analyzes: drawing, transition matrix, the walker's distribution, complex spectrum | transition matrix of a random walk on a directed graph | `figures/directed_graph.py` |
| Figure 9 | `arnoldi.svg` | Proposition 5, Remark 18 and Lemma 7 (Arnoldi method), against power iteration | random walk on a directed graph, convection–diffusion operator | `figures/arnoldi_nonsymmetric.py` |
| popover of Theorem 1 | `angle-to-eigenspace.svg` | Theorem 1 (the angle between the starting vector and the dominant eigenspace) | schematic | `figures/angle_diagram.py` |

## Reproducing the figures

```bash
pip install -r requirements.txt
python make_figures.py                       # all figures, written to figures/output/
python make_figures.py --only lanczos_ritz   # a single figure
python make_figures.py --preview previews    # PNG copies as well
```

All random draws are seeded, so the outputs are deterministic for a given software stack. The ten
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

For a nonsymmetric matrix, the Arnoldi method returns complex Ritz values when they are needed:

```python
from mvp import arnoldi, directed_community_walk
from mvp.arnoldi import ritz_pairs

rng = np.random.default_rng(1)
# Random walk on a directed graph whose three groups of 20 nodes form a cycle (Section 5),
# known through products with the transpose of its transition matrix.
apply_pt, dense_p, group, edges = directed_community_walk(60, 3, 4, 1, np.random.default_rng(0))
hessenberg, basis = arnoldi(apply_pt, rng.standard_normal(60), 20)
values, vectors, residuals = ritz_pairs(hessenberg, 20)
top = np.argsort(-np.abs(values))[:3]
print(np.round(values[top], 4), np.array2string(residuals[top], precision=1))
# [ 1.    +0.j     -0.3937+0.7046j -0.3937-0.7046j] [1.6e-07 4.7e-06 4.7e-06]
```

The eigenvalue 1 belongs to the stationary distribution of the walk, and the complex pair to its circulation
through the three groups; power iteration cannot separate the two members of the pair, which have the same
modulus.

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
