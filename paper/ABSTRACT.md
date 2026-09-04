# Abstract (plain-text fallback, no math rendering required)

**Cross-Architecture Transferability is Width-Confounded: Diagnosis and Subspace Correction**

Selecting a pretrained encoder for a downstream task without retraining or labels is a
standard preprocessing step in transfer learning.  The dominant label-free score, RankMe,
was validated within architecture families and is widely applied across them.  We show that
its cross-architecture use is unsound: rankings are systematically driven by the
penultimate-layer dimension rather than representational quality, and on heterogeneous model
pools the score performs worse than uniform random selection.

Under an isotropic random-feature model, we show specifically that RankMe effective rank
scales as Theta(d). We propose Subspace-Corrected Ranking (SCR), which adds the relative
geometry between each candidate's target features and its features on a fixed reference
dataset. On a common support of 22 pool-task units, SSR-dagger reduces mean regret from
0.169 (RankMe) to 0.066; the paired difference is -0.1025 (target-name-clustered 95% CI
[-0.164, -0.039], p=0.00114). SSR-dagger is intended for cross-family selection, whereas
RankMe remains preferable within a family. A five-task two-stage result is reported as a
proof of concept, not a general cost guarantee.

Code: https://anonymous.4open.science/r/crossarchitecturetransferability-FEC8

---

## Key numbers (no-LaTeX summary)

| Setting                        | Score          | Mean regret | 95% CI             |
|-------------------------------|----------------|-------------|--------------------|
| MV pool flat (132v x 5t)       | RankMe         | +0.319      | [+0.149, +0.509]   |
| MV pool flat (132v x 5t)       | SSR-dagger     | +0.098      | [+0.048, +0.148]   |
| Common support (22 pool-task units) | RankMe     | +0.169      | —                  |
| Common support (22 pool-task units) | SSR-dagger | +0.066     | paired difference CI [-0.164, -0.039] |
| MV pool two-stage (3.1x cheaper) | SSR-dagger -> RankMe | +0.102 | [+0.044, +0.153] |
| Mixed budget K=5 (label-free equiv.) | H-score -> RankMe | ~0.089 | —          |
| Mixed budget K=35              | H-score -> RankMe | +0.018   | —                  |

## Notation guide

| LaTeX symbol     | Plain-text equivalent       | Meaning                                  |
|------------------|-----------------------------|------------------------------------------|
| r_eff(H)         | reff(H)                     | Spectral effective rank of feature matrix|
| SA_k(H, H_ref)   | SA_k(H, H_ref)              | Top-k subspace alignment (principal angles)|
| SSR-dagger       | SSR-dagger                  | [log reff - log SA_k] / log C            |
| rho_Sp           | Spearman rho                | Spearman rank correlation coefficient    |
| SCR              | SCR                         | Subspace-Corrected Ranking (score family)|
| regret           | regret = oracle LP - strategy LP | Gap to oracle linear-probe accuracy |

## One-sentence summary

Label-free transferability scores are secretly ranking pretrained models by output
dimension rather than quality; a subspace alignment correction (SSR-dagger) fixes this
and halves selection error across three heterogeneous model pools.
