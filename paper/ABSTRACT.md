# Abstract (plain-text fallback, no math rendering required)

**Cross-Architecture Transferability is Width-Confounded: Diagnosis and Subspace Correction**

Selecting a pretrained encoder for a downstream task without retraining or labels is a
standard preprocessing step in transfer learning.  The dominant label-free score, RankMe,
was validated within architecture families and is widely applied across them.  We show that
its cross-architecture use is unsound: rankings are systematically driven by the
penultimate-layer dimension rather than representational quality, and on heterogeneous model
pools the score performs worse than uniform random selection.

We give a random-matrix explanation establishing this failure as inherent to any
rotation-invariant score, and propose Subspace-Corrected Ranking (SCR), a family of
label-free corrections that breaks rotation invariance via alignment with a fixed reference
encoder.  Across three architecturally distinct pools and 23 aggregated task units, the
proposed SSR-dagger reduces mean regret from +0.165 (RankMe) to +0.071
(95% CI [+0.045, +0.099], p < 1e-4).  Paired with a within-family score in a two-stage
pipeline, it matches this quality at one-third of the forward-pass cost.  Five labelled
architecture evaluations suffice to match the best fully label-free strategy; K=35 labelled
evaluations reduce regret to +0.018 --- a 5.7x improvement at the same forward-pass cost.

Code: https://anonymous.4open.science/r/sdg-framework

---

## Key numbers (no-LaTeX summary)

| Setting                        | Score          | Mean regret | 95% CI             |
|-------------------------------|----------------|-------------|--------------------|
| MV pool flat (132v x 5t)       | RankMe         | +0.319      | [+0.149, +0.509]   |
| MV pool flat (132v x 5t)       | SSR-dagger     | +0.098      | [+0.048, +0.148]   |
| Cross-pool flat (23 tasks)     | RankMe         | +0.165      | —                  |
| Cross-pool flat (23 tasks)     | SSR-dagger     | +0.071      | [+0.045, +0.099]   |
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
