# Response to Reviewer ek3T

We thank the reviewer for recognizing the practical importance of the problem, the width-confounding diagnosis, the thorough empirical validation, and the originality of using subspace geometry as a label-free correction. We agree that the submission did not define the alignment term clearly enough and did not provide adequate guidance for reference selection or the method's scope.

## 1. Clarification: we fix a reference dataset, not a reference encoder

The phrase “fixed reference encoder” in the submission is misleading. We fix a reference **dataset** $R$, while each candidate encoder processes both the target and reference datasets.

For candidate encoder $m$, we extract centered feature matrices

$$
H_{m,T}\in\mathbb R^{n_T\times d_m},
\qquad
H_{m,R}\in\mathbb R^{n_R\times d_m},
$$

where $T$ is the target task and $R$ is STL-10 in the main experiments. Because both matrices are produced by the same candidate $m$, their feature dimensions are identical. We do not align a candidate's features with features from a separate generic encoder.

This distinction is important for interpreting the method. The reference features describe which directions candidate $m$ uses for generic visual variation; the target features describe which directions the same candidate uses on the downstream data. Their overlap is measured within $m$'s own representation space.

## 2. Formal definition of $\mathrm{SA}_k$

Let

$$
H_{m,T}=U_T\Sigma_TV_T^\top,\qquad
H_{m,R}=U_R\Sigma_RV_R^\top,
$$

and set $k_m=\min(k,n_T,n_R,d_m)$. The implemented alignment is

$$
\boxed{
\mathrm{SA}_k(H_{m,T},H_{m,R})
=
\frac{1}{k_m}
\left\|V_{T,1:k_m}^{\top}V_{R,1:k_m}\right\|_F^2
=
\frac{1}{k_m}\sum_{i=1}^{k_m}\cos^2\theta_i
}.
$$

Here $\theta_i$ are the principal angles between the top right-singular subspaces. The value lies in $[0,1]$: a larger value means that the candidate uses more similar dominant feature directions on the target and generic reference data.

The complete computation for each candidate is:

1. extract and center target and reference features using that candidate;
2. compute the two SVDs;
3. retain the top $k_m$ right singular vectors;
4. compute the mean squared principal-angle cosine above;
5. compute RankMe effective rank on the target features; and
6. return the scalar SSR score.

The RankMe-consistent effective rank and the score used in the experiments are

$$
p_i=\frac{\sigma_i}{\sum_j\sigma_j},
\qquad
r_{\mathrm{eff}}(H)
=
\exp\left(-\sum_i p_i\log p_i\right),
$$

$$
\boxed{
\mathrm{SSR}^{\ddagger}(H,T)
=
\frac{\log r_{\mathrm{eff}}(H)
-\log\mathrm{SA}_k(H,H_{\mathrm{ref}})}
{\log C(T)}
}.
$$

Only the resulting scalar score is compared across heterogeneous candidates.

## 3. Criteria for selecting the reference dataset

We agree that reference selection was not principled enough in the submission. STL-10 is a practical default, not a theoretically optimal reference. We selected it because it is small, contains generic natural objects, is available to all candidate encoders, and is disjoint from the seven non-STL VTAB targets.

Our recommended criteria are:

- **target disjointness:** the reference must not be the evaluated target dataset;
- **no target-performance feedback:** neither target labels nor target accuracy should be used to select it;
- **broad visual coverage:** generic visual content is preferable to a narrow task-specific domain;
- **stable subspace estimation:** it should contain enough diverse samples to estimate the top-$k$ directions; and
- **low reusable cost:** reference features should be inexpensive to extract once and cache across targets.

The reviewer suggests accounting for the downstream domain. We agree with this direction, with an important constraint: domain adaptation of the reference must not use target labels or select the reference by observing target performance. A safe approach is to choose, before evaluation, a target-disjoint reference from the same broad modality (e.g., natural objects, medical images, or remote sensing), or to use a predetermined multi-reference ensemble. Learning or tuning the reference using target accuracy would turn reference selection into a supervised model-selection step.

Our ablation illustrates both the opportunity and the leakage risk. On VTAB, STL-10 gives SSR$^{\ddagger}$ regret $0.066$, CIFAR-10 gives $0.098$, and a four-reference median gives $0.051$. Flowers-102 gives $0.013$, but Flowers-102 is itself a target task; this unusually strong result reflects in-domain leakage and should not be used as evidence for the method.

One reference requires one additional unlabeled feature-extraction pass per candidate, after which its features are cached and reused for every target. An $M$-reference ensemble requires $M$ reference-side passes; target-side cost is unchanged. We therefore recommend one generic, target-disjoint reference as the default and a small predetermined ensemble only when additional robustness justifies its cost.

## 4. Single-family degeneration and intended applicability

We agree that SSR$^{\ddagger}$ does not provide a stable additional signal within a single architecture family. This is a real limitation. Within-family variants often have the same feature dimension and similar top-$k$ subspaces, so the alignment term has little discriminative variation.

The evidence supports a narrower operational role:

- use SSR$^{\ddagger}$ to compare heterogeneous architecture families;
- use RankMe within one family; and
- in a mixed model zoo, use SSR$^{\ddagger}$ to select the family and RankMe to select a variant inside it.

On the five-task MV multi-variant benchmark, SSR$^{\ddagger}$-to-RankMe achieves regret $0.102$, compared with $0.098$ for flat SSR$^{\ddagger}$, while reducing mean forward evaluations from $118.8$ to $38.0$. This supports the feasibility of this division of labor on MV, but it is not a general guarantee for arbitrary model zoos.

## 5. SSL, DiSCO, and the scope of the significance claim

We agree that the practical advantage is pool-dependent and that the submitted DiSCO comparison requires correction.

Our protocol audit confirms that official DiSCO uses **target ground-truth labels** to form class-conditional covariance and posterior quantities. It is therefore a label-dependent comparator, not a label-free baseline. The fair label-free comparison is SSR$^{\ddagger}$ versus RankMe.

We also found that our earlier DiSCO implementation was not faithful to the official class-conditional Mahalanobis-posterior procedure. After implementing the official method with PCA-128, recovering all failed feature extractions, and filling all missing LP cells, its corrected $G=6/G=8$ regrets are:

| Pool | $G=6$ | $G=8$ |
|---|---:|---:|
| MV | 0.146 | 0.146 |
| VTAB | 0.054 | 0.043 |
| SSL | 0.045 | 0.042 |
| EX | 0.045 | 0.049 |

Thus DiSCO is stronger than the submission reported, especially on EX, and remains highly competitive on SSL and VTAB. We withdraw the claim that SSR$^{\ddagger}$ generally outperforms DiSCO and do not use the old non-significant cross-pool comparison.

The central same-label-budget result remains supported. After uniformly excluding STL-10 as a target because it is the reference, the common support contains

$$
5\ \mathrm{MV}+7\ \mathrm{VTAB}+10\ \mathrm{EX}=22
$$

task units. SSR$^{\ddagger}$ has mean regret $0.071$, compared with $0.166$ for RankMe; their paired difference is $-0.095$, with 95% CI $[-0.160,-0.037]$ and $p=0.0002$.

In summary, the defensible contribution is narrower than some wording in the submission suggested: RankMe is strongly width-confounded in heterogeneous architecture pools, and SSR$^{\ddagger}$ is a lightweight, parameter-free, label-free correction for cross-family selection. It is not a universal replacement for RankMe, reference selection remains an empirical design choice, and we do not claim superiority over label-dependent DiSCO.
