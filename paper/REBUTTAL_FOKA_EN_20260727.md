# Response to Reviewer Foka

We thank the reviewer for recognizing the importance of the problem, the theoretical diagnosis, the practical design, the breadth of the evaluation, and the reproducibility package. We agree that the submission should have made the algorithm, reference selection, applicability rule, and empirical nature of the geometric premise more explicit.

## 1. Step-by-step algorithm and heterogeneous feature dimensions

The phrase “fixed reference encoder” in the submission is misleading. We fix a reference **dataset** $R$ (STL-10 in the main experiments), not one encoder shared across candidates.

For each candidate encoder $m$, the same encoder processes both the target and reference images:

$$
H_{m,T}\in\mathbb R^{n_T\times d_m},
\qquad
H_{m,R}\in\mathbb R^{n_R\times d_m}.
$$

Therefore, within each alignment computation, the two feature dimensions are identical. Different candidates may have different $d_m$, but their right-singular subspaces are never compared with one another. No common feature space or cross-encoder projection is needed.

For centered feature matrices, let

$$
H_{m,T}=U_T\Sigma_TV_T^\top,\qquad
H_{m,R}=U_R\Sigma_RV_R^\top,
$$

and $k_m=\min(k,n_T,n_R,d_m)$. The implemented alignment is

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

Here $\theta_i$ are the principal angles between the top right-singular subspaces in candidate $m$'s own $d_m$-dimensional representation space.

A compact algorithm is:

```text
Input: candidate encoder m, target T, reference R, subspace size k
1. H_T   <- centered features of T extracted by m
2. H_ref <- centered features of R extracted by m
3. V_T   <- top-k_m right singular vectors of H_T
4. V_ref <- top-k_m right singular vectors of H_ref
5. SA    <- ||V_T^T V_ref||_F^2 / k_m
6. p_i   <- sigma_i(H_T) / sum_j sigma_j(H_T)
7. r_eff <- exp(-sum_i p_i log p_i)
8. score <- (log r_eff - log SA) / log C(T)
Return: score
```

Thus the effective rank and final experimental score are

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

Only the returned scalar is compared across candidates. This is why heterogeneous feature dimensions do not require an additional projection mechanism.

## 2. Reference selection when STL-10 is unsuitable

STL-10 is a practical default, not a theoretically optimal reference. We selected it because it is small, contains diverse natural objects, can be processed by all candidates, and is disjoint from the seven non-STL VTAB target tasks.

When STL-10 is unavailable or inappropriate, we recommend the following ordered criteria:

1. **Target disjointness:** never use the evaluated target itself as its reference.
2. **No target-performance selection:** do not choose or tune the reference using target labels or target accuracy.
3. **Modality compatibility:** use the same broad modality as the deployment domain, such as natural images, medical images, or remote-sensing images.
4. **Broad rather than narrow coverage:** prefer diverse generic content over a highly task-specific dataset.
5. **Stable estimation:** include enough diverse samples to estimate the top-$k$ subspace.
6. **Universal availability:** every candidate must be able to process the reference with the same feature interface.

The reference may account for the broad nature of the downstream domain, but it must be chosen independently of downstream performance. Otherwise reference selection itself becomes a supervised model-selection procedure.

Our VTAB ablation illustrates the trade-off. SSR$^{\ddagger}$ regret is $0.066$ with STL-10, $0.098$ with CIFAR-10, and $0.051$ with the median over four references. Flowers-102 gives $0.013$, but it is itself a target task; this unusually low value reflects target/reference leakage and is not a valid advantage.

## 3. Should an ensemble be the default, and what does it cost?

One reference requires one additional unlabeled reference-side feature extraction per candidate. These features are computed once, cached, and reused across every target. If there are $M$ references, the reference-side cost and storage increase approximately $M$-fold; the target-side cost is unchanged. Computing the median alignment over cached references is negligible compared with feature extraction.

The four-reference ensemble improves VTAB regret from $0.066$ to $0.051$, but the improvement is not large enough to justify making four references mandatory in all deployments. Our recommendation is:

- default: one generic, target-disjoint reference;
- robustness-sensitive deployment: a small, predetermined ensemble, aggregated by the median;
- never include the evaluated target among its own references.

This preserves the lightweight character of the default method while providing a clear option when reference sensitivity matters.

## 4. Practical decision rule: SSR$^{\ddagger}$ or RankMe?

We agree that the submission lacked an operational decision rule. SSR$^{\ddagger}$ is not a universal replacement for RankMe.

- **Across heterogeneous architecture families:** use SSR$^{\ddagger}$. This is the setting in which width varies and RankMe is confounded.
- **Within a single architecture family:** use RankMe. Dimensions and dominant subspaces are similar, and the alignment term does not provide a stable additional signal.
- **Mixed model zoo with multiple variants per family:** use SSR$^{\ddagger}$ for family selection, followed by RankMe for within-family variant selection.

On the five-task MV multi-variant benchmark, flat SSR$^{\ddagger}$ has regret $0.098$ at mean cost $118.8$, while SSR$^{\ddagger}$-to-RankMe has regret $0.102$ at mean cost $38.0$. This supports the two-stage division of labor on MV, but we agree that five tasks constitute a proof of concept rather than a general cost-quality guarantee.

## 5. Status of the geometric premise

We agree that there is no formal bound connecting $\mathrm{SA}_k$ to linear-probe accuracy. The evidence should be described as an empirical mechanism, not a theorem about downstream performance.

On 126 VTAB architecture-task pairs, $\mathrm{SA}_k$ is strongly negatively correlated with the fraction of between-class variance lying outside the generic reference subspace ($\rho=-0.964$). That architecture-specific variance fraction, in turn, positively predicts LP accuracy. This explains why subtracting alignment can be useful, but it does not guarantee success on every pool or within every family.

## 6. Updated scope relative to DiSCO

Our audit confirms that official DiSCO uses target ground-truth labels to construct class-conditional quantities. It is therefore a label-dependent comparator, not a label-free baseline. We also reimplemented the official PCA-128 procedure and completed all missing feature and LP cells. Its corrected $G=6/G=8$ regrets are MV $0.146/0.146$, VTAB $0.054/0.043$, SSL $0.045/0.042$, and EX $0.045/0.049$.

These results show that DiSCO is stronger than the submission reported, particularly on EX. We therefore withdraw any claim that SSR$^{\ddagger}$ generally outperforms DiSCO. The fair same-label-budget comparison remains SSR$^{\ddagger}$ versus RankMe. On the 22-task common support that uniformly excludes STL-10 as a target, their regrets are $0.071$ and $0.166$, respectively; the paired difference is $-0.095$, with 95% CI $[-0.160,-0.037]$ and $p=0.0002$.

In summary, the intended claim is narrower: SSR$^{\ddagger}$ is a lightweight, parameter-free, label-free correction for cross-family width confounding. RankMe remains appropriate within a family; reference selection and the link to downstream accuracy remain empirical limitations; and the two-stage result should be interpreted only within the evaluated benchmark.
