# Author Response to the Initial Meta-Review

We thank the AC and reviewers for identifying several places where the submission was insufficiently precise. We performed an internal audit of the definitions, code, cached features, and evaluation accounting. Below we distinguish corrections from claims that remain supported. These responses clarify the submitted work; we do not rely on changes to the submission or supplementary material.

## 1. Exact alignment computation and feature dimensions

The phrase “fixed reference encoder” in the submission was incorrect. We fix a reference **dataset**, not one encoder shared by all candidates.

For every candidate encoder $m$, we separately extract

$$
H_{m,T}\in\mathbb R^{n_T\times d_m},
\qquad
H_{m,R}\in\mathbb R^{n_R\times d_m},
$$

where $T$ is the target and $R$ is STL-10. Both matrices are produced by the same encoder $m$, so their feature dimensions are identical. We never compare right-singular subspaces belonging to different encoders.

Let

$$
H_{m,T}=U_T\Sigma_TV_T^\top,\qquad
H_{m,R}=U_R\Sigma_RV_R^\top,
$$

and $k_m=\min(k,n_T,n_R,d_m)$. The implemented alignment is

$$
\mathrm{SA}_k(H_{m,T},H_{m,R})
=
\frac{1}{k_m}
\left\|V_{T,1:k_m}^{\top}V_{R,1:k_m}\right\|_F^2
=
\frac{1}{k_m}\sum_{i=1}^{k_m}\cos^2\theta_i.
$$

The complete algorithm is:

1. extract target and STL-10 features with the same candidate encoder;
2. center the feature matrices and compute their SVDs;
3. retain the top $k_m$ right singular vectors;
4. compute the mean squared principal-angle cosine above;
5. compute RankMe effective rank on the target features; and
6. return the scalar SSR score.

Different candidates may have different $d_m$, but only their final scalar scores are ranked. No cross-encoder projection or dimension matching is needed.

## 2. Effective rank and the score actually used

The audit confirms that effective rank is consistently defined as in RankMe:

$$
p_i=\frac{\sigma_i}{\sum_j\sigma_j},
\qquad
r_{\mathrm{eff}}(H)
=
\exp\left(-\sum_i p_i\log p_i\right).
$$

The score used to generate the main SSR results is

$$
\boxed{
\mathrm{SSR}^{\ddagger}(H,T)
=
\frac{\log r_{\mathrm{eff}}(H)
-\log\mathrm{SA}_k(H,H_{\mathrm{ref}})}
{\log C(T)}
}.
$$

Reviewer Vfak is correct that the submitted general SCR equation did not algebraically recover this displayed SSR expression. This is a presentation error. A consistent family is

$$
\mathrm{SCR}(H,T;\beta,g)
=
\frac{\log r_{\mathrm{eff}}(H)
+\beta g(\mathrm{SA}_k)}
{\log C(T)}.
$$

Setting $g(s)=-\log s$ and $\beta=1$ recovers SSR$^{\ddagger}$ exactly. For ranking candidates on one target, the common positive factor $1/\log C(T)$ does not change the selected model.

## 3. Reference choice, leakage, and cost

STL-10 is a practical default, not a theoretically optimal reference. We chose it because it is small, generic, available to every candidate, and disjoint from the seven non-STL VTAB targets. A suitable reference should have broad natural-image coverage, be target-disjoint, require no target labels, and be inexpensive to cache.

One reference requires one additional unlabeled feature-extraction pass per candidate. These reference features are cached and reused for every target. An $M$-reference ensemble requires $M$ reference-side passes, while target-side cost is unchanged. On VTAB, the four-reference median gives regret $0.051$, compared with $0.066$ for STL-10 alone. We therefore recommend one generic target-disjoint reference as the default and a small ensemble only when additional robustness justifies its cost. A target must not be evaluated using itself as the reference.

The submitted aggregation treated STL-10 inconsistently: it was an ordinary target for methods that did not require a reference, but SSR$^{\ddagger}$ was undefined because STL-10 was also its reference. The fair common-support analysis excludes STL-10 for every method:

$$
5\ \mathrm{MV}+7\ \mathrm{VTAB}+10\ \mathrm{EX}=22
$$

task units. On these 22 units, SSR$^{\ddagger}$ regret remains $0.071$, RankMe regret is $0.166$, and the paired difference is $-0.095$, with 95% CI $[-0.160,-0.037]$ and $p=0.0002$. Thus the central SSR$^{\ddagger}$-versus-RankMe conclusion is unchanged, while the task count and accounting must be corrected.

## 4. Corrected Marchenko--Pastur argument and scope

The submitted proof incorrectly mixed normalized squared singular values with the RankMe definition and incorrectly stated that the MP support is always bounded away from zero. For RankMe, write $\lambda_i=\sigma_i^2/n$, let their empirical distribution converge to the MP law $\mu_\gamma$, and define

$$
m_{1/2}=\int\sqrt{x}\,d\mu_\gamma(x).
$$

Then

$$
H(\mathbf p)
=
\log d+\log m_{1/2}
-\frac{1}{2m_{1/2}}
\int\sqrt{x}\log x\,d\mu_\gamma(x)+o(1).
$$

Using $0\log0=0$, the integrals remain finite even when the support reaches zero. Consequently,

$$
\frac{r_{\mathrm{eff}}(H)}{d}\rightarrow c(\gamma)>0,
$$

and hence $r_{\mathrm{eff}}(H)=\Theta(d)$.

We also agree that rotation invariance alone does not imply that every conceivable rotation-invariant quantity is monotone in $d$. The claim should be restricted to continuous, unnormalized effective-rank or spectral-diversity functionals whose isotropic limit grows with the number of active spectral dimensions. RankMe belongs to this class; dimension-normalized and deliberately non-monotone functionals do not.

## 5. When to use SSR$^{\ddagger}$ and when to use RankMe

SSR$^{\ddagger}$ is not intended as a universal replacement for RankMe. Our evidence supports a narrower operational rule:

- use SSR$^{\ddagger}$ to compare heterogeneous architecture families;
- use RankMe within one family, where width is fixed or nearly fixed; and
- in a mixed zoo, use SSR$^{\ddagger}$ for architecture selection and RankMe for within-family variant selection.

Within-family experiments show that alignment variation does not provide a stable additional ranking signal. This limitation motivates, rather than contradicts, the two-stage design.

On the five-task MV multi-variant benchmark, SSR$^{\ddagger}$$\rightarrow$RankMe obtains regret $0.102$, versus $0.098$ for flat SSR$^{\ddagger}$, while reducing mean forward evaluations from $118.8$ to $38.0$. We agree that five tasks support only a proof of concept, not a general cost-quality guarantee for arbitrary model zoos.

## 6. DiSCO protocol and corrected results

Reviewer Vfak correctly questioned DiSCO's label status. Official DiSCO computes class-conditional quantities using target class labels. It is therefore a **label-dependent comparator**, not a label-free baseline. Our previous categorization was wrong. The fair label-free comparison is SSR$^{\ddagger}$ versus RankMe.

We also found that the earlier DiSCO implementation did not faithfully implement the official class-conditional Mahalanobis-posterior procedure. We reimplemented the official algorithm, including PCA-128 preprocessing, and reran all four pools after completing every failed feature extraction and missing linear-probe cell. The complete caches contain 660/660 MV and 510/510 EX model-task cells; the LP tables contain no missing cells.

For the two recommended group settings $G\in\{6,8\}$, the corrected DiSCO regrets are:

| Pool | $G=6$ | $G=8$ |
|---|---:|---:|
| MV | 0.146 | 0.146 |
| VTAB | 0.054 | 0.043 |
| SSL | 0.045 | 0.042 |
| EX | 0.045 | 0.049 |

These results materially change our interpretation. SSR$^{\ddagger}$ remains stronger on MV ($0.098$ versus $0.146$), but DiSCO is stronger than the submitted SSR$^{\ddagger}$ result on EX ($0.045$--$0.049$ versus $0.060$) and is highly competitive on VTAB and SSL. Therefore:

- we withdraw the claim that SSR$^{\ddagger}$ generally outperforms DiSCO;
- the previously reported DiSCO cross-pool comparison and significance test should not be used;
- we withdraw the proposed $d_{\max}/d_{\min}\approx7$ decision threshold, because EX is a counterexample; and
- DiSCO should be retained only as a strong label-dependent reference, not as evidence about label-free superiority.

This correction does not affect the main like-for-like result that SSR$^{\ddagger}$ significantly improves over RankMe without labels.

## 7. Strength and limitations of the geometric evidence

There is no formal bound from $\mathrm{SA}_k$ to linear-probe accuracy. The strongest mechanistic evidence is empirical: on 126 VTAB architecture-task pairs, $\mathrm{SA}_k$ is strongly negatively correlated with the fraction of between-class variance outside the generic reference subspace ($\rho=-0.964$), and that architecture-specific fraction positively predicts downstream accuracy. This supports the correction's mechanism but is not a guarantee.

The robust conclusions are therefore:

1. RankMe is strongly width-confounded in heterogeneous architecture pools.
2. The corrected MP argument explains the $\Theta(d)$ scaling of RankMe effective rank under near-isotropy.
3. SSR$^{\ddagger}$ is a parameter-free, label-free cross-family correction that significantly improves over RankMe on the evaluated common support.
4. SSR$^{\ddagger}$ does not reliably improve within-family ranking, and the two-stage result is limited to the evaluated benchmark.
5. Reference selection and the link from alignment to downstream accuracy remain empirical limitations.
6. DiSCO uses target labels and, when implemented faithfully, is stronger than our submission reported.

We thank the reviewers for identifying the definition, proof, protocol, and accounting issues. The audit narrows several claims and corrects the DiSCO comparison, while preserving the paper's central diagnosis and its label-free SSR$^{\ddagger}$-versus-RankMe result.
