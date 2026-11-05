# Author Response to the Initial Meta-Review

We thank the AC and reviewers for the careful and constructive feedback. We agree that the submission should have defined the method and evaluation protocol more precisely. We clarify the main points below. Where the original wording or accounting was incorrect, we state this directly; these clarifications do not rely on changes to the submitted paper or supplementary material.

## 1. Exact method and the apparent feature-dimension mismatch

Our phrase “fixed reference encoder” was inaccurate. We fix a reference **dataset**, not one encoder shared across architectures.

For each candidate encoder $m$, we extract target features $H_{m,T}\in\mathbb R^{n_T\times d_m}$ and reference features $H_{m,R}\in\mathbb R^{n_R\times d_m}$, where $R$ is STL-10. Both matrices are produced by the same candidate encoder, so their feature dimensions always match. Subspaces from two different encoders are never directly compared.

Let

$$
H_{m,T}=U_T\Sigma_TV_T^\top,\qquad
H_{m,R}=U_R\Sigma_RV_R^\top,
$$

and $k_m=\min(k,n_T,n_R,d_m)$. The alignment used in the experiments is

$$
\mathrm{SA}_k(H_{m,T},H_{m,R})
=
\frac{1}{k_m}
\left\|V_{T,1:k_m}^{\top}V_{R,1:k_m}\right\|_F^2
=
\frac{1}{k_m}\sum_{i=1}^{k_m}\cos^2\theta_i.
$$

Thus the computation is: (1) extract target and reference features with the same candidate; (2) compute their top-$k$ right singular vectors; (3) compute principal-angle alignment; (4) compute RankMe effective rank; and (5) return a scalar score. Heterogeneous candidates can be ranked because only their scalar scores are compared; no cross-encoder projection is required.

RankMe effective rank is

$$
p_i=\frac{\sigma_i}{\sum_j\sigma_j},\qquad
r_{\mathrm{eff}}=\exp\left(-\sum_i p_i\log p_i\right).
$$

The score that produced the reported main results is

$$
\boxed{
\mathrm{SSR}^{\dagger}(H,T)
=
\frac{\log r_{\mathrm{eff}}(H)-\log\mathrm{SA}_k(H,H_{\mathrm{ref}})}
{\log C(T)}
}.
$$

## 2. SCR/SSR formula inconsistency

The reviewers are correct that the displayed general SCR family did not algebraically recover the displayed SSR† expression. This is a presentation error, not ambiguity about which score generated the main results: the boxed expression above was used.

An algebraically consistent family is

$$
\mathrm{SCR}(H,T;\beta,g)
=
\frac{\log r_{\mathrm{eff}}(H)+\beta g(\mathrm{SA}_k)}
{\log C(T)}.
$$

Taking $g(s)=-\log s$ and $\beta=1$ gives SSR† exactly. For ranking candidates within one target task, the common positive factor $1/\log C(T)$ does not affect the selected model.

## 3. Why STL-10, reference choice, and cost

STL-10 is a practical rather than theoretically optimal choice. We chose it because it is small, contains generic natural objects, can be processed by all candidates, and is disjoint from the seven non-STL VTAB targets. The intended reference-selection criteria are: generic coverage, target disjointness, no target labels, modest size, and availability for all candidates.

A single reference requires one additional unlabeled feature-extraction pass per candidate, but its features can be cached and reused across all targets. A four-reference ensemble requires four reference-side passes; target-side cost is unchanged. In our reference ablation, the four-reference median improves VTAB regret from $0.066$ to $0.051$. We therefore view one generic, target-disjoint reference as the default and a small ensemble as an optional robustness measure, not a mandatory default. A target dataset should not be used as its own reference.

## 4. Applicability: SSR† versus RankMe

We do not intend SSR† as a universal replacement for RankMe. Our evidence supports the following operational rule:

- across heterogeneous architecture families, particularly when penultimate dimensions vary substantially, use SSR†;
- within one architecture family, where feature dimension is fixed or nearly fixed, use RankMe;
- in a mixed model zoo, use SSR† for architecture selection and RankMe for within-family variant selection.

This also explains the weaker SSL-pool result. When candidates differ mainly in pretraining objective and have comparatively similar backbone widths, DiSCO performs better; when architecture/width variation dominates, SSR† is more useful. We therefore make no claim that SSR† dominates on every type of model pool.

## 5. Marchenko--Pastur derivation

Reviewer Vfak is correct that the submitted proof mixed normalized squared singular values with the RankMe definition and that the MP support is not bounded away from zero in every aspect-ratio regime.

For RankMe, $p_i=\sigma_i/\sum_j\sigma_j$. Let $\lambda_i=\sigma_i^2/n$, let their empirical distribution converge to the MP law $\mu_\gamma$, and define $m_{1/2}=\int\sqrt{x}\,d\mu_\gamma(x)$. Then

$$
H(\mathbf p)
=
\log d+\log m_{1/2}
-
\frac{1}{2m_{1/2}}
\int\sqrt{x}\log x\,d\mu_\gamma(x)
+o(1).
$$

With $0\log0=0$, these integrals remain finite when the support reaches zero. Hence

$$
r_{\mathrm{eff}}(H)/d\rightarrow c(\gamma)>0,
$$

which gives $r_{\mathrm{eff}}(H)=\Theta(d)$. This is the corrected RankMe-consistent argument.

We also agree that rotation invariance alone is insufficient to show that every conceivable rotation-invariant score ranks monotonically by $d$. The intended scope is continuous, unnormalized effective-rank/spectral-diversity scores whose isotropic limit increases with the number of active spectral dimensions. RankMe belongs to this class. Dimension-normalized or deliberately non-monotone spectral functionals are outside the claim.

## 6. DiSCO labeling protocol

Our audit confirms that the reported DiSCO implementation uses target ground-truth labels to form class-conditional quantities; it does not use pseudo-labels. It should therefore be regarded as a **label-dependent comparator**, not a label-free baseline. We apologize for the incorrect categorization.

Accordingly, the main like-for-like label-free comparison is SSR† versus RankMe. We do not claim significant superiority over DiSCO: their paired difference is $\Delta=-0.035$ with $p\approx0.15$, and DiSCO is stronger on the SSL pool.

## 7. STL-10 reference/target accounting

The previous “23 task units” wording used an inconsistent convention: STL-10 contributed to methods for which it was an ordinary target, whereas SSR† was undefined because STL-10 was also the reference. The fair common-support analysis excludes STL-10 from every method:

$$
5\ {\rm MV}+7\ {\rm VTAB}+10\ {\rm EX}=22
$$

target-task units. Under this uniform exclusion:

- SSR† mean regret remains $+0.071$;
- RankMe mean regret is $+0.166$;
- the paired SSR†--RankMe difference is
  $\Delta=-0.095$, $95\%\ {\rm CI}=[-0.160,-0.037]$, $p=0.0002$.

Thus the central conclusion is unchanged, while the common-support task count and RankMe aggregate should be interpreted using these corrected values.

## 8. Evidence strength and two-stage claim

We agree that there is currently no formal bound connecting $\mathrm{SA}_k$ to linear-probe accuracy. The observed relationship among alignment, architecture-specific class-mean variance, and downstream accuracy is empirical. This should be treated as mechanistic evidence, not a theoretical guarantee.

We also agree that the two-stage claim should be limited to the evaluated setting. On the five-task MV multi-variant benchmark, SSR†→RankMe obtains regret $+0.102$, compared with $+0.098$ for flat SSR†, while reducing mean forward evaluations from $118.8$ to $38.0$. This is evidence that the pipeline is feasible on this benchmark, not a general cost-quality guarantee for arbitrary model zoos. Additional multi-variant, multi-task evaluation is needed.

## Closing

The key points are therefore narrower than some wording in the submission suggested:

1. RankMe is strongly width-confounded in heterogeneous architecture pools.
2. SSR† is a parameter-free, label-free correction that significantly improves over RankMe on the evaluated heterogeneous pools.
3. SSR† is intended for cross-family selection; RankMe remains appropriate within a family.
4. Reference selection and the link to downstream accuracy remain empirical limitations.
5. DiSCO is label-dependent in our evaluation, and the two-stage result is a five-task proof of concept.

We thank the reviewers for identifying the definition, theory, and accounting issues. These clarifications preserve the central RankMe-versus-SSR† result while substantially narrowing and sharpening its scope.
