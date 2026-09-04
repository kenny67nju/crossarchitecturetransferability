# Submission 14729 Rebuttal

We thank the Area Chair and all reviewers for their careful and constructive feedback. We agree that the original submission did not describe the method and evaluation protocol with sufficient precision. We conducted an internal audit of the paper, code, and released result files. This audit identified several presentation issues and two evaluation-accounting issues that require correction. We respond point by point below.

## 1. Exact definition and implementation of $SA_k$

Our original phrase “a fixed reference encoder” was inaccurate. We use a fixed reference **dataset**, not one encoder shared by all architectures.

For every candidate encoder $m$, we extract

$$
H_{m,T}\in\mathbb R^{n_T\times d_m},\qquad
H_{m,D_{\mathrm{ref}}}\in\mathbb R^{n_{\mathrm{ref}}\times d_m},
$$

where $T$ is the target dataset and $D_{\mathrm{ref}}$ is STL-10. Both matrices are produced by the same candidate encoder $m$, so they always share feature dimension $d_m$. Right singular vectors belonging to different encoders are never directly compared.

Let

$$
H_{m,T}=U_T\Sigma_TV_T^\top,\qquad
H_{m,D_{\mathrm{ref}}}=U_R\Sigma_RV_R^\top,
$$

and define

$$
k_{\mathrm{eff}}=\min(k,n_T,n_{\mathrm{ref}},d_m).
$$

We compute

$$
\mathrm{SA}_k(H_{m,T},H_{m,D_{\mathrm{ref}}})
=
\frac{1}{k_{\mathrm{eff}}}
\left\|
V_{T,1:k_{\mathrm{eff}}}^{\top}
V_{R,1:k_{\mathrm{eff}}}
\right\|_F^2
=
\frac{1}{k_{\mathrm{eff}}}
\sum_{i=1}^{k_{\mathrm{eff}}}\cos^2\theta_i,
$$

where $\theta_i$ are the principal angles between the two top-$k$ right-singular subspaces.

The complete procedure is:

1. Extract unlabeled target features $H_{m,T}$ with candidate encoder $m$.
2. Extract STL-10 features $H_{m,D_{\mathrm{ref}}}$ with the same encoder.
3. Compute the top-$k$ right singular vectors of both matrices.
4. Compute the principal-angle alignment $\mathrm{SA}_k$.
5. Compute RankMe effective rank using
   $$
   p_i=\frac{\sigma_i}{\sum_j\sigma_j},\qquad
   r_{\mathrm{eff}}=\exp\left(-\sum_i p_i\log p_i\right).
   $$
6. Score the candidate using
   $$
   \mathrm{SSR}^{\dagger}(H_{m,T},T)
   =
   \frac{
   \log r_{\mathrm{eff}}(H_{m,T})
   -
   \log\mathrm{SA}_k(H_{m,T},H_{m,D_{\mathrm{ref}}})
   }{\log C(T)}.
   $$

Different architectures are compared only through their scalar SSR† scores; no common cross-encoder feature space is required. We will replace “fixed reference encoder” with “a fixed reference dataset encoded separately by each candidate encoder” throughout the paper and add the above algorithm explicitly.

## 2. Reconciliation of SCR and SSR†

The reviewers correctly identified an algebraic inconsistency. The original paper defined

$$
\mathrm{SCR}=\log r_{\mathrm{eff}}+\beta g(\mathrm{SA}_k)
$$

and claimed that $g(s)=-\log s$ and $\beta=1/\log C(T)$ recovered the displayed SSR† formula. Direct substitution does not yield that expression.

The score used in all reported main experiments is

$$
\boxed{
\mathrm{SSR}^{\dagger}(H,T)
=
\frac{\log r_{\mathrm{eff}}(H)-\log\mathrm{SA}_k(H,H_{\mathrm{ref}})}
{\log C(T)}
}.
$$

We correct the method-family definition to

$$
\mathrm{SCR}(H,T;\beta,g)
=
\frac{
\log r_{\mathrm{eff}}(H)+\beta g(\mathrm{SA}_k)
}{\log C(T)}.
$$

Then $g(s)=-\log s,\beta=1$ exactly recovers SSR†. Within a fixed task, $1/\log C(T)$ is a common positive factor and may be omitted without changing model selection.

We have unified the active code paths around the RankMe definition of $r_{\mathrm{eff}}$ and the boxed SSR† formula. Superseded UID-based implementations have been removed from the active reproduction path and retained under `archive/` for provenance.

## 3. Reference-dataset selection and cost

We agree that the original paper did not provide sufficient practical guidance. A suitable reference dataset should:

- require no labels for score computation;
- be disjoint from the evaluated target task;
- contain broad rather than narrowly target-specific visual content;
- be available to every candidate encoder;
- be small enough that its features can be cached cheaply.

We chose STL-10 because it is small, generic, and disjoint from the seven non-STL VTAB target tasks.

A single reference requires one additional unlabeled reference-side feature pass per candidate encoder. These features can be cached and reused across target tasks. A four-reference ensemble requires four reference-side passes rather than one, while target-side cost is unchanged. The four-reference median ensemble reduces VTAB regret from $0.066$ to $0.051$, but increases reference-side cost. We therefore recommend:

- one generic, target-disjoint reference as the default;
- a small ensemble when reference sensitivity is a concern or reference features are already cached;
- never using the evaluated target dataset itself as a reference when reporting that target’s performance.

## 4. When to use SSR† versus RankMe

SSR† is not intended as a universal replacement for RankMe. The practical rule supported by our evidence is:

- use SSR† across heterogeneous architecture families, particularly when penultimate dimensions vary substantially;
- use RankMe within one architecture family, where feature dimension is fixed or nearly fixed;
- in a mixed model zoo, use SSR† for architecture-family selection and RankMe for within-family variant selection.

The SSL pool further delineates the boundary: when candidates differ mainly in pretraining objective but have similar backbone widths, DiSCO performs better than SSR†. We now present this as a limitation and pool-dependent complementarity result, not as evidence that SSR† dominates on every pool.

## 5. Marchenko–Pastur derivation

Reviewer Vfak is correct that the original proof mixed normalized-eigenvalue entropy with the RankMe definition and incorrectly assumed that the MP support is always bounded away from zero.

RankMe uses

$$
p_i=\frac{\sigma_i}{\sum_j\sigma_j}.
$$

Let $\lambda_i=\sigma_i^2/n$, and let the empirical distribution of $\lambda_i$ converge to the Marchenko–Pastur distribution $\mu_\gamma$. Define

$$
m_{1/2}=\int\sqrt{x}\,d\mu_\gamma(x).
$$

Then

$$
H(\mathbf p)
=
\log d+\log m_{1/2}
-
\frac{1}{2m_{1/2}}
\int\sqrt{x}\log x\,d\mu_\gamma(x)
+o(1).
$$

Using $0\log0=0$, the relevant integrals remain finite even in aspect-ratio regimes where the MP support reaches zero. Consequently,

$$
r_{\mathrm{eff}}(H)/d\rightarrow c(\gamma)>0,
$$

which establishes $r_{\mathrm{eff}}(H)=\Theta(d)$. We have replaced the original proof with this RankMe-consistent derivation and removed the incorrect support assumption.

## 6. Scope of the rotation-invariance claim

We agree that rotation invariance alone does not imply that every possible rotation-invariant score ranks monotonically by $d$. Dimension-normalized or deliberately non-monotone spectral functionals provide counterexamples.

We narrow the result to the class actually studied:

> Continuous effective-rank or spectral-diversity-type scores whose isotropic large-sample limit is a strictly increasing function of the number of active spectral dimensions.

RankMe and the unnormalized spectral-diversity scores analyzed in the paper belong to this class. We no longer claim the conclusion for arbitrary rotation-invariant functionals. The retained diagnosis is that rotation invariance prevents a score from using feature orientation, while common unnormalized spectral-diversity scores become dominated by feature width under near-isotropy.

## 7. DiSCO labeling protocol

We thank Reviewer Vfak for identifying this ambiguity. Our internal audit confirms that our DiSCO implementation uses target ground-truth class labels to compute class-conditional quantities; it does not use pseudo-labels. It must therefore be classified as a **label-dependent comparator**, not a label-free baseline.

We will:

- mark DiSCO as label-dependent in every table;
- remove statements calling it the strongest label-free baseline;
- avoid presenting SSR† versus DiSCO as a like-for-like label-free comparison;
- retain DiSCO only as an informative class-conditional comparator.

The observed SSR†–DiSCO difference was already non-significant:

$$
\Delta=-0.035,\qquad p\approx0.15.
$$

We therefore make no claim that SSR† significantly outperforms DiSCO. The principal label-free comparison is SSR† versus RankMe.

## 8. STL-10 reference/target accounting

The reviewer is also correct that the previous “23 task units” aggregation used an inconsistent convention: STL-10 was included as a VTAB target for methods not requiring a reference, while SSR† was undefined on that row because STL-10 was the reference.

For a fair comparison, the revision excludes STL-10 from every method in the cross-pool aggregate. The corrected aggregate contains

$$
5\ \text{MV}+7\ \text{VTAB}+10\ \text{EX}=22
$$

target-task units.

Under this uniform exclusion:

- SSR† mean regret remains $+0.071$;
- RankMe mean regret is $+0.166$;
- SSR† versus RankMe is
  $$
  \Delta=-0.095,\quad
  95\%\ \mathrm{CI}=[-0.160,-0.037],\quad
  p=0.0002.
  $$

The principal conclusion is unchanged, but the task count, RankMe baseline, and corresponding statistics will be corrected throughout the paper.

## 9. Strength and scope of evidence

We agree with the reviewers on the following limitations:

1. SSR† does not significantly outperform DiSCO in the paired comparison.
2. SSR† is weaker than DiSCO on the SSL pool.
3. We do not have a formal bound connecting $\mathrm{SA}_k$ to linear-probe accuracy.

Our revised claim is therefore narrower:

- the RankMe width confound is consistently observed in heterogeneous model pools;
- SSR† is a lightweight, label-free correction that significantly improves over RankMe;
- the relationship between $\mathrm{SA}_k$, architecture-specific class-mean variance, and downstream accuracy is empirical rather than a formal guarantee;
- the appropriate correction depends on the dominant source of variation in the model pool.

## 10. Two-stage cost-quality claim

We agree that the original wording was too broad. The $3.1\times$ cost reduction is measured on the MV multi-variant pool:

$$
119\ \text{flat evaluations}
\quad\text{versus}\quad
35+3=38\ \text{two-stage evaluations}.
$$

The quality comparison uses only five target tasks. We therefore moderate this result to a proof of concept on the MV pool rather than a general guarantee for arbitrary model zoos.

The revised statement is:

> On the five-task MV multi-variant benchmark, SSR†→RankMe attains regret $+0.102$, compared with $+0.098$ for flat SSR†, while reducing mean forward evaluations from $118.8$ to $38.0$. This supports the feasibility of the two-stage design, but further validation on additional multi-variant, multi-task model zoos is required.

## Closing

We thank the reviewers for identifying issues that materially improved the paper. The revision will provide:

- an exact definition and step-by-step algorithm for $SA_k$;
- algebraically consistent SCR and SSR† formulas;
- a unified RankMe effective-rank implementation;
- a corrected MP proof;
- a properly qualified rotation-invariance result;
- explicit reference-dataset selection and cost guidance;
- correct DiSCO label accounting;
- uniform exclusion of STL-10 from cross-pool aggregation;
- more conservative claims regarding SSL performance, theoretical guarantees, and two-stage generality.

The central empirical conclusion remains unchanged: in heterogeneous architecture pools, RankMe is strongly width-confounded, and the parameter-free, label-free SSR† correction substantially reduces selection regret relative to RankMe.
