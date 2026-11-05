# Response to Reviewer Vfak

We thank the reviewer for the careful technical reading and for recognizing the practical diagnosis, use of oracle regret, broad empirical coverage, and diagnostic ablations. The review identifies genuine ambiguities and two errors in the presentation. We answer the concerns in the same order as the review.

## 1. Exact definition of $\mathrm{SA}_k$ and heterogeneous dimensions

The wording “fixed reference encoder” is incorrect. We fix a reference **dataset** $R$ (STL-10), not one encoder shared across candidates.

For each candidate encoder $m$, we extract centered target and reference features

$$
H_{m,T}\in\mathbb R^{n_T\times d_m},
\qquad
H_{m,R}\in\mathbb R^{n_R\times d_m}.
$$

Both are produced by the same candidate $m$, so their feature dimensions always match. The method never compares the right-singular subspace of one encoder with that of another encoder.

Let

$$
H_{m,T}=U_T\Sigma_TV_T^\top,\qquad
H_{m,R}=U_R\Sigma_RV_R^\top,
$$

and $k_m=\min(k,n_T,n_R,d_m)$. The implemented alignment is

Equivalently, using the principal angles directly,

$$
\mathrm{SA}_k(H_{m,T},H_{m,R})
=
\frac{1}{k_m}\sum_{i=1}^{k_m}\cos^2\theta_i.
$$

The $V$'s are the top right-singular directions in the candidate's own $d_m$-dimensional feature space. Different candidates may have different $d_m$, but only the final scalar $\mathrm{SA}_k$ and SSR scores are compared across candidates. No common cross-encoder projection is required.

This also resolves the Proposition 3 ambiguity. If $H_{m,T}$ is right-rotated without applying the same rotation to $H_{m,R}$, their principal-angle alignment changes. Left singular vectors are not used.

The complete computation for candidate $m$ is: extract target and STL-10 features with $m$; center both matrices; compute their top-$k_m$ right-singular vectors; compute the alignment above; compute target effective rank; and return one scalar score.

## 2. SCR/SSR inconsistency and the score actually used

The reviewer is correct: direct substitution into the submitted general SCR equation does not recover the displayed SSR$^{\ddagger}$ equation. This is a presentation error, and the two expressions are not equivalent.

The effective rank and score used to produce the reported SSR results are

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

An algebraically consistent general family is

$$
\mathrm{SCR}(H,T;\beta,g)
=
\frac{\log r_{\mathrm{eff}}(H)+\beta g(\mathrm{SA}_k)}
{\log C(T)}.
$$

Choosing $g(s)=-\log s$ and $\beta=1$ gives the boxed expression exactly. For a fixed target, $1/\log C(T)$ is common to all candidates and does not alter their ranking.

## 3. Corrected Marchenko--Pastur derivation

We agree that the submitted derivation is incorrect as written: it mixes normalized squared singular values with RankMe's singular-value normalization, omits the size-biased term identified by the reviewer, and incorrectly claims that the MP support is always bounded away from zero.

Let $H\in\mathbb R^{n\times d}$ have i.i.d. isotropic rows and set the standard MP aspect ratio $\rho=d/n$ (the submission uses the reciprocal convention $\gamma=n/d$). Let
$\lambda_i=\sigma_i^2/(n\sigma_0^2)$, where $\sigma_0^2$ is the entry variance. We use all $d$ eigenvalues of $H^\top H/(n\sigma_0^2)$, including zeros when $d>n$. Their empirical distribution converges almost surely to the MP law $\mu_\rho$. Define

$$
m_{1/2}=\int\sqrt{x}\,d\mu_\gamma(x).
$$

The common factor $\sqrt n\,\sigma_0$ cancels in RankMe, so
$p_i=\sqrt{\lambda_i}/\sum_{j=1}^d\sqrt{\lambda_j}$. Writing
$S_d=d^{-1}\sum_{j=1}^d\sqrt{\lambda_j}$, we have
$S_d\to m_{1/2}$ and the exact finite-$d$ identity

$$
H(\mathbf p)
=
\log d+\log S_d
-\frac{1}{2S_d}\frac{1}{d}
\sum_{i=1}^d\sqrt{\lambda_i}\log\lambda_i.
$$

Consequently, the correct entropy limit is

$$
H(\mathbf p)
=
\log d+\log m_{1/2}
-\frac{1}{2m_{1/2}}
\int\sqrt{x}\log x\,d\mu_\gamma(x)+o(1).
$$

This contains the required size-biased spectral term. With the convention
$\sqrt0\log0=0$, the integral is finite even when the MP law has support at zero or an atom at zero. Moreover, $m_{1/2}>0$ for every fixed
$0<\rho<\infty$. Therefore

$$
\frac{r_{\mathrm{eff}}(H)}{d}\longrightarrow
\exp\left(
\log m_{1/2}
-\frac{1}{2m_{1/2}}
\int\sqrt{x}\log x\,d\mu_\gamma(x)
\right)
=c(\rho)>0,
$$

which gives $r_{\mathrm{eff}}(H)=\Theta(d)$ for fixed $\rho$. In the submission's notation this constant is $c(1/\gamma)$. Thus the scaling conclusion remains valid, including the $d>n$ regime, although the constant then necessarily satisfies $c(\rho)\le 1/\rho$ because $r_{\mathrm{eff}}\le n$.

## 4. Scope of Corollary 2

We agree that “any rotation-invariant score” is too broad. Rotation invariance only implies dependence on singular values; it does not imply monotonicity in $d$. Dimension-normalized effective rank, condition numbers, and deliberately non-monotone spectral-shape statistics are counterexamples.

The defensible statement is restricted to continuous, unnormalized spectral-diversity/effective-rank functionals whose isotropic limit increases with the number of active spectral dimensions. RankMe belongs to this class. VNE requires separate normalization-sensitive treatment, and arbitrary rotation-invariant quantities should not be covered by the corollary.

## 5. DiSCO protocol and corrected evaluation

The review correctly identifies an evaluation-protocol problem. Official DiSCO uses **target ground-truth labels** to form class-conditional covariance and posterior quantities. It does not use pseudo-labels or unlabeled clustering in our evaluation. Therefore DiSCO is a **label-dependent comparator**, not a label-free baseline, and it is not directly comparable to RankMe or SSR$^{\ddagger}$ under the same label budget.

Our audit also found that the earlier implementation did not faithfully reproduce the official class-conditional Mahalanobis-posterior algorithm. We implemented the official procedure with PCA-128 preprocessing, recovered all failed model-task feature extractions, and completed the 22 previously missing LP cells. The final MV and EX tables contain 660/660 and 510/510 cells, respectively.

For consistency with the common support used below, the VTAB entries exclude
STL-10; all other pool entries use their complete target sets. The corrected
DiSCO regrets for the two evaluated group settings are:

| Pool | $G=6$ | $G=8$ |
|---|---:|---:|
| MV | 0.146 | 0.146 |
| VTAB (7 targets; STL-10 excluded) | 0.061 | 0.044 |
| SSL | 0.045 | 0.042 |
| EX | 0.045 | 0.049 |

These results invalidate the submitted DiSCO aggregate and its associated significance comparison. SSR$^{\ddagger}$ remains stronger on MV ($0.098$ versus $0.146$), but DiSCO is stronger on EX ($0.045$--$0.049$ versus $0.060$) and highly competitive on VTAB and SSL. We therefore withdraw any claim that SSR$^{\ddagger}$ generally outperforms DiSCO, as well as the proposed $d_{\max}/d_{\min}\approx7$ rule for deciding between them.

The main like-for-like label-free comparison is SSR$^{\ddagger}$ versus RankMe.

## 6. STL-10 reference/target accounting

The reviewer is correct that the submitted 23-task accounting is inconsistent. STL-10 was treated as an ordinary VTAB target for methods that do not require a reference, whereas SSR$^{\ddagger}$ is undefined when STL-10 is simultaneously the target and reference. There is no valid way to claim that unit as an independent SSR$^{\ddagger}$ evaluation.

The fair common-support analysis excludes STL-10 from **all** methods:

$$
5\ \mathrm{MV}+7\ \mathrm{VTAB}+10\ \mathrm{EX}=22
$$

task units. On this common support:

- SSR$^{\ddagger}$ mean regret is $0.071$;
- RankMe mean regret is $0.166$; and
- the paired difference is $-0.095$, with 95% CI $[-0.160,-0.037]$ and $p=0.0002$, using a two-sided paired task-level bootstrap test.

Thus the central SSR$^{\ddagger}$-versus-RankMe conclusion is unchanged, but Table 7's 23-task wording and any statistic using inconsistent support should be corrected.

## 7. Two-stage claim and evidence strength

We agree that the one-third-cost statement was expressed too broadly. The direct multi-variant evidence consists of five MV tasks. On that benchmark:

- flat SSR$^{\ddagger}$: regret $0.098$, mean cost $118.8$ forward evaluations;
- SSR$^{\ddagger}$-to-RankMe: regret $0.102$, mean cost $38.0$.

This shows that the pipeline preserved the point-estimate quality at approximately one-third cost **on this benchmark**. It is a proof of concept, not a general cost-quality guarantee for arbitrary heterogeneous model zoos. Additional multi-variant pools and tasks are required.

Finally, there is no formal bound connecting $\mathrm{SA}_k$ to linear-probe accuracy. The supporting mechanism is empirical: on 126 VTAB architecture-task pairs, $\mathrm{SA}_k$ correlates at $-0.964$ with the fraction of between-class variance outside the generic reference subspace, which in turn positively predicts LP accuracy. This is mechanistic evidence, not a theorem about downstream performance.

In summary, the audit narrows the defensible contribution: RankMe is strongly width-confounded in heterogeneous architecture pools, and SSR$^{\ddagger}$ is a parameter-free, label-free cross-family correction that significantly improves over RankMe on the common evaluation support. We do not claim universal superiority over label-dependent methods, a theorem linking alignment to accuracy, or a general two-stage deployment guarantee.
