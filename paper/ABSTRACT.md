# Camera-ready metadata

## Title

Width Confounding in Label-Free Cross-Architecture Model Selection: Diagnosis and Subspace Correction

## Abstract

Choosing a pretrained encoder without target labels is a practical problem in transfer learning. RankMe, validated primarily within architecture families, can be confounded by feature width when architectures differ. Under an isotropic random-feature model at fixed aspect ratio, its effective rank scales as Theta(d). We propose Subspace-Corrected Ranking (SCR), which augments the target spectrum with the alignment between target and reference features extracted by the same candidate encoder. Its default member SSR‡ subtracts log subspace alignment from log effective rank. Across 22 pool–task units from three heterogeneous model pools (MV/VTAB/EX), SSR‡ reduces mean selection regret from 0.133 to 0.069 relative to RankMe (paired difference: -0.0645; target-name-clustered 95% CI: [-0.122,-0.020]). On MV, the mean score–width correlation decreases from 0.609 to 0.305, indicating that width dependence is reduced but remains present. On five MV tasks, a two-stage pipeline using SSR‡ for architecture selection and RankMe for within-family selection requires fewer candidate evaluations while achieving regret similar to evaluating SSR‡ over all candidates.

## Keywords

Label-free model selection; transferability estimation; effective rank; width confounding; subspace alignment; pretrained encoders.

## TL;DR

RankMe is width-associated across heterogeneous encoders; target–reference subspace correction reduces width association and selection regret on the evaluated common support, without universal accuracy or pipeline-optimality guarantees.
