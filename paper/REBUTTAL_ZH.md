# Submission 14729 Rebuttal（中文）

感谢 Area Chair 和各位审稿人的细致审阅。我们同意，原稿对方法定义和评估协议的说明不够精确。根据审稿意见，我们对论文、代码和发布的结果文件进行了内部核查，发现若干表述问题以及两处需要修正的评估口径。下面逐项回应。

## 1. $SA_k$ 的精确定义与跨架构维度问题

原稿使用“固定参考编码器”的措辞不准确。实际固定的是**参考数据集**，而不是一个供所有架构共同使用的编码器。

对每个候选编码器 $m$，我们分别提取

$$
H_{m,T}\in\mathbb R^{n_T\times d_m},\qquad
H_{m,D_{\mathrm{ref}}}\in\mathbb R^{n_{\mathrm{ref}}\times d_m},
$$

其中 $T$ 是目标数据集，$D_{\mathrm{ref}}$ 是 STL-10。两个特征矩阵均由同一个候选编码器 $m$ 产生，因此特征维度始终相同。不同编码器的右奇异向量从不直接比较。

令

$$
H_{m,T}=U_T\Sigma_TV_T^\top,\qquad
H_{m,D_{\mathrm{ref}}}=U_R\Sigma_RV_R^\top,
$$

并设

$$
k_{\mathrm{eff}}=\min(k,n_T,n_{\mathrm{ref}},d_m).
$$

我们计算

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
\sum_{i=1}^{k_{\mathrm{eff}}}\cos^2\theta_i.
$$

其中 $\theta_i$ 是两个 top-$k$ 右奇异子空间的主角。

完整计算流程为：

1. 使用候选编码器 $m$ 提取目标数据集的无标签特征 $H_{m,T}$；
2. 使用同一个编码器提取 STL-10 特征 $H_{m,D_{\mathrm{ref}}}$；
3. 分别计算两个矩阵的 top-$k$ 右奇异向量；
4. 计算主角对齐度 $\mathrm{SA}_k$；
5. 使用 RankMe 定义计算有效秩：
   $$
   p_i=\frac{\sigma_i}{\sum_j\sigma_j},\qquad
   r_{\mathrm{eff}}=\exp\left(-\sum_i p_i\log p_i\right);
   $$
6. 计算最终分数：
   $$
   \mathrm{SSR}^{\dagger}(H_{m,T},T)
   =
   \frac{
   \log r_{\mathrm{eff}}(H_{m,T})
   -
   \log\mathrm{SA}_k(H_{m,T},H_{m,D_{\mathrm{ref}}})
   }{\log C(T)}.
   $$

不同架构最终只比较标量 SSR†，不需要把不同编码器的特征投影到共同空间。修订稿将全文的“固定参考编码器”改为“由每个候选编码器分别编码的固定参考数据集”，并加入上述算法框。

## 2. SCR 与 SSR† 公式不一致

审稿人正确指出了原稿中的代数错误。原稿定义

$$
\mathrm{SCR}=\log r_{\mathrm{eff}}+\beta g(\mathrm{SA}_k),
$$

但声称取 $g(s)=-\log s$、$\beta=1/\log C(T)$ 可以得到所展示的 SSR†；直接代入并不能得到该公式。

所有论文主实验实际使用的是

$$
\boxed{
\mathrm{SSR}^{\dagger}(H,T)
=
\frac{\log r_{\mathrm{eff}}(H)-\log\mathrm{SA}_k(H,H_{\mathrm{ref}})}
{\log C(T)}
}.
$$

修订后的方法族定义为

$$
\mathrm{SCR}(H,T;\beta,g)
=
\frac{
\log r_{\mathrm{eff}}(H)+\beta g(\mathrm{SA}_k)
}{\log C(T)}.
$$

此时令 $g(s)=-\log s,\beta=1$ 即可严格得到 SSR†。在单个任务内部，$1/\log C(T)$ 是共同的正常数，省略它不会改变模型选择结果。

我们已经统一有效代码路径中的 $r_{\mathrm{eff}}$ 和 SSR† 实现，并将旧的 UID 版本移出主复现路径、保留在 `archive/` 中用于追溯。

## 3. 参考数据集的选择与成本

我们同意原稿缺少可操作的选择原则。合适的参考数据集应当：

- 在计算分数时不使用标签；
- 与被评估的目标任务相互独立；
- 覆盖较通用的视觉内容，而不是高度特化于某个目标；
- 能够供所有候选编码器提取特征；
- 规模较小，便于缓存参考特征。

我们选择 STL-10，是因为它规模较小、内容通用，并且与七个非 STL-10 的 VTAB 目标任务分离。

单参考方案需要每个候选编码器额外执行一次无标签参考特征提取；该特征可以缓存并在多个目标任务间复用。四参考集成需要四次参考侧特征提取，但目标侧成本不变。实验中，四参考中位数集成将 VTAB regret 从 $0.066$ 降至 $0.051$，但同时增加了参考侧成本。因此我们的建议是：

- 默认使用一个通用、与目标独立的参考数据集；
- 当参考敏感性较高或多个参考特征已缓存时，使用小规模集成；
- 报告目标任务结果时，不使用该目标数据集本身作为参考。

## 4. SSR† 与 RankMe 的适用场景

SSR† 并非 RankMe 在所有场景下的替代品。根据现有证据，我们给出的实践规则是：

- 跨异构架构家族选择、且倒数第二层维度差异较大时，使用 SSR†；
- 在同一架构家族内部、特征维度固定或近似固定时，使用 RankMe；
- 在混合模型库中，第一阶段使用 SSR† 选择架构家族，第二阶段使用 RankMe 选择家族内变体。

SSL pool 进一步说明了方法边界：当候选模型的主要差异来自预训练目标，而骨干宽度相近时，DiSCO 优于 SSR†。修订稿将其作为明确限制，而不再暗示 SSR† 对所有模型池均占优。

## 5. Marchenko–Pastur 推导

Reviewer Vfak 正确指出，原证明混用了归一化特征值熵与 RankMe 定义，并错误地声称 MP 支撑在所有纵横比下都远离零。

RankMe 使用

$$
p_i=\frac{\sigma_i}{\sum_j\sigma_j}.
$$

令 $\lambda_i=\sigma_i^2/n$，并令 $\lambda_i$ 的经验分布收敛到 Marchenko–Pastur 分布 $\mu_\gamma$。定义

$$
m_{1/2}=\int\sqrt{x}\,d\mu_\gamma(x).
$$

则

$$
H(\mathbf p)
=
\log d+\log m_{1/2}
-
\frac{1}{2m_{1/2}}
\int\sqrt{x}\log x\,d\mu_\gamma(x)
+o(1).
$$

采用 $0\log0=0$ 后，即使 MP 支撑到达零，相关积分仍然有限。因此

$$
r_{\mathrm{eff}}(H)/d\rightarrow c(\gamma)>0,
$$

从而得到 $r_{\mathrm{eff}}(H)=\Theta(d)$。修订稿已改用与 RankMe 一致的证明，并删除“支撑远离零”的错误假设。

## 6. 旋转不变结论的范围

我们同意，仅凭旋转不变性，不能推出任意旋转不变分数都按 $d$ 单调排序。经过维度归一化或人为构造的非单调谱函数可以构成反例。

修订后的结论仅适用于本文实际研究的函数类：

> 连续的、有效秩或谱多样性类型的分数，且其各向同性大样本极限是有效谱维数的严格递增函数。

RankMe 以及本文分析的未归一化谱多样性指标属于这一类。我们不再对任意旋转不变函数作普遍性声明。保留的核心结论是：旋转不变性使分数无法使用特征方向信息，而常见的未归一化谱多样性指标在近各向同性条件下会受到特征宽度主导。

## 7. DiSCO 的标签协议

感谢 Reviewer Vfak 指出这一问题。内部核查表明，我们的 DiSCO 实现使用了目标任务的真实类别标签来计算类条件量，没有使用伪标签。因此该结果必须归类为**需要标签的比较方法**，而不是 label-free baseline。

修订稿将：

- 在所有表格中把 DiSCO 标注为 label-dependent；
- 删除“最强 label-free baseline”等不准确表述；
- 不再把 SSR† 与 DiSCO 描述为同等标签预算下的直接比较；
- 仅将 DiSCO 保留为类条件的参考比较方法。

原结果中 SSR† 相对 DiSCO 的差异本来就不显著：

$$
\Delta=-0.035,\qquad p\approx0.15.
$$

因此我们不主张 SSR† 显著优于 DiSCO。论文的主要无标签比较是 SSR† 与 RankMe。

## 8. STL-10 的参考/目标统计口径

审稿人正确指出了旧的“23 task units”统计中的口径问题：STL-10 对 RankMe 等方法作为 VTAB 目标任务存在，但由于它同时是参考数据集，SSR† 在该行没有定义。

公平的修订方案是从所有方法的跨池汇总中统一排除 STL-10。修订后的任务数为

$$
5\ \text{MV}+7\ \text{VTAB}+10\ \text{EX}=22.
$$

统一排除后：

- SSR† mean regret：$+0.071$；
- RankMe mean regret：$+0.166$；
- SSR† 相对 RankMe：
  $$
  \Delta=-0.095,\quad
  95\%\ \mathrm{CI}=[-0.160,-0.037],\quad
  p=0.0002.
  $$

因此核心结论不变，但任务数、RankMe 基线和对应统计量将在全文中统一修正。

## 9. 证据强度与方法边界

我们同意以下限制：

1. SSR† 相对 DiSCO 的差异在配对比较中不显著；
2. SSR† 在 SSL pool 上弱于 DiSCO；
3. 当前没有将 $\mathrm{SA}_k$ 与 linear-probe accuracy 联系起来的形式化界。

因此修订后的主张是：

- RankMe 的宽度混淆在异构模型池中稳定存在；
- SSR† 是一种轻量、无标签、相对 RankMe 有显著改善的修正；
- $\mathrm{SA}_k$、架构特异类均值方差和下游准确率之间的关系是经验观察，而非理论保证；
- 最适合的修正取决于模型池的主要变化来源。

## 10. Two-stage 成本—质量结论

我们同意原稿表述过于宽泛。$3.1\times$ 成本下降来自 MV multi-variant pool：

$$
119\ \text{次 flat evaluation}
\quad\text{对比}\quad
35+3=38\ \text{次 two-stage evaluation}.
$$

质量比较只有五个目标任务。因此我们将其降调为 MV pool 上的 proof of concept，而不是任意模型库上的普遍保证。

修订后的表述为：

> 在五任务 MV multi-variant benchmark 上，SSR†→RankMe 的 regret 为 $+0.102$，flat SSR† 为 $+0.098$，同时平均前向评估次数从 $118.8$ 降至 $38.0$。这一结果支持两阶段设计的可行性，但仍需要在更多具有多变体、多任务结构的模型库中验证。

## 总结

感谢审稿人指出这些问题。修订稿将提供：

- $SA_k$ 的精确定义和逐步算法；
- 正确一致的 SCR/SSR† 公式；
- 统一的 RankMe 有效秩实现；
- 修正后的 MP 证明；
- 经过限定的旋转不变性结论；
- 明确的参考数据集选择和成本指南；
- 正确的 DiSCO 标签属性；
- 从所有跨池统计中统一排除 STL-10；
- 对 SSL、理论保证和 two-stage 泛化范围更克制的表述。

核心经验结论保持不变：在异构架构模型池中，RankMe 会受到显著的宽度混淆；参数无关、无标签的 SSR† 可以显著降低相对 RankMe 的模型选择 regret。
