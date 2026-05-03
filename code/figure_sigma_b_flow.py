"""
figure_sigma_b_flow.py
生成论文 Figure 3：SA_k vs frac_spe 的散点图
说明 SSR^‡ 的真实工作机制：SA_k 与"类间方差落入特异子空间的比例"近乎完美负相关
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.stats import spearmanr, pearsonr

ROOT = Path(__file__).resolve().parent.parent
OUT  = ROOT / "paper"

# 论文样式
mpl.rcParams.update({
    'font.size': 9,
    'axes.labelsize': 10,
    'axes.titlesize': 10,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 8,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
    'axes.spines.top':   False,
    'axes.spines.right': False,
})

# 读取已生成的数据
df_vtab = pd.read_csv(ROOT / "analysis" / "fisher_validation.csv")

# 7 任务，颜色
TASK_COLORS = {
    'cifar10':       '#e41a1c',
    'cifar100':      '#377eb8',
    'dtd':           '#4daf4a',
    'fashion_mnist': '#984ea3',
    'flowers102':    '#ff7f00',
    'pets':          '#a65628',
    'svhn':          '#f781bf',
}
TASK_LABELS = {
    'cifar10':'CIFAR-10','cifar100':'CIFAR-100','dtd':'DTD',
    'fashion_mnist':'Fashion-MNIST','flowers102':'Flowers-102',
    'pets':'Pets','svhn':'SVHN',
}

fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2))

# ─────────────────────────────────────────────────────────────
# 左图：SA_k vs frac_spe 散点（按任务上色）
# ─────────────────────────────────────────────────────────────
ax = axes[0]
for task, color in TASK_COLORS.items():
    sub = df_vtab[df_vtab.task == task]
    ax.scatter(sub.SA, sub.frac_spe, s=18, alpha=0.75,
               edgecolor='white', linewidth=0.4,
               color=color, label=TASK_LABELS[task])

# 拟合直线（log-log 空间外但用线性 x、y 都是 [0,1]）
x_all = df_vtab.SA.values
y_all = df_vtab.frac_spe.values
mask = ~(np.isnan(x_all) | np.isnan(y_all))
x_all, y_all = x_all[mask], y_all[mask]
rho_sp, p_sp = spearmanr(x_all, y_all)
rho_pe, _    = pearsonr(x_all, y_all)

# 全局拟合
x_fit = np.linspace(0.01, 0.65, 100)
slope, intercept = np.polyfit(x_all, y_all, 1)
ax.plot(x_fit, slope * x_fit + intercept, 'k--', lw=0.8, alpha=0.6,
        label=f'linear fit')

ax.set_xlabel(r'$\mathrm{SA}_k(H, H_{\mathrm{ref}})$  (k=20, ref=STL-10)')
ax.set_ylabel(r'$f_{\mathrm{spe}} = \mathrm{tr}(Q\Sigma_B Q)\,/\,\mathrm{tr}(\Sigma_B)$')
ax.set_title(f'(a)  Class-mean variance flow into specific subspace\n'
             rf'$\rho_{{\mathrm{{Sp}}}}={rho_sp:+.3f}$, $\rho_{{\mathrm{{Pe}}}}={rho_pe:+.3f}$  '
             f'(n={len(x_all)})', pad=4)
ax.set_xlim(-0.02, 0.70)
ax.set_ylim(0.10, 1.05)
ax.grid(alpha=0.3, linewidth=0.3)

# ─────────────────────────────────────────────────────────────
# 右图：frac_spe vs LP（每任务一个面板的简化版本——单图分色）
# 验证：高 frac_spe → 高 LP（在每个任务内）
# ─────────────────────────────────────────────────────────────
ax = axes[1]

# 按任务标准化 LP（z-score），让多任务可在同一图比较
df_vtab['lp_z'] = df_vtab.groupby('task').lp_acc.transform(
    lambda x: (x - x.mean()) / (x.std() + 1e-12))
df_vtab['fspe_z'] = df_vtab.groupby('task').frac_spe.transform(
    lambda x: (x - x.mean()) / (x.std() + 1e-12))

for task, color in TASK_COLORS.items():
    sub = df_vtab[df_vtab.task == task]
    ax.scatter(sub.fspe_z, sub.lp_z, s=18, alpha=0.75,
               edgecolor='white', linewidth=0.4, color=color)

# 计算每任务 ρ(frac_spe, LP) 和均值
per_task_rho = []
for task in TASK_COLORS:
    sub = df_vtab[df_vtab.task == task]
    if len(sub) > 5:
        r, _ = spearmanr(sub.frac_spe, sub.lp_acc)
        per_task_rho.append(r)
mean_within_rho = np.mean(per_task_rho)

# 拟合（z-score 后整体）
x_z = df_vtab.fspe_z.values
y_z = df_vtab.lp_z.values
m = ~(np.isnan(x_z) | np.isnan(y_z))
slope_z, intercept_z = np.polyfit(x_z[m], y_z[m], 1)
xf = np.linspace(-2.5, 2.5, 100)
ax.plot(xf, slope_z * xf + intercept_z, 'k--', lw=0.8, alpha=0.6)

ax.axhline(0, color='grey', lw=0.4, alpha=0.5)
ax.axvline(0, color='grey', lw=0.4, alpha=0.5)

ax.set_xlabel(r'$f_{\mathrm{spe}}$  (per-task $z$-score)')
ax.set_ylabel(r'$\mathrm{lp\_acc}$  (per-task $z$-score)')
ax.set_title(f'(b)  Specific-subspace variance predicts LP accuracy\n'
             rf'within-task mean $\rho_{{\mathrm{{Sp}}}}={mean_within_rho:+.3f}$  '
             f'(7 tasks)', pad=4)
ax.set_xlim(-2.6, 2.6)
ax.set_ylim(-2.6, 2.6)
ax.grid(alpha=0.3, linewidth=0.3)

# 共享图例放置在两子图下方，避免遮挡数据
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels,
           loc='lower center', bbox_to_anchor=(0.5, -0.02),
           ncol=8, frameon=False,
           handletextpad=0.4, columnspacing=1.2,
           markerscale=1.0, fontsize=8)

plt.tight_layout(rect=[0, 0.06, 1, 1])
out_pdf = OUT / "fig_sigma_b_flow.pdf"
plt.savefig(out_pdf)
print(f"Saved: {out_pdf}")
out_png = OUT / "fig_sigma_b_flow.png"
plt.savefig(out_png, dpi=200)
print(f"Saved: {out_png}")

# 同步到 repository
import shutil
repo_paper = ROOT / "repository" / "paper"
if repo_paper.exists():
    shutil.copy(out_pdf, repo_paper / "fig_sigma_b_flow.pdf")
    print(f"Copied to: {repo_paper}/fig_sigma_b_flow.pdf")

print("\n===  关键统计  ===")
print(f"全池  ρ_Sp(SA_k, frac_spe) = {rho_sp:+.4f}  (p={p_sp:.2e})")
print(f"全池  ρ_Pe(SA_k, frac_spe) = {rho_pe:+.4f}")
print(f"逐任务 ρ(SA_k, frac_spe) 范围：")
for task in TASK_COLORS:
    sub = df_vtab[df_vtab.task == task]
    if len(sub) > 5:
        r, p = spearmanr(sub.SA, sub.frac_spe)
        print(f"  {task:15s}  ρ={r:+.3f}  p={p:.2e}  n={len(sub)}")

print(f"\n逐任务 ρ(frac_spe, LP)：")
for task in TASK_COLORS:
    sub = df_vtab[df_vtab.task == task]
    if len(sub) > 5:
        r, p = spearmanr(sub.frac_spe, sub.lp_acc)
        print(f"  {task:15s}  ρ={r:+.3f}  p={p:.2e}")
print(f"\n均值（所有 7 任务）           = {mean_within_rho:+.3f}")
