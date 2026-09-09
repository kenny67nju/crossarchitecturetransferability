"""Camera-ready MV cost-regret figure using audited strategy definitions."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

# Strategies (regret, cost, color, marker, label, group)
strategies = [
    ('S0 Oracle',                          0.000, 118.8, 'gray',    'o', 'Oracle (upper bound)',         'ref'),
    ('S1 Random',                          0.194,   1.0, 'gray',    's', 'Random',                       'ref'),
    ('S2 Flat-RankMe',                     0.319, 118.8, '#d62728', '^', 'Flat-RankMe ⚠ <Random',       'lf'),
    ('S3 Flat-SSR$^\\ddagger$',            0.098, 118.8, '#1f77b4', 'D', 'Flat-SSR$^\\ddagger$',        'lf'),
    ('S4 2S RankMe→RankMe',                0.144,  38.0, '#9467bd', 'v', '2S RankMe$\\to$RankMe',       'lf'),
    ('S5 2S SSR$^\\ddagger$→RankMe',       0.102,  38.0, '#2ca02c', '*', '2S SSR$^\\ddagger\\to$RankMe (proposed)', 'lf'),
    ('2S H-score→RankMe (mixed)',          0.018,  37.0, '#e377c2', '*', '2S H-score$\\to$RankMe (mixed budget)',     'mb'),
]
ci = {
    'S1 Random':                          (0.118, 0.289),
    'S2 Flat-RankMe':                     (0.149, 0.509),
    'S3 Flat-SSR$^\\ddagger$':            (0.048, 0.148),
    'S4 2S RankMe→RankMe':                (0.093, 0.220),
    'S5 2S SSR$^\\ddagger$→RankMe':       (0.044, 0.153),
}

fig, ax = plt.subplots(figsize=(7.6, 4.9))
for name, regret, cost, color, marker, label, group in strategies:
    if name in ci:
        lo, hi = ci[name]
        ax.errorbar(cost, regret, yerr=[[regret-lo],[hi-regret]], fmt='none',
                    ecolor=color, alpha=0.45, capsize=3, lw=1.0)
    sz = 380 if 'proposed' in label or 'mixed' in label else 200
    ax.scatter(cost, regret, c=color, marker=marker, s=sz,
               edgecolors='black', linewidths=1.0, label=label, zorder=5)

ax.axhline(0, color='gray', lw=0.7, ls='--', alpha=0.5)

ax.annotate('Observed on five MV tasks:\n3.1$\\times$ fewer evaluations,\nsimilar regret to Flat-SSR$^\\ddagger$',
            xy=(38, 0.102), xytext=(60, 0.180),
            arrowprops=dict(arrowstyle='->', color='black', lw=1.0),
            fontsize=9.5, ha='center',
            bbox=dict(boxstyle='round,pad=0.4', fc='lightyellow', ec='black'))
ax.annotate('Mixed budget result:\nregret 0.018 with\n35 labeled evals',
            xy=(37, 0.018), xytext=(55, -0.025),
            arrowprops=dict(arrowstyle='->', color='black', lw=1.0),
            fontsize=9.5, ha='center',
            bbox=dict(boxstyle='round,pad=0.4', fc='lavenderblush', ec='black'))
ax.annotate('Flat-RankMe is\nworse than random!\n(VGG bias)',
            xy=(118.8, 0.319), xytext=(86, 0.41),
            arrowprops=dict(arrowstyle='->', color='red', lw=1.0),
            fontsize=9.5, ha='center', color='darkred',
            bbox=dict(boxstyle='round,pad=0.4', fc='mistyrose', ec='red'))

ax.set_xlabel('Forward-pass cost per task', fontsize=12)
ax.set_ylabel('Regret = Oracle$-$Strategy (LP-acc)', fontsize=12)
ax.set_title('Cost--regret comparison (MV, 35 multi-variant archs $\\times$ 5 tasks)\n'
             'Label-free and mixed-budget pipelines; 95\\% task-bootstrap CI',
             fontsize=11)
ax.set_xlim(-5, 130)
ax.set_ylim(-0.06, 0.55)
ax.grid(True, alpha=0.3)
ax.legend(loc='upper right', fontsize=8.5, framealpha=0.94, ncol=1)
plt.tight_layout()
plt.savefig('paper/fig_pareto.pdf', bbox_inches='tight', dpi=200)
plt.savefig('paper/fig_pareto.png', bbox_inches='tight', dpi=150)
print('Saved → paper/fig_pareto.pdf')
