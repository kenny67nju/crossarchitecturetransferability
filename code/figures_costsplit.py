"""v1.7 cost-split figure: compute axis (forward passes) + label axis (labeled evals)."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

p2 = pd.read_csv('results/p2_label_budget_sweep.csv')

fig, ax = plt.subplots(figsize=(7.5, 5.0))

# Mixed budget curve points
for _, row in p2.iterrows():
    K = int(row['budget'])
    compute = K + 4
    label = K
    regret = row['regret']
    sd = row['sd_lp']
    color = plt.cm.viridis(K / 35)
    ax.scatter(compute, label, s=900*(0.4-min(regret,0.4)), c=[color],
               edgecolors='black', linewidths=0.8, alpha=0.85,
               zorder=5)
    ax.text(compute+1, label+0.5, f'K={K}\nr={regret:.3f}',
            fontsize=8.5, ha='left', va='bottom')

# Pure label-free reference points (label_cost = 0)
ax.scatter(38, 0, s=420, marker='*', c='#2ca02c', edgecolors='black',
           linewidths=1.2, zorder=10, label='S5 2S SSR$^\\ddagger\\to$RankMe (regret 0.102)')
ax.scatter(119, 0, s=420, marker='D', c='#1f77b4', edgecolors='black',
           linewidths=1.0, zorder=10, label='Flat-SSR$^\\ddagger$ (regret 0.098)')
ax.scatter(119, 0, s=420, marker='^', c='#d62728', edgecolors='black',
           linewidths=1.0, zorder=10,
           label='Flat-RankMe (regret 0.319, biased)', alpha=0.7)

# Annotate key crossover
ax.annotate('K=5: observed regret 0.089\n(five-task MV benchmark)',
            xy=(9, 5), xytext=(50, 8),
            arrowprops=dict(arrowstyle='->', color='black', lw=1.0),
            fontsize=10, ha='left',
            bbox=dict(boxstyle='round,pad=0.4', fc='lightyellow', ec='black'))
ax.annotate('K=35: observed\nregret 0.018',
            xy=(39, 35), xytext=(70, 28),
            arrowprops=dict(arrowstyle='->', color='black', lw=1.0),
            fontsize=10, ha='left',
            bbox=dict(boxstyle='round,pad=0.4', fc='lightgreen', ec='black'))

ax.set_xlabel('Compute cost (forward passes per task)', fontsize=12)
ax.set_ylabel('Label cost (labeled evaluations per task)', fontsize=12)
ax.set_title('Cost--quality trade-off: compute vs label budget axes\n'
             'Bubble size $\\propto$ regret reduction; mixed pipeline (H-score$\\to$RankMe) sweep K=1..35',
             fontsize=11)
ax.set_xlim(-5, 130)
ax.set_ylim(-3, 42)
ax.grid(True, alpha=0.3)
ax.legend(loc='upper right', fontsize=9, framealpha=0.95)
plt.tight_layout()
plt.savefig('paper/fig_costsplit.pdf', bbox_inches='tight', dpi=200)
plt.savefig('paper/fig_costsplit.png', bbox_inches='tight', dpi=150)
print('Saved → paper/fig_costsplit.pdf')
