"""Render article figures from the retained deep-dive matrix (requires matplotlib).

Run: python benchmarks/decoder/results/deep-dive/figures/render.py
No benchmarks are executed. Error bars are observed min/max, not confidence intervals.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
MATRIX = json.loads((HERE.parent / 'matrix.json').read_text())
CASES = {case['name']: {row['variant']: row for row in case['results']}
         for case in MATRIX['cases']}
plt.rcParams.update({'font.size': 11, 'axes.spines.top': False,
                     'axes.spines.right': False, 'svg.fonttype': 'none'})
COLORS = ['#64748b', '#d97706', '#2563eb']


def save(fig, name):
    fig.savefig(HERE / (name + '.png'), dpi=180, facecolor='white')
    fig.savefig(HERE / (name + '.svg'), facecolor='white')
    plt.close(fig)


names = list(CASES)
y = np.arange(len(names))
fig, ax = plt.subplots(figsize=(11, 6.8), layout='constrained')
for i, (variant, label) in enumerate([('sequential', 'Ordinary'),
                                      ('capture', 'Single-stream capture'),
                                      ('inferred', 'Inferred dependencies')]):
    timings = [CASES[name][variant]['timings']['stream_us'] for name in names]
    med = np.array([t['median'] for t in timings])
    errors = np.array([[t['median'] - t['minimum'] for t in timings],
                       [t['maximum'] - t['median'] for t in timings]])
    ax.barh(y + (i - 1) * .24, med, height=.22, color=COLORS[i], label=label,
            xerr=errors, error_kw={'elinewidth': .8, 'capsize': 2, 'ecolor': '#334155'})
ax.set_yticks(y, names)
ax.invert_yaxis()
ax.set_xlabel('Native stream time per complete decode step (µs; lower is better)')
ax.set_title('Same unfused FP32 kernels, different execution plans', loc='left', pad=16)
ax.legend(loc='lower right')
ax.set_axisbelow(True)
ax.grid(axis='x', alpha=.2)
fig.supxlabel('RTX 5060 Ti · retained October 2026 data · 7 samples · bars: median; whiskers: min–max', fontsize=10)
save(fig, 'execution-plans')

variants = ['inferred_reuse', 'group2_inferred', 'group4_inferred',
            'group8_inferred', 'batched_inferred']
rows = [CASES['small'][variant] for variant in variants]
x = np.arange(5)
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), layout='constrained')
for ax, values, label, color in [
    (axes[0], [r['timings']['generation_ms']['median'] for r in rows],
     '16-step Python completion (ms)', '#2563eb'),
    (axes[1], [r['graph_setup_ms'] for r in rows],
     'Graph construction (ms)', '#d97706'),
]:
    ax.plot(x, values, marker='o', color=color, linewidth=2)
    for xx, value in zip(x, values):
        ax.annotate(f'{value:.3f}', (xx, value), xytext=(0, 10),
                    textcoords='offset points', ha='center', fontsize=10)
    ax.set_xticks(x, ['1', '2', '4', '8', '16'])
    ax.set_xlabel('Successive decode steps per submission')
    ax.set_ylabel(label)
    ax.margins(y=.25)
    if ax is axes[1]:
        ax.set_ylim(bottom=0)
    ax.grid(alpha=.2)
fig.suptitle('Grouping saves steady-state time but increases setup cost', x=.02, ha='left')
fig.supxlabel('Small model · storage reuse enabled · setup excluded from steady-state completion', fontsize=10)
save(fig, 'grouping-costs')
