"""Regenerate every chart in results/ from results/results.json.

Single source of truth: edit the JSON, re-run this, the README updates. Charts
are SVG so they diff, scale, and render natively on GitHub.

    python scripts/make_charts.py
"""
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(ROOT, 'results')
DATA = json.load(open(os.path.join(OUT, 'results.json')))

# A palette that stays legible on GitHub's light and dark themes, so the charts
# carry their own background rather than assuming one.
INK = '#1c1f26'
MUTED = '#6b7280'
GRID = '#dfe3e8'
BG = '#fbfbfc'
OURS = '#2f6f4f'          # our own results
RIVAL = '#8a5a2b'         # published rivals
BASE = '#9aa3af'          # classical baselines
BAD = '#a33a3a'           # regressions / refuted


def _fig(w=8.6, h=None, n=1):
    h = h or max(2.4, 0.52 * n + 1.5)
    fig, ax = plt.subplots(figsize=(w, h))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    for side in ('top', 'right', 'left'):
        ax.spines[side].set_visible(False)
    ax.spines['bottom'].set_color(GRID)
    ax.tick_params(colors=MUTED, length=0, labelsize=9)
    return fig, ax


def _save(fig, name):
    path = os.path.join(OUT, name)
    fig.tight_layout()
    fig.savefig(path, format='svg', facecolor=BG, bbox_inches='tight')
    plt.close(fig)
    print('wrote', os.path.relpath(path, ROOT))


def barh(name, title, rows, xlabel='bits per byte (lower is better)', note=None):
    """rows: list of (label, value, colour)."""
    labels = [r[0] for r in rows]
    vals = [r[1] for r in rows]
    cols = [r[2] for r in rows]
    fig, ax = _fig(n=len(rows))
    y = range(len(rows))
    ax.barh(list(y), vals, color=cols, height=0.62, zorder=3)
    ax.set_yticks(list(y))
    ax.set_yticklabels(labels, color=INK, fontsize=9.5)
    ax.invert_yaxis()
    ax.xaxis.grid(True, color=GRID, zorder=0, linewidth=0.8)
    ax.set_xlabel(xlabel, color=MUTED, fontsize=9)
    ax.set_title(title, color=INK, fontsize=11.5, loc='left', pad=12, fontweight='bold')
    span = max(vals) or 1
    for i, v in enumerate(vals):
        ax.text(v + span * 0.012, i, f'{v:.3f}', va='center',
                color=INK, fontsize=9, fontweight='bold')
    ax.set_xlim(0, span * 1.14)
    if note:
        ax.text(0, 1.0, note, transform=ax.transAxes, color=MUTED, fontsize=8,
                va='bottom', ha='left')
    _save(fig, name)


def chart_headline():
    r = DATA['headline_alice29']['results']
    order = sorted(r, key=lambda k: r[k]['bpb'])
    rows = []
    for k in order:
        c = OURS if 'llm_ptc' in k or k == 'ptc' else (RIVAL if 'ts_zip' in k else BASE)
        rows.append((k, r[k]['bpb'], c))
    barh('chart_headline.svg',
         'alice29.txt, 152,089 bytes - every codec on the identical file',
         rows, note='green = this repo   brown = published rival   grey = classical baseline')


def chart_models():
    m = DATA['model_comparison']['models']
    rows = [(f"{x['name']}  ({x['params_m']}M, {x['kind']})", x['bpb'], OURS) for x in m]
    rows.sort(key=lambda t: t[1])
    barh('chart_models.svg',
         'Same coder, same 8 KB, only the model changes',
         rows, note='A 135M base model beats a 600M instruct model. Data and calibration beat size.')


def chart_ablation():
    a = DATA['enwik8_ablation']
    rows = []
    for v in a['variants']:
        if 'note' in v:
            continue                      # measured on the biased slice, not comparable
        col = OURS if v.get('default') else (BAD if v['bpb'] > 0.928 else BASE)
        rows.append((v['config'], v['bpb'], col))
    barh('chart_ablation.svg',
         'enwik8 mid-file slice - what actually helped',
         rows, note='Only the match model pays. SSE/APM, the CM standard, measured neutral to harmful.')


def chart_contamination():
    c = DATA['contamination_test']
    fig, ax = _fig(h=3.2)
    groups = ['alice29\n(likely memorised)', 'post2026 arXiv\n(definitely unseen)']
    ours = [c['alice29_likely_memorised']['llm_ptc'], c['post2026_definitely_unseen']['llm_ptc']]
    xz = [c['alice29_likely_memorised']['xz -9'], c['post2026_definitely_unseen']['xz -9']]
    x = range(len(groups))
    w = 0.34
    ax.bar([i - w / 2 for i in x], ours, w, color=OURS, label='llm_ptc', zorder=3)
    ax.bar([i + w / 2 for i in x], xz, w, color=BASE, label='xz -9 (control)', zorder=3)
    for i, (o, z) in enumerate(zip(ours, xz)):
        ax.text(i - w / 2, o + 0.05, f'{o:.3f}', ha='center', color=INK, fontsize=9, fontweight='bold')
        ax.text(i + w / 2, z + 0.05, f'{z:.3f}', ha='center', color=INK, fontsize=9)
        ax.text(i, max(o, z) + 0.28, f'{z / o:.2f}x advantage', ha='center',
                color=OURS, fontsize=9.5, fontweight='bold')
    ax.set_xticks(list(x))
    ax.set_xticklabels(groups, color=INK, fontsize=9.5)
    ax.yaxis.grid(True, color=GRID, zorder=0, linewidth=0.8)
    ax.yaxis.set_major_locator(MultipleLocator(1))
    ax.set_ylabel('bits per byte', color=MUTED, fontsize=9)
    ax.set_ylim(0, 3.9)
    ax.set_title('Is it compression or memorisation?', color=INK, fontsize=11.5,
                 loc='left', pad=22, fontweight='bold')
    ax.text(0, 1.0, 'xz cannot memorise, so a surviving advantage over it is real. '
                    'It survives - but shrinks from 2.96x to 2.13x.',
            transform=ax.transAxes, color=MUTED, fontsize=8, va='bottom')
    leg = ax.legend(frameon=False, fontsize=9, loc='upper right')
    for t in leg.get_texts():
        t.set_color(MUTED)
    _save(fig, 'chart_contamination.svg')


def chart_ptc_evolution():
    e = DATA['ptc_evolution']
    files = list(e['order124_only'])
    before = [e['order124_only'][f] for f in files]
    after = [e['plus_match_and_stride'][f] for f in files]
    fig, ax = _fig(h=3.2)
    x = range(len(files))
    w = 0.34
    ax.bar([i - w / 2 for i in x], before, w, color=BASE, label='order 1/2/4 contexts only', zorder=3)
    ax.bar([i + w / 2 for i in x], after, w, color=OURS, label='+ match model + stride detector', zorder=3)
    for i, (b, a) in enumerate(zip(before, after)):
        ax.text(i + w / 2, a + 0.04, f'-{(1 - a / b) * 100:.0f}%', ha='center',
                color=OURS, fontsize=9, fontweight='bold')
    ax.set_xticks(list(x))
    ax.set_xticklabels(files, color=INK, fontsize=9)
    ax.yaxis.grid(True, color=GRID, zorder=0, linewidth=0.8)
    ax.set_ylabel('bits per byte', color=MUTED, fontsize=9)
    ax.set_title('ptc: the pure-maths compressor, before and after', color=INK,
                 fontsize=11.5, loc='left', pad=12, fontweight='bold')
    ax.text(0, 1.0, 'kennedy.xls gained most (-20%), as predicted: it is a fixed-record '
                    'spreadsheet and the model was 1-D.',
            transform=ax.transAxes, color=MUTED, fontsize=8, va='bottom')
    leg = ax.legend(frameon=False, fontsize=9)
    for t in leg.get_texts():
        t.set_color(MUTED)
    _save(fig, 'chart_ptc_evolution.svg')


def chart_speed_ratio():
    """The tradeoff that decides whether any of this is usable."""
    fig, ax = _fig(h=3.6)
    pts = [('xz -9', 500_000, 2.551, BASE), ('bz2 -9', 400_000, 2.272, BASE),
           ('brotli 11', 60_000, 2.445, BASE), ('ptc (pure maths)', 24_000, 2.606, OURS),
           ('llm_ptc + GPT-2', 54, 1.834, OURS),
           ('llm_ptc + SmolLM2', 963, 0.939, OURS)]
    for name, speed, bpb, col in pts:
        ax.scatter(speed, bpb, s=90, color=col, zorder=3, edgecolor=BG, linewidth=1.5)
        ax.annotate(name, (speed, bpb), textcoords='offset points', xytext=(9, 4),
                    color=INK, fontsize=9)
    ax.set_xscale('log')
    ax.invert_yaxis()
    ax.grid(True, color=GRID, zorder=0, linewidth=0.8)
    ax.set_xlabel('encode speed, bytes/second (log scale)', color=MUTED, fontsize=9)
    ax.set_ylabel('bits per byte (better upward)', color=MUTED, fontsize=9)
    ax.set_title('The whole tradeoff, on one chart', color=INK, fontsize=11.5,
                 loc='left', pad=12, fontweight='bold')
    ax.text(0, 1.0, 'Four orders of magnitude of speed buys about 2.7x of ratio. '
                    'There is no fast-and-best corner.',
            transform=ax.transAxes, color=MUTED, fontsize=8, va='bottom')
    ax.set_xlim(20, 3e6)
    _save(fig, 'chart_speed_ratio.svg')


if __name__ == '__main__':
    chart_headline()
    chart_models()
    chart_ablation()
    chart_contamination()
    chart_ptc_evolution()
    chart_speed_ratio()
