#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VE_ITD and VE_ILD vs channel number, one figure per cue, one column per rate.
x = VE (%), y = channel number (channel 1 = deepest, at the bottom).
Per channel: box plot across units, with individual units overlaid (colour = animal).
An arrow on the left indicates recording depth (deep at ch 1, superficial at ch 32).
"""

import os
from sys import platform

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# ---------------------------------------------------------------------------
# 1. Settings and data loading
# ---------------------------------------------------------------------------
if platform == "darwin":
    results_path = '/Volumes/Disk/4_ITD_ILD_cueTrading/Data/AMUA_analysis_v2/'
    Figure_path = '/Volumes/Disk/4_ITD_ILD_cueTrading/Figures_paper/'
elif platform == "linux":
    results_path = '/disks/CIdata/2023_02_23Results/'
    Figure_path = '/disks/CIdata/2_ITD_ILD_CueTrading/Figures/'
else:
    results_path = './Data/AMUA_analysis_v2/'
    Figure_path = './Figures/'
os.makedirs(Figure_path, exist_ok=True)

npz_name = 'all_files_VE_data_3to30ms.npz'
stiRate = [1, 100, 900]
ITD, ILD = 0, 1

INVERT_Y = True        # False: channel 1 (deepest) at the bottom
DOT_JITTER = 0.25       # vertical jitter of dots within each box (channels)
BOX_WIDTH = 0.7         # box height in channel units
XMAX = 95
ARROW_X = -0.20         # arrow position in axes fraction (more negative = further left)

with np.load(os.path.join(results_path, npz_name), allow_pickle=True) as d:
    ve_value = d['VE_value']       # (nfiles, nrates, nchans, 2)
    unit_index = d['unit_index']   # (nfiles, nchans)
    fnames = d['fnames'].tolist()

nfiles, nrates, nchans = ve_value.shape[:3]

rows = []
for ff in range(nfiles):
    animal = fnames[ff].split('_')[0]
    for cc in np.where(unit_index[ff] == 1)[0]:
        for rr in range(nrates):
            rows.append(dict(
                animal=animal, file=ff, chan=int(cc) + 1,   # 1-based channel number
                rate=stiRate[rr],
                ve_itd=float(ve_value[ff, rr, cc, ITD]),
                ve_ild=float(ve_value[ff, rr, cc, ILD]),
            ))
df = pd.DataFrame(rows)

animals = sorted(df.animal.unique())
animal_colors = {a: plt.colormaps['tab20'](i % 20) for i, a in enumerate(animals)}
animal_labels = {a: f'Rat {i + 1}' for i, a in enumerate(animals)}
print(f'{len(animals)} animals, {len(df) // nrates} responsive units')

rng = np.random.default_rng(0)
unit_key = df.file.astype(str) + '_' + df.chan.astype(str)
jit_map = {k: rng.uniform(-DOT_JITTER, DOT_JITTER) for k in unit_key.unique()}
df['y'] = df.chan + unit_key.map(jit_map)   # same jitter for a unit across rates

# ---------------------------------------------------------------------------
# 2. Plot: one figure per cue
# ---------------------------------------------------------------------------
def plot_cue(cue):
    ve_col = f've_{cue.lower()}'
    fig, axes = plt.subplots(1, nrates, figsize=(15, 9), sharey=True, sharex=True,
                             gridspec_kw={'wspace': 0.08})
    all_chans = np.arange(1, nchans + 1)

    for ax, rate in zip(axes, stiRate):
        sub = df[df.rate == rate]

        # one box per channel (only channels with data)
        chans = [c for c in all_chans if (sub.chan == c).any()]
        data = [sub.loc[sub.chan == c, ve_col].values for c in chans]
        ax.boxplot(data, positions=chans, vert=False, widths=BOX_WIDTH,
                   showfliers=False, patch_artist=True,
                   boxprops=dict(facecolor='whitesmoke', edgecolor='k', linewidth=0.8),
                   medianprops=dict(color='k', linewidth=1.5),
                   whiskerprops=dict(linewidth=0.8), capprops=dict(linewidth=0.8),
                   zorder=1)

        # individual units on top
        for animal in animals:
            a = sub[sub.animal == animal]
            if a.empty:
                continue
            ax.scatter(a[ve_col], a.y, marker='o', s=18,
                       color=animal_colors[animal], edgecolors='k',
                       linewidth=0.3, alpha=0.85, zorder=3)

        ax.set_title(f'{rate} pps')
        ax.set_xlabel(f'VE_{cue} (%)')
        ax.set_xlim(0, XMAX)
        ax.set_yticks(all_chans)
        if INVERT_Y:
            ax.set_ylim(nchans + 0.5, 0.5)
        else:
            ax.set_ylim(0.5, nchans + 0.5)
        ax.grid(axis='x', linestyle=':', alpha=0.4)
        if rate == stiRate[0]:
            ax.set_ylabel('Channel number')
            ax.yaxis.set_label_coords(-0.08, 0.5)

    # depth arrow on the far left of the first panel (axes coordinates)
    # top = channel 32 (superficial), bottom = channel 1 (deep); arrow points to deep
    ax0 = axes[0]
    ARROW_TOP, ARROW_BOT = 0.62, 0.38     # shorter: closer together = shorter arrow
    ax0.annotate('', xy=(ARROW_X, ARROW_BOT), xytext=(ARROW_X, ARROW_TOP),
                 xycoords='axes fraction', textcoords='axes fraction',
                 arrowprops=dict(arrowstyle='-|>', lw=2, color='k',
                                 mutation_scale=18, shrinkA=0, shrinkB=0),
                 annotation_clip=False)
    ax0.text(ARROW_X - 0.04, ARROW_TOP, 'Superficial', transform=ax0.transAxes,
             rotation=90, ha='center', va='top', fontsize=11, clip_on=False)
    ax0.text(ARROW_X - 0.04, ARROW_BOT, 'Deep', transform=ax0.transAxes,
             rotation=90, ha='center', va='bottom', fontsize=11, clip_on=False)

    # animal legend
    animal_handles = [Line2D([0], [0], marker='o', color='w',
                             markerfacecolor=animal_colors[a], markeredgecolor='k',
                             markersize=8, label=animal_labels[a]) for a in animals]
    fig.legend(handles=animal_handles, title='Animal', loc='upper left',
               bbox_to_anchor=(0.90, 0.92), fontsize=9, title_fontsize=10)

    fig.tight_layout(rect=[0, 0, 0.90, 1])
    out = os.path.join(Figure_path, f'VE_{cue}_boxplot_by_channel.png')
    fig.savefig(out, dpi=300, bbox_inches='tight')
    print('Saved', out)

for cue in ('ITD', 'ILD'):
    plot_cue(cue)

# ---------------------------------------------------------------------------
# 3. Sanity check: units per animal
# ---------------------------------------------------------------------------
for rate in stiRate:
    sub = df[df.rate == rate]
    print(f'\n{rate} pps: units per animal')
    print(sub.groupby('animal').size())

plt.show()