#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ITD-ILD index = (VE_ITD - VE_ILD) / (VE_ITD + VE_ILD) vs channel number.
All units are plotted. Filled circle = both VE_ITD and VE_ILD significant;
open circle = at least one of them not significant. Colour = animal.
One figure, one column per stimulus rate.
x = index (-1 = ILD only, +1 = ITD only), y = channel number
(channel 1 = most superficial, at the top).
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

BOX_ON_BOTH_SIG_ONLY = False   # False: boxes use all units; True: only both-significant units
INVERT_Y = True                # True: channel 1 (most superficial) at the top
DOT_JITTER = 0.25
BOX_WIDTH = 0.7
XLIM = (-1.05, 1.05)
ARROW_X = -0.20

with np.load(os.path.join(results_path, npz_name), allow_pickle=True) as d:
    ve_value = d['VE_value']       # (nfiles, nrates, nchans, 2)
    ve_stats = d['VE_stats']       # 1 = significant
    unit_index = d['unit_index']   # (nfiles, nchans)
    fnames = d['fnames'].tolist()

nfiles, nrates, nchans = ve_value.shape[:3]

rows = []
for ff in range(nfiles):
    animal = fnames[ff].split('_')[0]
    for cc in np.where(unit_index[ff] == 1)[0]:
        for rr in range(nrates):
            rows.append(dict(
                animal=animal, file=ff, chan=int(cc) + 1,
                rate=stiRate[rr],
                ve_itd=float(ve_value[ff, rr, cc, ITD]),
                ve_ild=float(ve_value[ff, rr, cc, ILD]),
                sig_itd=int(ve_stats[ff, rr, cc, ITD]),
                sig_ild=int(ve_stats[ff, rr, cc, ILD]),
            ))
df = pd.DataFrame(rows)
df['both_sig'] = (df.sig_itd == 1) & (df.sig_ild == 1)

den = df.ve_itd + df.ve_ild
df['index'] = np.where(den > 0, (df.ve_itd - df.ve_ild) / den, np.nan)
n_nan = int(df['index'].isna().sum())
if n_nan:
    print(f'Warning: {n_nan} unit-rate entries have VE_ITD + VE_ILD <= 0 and are excluded')
df = df.dropna(subset=['index']).copy()

print('Units per rate (all / both significant):')
for rate in stiRate:
    s = df[df.rate == rate]
    print(f'  {rate} pps: {len(s)} / {int(s.both_sig.sum())}')

animals = sorted(df.animal.unique())
animal_colors = {a: plt.colormaps['tab20'](i % 20) for i, a in enumerate(animals)}
animal_labels = {a: f'Rat {i + 1}' for i, a in enumerate(animals)}

rng = np.random.default_rng(0)
unit_key = df.file.astype(str) + '_' + df.chan.astype(str)
jit_map = {k: rng.uniform(-DOT_JITTER, DOT_JITTER) for k in unit_key.unique()}
df['y'] = df.chan + unit_key.map(jit_map)

# ---------------------------------------------------------------------------
# 2. Plot
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, nrates, figsize=(15, 9), sharey=True, sharex=True,
                         gridspec_kw={'wspace': 0.08})
all_chans = np.arange(1, nchans + 1)

for ax, rate in zip(axes, stiRate):
    sub = df[df.rate == rate]
    box_src = sub[sub.both_sig] if BOX_ON_BOTH_SIG_ONLY else sub

    chans = [c for c in all_chans if (box_src.chan == c).any()]
    data = [box_src.loc[box_src.chan == c, 'index'].values for c in chans]
    if data:
        ax.boxplot(data, positions=chans, vert=False, widths=BOX_WIDTH,
                   showfliers=False, patch_artist=True,
                   boxprops=dict(facecolor='whitesmoke', edgecolor='k', linewidth=0.8),
                   medianprops=dict(color='k', linewidth=1.5),
                   whiskerprops=dict(linewidth=0.8), capprops=dict(linewidth=0.8),
                   zorder=1)

    for animal in animals:
        a = sub[sub.animal == animal]
        if a.empty:
            continue
        col = animal_colors[animal]
        f, o = a[a.both_sig], a[~a.both_sig]
        ax.scatter(f['index'], f.y, marker='o', s=22, color=col,
                   edgecolors='k', linewidth=0.3, alpha=0.85, zorder=3)
        ax.scatter(o['index'], o.y, marker='o', s=22, facecolors='none',
                   edgecolors=col, linewidth=0.9, alpha=0.85, zorder=3)

    ax.axvline(0, color='k', linestyle='--', linewidth=1, zorder=2)
    ax.set_title(f'{rate} pps')
    ax.set_xlabel('(VE_ITD - VE_ILD) / (VE_ITD + VE_ILD)')
    ax.set_xlim(*XLIM)
    ax.set_yticks(all_chans)
    if INVERT_Y:
        ax.set_ylim(nchans + 0.5, 0.5)
    else:
        ax.set_ylim(0.5, nchans + 0.5)
    ax.grid(axis='x', linestyle=':', alpha=0.4)
    if rate == stiRate[0]:
        ax.set_ylabel('Channel number')
        ax.yaxis.set_label_coords(-0.08, 0.5)

# depth arrow: top = channel 1 (superficial), bottom = deepest channel
ax0 = axes[0]
ARROW_TOP, ARROW_BOT = 0.62, 0.38
ax0.annotate('', xy=(ARROW_X, ARROW_BOT), xytext=(ARROW_X, ARROW_TOP),
             xycoords='axes fraction', textcoords='axes fraction',
             arrowprops=dict(arrowstyle='-|>', lw=2, color='k',
                             mutation_scale=18, shrinkA=0, shrinkB=0),
             annotation_clip=False)
ax0.text(ARROW_X - 0.04, ARROW_TOP, 'Superficial', transform=ax0.transAxes,
         rotation=90, ha='center', va='top', fontsize=11, clip_on=False)
ax0.text(ARROW_X - 0.04, ARROW_BOT, 'Deep', transform=ax0.transAxes,
         rotation=90, ha='center', va='bottom', fontsize=11, clip_on=False)



# legends
def h(label, face, edge):
    return Line2D([0], [0], marker='o', color='w', markerfacecolor=face,
                  markeredgecolor=edge, markersize=8, label=label)

sig_handles = [h('Both significant', 'dimgrey', 'k'),
               h('Not both significant', 'none', 'dimgrey')]
animal_handles = [h(animal_labels[a], animal_colors[a], 'k') for a in animals]

# significance legend: top centre of the figure, horizontal
fig.legend(handles=sig_handles, loc='upper right', ncol=2,
           bbox_to_anchor=(0.9, 0.95), fontsize=10, frameon=True)

# animal legend: right side
fig.legend(handles=animal_handles, title='Animal', loc='upper left',
           bbox_to_anchor=(0.90, 0.92), fontsize=9, title_fontsize=10)

fig.tight_layout(rect=[0, 0, 0.90, 0.94])   # leave room at the top for the legend
# legends
# def h(label, face, edge):
#     return Line2D([0], [0], marker='o', color='w', markerfacecolor=face,
#                   markeredgecolor=edge, markersize=8, label=label)

# sig_handles = [h('Both significant', 'dimgrey', 'k'),
#                h('Not both significant', 'none', 'dimgrey')]
# animal_handles = [h(animal_labels[a], animal_colors[a], 'k') for a in animals]
# leg1 = fig.legend(handles=sig_handles, title='Significance', loc='upper left',
#                   bbox_to_anchor=(0.90, 0.92), fontsize=9, title_fontsize=10)
# fig.add_artist(leg1)
# fig.legend(handles=animal_handles, title='Animal', loc='upper left',
#            bbox_to_anchor=(0.90, 0.75), fontsize=9, title_fontsize=10)

# fig.tight_layout(rect=[0, 0, 0.90, 1])
out = os.path.join(Figure_path, 'ITD_ILD_index_allUnits_boxplot_by_channel.png')
fig.savefig(out, dpi=300, bbox_inches='tight')
print('Saved', out)

plt.show()