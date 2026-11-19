#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 28 15:03:32 2026

@author: feipeng
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Scatter plot of VE_ITD vs VE_ILD per stimulus rate, with a zoom-in inset.
Two variables are encoded per dot: significance category and animal.
MODE = 'color_by_animal' : colour = animal, marker shape = category
MODE = 'shape_by_animal' : colour = category, marker shape = animal
"""

import os
from sys import platform

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

# ---------------------------------------------------------------------------
# 1. Settings and data loading
# ---------------------------------------------------------------------------
MODE = 'color_by_animal'          # or 'shape_by_animal'

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
INSET_MAX = 20

with np.load(os.path.join(results_path, npz_name), allow_pickle=True) as d:
    ve_value = d['VE_value']       # (nfiles, nrates, nchans, 2)
    ve_stats = d['VE_stats']
    unit_index = d['unit_index']   # (nfiles, nchans)
    fnames = d['fnames'].tolist()

nfiles, nrates, nchans = ve_value.shape[:3]

rows = []
for ff in range(nfiles):
    animal = fnames[ff].split('_')[0]
    for cc in np.where(unit_index[ff] == 1)[0]:
        for rr in range(nrates):
            rows.append(dict(
                animal=animal, rate=stiRate[rr],
                ve_itd=float(ve_value[ff, rr, cc, ITD]),
                ve_ild=float(ve_value[ff, rr, cc, ILD]),
                sig_itd=int(ve_stats[ff, rr, cc, ITD]),
                sig_ild=int(ve_stats[ff, rr, cc, ILD]),
            ))
df = pd.DataFrame(rows)

animals = sorted(df.animal.unique())
print(f'{len(animals)} animals, {len(df) // nrates} responsive units')

# ---------------------------------------------------------------------------
# 2. Visual encoding
# ---------------------------------------------------------------------------
categories = [  # (sig_itd, sig_ild, label)
    (0, 0, 'Neither significant'),
    (1, 0, 'ITD significant only'),
    (0, 1, 'ILD significant only'),
    (1, 1, 'Both significant'),
]
cat_colors = {'Neither significant': 'grey', 'ITD significant only': 'blue',
              'ILD significant only': 'magenta', 'Both significant': 'red'}
cat_markers = {'Neither significant': '^', 'ITD significant only': 's',
               'ILD significant only': 'D', 'Both significant': 'o'}

marker_cycle = ['^', 's', '*', 'o', 'D', 'P', 'X', '*']
animal_markers = {a: marker_cycle[i % len(marker_cycle)] for i, a in enumerate(animals)}
animal_colors = {a: plt.colormaps['tab20'](i % 20) for i, a in enumerate(animals)}
animal_labels = {a: f'Rat {i + 1}' for i, a in enumerate(animals)}


def style(animal, cat_label):
    """Return (colour, marker) for a dot, depending on MODE."""
    if MODE == 'color_by_animal':
        return animal_colors[animal], cat_markers[cat_label]
    return cat_colors[cat_label], animal_markers[animal]


# ---------------------------------------------------------------------------
# 3. Scatter plot
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, nrates, figsize=(14, 5))

for ax, rate in zip(axes, stiRate):
    sub = df[df.rate == rate]
    ax_in = inset_axes(ax, width='30%', height='30%', loc='upper right')

    for animal in animals:
        for s_itd, s_ild, cat_label in categories:
            c = sub[(sub.animal == animal) & (sub.sig_itd == s_itd) & (sub.sig_ild == s_ild)]
            if c.empty:
                continue
            col, mk = style(animal, cat_label)
            for a in (ax, ax_in):
                a.scatter(c.ve_itd, c.ve_ild, color=col, marker=mk, s=16,
                          alpha=0.7, edgecolors='grey', linewidth=0.3)

    ax.axline((0, 0), (1, 1), color='black', linestyle='--')
    ax_in.plot([0, INSET_MAX], [0, INSET_MAX], 'k--', linewidth=1, zorder=1)

    ax.set_xlim(0, 95)
    ax.set_ylim(0, 95)
    ax.set_aspect('equal', adjustable='box')
    if rate == 1:
        ax.set_title('single pulse')
    else:
        ax.set_title(f'{rate} pps')
    ax.set_xlabel('VE_ITD (%)')
    if rate == stiRate[0]:
        ax.set_ylabel('VE_ILD (%)')

    ax_in.set_xlim(0, INSET_MAX)
    ax_in.set_ylim(0, INSET_MAX)
    ax_in.set_xticks([0, INSET_MAX])
    ax_in.set_yticks([0, INSET_MAX])
    ax_in.tick_params(labelsize=8)
    ax_in.set_aspect('equal', adjustable='box')

# ---------------------------------------------------------------------------
# 4. Legends (one for category, one for animal)
# ---------------------------------------------------------------------------
def handle(label, face, marker):
    return Line2D([0], [0], marker=marker, color='w', markerfacecolor=face,
                  markeredgecolor='grey', markersize=7, label=label)

if MODE == 'color_by_animal':
    cat_handles = [handle(lab, 'lightgrey', cat_markers[lab]) for _, _, lab in categories]
    animal_handles = [handle(animal_labels[a], animal_colors[a], 'o') for a in animals]
else:
    cat_handles = [handle(lab, cat_colors[lab], 'o') for _, _, lab in categories]
    animal_handles = [handle(animal_labels[a], 'lightgrey', animal_markers[a]) for a in animals]

leg1 = fig.legend(handles=cat_handles, title='Significance', loc='upper left',
                  bbox_to_anchor=(0.87, 0.92), fontsize=8, title_fontsize=9)
fig.add_artist(leg1)
fig.legend(handles=animal_handles, title='Animal', loc='upper left',
           bbox_to_anchor=(0.87, 0.62), fontsize=8, title_fontsize=9)

fig.tight_layout(rect=[0, 0, 0.86, 1])
out = os.path.join(Figure_path, f'Scatter_VE_ITD_vs_ILD_{MODE}.png')
fig.savefig(out, dpi=300, bbox_inches='tight')
print('Saved', out)

# Units per animal and category (to check that every animal contributes)
for rate in stiRate:
    sub = df[df.rate == rate]
    print(f'\n{rate} pps: units per animal x category')
    print(pd.crosstab(sub.animal, sub.sig_itd.astype(str) + sub.sig_ild.astype(str)
                      ).rename(columns={'00': 'neither', '10': 'ITD only',
                                        '01': 'ILD only', '11': 'both'}))
plt.show()





