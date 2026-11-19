#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Box plot of VE_ILD across channels (1-32), pooled across animals.
One panel per stimulus rate. Each dot is one responsive unit, coloured by animal.
"""

import os
from sys import platform

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

np.random.seed(0)  # reproducible jitter

# ---------------------------------------------------------------------------
# 1. Settings
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

npz_name = 'all_files_VE_data_3to30ms.npz'   # choose the time window you want
stiRate = [1, 100, 900]                      # pps, same order as the data
VE_TYPE = 'ILD'                              # 'ILD' or 'ITD'
ONLY_SIGNIFICANT = False                     # True: keep only units with significant VE of VE_TYPE

METRIC_INDEX = {'ITD': 0, 'ILD': 1}[VE_TYPE]  # last-axis index in VE_value / VE_stats

# ---------------------------------------------------------------------------
# 2. Load data
# ---------------------------------------------------------------------------
npz_file = os.path.join(results_path, npz_name)
if not os.path.exists(npz_file):
    raise FileNotFoundError(f'Data file not found: {npz_file}')

with np.load(npz_file, allow_pickle=True) as data:
    ve_value = data['VE_value']        # (nfiles, nrates, nchans, 2)
    ve_stats = data['VE_stats']        # (nfiles, nrates, nchans, 2)
    unit_index = data['unit_index']    # (nfiles, nchans), 1 = responsive
    fnames = data['fnames'].tolist()

nfiles, nrates, nchans = ve_value.shape[:3]
animal_of_file = [f.split('_')[0] for f in fnames]

print(f'Loaded {npz_name}: {nfiles} files, {nrates} rates, {nchans} channels')
print(f'Responsive units: {int(np.sum(unit_index))}')



#%%
import pandas as pd

animal_of_file = [f.split('_')[0] for f in fnames]
rows = []

for animal in sorted(set(animal_of_file)):
    files = [ff for ff, a in enumerate(animal_of_file) if a == animal]
    resp_files = [ff for ff in files if np.any(unit_index[ff] == 1)]
    n_units = int(sum(np.sum(unit_index[ff] == 1) for ff in files))
    rows.append({
        'Animal': animal,
        'Penetrations (files)': len(files),
        'Penetrations with responsive units': len(resp_files),
        'Responsive units': n_units,
    })

pen_df = pd.DataFrame(rows)
pen_df.loc[len(pen_df)] = ['Total'] + pen_df.iloc[:, 1:].sum().tolist()
print(pen_df.to_string(index=False))

pen_df.to_csv(os.path.join(Figure_path, 'penetrations_per_animal.csv'), index=False)


#%%
# ---------------------------------------------------------------------------
# 3. Organise data: VE per rate and channel, with the animal of each unit
# ---------------------------------------------------------------------------
ch_values = [[[] for _ in range(nchans)] for _ in range(nrates)]
ch_animals = [[[] for _ in range(nchans)] for _ in range(nrates)]

for ff in range(nfiles):
    for cc in np.where(unit_index[ff] == 1)[0]:
        for rr in range(nrates):
            if ONLY_SIGNIFICANT and ve_stats[ff, rr, cc, METRIC_INDEX] != 1:
                continue
            ch_values[rr][cc].append(ve_value[ff, rr, cc, METRIC_INDEX])
            ch_animals[rr][cc].append(animal_of_file[ff])

# Colours and labels per animal (animals with at least one responsive unit)
animals = sorted(set(animal_of_file[ff] for ff in range(nfiles)
                     if np.any(unit_index[ff] == 1)))
animal_colors = {a: plt.colormaps['tab20'](i % 20 / 20) for i, a in enumerate(animals)}
animal_labels = {a: f'Rat {i + 1}' for i, a in enumerate(animals)}

# ---------------------------------------------------------------------------
# 4. Plot
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(nrates, 1, figsize=(12, 3.6 * nrates),
                         sharex=True, sharey=True)
axes = np.atleast_1d(axes)

for rr, ax in enumerate(axes):

    channels_with_data = [cc for cc in range(nchans) if len(ch_values[rr][cc]) > 0]

    # Box plot (one box per channel that has data)
    if channels_with_data:
        ax.boxplot(
            [ch_values[rr][cc] for cc in channels_with_data],
            positions=[cc + 1 for cc in channels_with_data],
            widths=0.6,
            showfliers=False,
            patch_artist=True,
            boxprops=dict(facecolor='lightgrey', alpha=0.5, edgecolor='black'),
            medianprops=dict(color='black', linewidth=2),
            whiskerprops=dict(color='black'),
            capprops=dict(color='black'),
        )

    # Individual units, coloured by animal, plus the unit count above each box
    for cc in channels_with_data:
        values = ch_values[rr][cc]
        x = np.random.uniform(cc + 1 - 0.18, cc + 1 + 0.18, len(values))
        ax.scatter(
            x, values,
            c=[animal_colors[a] for a in ch_animals[rr][cc]],
            s=10, alpha=0.6, edgecolor='grey', linewidth=0.5, zorder=3
        )
        # ax.text(cc + 1, 1.01, f'n={len(values)}',
        #         ha='center', va='bottom', fontsize=6, rotation=90,
        #         transform=ax.get_xaxis_transform())

    ax.set_title(f'{stiRate[rr]} pps', pad=22)
    ax.set_ylabel(f'VE_{VE_TYPE} (%)')
    ax.set_xticks(range(1, nchans + 1))
    ax.set_xlim(0.3, nchans + 0.7)
    # ax.set_ylim(bottom=0)
    ax.grid(True, axis='y', alpha=0.3)

# After the "for rr, ax in enumerate(axes):" loop
all_vals = [v for rr in range(nrates) for cc in range(nchans) for v in ch_values[rr][cc]]
y_max = max(all_vals) if all_vals else 100
axes[0].set_ylim(0, y_max * 1.05)   # sharey=True applies it to all panels

axes[-1].set_xlabel('Channel Number')


# Legend (animals) beside the top panel
legend_elements = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor=animal_colors[a],
           markeredgecolor='grey', markersize=7, label=animal_labels[a])
    for a in animals
]
axes[0].legend(handles=legend_elements, loc='upper left',
               bbox_to_anchor=(1.01, 1.0), fontsize=9, frameon=True)

title_extra = f' ({VE_TYPE}-significant units only)' if ONLY_SIGNIFICANT else ''
fig.suptitle(f'VE_{VE_TYPE} across channels, pooled across animals{title_extra}')
fig.tight_layout(rect=[0, 0, 0.9, 0.97])

suffix = f'_{VE_TYPE}sigOnly' if ONLY_SIGNIFICANT else ''
out_file = os.path.join(Figure_path, f'VE_{VE_TYPE}_boxplot_by_channel{suffix}.png')
fig.savefig(out_file, dpi=300, bbox_inches='tight')
print(f'Saved figure to {out_file}')
plt.show()