#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 27 17:13:25 2026

@author: feipeng
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 14 16:52:45 2026

This script compares variance explained (VE) by ILD and ITD for AMUA averaged
over two time windows (3-30 ms and 30-200 ms) at 100 pps and 900 pps separately.

It generates a figure with two vertically stacked panels:
- Top: VE_ILD distributions across animals for four conditions.
- Bottom: VE_ITD distributions for the same conditions.

Violin plots include all responsive units. All units are plotted as dots
coloured by animal (significant and nonsignificant units are not distinguished
in the plot).

Wilcoxon signed-rank tests compare 3-30 ms vs. 30-200 ms for 100 pps and
900 pps separately in each panel. Significant results are annotated with '*'
using a Bonferroni-corrected threshold of p < 0.0125.

@author: fei
"""

import numpy as np
import os
from sys import platform
import matplotlib.pyplot as plt
from scipy import stats
from matplotlib.lines import Line2D
import pandas as pd

np.random.seed(0)  # reproducible jitter

# Set paths based on operating system
if platform == "darwin":
    results_path = '/Volumes/Disk/4_ITD_ILD_cueTrading/Data/AMUA_analysis_v2/'
    Figure_path = '/Volumes/Disk/4_ITD_ILD_cueTrading/Figures_paper/'
elif platform == "linux":
    results_path = '/disks/CIdata/2023_02_23Results/'
    Figure_path = '/disks/CIdata/2_ITD_ILD_CueTrading/Figures/'
else:
    results_path = './Data/AMUA_analysis_v2/'
    Figure_path = './Figures/'

# Create output directory
os.makedirs(Figure_path, exist_ok=True)

# Configuration
conditions = ['3-30 ms', '30-200 ms', '3-30 ms', '30-200 ms']
group_labels = ['100 pps', '900 pps']

npz_file_3to30 = os.path.join(results_path, 'all_files_VE_data_v2.npz')
npz_file_30to200 = os.path.join(results_path, 'all_files_VE_data_30to200ms.npz')

figname = 'Fig3_VE_ILD_ITD_TimeWindows_100_900pps_Separate'

#%% Load precomputed data
if not (os.path.exists(npz_file_3to30) and os.path.exists(npz_file_30to200)):
    print(f"Error: One or both data files missing: {npz_file_3to30}, {npz_file_30to200}")
    exit()

# Load 3-30 ms data
with np.load(npz_file_3to30, allow_pickle=True) as data:
    ve_value_3to30 = data['VE_value']       # (nfiles, nrates, nchans, 2)
    ve_stats_3to30 = data['VE_stats']       # (nfiles, nrates, nchans, 2)
    unit_index_3to30 = data['unit_index']   # (nfiles, nchans)
    all_fnames = data['fnames'].tolist()

# Load 30-200 ms data
with np.load(npz_file_30to200, allow_pickle=True) as data:
    ve_value_30to200 = data['VE_value']
    ve_stats_30to200 = data['VE_stats']
    unit_index_30to200 = data['unit_index']

# Verify data dimensions
nfiles, nrates, nchans = ve_value_3to30.shape[:3]
expected_shape = (nfiles, nrates, nchans, 2)

if (
    ve_value_3to30.shape != expected_shape
    or ve_stats_3to30.shape != expected_shape
    or ve_value_30to200.shape != expected_shape
    or ve_stats_30to200.shape != expected_shape
    or unit_index_3to30.shape != (nfiles, nchans)
    or unit_index_30to200.shape != (nfiles, nchans)
):
    print(f"Error: Data dimension mismatch. Expected {expected_shape}, got "
          f"VE_value_3to30: {ve_value_3to30.shape}, "
          f"VE_stats_3to30: {ve_stats_3to30.shape}, "
          f"VE_value_30to200: {ve_value_30to200.shape}, "
          f"VE_stats_30to200: {ve_stats_30to200.shape}")
    exit()

#%% Combine data for 100 pps and 900 pps
# stiRate = [1, 100, 900]
# rate index 1 = 100 pps; rate index 2 = 900 pps

# Combined condition order:
# 0: 100 pps, 3-30 ms
# 1: 100 pps, 30-200 ms
# 2: 900 pps, 3-30 ms
# 3: 900 pps, 30-200 ms

ve_value = np.zeros((nfiles, 4, nchans, 2), dtype='float32')
ve_stats = np.zeros((nfiles, 4, nchans, 2), dtype='float32')
unit_index = np.zeros((nfiles, nchans), dtype='float32')

for ff in range(nfiles):
    unit_index[ff] = np.logical_and(
        unit_index_3to30[ff],
        unit_index_30to200[ff]
    ).astype('float32')

    ve_value[ff, 0, :, :] = ve_value_3to30[ff, 1, :, :]
    ve_value[ff, 1, :, :] = ve_value_30to200[ff, 1, :, :]
    ve_value[ff, 2, :, :] = ve_value_3to30[ff, 2, :, :]
    ve_value[ff, 3, :, :] = ve_value_30to200[ff, 2, :, :]

    ve_stats[ff, 0, :, :] = ve_stats_3to30[ff, 1, :, :]
    ve_stats[ff, 1, :, :] = ve_stats_30to200[ff, 1, :, :]
    ve_stats[ff, 2, :, :] = ve_stats_3to30[ff, 2, :, :]
    ve_stats[ff, 3, :, :] = ve_stats_30to200[ff, 2, :, :]

#%% Identify responsive animals
ild_units = []
itd_units = []
animal_ids = []

unique_animals = sorted(set([fname.split('_')[0] for fname in all_fnames]))

responsive_animals = []
for animal_id in unique_animals:
    animal_files = [
        ff for ff, fname in enumerate(all_fnames)
        if fname.startswith(animal_id)
    ]

    if not animal_files:
        continue

    has_responsive_unit = False
    for ff in animal_files:
        if np.any(unit_index[ff] == 1):
            has_responsive_unit = True
            break

    if has_responsive_unit:
        responsive_animals.append(animal_id)

# Assign colours and labels
animal_colors = {
    animal_id: plt.colormaps['tab20'](i % 20 / 20)
    for i, animal_id in enumerate(sorted(responsive_animals))
}

animal_labels = {
    animal_id: f'Rat {i + 1}'
    for i, animal_id in enumerate(sorted(responsive_animals))
}

#%% Collect all responsive units with per-condition significance flags
# ve_stats dimensions:
#   [file, condition, channel, metric]
# metric 0 = ITD
# metric 1 = ILD

for animal_id in responsive_animals:
    animal_files = [
        ff for ff, fname in enumerate(all_fnames)
        if fname.startswith(animal_id)
    ]

    for ff in animal_files:
        responsive_channels = np.where(unit_index[ff] == 1)[0]

        for cc in responsive_channels:

            # Per-condition significance: shape (4,)
            ild_sig_flags = ve_stats[ff, :, cc, 1] == 1
            itd_sig_flags = ve_stats[ff, :, cc, 0] == 1

            ild_units.append({
                'values': ve_value[ff, :, cc, 1],
                'sig_flags': ild_sig_flags.astype(bool),
                'animal': animal_id,
                'file': ff,
                'unit': cc,
            })

            itd_units.append({
                'values': ve_value[ff, :, cc, 0],
                'sig_flags': itd_sig_flags.astype(bool),
                'animal': animal_id,
                'file': ff,
                'unit': cc,
            })

    animal_ids.append(animal_id)

#%% Summary statistics and Wilcoxon tests
def print_summary_stats(units_data, ve_type, animal_ids, animal_labels):
    """
    Compute per-animal median VE values and perform Wilcoxon signed-rank tests.

    All responsive units are included in the per-animal medians.
    """

    if not units_data:
        print(f"No {ve_type} units found")
        return {}

    rate_data = [[] for _ in range(4)]

    valid_animal_ids = sorted(set(unit['animal'] for unit in units_data))

    for animal_id in valid_animal_ids:
        animal_units = [
            unit for unit in units_data
            if unit['animal'] == animal_id
        ]

        if not animal_units:
            continue

        animal_values = np.array([unit['values'] for unit in animal_units])

        if animal_values.size == 0:
            continue

        median_values = np.median(animal_values, axis=0)

        for j in range(4):
            rate_data[j].append(median_values[j])

    print(f"\n=== {ve_type} Median VE Summary Statistics (Across Animals) ===")
    print(f"Number of animals: {len(valid_animal_ids)}")

    for j, cond in enumerate(conditions):
        rate_data_arr = np.array(rate_data[j])

        if len(rate_data_arr) == 0:
            continue

        group = group_labels[j // 2]

        print(f"\nCondition {cond} ({group}):")
        print(f"  Median: {np.median(rate_data_arr):.2f}%")
        print(f"  Mean: {np.mean(rate_data_arr):.2f}%")
        print(f"  Q1 (25th %ile): {np.percentile(rate_data_arr, 25):.2f}%")
        print(f"  Q3 (75th %ile): {np.percentile(rate_data_arr, 75):.2f}%")
        print(
            f"  IQR: "
            f"{np.percentile(rate_data_arr, 75) - np.percentile(rate_data_arr, 25):.2f}%"
        )
        print(f"  Min: {np.min(rate_data_arr):.2f}%")
        print(f"  Max: {np.max(rate_data_arr):.2f}%")
        print(f"  Std: {np.std(rate_data_arr):.2f}%")

    print(f"\nPer-Animal Median VE Values for {ve_type}:")
    for animal_id in valid_animal_ids:
        animal_units = [
            unit for unit in units_data
            if unit['animal'] == animal_id
        ]

        if animal_units:
            animal_values = np.array([
                unit['values'] for unit in animal_units
            ])
            median_values = np.median(animal_values, axis=0)
            print(
                f"  {animal_labels[animal_id]} "
                f"(ID: {animal_id}): {median_values}"
            )

    p_values = {}

    if all(len(rate_data[j]) > 0 for j in [0, 1, 2, 3]) and len(rate_data[0]) > 1:
        print(
            f"\nWilcoxon Signed-Rank Tests for {ve_type} "
            f"(Bonferroni-corrected alpha = 0.0125):"
        )

        pairs = [
            (0, 1, '100 pps: 3-30 ms vs. 30-200 ms'),
            (2, 3, '900 pps: 3-30 ms vs. 30-200 ms'),
        ]

        for i, j, label in pairs:
            stat, p_val = stats.wilcoxon(
                rate_data[i],
                rate_data[j],
                alternative='two-sided'
            )

            p_values[(i, j)] = p_val

            print(f"  {label}:")
            print(f"    Statistic: {stat:.2f}")
            print(f"    p-value: {p_val:.4f}")
            print(
                f"    Significant: "
                f"{'Yes' if p_val < 0.0125 else 'No'} "
                f"(Bonferroni-corrected)"
            )
    else:
        print(
            f"\nInsufficient data for Wilcoxon tests for {ve_type} "
            f"(need at least 2 animals with data)"
        )

    return p_values

#%% Violin plotting function
def plot_violins_per_animal(
    units_data,
    ax,
    title,
    ve_type,
    conditions,
    animal_ids,
    p_values,
    animal_colors,
    animal_labels
):
    """
    Plot violin distributions for all responsive units.

    All units are plotted as dots in their animal's colour, regardless of
    statistical significance.
    """

    if not units_data:
        print(f"No {ve_type} units found")
        ax.text(
            0.5, 0.5,
            f'No {ve_type} units',
            ha='center',
            va='center',
            fontsize=10
        )
        ax.set_xlabel('Time Window')
        ax.set_ylabel(f'VE_{ve_type} (%)')
        ax.set_title(title)
        ax.set_ylim(0, 100)
        return

    n_conditions = len(conditions)
    valid_animal_ids = sorted(set(unit['animal'] for unit in units_data))
    n_animals = len(valid_animal_ids)

    # X-axis positions
    bar_width = 0.15
    group_width = 1.0

    # Collect data by animal and condition
    all_data = []
    max_y = 0

    for i in range(n_conditions):
        for animal_id in valid_animal_ids:
            animal_units = [
                unit for unit in units_data
                if unit['animal'] == animal_id
            ]

            if not animal_units:
                continue

            data_array = np.array([
                unit['values'][i] for unit in animal_units
            ])

            pos = (
                i * (n_animals * bar_width + group_width - bar_width)
                + valid_animal_ids.index(animal_id) * bar_width
            )

            all_data.append({
                'animal_id': animal_id,
                'position': pos,
                'values': data_array,
            })

            if data_array.size > 0:
                max_y = max(max_y, np.max(data_array))

    plotted_entries = [
        entry for entry in all_data
        if entry['values'].size > 0
    ]

    # Plot violins using all responsive units
    violin_parts = ax.violinplot(
        [entry['values'] for entry in plotted_entries],
        positions=[entry['position'] for entry in plotted_entries],
        widths=bar_width * 0.8,
        showmedians=True,
    )

    # Colour violins by animal
    for idx, pc in enumerate(violin_parts['bodies']):
        animal_id = plotted_entries[idx]['animal_id']
        pc.set_facecolor(animal_colors[animal_id])
        pc.set_alpha(0.25)
        pc.set_edgecolor('black')
        pc.set_linewidth(1)

    # Style median lines
    violin_parts['cmedians'].set_color('black')
    violin_parts['cmedians'].set_linewidth(2)

    for part in ['cmaxes', 'cmins', 'cbars']:
        if part in violin_parts:
            violin_parts[part].set_alpha(0)

    # Add quartile lines and individual unit dots
    for entry in plotted_entries:
        pos = entry['position']

        q1 = np.percentile(entry['values'], 25)
        q3 = np.percentile(entry['values'], 75)

        ax.hlines(
            q1,
            pos - bar_width / 2,
            pos + bar_width / 2,
            colors='black',
            linewidth=1,
            linestyle='--',
            alpha=0.8
        )

        ax.hlines(
            q3,
            pos - bar_width / 2,
            pos + bar_width / 2,
            colors='black',
            linewidth=1,
            linestyle='--',
            alpha=0.8
        )

        jitter_strength = bar_width * 0.3
        x_positions = np.random.uniform(
            pos - jitter_strength,
            pos + jitter_strength,
            len(entry['values'])
        )

        ax.scatter(
            x_positions,
            entry['values'],
            color=animal_colors[entry['animal_id']],
            s=10,
            alpha=0.6,
            edgecolor='grey',
            linewidth=0.5
        )

    # Add statistical annotations
    sig_pairs = [
        (0, 1, '100 pps: 3-30 ms vs. 30-200 ms'),
        (2, 3, '900 pps: 3-30 ms vs. 30-200 ms'),
    ]

    bar_heights = [max_y + 5, max_y + 15]

    for idx, (i, j, _) in enumerate(sig_pairs):
        if (i, j) in p_values and p_values[(i, j)] < 0.0125:
            start_i = (
                i * (n_animals * bar_width + group_width - bar_width)
                + (n_animals - 1) * bar_width / 2
            )

            start_j = (
                j * (n_animals * bar_width + group_width - bar_width)
                + (n_animals - 1) * bar_width / 2
            )

            y_bar = bar_heights[idx]

            ax.plot(
                [start_i, start_j],
                [y_bar, y_bar],
                color='black',
                linewidth=1.5
            )

            ax.text(
                (start_i + start_j) / 2,
                y_bar,
                '*',
                ha='center',
                va='bottom',
                fontsize=8
            )

    # Axis properties
    xtick_positions = [
        i * (n_animals * bar_width + group_width - bar_width)
        + (n_animals - 1) * bar_width / 2
        for i in range(n_conditions)
    ]

    ax.set_xticks(xtick_positions)
    ax.set_xticklabels(
        ['Onset', 'Sustained', 'Onset', 'Sustained'],
        ha='center'
    )

    ax.set_ylabel(f'VE_{ve_type} (%)', fontsize=12)
    ax.set_title(title)
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3)

    # Add group labels
    group_positions = [
        (xtick_positions[0] + xtick_positions[1]) / 2,
        (xtick_positions[2] + xtick_positions[3]) / 2,
    ]

    for pos, label in zip(group_positions, group_labels):
        ax.text(
            pos,
            -0.15,
            label,
            ha='center',
            va='top',
            fontsize=10,
            transform=ax.get_xaxis_transform()
        )

    # Legend beside the ITD panel (animals only)
    if ve_type == 'ITD':
        legend_elements = [
            Line2D(
                [0], [0],
                color=animal_colors[animal_id],
                lw=2,
                label=label
            )
            for animal_id, label in animal_labels.items()
            if animal_id in valid_animal_ids
        ]

        ax.legend(
            handles=legend_elements,
            loc='center left',
            bbox_to_anchor=(1.02, 0.5),
            borderaxespad=0.,
            fontsize=8,
        )


#%% Count nonsignificant values per animal per condition
def count_nonsignificant_values(units_data, ve_type, conditions, group_labels, animal_labels):
    """
    Count nonsignificant VE values per animal per condition.

    Returns a DataFrame with values formatted as 'count (percentage%)'.
    """

    if not units_data:
        print(f"\nNo {ve_type} units found; nothing to count.")
        return pd.DataFrame()

    valid_animal_ids = sorted(set(unit['animal'] for unit in units_data))

    condition_names = []
    for i, cond in enumerate(conditions):
        group = group_labels[i // 2]
        condition_names.append(f"{cond} ({group})")

    records = []

    for animal_id in valid_animal_ids:
        animal_units = [
            unit for unit in units_data
            if unit['animal'] == animal_id
        ]

        total_units = len(animal_units)
        counts = np.zeros(4, dtype=int)

        for unit in animal_units:
            sig_flags = np.asarray(unit['sig_flags'], dtype=bool)
            counts += (~sig_flags).astype(int)

        record = {
            'Animal': animal_labels.get(animal_id, animal_id),
            'Animal_ID': animal_id,
            'Total_units': total_units,
        }

        for i, cond_name in enumerate(condition_names):
            count = counts[i]
            percentage = 100 * count / total_units if total_units > 0 else np.nan

            record[cond_name] = f'{count} ({percentage:.1f}%)'

        records.append(record)

    count_df = pd.DataFrame(records)

    print(f"\n=== Number and Percentage of Nonsignificant {ve_type} Values per Animal per Condition ===")
    print(count_df.to_string(index=False))

    return count_df


#%% Count and save nonsignificant values: counts and percentages
count_df_ild = count_nonsignificant_values(
    ild_units,
    'ILD',
    conditions,
    group_labels,
    animal_labels
)

count_df_itd = count_nonsignificant_values(
    itd_units,
    'ITD',
    conditions,
    group_labels,
    animal_labels
)

csv_ild_path = os.path.join(
    Figure_path,
    f'{figname}_nonsignificant_counts_percentages_ILD.csv'
)

csv_itd_path = os.path.join(
    Figure_path,
    f'{figname}_nonsignificant_counts_percentages_ITD.csv'
)

count_df_ild.to_csv(csv_ild_path, index=False)
count_df_itd.to_csv(csv_itd_path, index=False)

print(f"\nSaved ILD nonsignificant counts and percentages to: {csv_ild_path}")
print(f"Saved ITD nonsignificant counts and percentages to: {csv_itd_path}")

#%% Generate violin plots
# fig, (ax_ild, ax_itd) = plt.subplots(
#     2, 1,
#     figsize=(9, 10),
#     sharey=True
# )

# p_values_ild = print_summary_stats(
#     ild_units,
#     'ILD',
#     animal_ids,
#     animal_labels
# )

# plot_violins_per_animal(
#     ild_units,
#     ax_ild,
#     ' ',
#     'ILD',
#     conditions,
#     animal_ids,
#     p_values_ild,
#     animal_colors,
#     animal_labels
# )

# p_values_itd = print_summary_stats(
#     itd_units,
#     'ITD',
#     animal_ids,
#     animal_labels
# )

# plot_violins_per_animal(
#     itd_units,
#     ax_itd,
#     ' ',
#     'ITD',
#     conditions,
#     animal_ids,
#     p_values_itd,
#     animal_colors,
#     animal_labels
# )

# plt.suptitle(
#     'VE by ILD and ITD Across Time Windows at 100 and 900 pps'
# )

# plt.tight_layout(rect=[0, 0.05, 0.85, 0.96])

# plt.savefig(
#     os.path.join(Figure_path, f'{figname}_v2.png'),
#     dpi=300
# )

# plt.show()

#%%
#%% Generate violin plots
# Top panel: VE_ITD, bottom panel: VE_ILD
fig, (ax_itd, ax_ild) = plt.subplots(
    2, 1,
    figsize=(9, 10),
    sharey=True
)

p_values_itd = print_summary_stats(
    itd_units,
    'ITD',
    animal_ids,
    animal_labels
)

plot_violins_per_animal(
    itd_units,
    ax_itd,
    ' ',
    'ITD',
    conditions,
    animal_ids,
    p_values_itd,
    animal_colors,
    animal_labels
)

p_values_ild = print_summary_stats(
    ild_units,
    'ILD',
    animal_ids,
    animal_labels
)

plot_violins_per_animal(
    ild_units,
    ax_ild,
    ' ',
    'ILD',
    conditions,
    animal_ids,
    p_values_ild,
    animal_colors,
    animal_labels
)

plt.suptitle(
    'VE by ITD and ILD Across Time Windows at 100 and 900 pps'
)

plt.tight_layout(rect=[0, 0.05, 0.85, 0.96])

plt.savefig(
    os.path.join(Figure_path, f'{figname}_v2.png'),
    dpi=300
)

plt.show()