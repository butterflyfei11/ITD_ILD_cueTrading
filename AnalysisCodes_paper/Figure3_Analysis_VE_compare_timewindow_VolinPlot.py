#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 03 17:45:00 2025

This script compares variance explained (VE) by ILD and ITD for AMUA averaged over two time windows
(3-30 ms and 30-200 ms) at 100 pps and 900 pps separately. It generates a figure with two panels:
- Left: VE_ILD distributions across animals for four conditions (3-30 ms, 30-200 ms at 100 pps; 3-30 ms, 30-200 ms at 900 pps).
- Right: VE_ITD distributions for the same conditions.
Wilcoxon signed-rank tests compare 3-30 ms vs. 30-200 ms for 100 pps and 900 pps separately in each panel.
Violin plots are used, with statistical annotations for significant Wilcoxon results (p < 0.025, Bonferroni-corrected).
X-tick labels show '3-30 ms' and '30-200 ms', with '100 pps' and '900 pps' centered below the respective condition pairs.

@author: fei
"""

import numpy as np
import os
from sys import platform
import matplotlib.pyplot as plt
from scipy import stats
from matplotlib.lines import Line2D

# Set paths based on operating system
if platform == "darwin":
    results_path = '/Volumes/Disk/4_ITD_ILD_cueTrading/Data/AMUA_analysis_v2/'
    Figure_path = '/Volumes/Disk/4_ITD_ILD_cueTrading/Figures/'
elif platform == "linux":
    results_path = '/disks/CIdata/2023_02_23Results/'
    Figure_path = '/disks/CIdata/2_ITD_ILD_CueTrading/Figures/'
else:
    results_path = './Data/AMUA_analysis_v2/'
    Figure_path = './Figures/'

# Create output directory
os.makedirs(Figure_path, exist_ok=True)

# Configuration
conditions = ['3-30 ms', '30-200 ms', '3-30 ms', '30-200 ms']  # Time windows for 100 pps and 900 pps
group_labels = ['100 pps', '900 pps']  # Group labels for x-axis
npz_file_3to30 = os.path.join(results_path, 'all_files_VE_data_v2.npz')  # 3-30 ms data
npz_file_30to200 = os.path.join(results_path, 'all_files_VE_data_30to200ms.npz')  # 30-200 ms data
figname = 'Fig_VE_ILD_ITD_TimeWindows_100_900pps_Separate'

# Load precomputed data
if not (os.path.exists(npz_file_3to30) and os.path.exists(npz_file_30to200)):
    print(f"Error: One or both data files missing: {npz_file_3to30}, {npz_file_30to200}")
    exit()

# Load 3-30 ms data
with np.load(npz_file_3to30, allow_pickle=True) as data:
    ve_value_3to30 = data['VE_value']  # Shape: (nfiles, nrates, nchans, 2)
    ve_stats_3to30 = data['VE_stats']  # Shape: (nfiles, nrates, nchans, 2)
    unit_index_3to30 = data['unit_index']  # Shape: (nfiles, nchans)
    all_fnames = data['fnames'].tolist()

# Load 30-200 ms data
with np.load(npz_file_30to200, allow_pickle=True) as data:
    ve_value_30to200 = data['VE_value']
    ve_stats_30to200 = data['VE_stats']
    unit_index_30to200 = data['unit_index']

# Verify data dimensions
nfiles, nrates, nchans = ve_value_3to30.shape[:3]
expected_shape = (nfiles, nrates, nchans, 2)
if (ve_value_3to30.shape != expected_shape or ve_stats_3to30.shape != expected_shape or
    ve_value_30to200.shape != expected_shape or ve_stats_30to200.shape != expected_shape or
    unit_index_3to30.shape != (nfiles, nchans) or unit_index_30to200.shape != (nfiles, nchans)):
    print(f"Error: Data dimension mismatch. Expected {expected_shape}, got VE_value_3to30: {ve_value_3to30.shape}, "
          f"VE_stats_3to30: {ve_stats_3to30.shape}, VE_value_30to200: {ve_value_30to200.shape}, "
          f"VE_stats_30to200: {ve_stats_30to200.shape}")
    exit()

# Combine data for 100 pps and 900 pps
stiRate = [1, 100, 900]  # From original script
rate_indices = [1, 2]  # 100 pps, 900 pps
ve_value = np.zeros((nfiles, 4, nchans, 2), dtype='float32')  # [100_3to30, 100_30to200, 900_3to30, 900_30to200] × nchans × [ITD, ILD]
ve_stats = np.zeros((nfiles, 4, nchans, 2), dtype='float32')
unit_index = np.zeros((nfiles, nchans), dtype='float32')

for ff in range(nfiles):
    unit_index[ff] = np.logical_and(unit_index_3to30[ff], unit_index_30to200[ff]).astype('float32')
    ve_value[ff, 0, :, :] = ve_value_3to30[ff, 1, :, :]  # 100 pps, 3-30 ms
    ve_value[ff, 1, :, :] = ve_value_30to200[ff, 1, :, :]  # 100 pps, 30-200 ms
    ve_value[ff, 2, :, :] = ve_value_3to30[ff, 2, :, :]  # 900 pps, 3-30 ms
    ve_value[ff, 3, :, :] = ve_value_30to200[ff, 2, :, :]  # 900 pps, 30-200 ms
    ve_stats[ff, 0, :, :] = ve_stats_3to30[ff, 1, :, :]
    ve_stats[ff, 1, :, :] = ve_stats_30to200[ff, 1, :, :]
    ve_stats[ff, 2, :, :] = ve_stats_3to30[ff, 2, :, :]
    ve_stats[ff, 3, :, :] = ve_stats_30to200[ff, 2, :, :]

# Collect significant units per animal
ild_units = []
itd_units = []
animal_ids = []
unique_animals = sorted(set([fname.split('_')[0] for fname in all_fnames]))

# Identify animals with responsive units
responsive_animals = []
for animal_id in unique_animals:
    animal_files = [ff for ff, fname in enumerate(all_fnames) if fname.startswith(animal_id)]
    if not animal_files:
        continue
    has_ild = False
    has_itd = False
    for ff in animal_files:
        responsive_channels = np.where(unit_index[ff] == 1)[0]
        for cc in responsive_channels:
            if np.all(ve_stats[ff, :, cc, 1] == 1):  # ILD significant in all conditions
                has_ild = True
            if np.all(ve_stats[ff, :, cc, 0] == 1):  # ITD significant in all conditions
                has_itd = True
    if has_ild or has_itd:
        responsive_animals.append(animal_id)

# Assign colors and labels
animal_colors = {animal_id: plt.colormaps['tab20'](i % 20 / 20) for i, animal_id in enumerate(sorted(responsive_animals))}
animal_labels = {animal_id: f'Rat {i+1}' for i, animal_id in enumerate(sorted(responsive_animals))}

# Collect significant units
for animal_id in responsive_animals:
    animal_files = [ff for ff, fname in enumerate(all_fnames) if fname.startswith(animal_id)]
    for ff in animal_files:
        responsive_channels = np.where(unit_index[ff] == 1)[0]
        for cc in responsive_channels:
            if np.all(ve_stats[ff, :, cc, 1] == 1):
                ild_units.append({'values': ve_value[ff, :, cc, 1], 'animal': animal_id, 'file': ff, 'unit': cc})
            if np.all(ve_stats[ff, :, cc, 0] == 1):
                itd_units.append({'values': ve_value[ff, :, cc, 0], 'animal': animal_id, 'file': ff, 'unit': cc})
    animal_ids.append(animal_id)

# Function to compute summary statistics and perform Wilcoxon tests
def print_summary_stats(units_data, ve_type, animal_ids, animal_labels):
    if not units_data:
        print(f"No {ve_type}-significant units found")
        return {}
    
    rate_data = [[] for _ in range(4)]  # Median VE per condition
    valid_animal_ids = sorted(set(unit['animal'] for unit in units_data))
    for animal_id in valid_animal_ids:
        animal_units = [unit for unit in units_data if unit['animal'] == animal_id]
        if not animal_units:
            continue
        animal_values = np.array([unit['values'] for unit in animal_units])
        median_values = np.median(animal_values, axis=0) if animal_values.size > 0 else np.zeros(4)
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
        print(f"  IQR: {np.percentile(rate_data_arr, 75) - np.percentile(rate_data_arr, 25):.2f}%")
        print(f"  Min: {np.min(rate_data_arr):.2f}%")
        print(f"  Max: {np.max(rate_data_arr):.2f}%")
        print(f"  Std: {np.std(rate_data_arr):.2f}%")
    
    print(f"\nPer-Animal Median VE Values for {ve_type}:")
    for animal_id in valid_animal_ids:
        animal_units = [unit for unit in units_data if unit['animal'] == animal_id]
        if animal_units:
            animal_values = np.array([unit['values'] for unit in animal_units])
            median_values = np.median(animal_values, axis=0)
            print(f"  {animal_labels[animal_id]} (ID: {animal_id}): {median_values}")
    
    p_values = {}
    if all(len(rate_data[j]) > 0 for j in [0, 1, 2, 3]) and len(rate_data[0]) > 1:
        print(f"\nWilcoxon Signed-Rank Tests for {ve_type} (Bonferroni-corrected α = 0.025):")
        pairs = [(0, 1, '100 pps: 3-30 ms vs. 30-200 ms'),
                 (2, 3, '900 pps: 3-30 ms vs. 30-200 ms')]
        for i, j, label in pairs:
            stat, p_val = stats.wilcoxon(rate_data[i], rate_data[j], alternative='two-sided')
            p_values[(i, j)] = p_val
            print(f"  {label}:")
            print(f"    Statistic: {stat:.2f}")
            print(f"    p-value: {p_val:.4f}")
            print(f"    Significant: {'Yes' if p_val < 0.025 else 'No'} (Bonferroni-corrected)")
    else:
        print(f"\nInsufficient data for Wilcoxon tests for {ve_type} (need at least 2 animals with data)")
    return p_values

# Function to plot violin plots
def plot_violins_per_animal(units_data, ax, title, ve_type, conditions, animal_ids, p_values, animal_colors, animal_labels):
    if not units_data:
        print(f"No {ve_type}-significant units found")
        ax.text(0.5, 0.5, f'No {ve_type}-significant units', ha='center', va='center', fontsize=10)
        ax.set_xlabel('Time Window')
        ax.set_ylabel(f'VE by {ve_type} (%)')
        ax.set_title(title)
        ax.set_ylim(0, 100)
        return
    
    n_conditions = len(conditions)
    valid_animal_ids = sorted(set(unit['animal'] for unit in units_data))
    n_animals = len(valid_animal_ids)
    
    # Set up x-axis positions
    bar_width = 0.15
    group_width = 1.0
    positions = []
    for i in range(n_conditions):
        start = i * (n_animals * bar_width + group_width - bar_width)
        positions.extend([start + j * bar_width for j in range(n_animals)])
    
    # Collect VE values
    rate_data = [[] for _ in range(n_conditions)]
    max_y = 0
    all_data = []
    for i in range(n_conditions):
        for animal_id in valid_animal_ids:
            animal_units = [unit for unit in units_data if unit['animal'] == animal_id]
            data_array = np.array([unit['values'][i] for unit in animal_units])
            all_data.append(data_array)
            rate_data[i].append(np.median(data_array) if data_array.size > 0 else np.nan)
            if data_array.size > 0:
                max_y = max(max_y, np.max(data_array))
    
    # Plot violins
    violin_parts = ax.violinplot([data for data in all_data if data.size > 0], 
                                positions=[pos for pos, data in zip(positions, all_data) if data.size > 0], 
                                widths=bar_width * 0.8, showmedians=True)
    
    # Color violins by animal
    for idx, pc in enumerate(violin_parts['bodies']):
        cond_idx = idx // n_animals
        animal_idx = idx % n_animals
        animal_id = valid_animal_ids[animal_idx]
        pc.set_facecolor(animal_colors[animal_id])
        pc.set_alpha(0.3)
        pc.set_edgecolor('black')
        pc.set_linewidth(1)
    
    # Style median lines
    violin_parts['cmedians'].set_color('black')
    violin_parts['cmedians'].set_linewidth(2)
    for part in ['cmaxes', 'cmins', 'cbars']:
        if part in violin_parts:
            violin_parts[part].set_alpha(0)
    
    # Add quartile lines and jittered points
    violin_idx = 0
    for idx, data_array in enumerate(all_data):
        if data_array.size == 0:
            continue
        pos = positions[idx]
        q1 = np.percentile(data_array, 25)
        q3 = np.percentile(data_array, 75)
        ax.hlines(q1, pos - bar_width/2, pos + bar_width/2, colors='black', linewidth=1, linestyle='--', alpha=0.8)
        ax.hlines(q3, pos - bar_width/2, pos + bar_width/2, colors='black', linewidth=1, linestyle='--', alpha=0.8)
        
        jitter_strength = bar_width * 0.3
        x_positions = np.random.uniform(pos - jitter_strength, pos + jitter_strength, len(data_array))
        animal_idx = idx % n_animals
        animal_id = valid_animal_ids[animal_idx]
        ax.scatter(x_positions, data_array, color=animal_colors[animal_id], s=10, alpha=0.6, edgecolor='grey', linewidth=0.5)
        violin_idx += 1
    
    # Add statistical annotations
    sig_pairs = [(0, 1, '100 pps: 3-30 ms vs. 30-200 ms'),
                 (2, 3, '900 pps: 3-30 ms vs. 30-200 ms')]
    bar_heights = [max_y + 5, max_y + 15]
    for idx, (i, j, _) in enumerate(sig_pairs):
        if (i, j) in p_values and p_values[(i, j)] < 0.025:
            start_i = i * (n_animals * bar_width + group_width - bar_width) + (n_animals - 1) * bar_width / 2
            start_j = j * (n_animals * bar_width + group_width - bar_width) + (n_animals - 1) * bar_width / 2
            y_bar = bar_heights[idx]
            ax.plot([start_i, start_j], [y_bar, y_bar], color='black', linewidth=1.5)
            ax.text((start_i + start_j)/2, y_bar, '***', ha='center', va='bottom', fontsize=8)
    
    # Set axis properties
    xtick_positions = [i * (n_animals * bar_width + group_width - bar_width) + (n_animals - 1) * bar_width / 2 for i in range(n_conditions)]
    ax.set_xticks(xtick_positions)
    # ax.set_xticklabels(['3-30 ms', '30-200 ms', '3-30 ms', '30-200 ms'], ha='center')
    ax.set_xticklabels(['Onset', 'Sustained', 'Onset', 'Sustained'], ha='center')

    # ax.set_xlabel('Time Window')
    ax.set_ylabel(f'VE by {ve_type} (%)')
    ax.set_title(title)
    ax.set_ylim(0, max_y + 30)
    ax.grid(True, alpha=0.3)
    
    # Add group labels (100 pps, 900 pps) centered between condition pairs
    group_positions = [
        (xtick_positions[0] + xtick_positions[1]) / 2,  # Center of 100 pps conditions
        (xtick_positions[2] + xtick_positions[3]) / 2   # Center of 900 pps conditions
    ]
    for pos, label in zip(group_positions, group_labels):
        ax.text(pos, -0.15, label, ha='center', va='top', fontsize=10, transform=ax.get_xaxis_transform())
    
    # Add legend
    legend_elements = [Line2D([0], [0], color=animal_colors[animal_id], lw=2, label=label) 
                       for animal_id, label in animal_labels.items() if animal_id in valid_animal_ids]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=8)

# Generate violin plots
fig, (ax_ild, ax_itd) = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
p_values_ild = print_summary_stats(ild_units, 'ILD', animal_ids, animal_labels)
plot_violins_per_animal(ild_units, ax_ild, 'ILD Significant Units', 'ILD', conditions, animal_ids, p_values_ild, animal_colors, animal_labels)
p_values_itd = print_summary_stats(itd_units, 'ITD', animal_ids, animal_labels)
plot_violins_per_animal(itd_units, ax_itd, 'ITD Significant Units', 'ITD', conditions, animal_ids, p_values_itd, animal_colors, animal_labels)
plt.suptitle(f'VE by ILD and ITD Across Time Windows at 100 and 900 pps')
plt.tight_layout(rect=[0, 0.1, 1, 0.95])  # Adjust for group labels
plt.savefig(os.path.join(Figure_path, f'{figname}_v2.png'), dpi=300)
plt.show()