#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 30 14:46:47 2023
Updated to process and analyze AMUA data with consolidated data matrix, Wilcoxon-based responsiveness check,
global responsive unit index, per-file responsive unit counts, optional loading of saved data,
bar plot with global counts of significant units, violin plots for ILD/ITD VE distributions across animals,
counts of units with VE_ITD > VE_ILD and vice versa in the scatter plot section


This script generates three figures:
- Figure 5: Scatter plot of VE values (ITD vs. ILD) for each rate with an inset for low VE values, plus counts of ITD > ILD and ILD > ITD units.
- Figure 6: Bar plot of significant unit proportions (Neither, ITD only, ILD only, Both) across rates.
- Figure 7: Violin plots of VE distributions for ILD and ITD units across animals and rates, with statistical tests.

@author: fei
"""

import numpy as np
import os
import sys
from sys import platform
import matplotlib.pyplot as plt
from scipy import stats
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

# Set paths based on operating system (macOS or Linux)
if platform == "darwin":
    libPath = '/Volumes/Disk/ratCageProgramsV3/'  # Path to custom library
    Sig_path = '/Volumes/Disk/4_ITD_ILD_cueTrading/Data/AMUA/'  # Path to AMUA data files
    results_path = '/Volumes/Disk/4_ITD_ILD_cueTrading/Data/AMUA_analysis_v2/'  # Path for results
    Figure_path = '/Volumes/Disk/4_ITD_ILD_cueTrading/Figures_paper/'  # Path for saving figures
elif platform == "linux":
    libPath = '/home/shiyi/PycharmProjects/ENVvsPS/'  # Path to custom library
    Sig_path = '/disks/CIdata/2023_02_23/'  # Path to AMUA data files
    results_path = '/disks/CIdata/2023_02_23Results/'  # Path for results
    Figure_path = '/disks/CIdata/2_ITD_ILD_cueTrading/Figures/'  # Path for saving figures
sys.path.append(libPath)
import RZ2ephys as ep  # Custom module for electrophysiological data processing

# Configuration parameters
LOAD_PREVIOUS_DATA = True  # If True, load precomputed data; if False, recompute
ResponsiveMethod = 2  # Use Wilcoxon test (p < 0.05) for unit responsiveness
t0 = 3 # Start of time window for VE at 1 pps 
t1 = 30  # End of time window for VE computation (ms) at 1 pps
ts1 = 3 # start of time window for VE computation at 100 and 900 pps 
ts2 = 30 # End of time window for VE computation (ms) at 100 and 900 pps 


figname1_pre = 'Fig5_VE_distribution'
figname2_pre = 'Fig6_VE_Count_of_significant_units'
figname3_pre = 'Fig7_VE_ILD_ITD_per_animal'
# Set file names based on time window
if t1 == 200 and ts1 == 3 and ts2 == 200:
    # rate0:3-200, rate 100/900, 3-200
    results_dataname = 'all_files_VE_data_v2_3to200ms.npz'  # Output file for 3–200 ms
    figname1 = figname1_pre+'_3to200ms'  # Scatter plot
    figname2 = figname2_pre+'_3to200ms'  # Bar plot
    figname3 = figname3_pre+'_3to200ms'  # Violin plot
elif t1 == 30 and ts1 == 3 and ts2 == 200:  
    # rate0:3-30, rate 100/900, 3-200
    results_dataname = 'all_files_VE_data_stiDur.npz'  # Output file for 3–30 ms
    figname1 = figname1_pre+'_stiDur_v1'  # Scatter plot
    figname2 = figname2_pre+'_stiDur_v1'  # Bar plot
    figname3 = figname3_pre+'_stiDur_v1'  # Violin plot
elif t1 == 30 and ts1 == 3 and ts2 == 30:
    # rate0:3-30, rate 100/900: 3-30
    results_dataname = 'all_files_VE_data_3to30ms.npz'  # Output file for 3–30 ms
    figname1 = figname1_pre+'_3to30ms'  # Scatter plot
    figname2 = figname2_pre+'_3to30ms'  # Bar plot
    figname3 = figname3_pre+'_3to30ms'  # Violin plot
elif t1 == 30 and ts1 == 30 and ts2 == 200:
    # rate0:3-30, rate 100/900: 30-200
    results_dataname = 'all_files_VE_data_30to200ms.npz'  # Output file for 3–30 ms
    figname1 = figname1_pre+'_30to200ms'  # Scatter plot
    figname2 = figname2_pre+'_30to200ms'  # Bar plot
    figname3 = figname3_pre+'_30to200ms'  # Violin plot

    

#%%

# Get list of AMUA data files (.npy) and initialize parameters
filelist = [f for f in os.listdir(Sig_path) if f.endswith('.npy') and 'AMUA' in f]
filelist.sort()  # Sort files for consistent processing
nfiles = len(filelist)  # Number of files
stiRate = [1, 100, 900]  # Stimulus rates (pps)
nrates = len(stiRate)  # Number of rates
nchans = 32  # Number of channels per file

# Initialize data arrays
allunitVE_value = np.zeros((nfiles, nrates, nchans, 2), dtype='float32')  # VE values: files × rates × channels × types (0=ITD, 1=ILD)
allunitVE_Stats = np.zeros((nfiles, nrates, nchans, 2), dtype='float32')  # Significance: 1 if VE > 95th percentile of permuted VE
all_Stats_perChan_pvalues = np.zeros((nfiles, nchans), dtype='float32')  # Wilcoxon p-values for responsiveness
all_unitIndex = np.zeros((nfiles, nchans), dtype='float32')  # Responsive units per file (1=responsive)
global_unitIndex = np.zeros(nchans, dtype='float32')  # Units responsive in all files
all_fnames = []  # List of filenames
VE_sigunit = np.zeros((nfiles, nrates, 4), dtype='float32')  # Unit counts: [Neither, ITD only, ILD only, Both]
ResponsivePidx = np.zeros(nfiles, dtype='float32')  # Files with responsive units
ResponsivePidxLen = np.zeros(nfiles, dtype='float32')  # Number of responsive units per file

# Load precomputed data if available
npz_file = os.path.join(results_path, results_dataname)
if LOAD_PREVIOUS_DATA and os.path.exists(npz_file):
    print(f"Loading saved data from {npz_file}")
    with np.load(npz_file, allow_pickle=True) as data:
        allunitVE_value = data['VE_value']
        allunitVE_Stats = data['VE_stats']
        all_Stats_perChan_pvalues = data['pvalues']
        all_unitIndex = data['unit_index']
        global_unitIndex = data['global_unit_index']
        all_fnames = data['fnames'].tolist()
    
    # Verify loaded data dimensions
    print(f"allunitVE_Stats shape after loading: {allunitVE_Stats.shape}")
    expected_shape = (nfiles, nrates, nchans, 2)
    if (allunitVE_value.shape != expected_shape or
        allunitVE_Stats.shape != expected_shape or
        all_Stats_perChan_pvalues.shape != (nfiles, nchans) or
        all_unitIndex.shape != (nfiles, nchans) or
        global_unitIndex.shape != (nchans,) or
        len(all_fnames) != nfiles):
        print("Warning: Loaded data dimensions or file list mismatch. Recomputing data.")
        LOAD_PREVIOUS_DATA = False
else:
    LOAD_PREVIOUS_DATA = False

# Process AMUA data if not loading precomputed data
if not LOAD_PREVIOUS_DATA:
    for ff, fname0 in enumerate(filelist):
        # Load AMUA data (channels × rates × ITDs × ILDs × trials × time points)
        Sig_AMUA = np.load(os.path.join(Sig_path, fname0), allow_pickle=True)
        fname = fname0[:-8]  # Remove '.npy' from filename
        all_fnames.append(fname)
        
        # Extract dimensions
        nITD = Sig_AMUA.shape[2]  # Number of ITD conditions
        nILD = Sig_AMUA.shape[3]  # Number of ILD conditions
        ntrials = Sig_AMUA.shape[4]  # Number of trials
        fs = 2000  # Sampling rate (Hz)
        xs = np.arange(0, Sig_AMUA.shape[5]/fs, 1/fs) - 0.1  # Time axis (s)
        xs = xs * 1000  # Convert to ms
        
        # Compute unit responsiveness using Wilcoxon test
        Sig_AMUA_avgTrial = np.mean(Sig_AMUA, 4)  # Average over trials
        Stats_perChan_pvalues = np.zeros(nchans, dtype='float32')  # Wilcoxon p-values
        UnitIndex = np.zeros(nchans, dtype='float32')  # Responsive units
        
        # Define time windows
        ind_baseline = np.where(xs < 0)[0]  # Pre-stimulus baseline
        ind_onset = np.where((xs > 3) & (xs < 30))[0]  # Onset response (3–30 ms)
        ind_artifact = np.where((xs >= 0) & (xs <= 2))[0]  # Artifact window (0–2 ms)
        
        for cc in range(nchans):
            # Compute baseline and onset activity
            temp_baseline = np.mean(Sig_AMUA[cc, :, :, :, :, ind_baseline[0]:ind_baseline[-1]], 4)
            temp_onset = np.mean(Sig_AMUA[cc, :, :, :, :, ind_onset[0]:ind_onset[-1]], 4)
            temp_artifact = np.max(Sig_AMUA_avgTrial[cc, :, :, :, ind_artifact[0]:ind_artifact[-1]], 3)
            
            # Check for artifacts
            temp_baseline_avg = np.mean(Sig_AMUA_avgTrial[cc, :, :, :, ind_baseline[0]:ind_baseline[-1]], 3)
            temp_baseline_SD = np.std(Sig_AMUA_avgTrial[cc, :, :, :, ind_baseline[0]:ind_baseline[-1]], 3)
            artifact_mask = temp_artifact > (temp_baseline_avg + 10 * temp_baseline_SD)
            if np.any(artifact_mask):
                Stats_perChan_pvalues[cc] = 1.0  # Mark as non-responsive due to artifact
                continue
            
            # Pool responses over all stimulus parameters
            temp_baseline_alldata = temp_baseline.flatten()
            temp_onset_alldata = temp_onset.flatten()
            
            # Wilcoxon test (onset > baseline)
            Stats_baseline_onset = stats.wilcoxon(temp_baseline_alldata, temp_onset_alldata, alternative='less')
            Stats_perChan_pvalues[cc] = Stats_baseline_onset.pvalue
            
            # Mark unit as responsive if p < 0.05
            if Stats_perChan_pvalues[cc] < 0.05:
                UnitIndex[cc] = 1
        
        all_Stats_perChan_pvalues[ff, :] = Stats_perChan_pvalues
        all_unitIndex[ff, :] = UnitIndex
        
        # Compute VE and permutation test for responsive units
       
        VE_array = np.zeros((nrates, nchans, 2, 1001), dtype='float32')  # VE values (real + 1000 permutations)
        PermutationTest = np.zeros((nrates, nchans, 2), dtype='float32')  # Significance results
        
        for rr in range(nrates):
            if rr == 0:
                idx_sig = np.where((xs > t0) & (xs < t1))[0]  # Time window for VE (3-30 ms Onset response for one pulse)
            else:
                idx_sig = np.where((xs > ts1) & (xs < ts2))[0] # stimulus duration is 200 ms 
                
            for cc in range(nchans):
                if UnitIndex[cc] == 0:  # Skip non-responsive units
                    continue
                for ii in range(1001):  # Real VE (ii=0) and permutations (ii>0)
                    dat_temp = np.mean(Sig_AMUA[cc, rr, :, :, :, idx_sig[0]:idx_sig[-1]], 3)
                    if ii > 0:
                        dat_temp_flatten = dat_temp.flatten()
                        np.random.shuffle(dat_temp_flatten)
                        dat_temp = dat_temp_flatten.reshape(nITD, nILD, ntrials)
                    
                    # Compute VE for ITD and ILD
                    dat_grand_mean = np.mean(dat_temp)
                    dat_ITD = np.mean(dat_temp, 1)
                    dat_ITD_mean = np.mean(dat_ITD, 1)
                    dat_ILD = np.mean(dat_temp, 0)
                    dat_ILD_mean = np.mean(dat_ILD, 1)
                    
                    dat_ITD_diff = dat_ITD_mean - np.repeat(dat_grand_mean, nITD)
                    SS_ITD = ntrials * nILD * sum(dat_ITD_diff**2)
                    dat_ILD_diff = dat_ILD_mean - np.repeat(dat_grand_mean, nILD)
                    SS_ILD = ntrials * nITD * sum(dat_ILD_diff**2)
                    dat_flatten = dat_temp.flatten()
                    dat_diff = dat_flatten - np.repeat(dat_grand_mean, len(dat_flatten))
                    SS_total = sum(dat_diff**2)
                    
                    VE_array[rr, cc, 0, ii] = (SS_ITD / SS_total) * 100  # VE for ITD (%)
                    VE_array[rr, cc, 1, ii] = (SS_ILD / SS_total) * 100  # VE for ILD (%)
                
                # Determine significance (VE > 95th percentile of permuted VE)
                threshold_ITD = np.percentile(VE_array[rr, cc, 0, 1:], 95)
                threshold_ILD = np.percentile(VE_array[rr, cc, 1, 1:], 95)
                PermutationTest[rr, cc, 0] = 1 if VE_array[rr, cc, 0, 0] > threshold_ITD else 0
                PermutationTest[rr, cc, 1] = 1 if VE_array[rr, cc, 1, 0] > threshold_ILD else 0
        
        allunitVE_value[ff, :, :, :] = VE_array[:, :, :, 0]
        allunitVE_Stats[ff, :, :, :] = PermutationTest
    
    # Compute global responsive unit index
    global_unitIndex = np.min(all_unitIndex, axis=0)  # Unit is responsive only if responsive in all files
    # np.save(os.path.join(results_path, 'global_unitIndex.npy'), global_unitIndex)
    
    # Save results to .npz file
    np.savez(os.path.join(results_path, results_dataname),
             VE_value=allunitVE_value,
             VE_stats=allunitVE_Stats,
             pvalues=all_Stats_perChan_pvalues,
             unit_index=all_unitIndex,
             global_unit_index=global_unitIndex,
             fnames=np.array(all_fnames))

# Analysis and Visualization
# Count responsive units per file
allunitIndex = [np.where(all_unitIndex[ff] == 1)[0] for ff in range(nfiles)]
NumResponsiveUnit = [len(idx) for idx in allunitIndex]
ResponsivePenetration = [ff for ff, count in enumerate(NumResponsiveUnit) if count > 0]
NgoodPenetration = len(ResponsivePenetration)
print(f"Total Per-File Responsive Units: {sum(NumResponsiveUnit)}")
print(f"Globally Responsive Units (responsive in all files): {np.sum(global_unitIndex)}")
for ff, fname in enumerate(all_fnames):
    print(f"File {fname}: {NumResponsiveUnit[ff]} responsive units")

#%% Figure 4: Scatter Plot with Inset (per-file responsive units) and ITD vs. ILD comparison
plt.figure(figsize=[12, 5])
for idx_rate in range(nrates):
    alldata_VE_sig_all = []
    colors_all = ['grey', 'blue', 'magenta', 'red']
    # Initialize counters for VE_ITD > VE_ILD and VE_ILD > VE_ITD
    itd_greater = 0
    ild_greater = 0
    total_compared_units = 0
    plt.subplot(1, 3, idx_rate + 1)
    for jj, (ITD_test, ILD_test) in enumerate([(0, 0), (1, 0), (0, 1), (1, 1)]):
        alldata_VE_sig = []
        for ii in range(NgoodPenetration):
            IdxP = ResponsivePenetration[ii]
            IdxGoodUnit = allunitIndex[IdxP]
            if len(IdxGoodUnit) == 0:
                continue
            VE_value_SingleP = allunitVE_value[IdxP, :, IdxGoodUnit, :]
            VE_stats_SingleP = allunitVE_Stats[IdxP, :, IdxGoodUnit, :]
            Idx_sigUnit = np.where((VE_stats_SingleP[:, idx_rate, 0] == ITD_test) & (VE_stats_SingleP[:, idx_rate, 1] == ILD_test))[0]
            if Idx_sigUnit.shape[0] > 0:
                alldata_VE_sig.append(VE_value_SingleP[Idx_sigUnit, idx_rate, :])
                
                # Count units where VE_ITD > VE_ILD or vice versa (only for ITD only, ILD only, or Both significant)
                if jj in [3]:  # Skip 'Neither significant' (jj=0)
                    for unit_idx in Idx_sigUnit:
                        ve_itd = VE_value_SingleP[unit_idx, idx_rate, 0]
                        ve_ild = VE_value_SingleP[unit_idx, idx_rate, 1]
                        if ve_itd > ve_ild:
                            itd_greater += 1
                        elif ve_ild > ve_itd:
                            ild_greater += 1
                        total_compared_units += 1
        
        alldata_VE_sig_temp = np.concatenate(alldata_VE_sig) if alldata_VE_sig else np.array([])
        alldata_VE_sig_all.append(alldata_VE_sig_temp)
        
        if alldata_VE_sig_temp.size > 0:
            plt.scatter(alldata_VE_sig_temp[:, 0], alldata_VE_sig_temp[:, 1], color=colors_all[jj], s=12, edgecolors='black', linewidth=0.2)

    plt.xlim([0, 95])
    plt.ylim([0, 95])
    plt.title(f'{stiRate[idx_rate]} pps')
    plt.xlabel('VE by ITD (%)')
    if idx_rate == 0:
        plt.ylabel('VE by ILD (%)')
    plt.gca().set_aspect('equal', adjustable='box')
    # Print counts for VE_ITD vs. VE_ILD comparison
    print(f"\nRate {stiRate[idx_rate]} pps VE Comparison:")
    print(f"  Units with VE_ITD > VE_ILD: {itd_greater}")
    print(f"  Units with VE_ILD > VE_ITD: {ild_greater}")
    print(f"  Total units compared: {total_compared_units}")
    
    data_range = max([np.max(data) for data in alldata_VE_sig_all if data.size > 0], default=95)
    plt.axline((0, 0), (data_range, data_range), color="black", linestyle='--')
    
    # Add inset plot for low VE values
    ax_inset = inset_axes(plt.gca(), width="30%", height="30%", loc='upper right')
    for data, color in zip(alldata_VE_sig_all, colors_all):
        if data.size > 0:
            ax_inset.scatter(data[:, 0], data[:, 1], color=color, s=12, edgecolors='black', linewidth=0.2)
    ax_inset.set_xlim(0, 5)
    ax_inset.set_ylim(0, 5)
    ax_inset.set_xticks([0, 5])
    ax_inset.set_yticks([0, 5])
    ax_inset.tick_params(labelsize=8)
    ax_inset.set_aspect('equal', adjustable='box')
    ax_inset.patch.set_edgecolor('black')
    ax_inset.patch.set_linewidth(1)
    plt.axline((0, 0), (data_range, data_range), color="black", linestyle='--')
    
    if idx_rate == 2:
        plt.legend(['Neither significant', 'ITD significant only', 'ILD significant only', 'Both significant'], bbox_to_anchor=(0.0001, 0.99))

plt.subplots_adjust(left=0.08, bottom=0.1, right=0.95, top=0.9, wspace=0.15)
plt.suptitle(figname1)
plt.tight_layout()
plt.savefig(os.path.join(Figure_path, figname1+'.png'), dpi=300)
plt.show()

#%%
# idx_rate = 1
# ITD_test = 1
# ILD_test = 1
# alldata_VE_sig = []
# plt.figure()
# for ii in range(3):
#     IdxGoodUnit = []
#     IdxP = ResponsivePenetration[ii]
#     IdxGoodUnit = allunitIndex[IdxP]
#     VE_value_SingleP_sig_ITD = []
#     VE_value_SingleP_sig_ILD = []
#     # if len(IdxGoodUnit) == 0:
#     #     continue
#     VE_value_SingleP = allunitVE_value[IdxP, :, IdxGoodUnit, :]
#     VE_stats_SingleP = allunitVE_Stats[IdxP, :, IdxGoodUnit, :]
#     Idx_sigUnit = np.where((VE_stats_SingleP[:, idx_rate, 0] == ITD_test) & (VE_stats_SingleP[:, idx_rate, 1] == ILD_test))[0]
#     if Idx_sigUnit.shape[0] > 0:
#         alldata_VE_sig.append(VE_value_SingleP[Idx_sigUnit, idx_rate, :])
#     VE_value_SingleP_sig_ITD = VE_value_SingleP[Idx_sigUnit,idx_rate,0]
#     VE_value_SingleP_sig_ILD = VE_value_SingleP[Idx_sigUnit,idx_rate,1]
    
#     plt.plot(Idx_sigUnit,VE_value_SingleP_sig_ITD/VE_value_SingleP_sig_ILD,'-o')   

#%%
# =============================================================================
#  VE_ITD / VE_ILD ratio — grouped by animal (one panel per animal)
#  Only units with BOTH ITD and ILD significant at selected rate
# =============================================================================

import matplotlib.pyplot as plt
import numpy as np

rate_idx = 0                         # 0=1 pps, 1=100 pps, 2=900 pps
rate_str = f"{stiRate[rate_idx]} pps"

# ─── Define how to get animal ID from filename ─────────────────────────────
# <--- CHANGE THIS PART according to your filenames ------------------------->
def get_animal_id(fname):
    parts = fname.split('_')
    # Examples — uncomment / adapt the one that matches your data
    # return parts[0]                             # "R123", "rat07", "AnimalA"
    # return parts[1]                             # second field
    # return f"{parts[0]}_{parts[1]}"             # first two fields
    # return parts[0][:4]                         # first 4 characters
    return parts[0]   # ← most common — change if needed !

# ─── Collect data by animal ────────────────────────────────────────────────

data_by_animal = {}

for ff in range(nfiles):
    fname = all_fnames[ff]
    animal = get_animal_id(fname)
    
    # Responsive channels
    resp_ch = np.nonzero(all_unitIndex[ff])[0]   # faster than where+==
    if len(resp_ch) == 0:
        continue
    
    # Both significant at this rate?
    both = (allunitVE_Stats[ff, rate_idx, resp_ch, 0] == 1) & \
           (allunitVE_Stats[ff, rate_idx, resp_ch, 1] == 1)
    
    if not both.any():
        continue
    
    ch_both = resp_ch[both]
    
    ve_itd = allunitVE_value[ff, rate_idx, ch_both, 0]
    ve_ild = allunitVE_value[ff, rate_idx, ch_both, 1]
    
    ratio = np.full_like(ve_itd, np.nan, dtype=float)
    mask = ve_ild > 0.1   # very small threshold to avoid artifacts
    # ratio[mask] = ve_itd[mask] / ve_ild[mask]
    # ratio[mask] = ve_itd[mask] - ve_ild[mask]
    ratio[mask] = ve_itd[mask] / (ve_itd[mask] + ve_ild[mask])
    
    if animal not in data_by_animal:
        data_by_animal[animal] = []
    
    data_by_animal[animal].append({
        'channels': ch_both,
        'ratio': ratio,
        'pen_idx': len(data_by_animal[animal]),  # 0,1,2,... within this animal
        'fname_short': fname[-20:]               # last part for legend/debug
    })

# ─── Plot ──────────────────────────────────────────────────────────────────

animals = sorted(data_by_animal.keys())
n_animals = len(animals)

print(f"Found {n_animals} animals with both-significant units at {rate_str}")
print("Animals:", animals)
print(f"Total penetrations used: {sum(len(v) for v in data_by_animal.values())} / 27")

if n_animals == 0:
    print("→ No data to plot. Check filename parsing or data.")
else:
    ncols = min(5, max(2, n_animals))
    nrows = (n_animals + ncols - 1) // ncols
    
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(4.0 * ncols, 3.3 * nrows),
                             sharey=False, squeeze=False)
    
    all_ratios = []
    
    for i, animal in enumerate(animals):
        ax = axes[i // ncols, i % ncols]
        pen_data_list = data_by_animal[animal]
        
        colors = plt.cm.tab10(np.linspace(0.1, 0.9, max(5, len(pen_data_list))))
        
        for j, pen in enumerate(pen_data_list):
            ch = pen['channels']
            rat = pen['ratio']
            valid = ~np.isnan(rat)
            
            if not valid.any():
                continue
                
            ax.plot(ch[valid], rat[valid], '-o',
                    color=colors[j],
                    ms=5.5, lw=1.1, mec='k', mew=0.5,
                    label=f"p{j+1} ({pen['fname_short']})")
            
            all_ratios.extend(rat[valid])
        
        ax.axhline(1.0, color='crimson', ls='--', lw=1.3, alpha=0.85)
        
        # ax.set_yscale('log')
        # ax.set_ylim(0.01, 20)
        # ax.set_yticks([0.1, 0.3, 1, 3, 10])
        # ax.set_yticklabels(['0.1','0.3','1','3','10'])
        
        ax.set_xlabel('Channel')
        if i % ncols == 0:
            ax.set_ylabel('VE_ITD / (VE_ITD + VE_ILD)')
        
        ax.set_title(animal, fontsize=11)
        ax.grid(True, ls=':', alpha=0.4)
        
        # if len(pen_data_list) > 1 and i == 0:
        #     ax.legend(title='Penetration', fontsize=8,
        #               loc='upper right', bbox_to_anchor=(1.03, 1.03))
    
    # Hide unused subplots
    for j in range(i+1, nrows*ncols):
        axes[j//ncols, j%ncols].set_visible(False)
    
    if all_ratios:
        med = np.median(all_ratios)
        fig.suptitle(f"VE_ITD / (VE_ITD + VE_ILD) – both significant units @ {rate_str}\n",fontsize=13)
    
    plt.tight_layout(rect=[0,0,1,0.95])
    
    savename = f"ratio_ITD over sum of ITD ILD_both-sig_per_animal_{rate_str.replace(' ','')}.png"
    plt.savefig(os.path.join(Figure_path, savename), dpi=300, bbox_inches='tight')
    plt.show()


#%%
# Figure 5: Bar Plot for Count of Significant Units
VE_counts = np.zeros((nrates, 4), dtype='float32')  # Counts: [Neither, ITD only, ILD only, Both]
total_units = 0

# Verify allunitVE_Stats shape
print(f"allunitVE_Stats shape: {allunitVE_Stats.shape}")
if allunitVE_Stats.shape[1] != nrates or allunitVE_Stats.shape[3] != 2:
    print(f"Error: allunitVE_Stats has incorrect shape {allunitVE_Stats.shape}. Expected (nfiles, {nrates}, {nchans}, 2).")
    VE_counts[:] = 0
else:
    for ff in range(nfiles):
        UnitIndex = np.where(all_unitIndex[ff] == 1)[0]
        if len(UnitIndex) == 0:
            continue
        total_units += len(UnitIndex)
        print(f"File {ff} ({all_fnames[ff]}): {len(UnitIndex)} responsive units")
        
        for cc in UnitIndex:
            for rr in range(nrates):
                VE_significant_ITD = allunitVE_Stats[ff, rr, cc, 0]
                VE_significant_ILD = allunitVE_Stats[ff, rr, cc, 1]
                if VE_significant_ITD == 0 and VE_significant_ILD == 0:
                    VE_counts[rr, 0] += 1
                elif VE_significant_ITD == 1 and VE_significant_ILD == 0:
                    VE_counts[rr, 1] += 1
                elif VE_significant_ITD == 0 and VE_significant_ILD == 1:
                    VE_counts[rr, 2] += 1
                elif VE_significant_ITD == 1 and VE_significant_ILD == 1:
                    VE_counts[rr, 3] += 1

if total_units == 0:
    print("Warning: No responsive units found across all files. Skipping bar plot.")
else:
    fig, ax = plt.subplots(figsize=(6, 4.5))
    barWidth = 0.15
    br1 = np.arange(nrates)
    br2 = [x + barWidth for x in br1]
    br3 = [x + barWidth for x in br2]
    br4 = [x + barWidth for x in br3]
    
    # Plot bars for each category
    bars1 = plt.bar(br1, VE_counts[:, 0]/total_units*100, color='grey', width=barWidth, label='Neither significant')
    bars2 = plt.bar(br2, VE_counts[:, 1]/total_units*100, color='blue', width=barWidth, label='ITD significant only')
    bars3 = plt.bar(br3, VE_counts[:, 2]/total_units*100, color='magenta', width=barWidth, label='ILD significant only')
    bars4 = plt.bar(br4, VE_counts[:, 3]/total_units*100, color='red', width=barWidth, label='Both significant')
    
    def add_value_labels(bars):
        """Add percentage labels above bars."""
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2., height,
                     f'{height:.1f}', ha='center', va='bottom', fontsize=6, fontweight='bold')
    
    add_value_labels(bars1)
    add_value_labels(bars2)
    add_value_labels(bars3)
    add_value_labels(bars4)
    
    plt.legend()
    plt.xticks([r + barWidth * 1.5 for r in range(nrates)], ['1 pps', '100 pps', '900 pps'])
    plt.xlabel('Stimulus rate')
    plt.ylabel('% of multiunits')
    plt.ylim([0, 100])
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    
    print(f"Total responsive units: {total_units}")
    print(f"Counts for Neither significant: {VE_counts[:, 0]}")
    print(f"Counts for ITD significant only: {VE_counts[:, 1]}")
    print(f"Counts for ILD significant only: {VE_counts[:, 2]}")
    print(f"Counts for Both significant: {VE_counts[:, 3]}")
    
    plt.suptitle(figname2)
    plt.tight_layout()
    plt.savefig(os.path.join(Figure_path, figname2+'.png'), dpi=300)
    plt.show()


#%%
# Figure 6: Violin Plot with Statistical Annotations
def plot_violins_per_animal(units_data, ax, title, ve_type, stiRate, animal_ids, p_values, animal_colors, animal_labels):
    """
    Generate violin plot for VE values of significant units (ILD or ITD) across animals at three rates.
    
    Parameters:
    - units_data: List of dictionaries with VE values, animal ID, file, and unit for significant units.
    - ax: Matplotlib axes object for plotting.
    - title: Plot title (e.g., 'ILD Significant Units').
    - ve_type: Type of units ('ILD' or 'ITD').
    - stiRate: List of stimulus rates (e.g., [1, 100, 900]).
    - animal_ids: List of animal IDs with responsive units.
    - p_values: Dictionary of p-values from post-hoc Wilcoxon tests.
    - animal_colors: Dictionary mapping animal IDs to colors.
    - animal_labels: Dictionary mapping animal IDs to labels (e.g., 'Rat 1').
    """
    if not units_data:
        print(f"No {ve_type}-significant units found")
        ax.text(0.5, 0.5, f'No {ve_type}-significant units', ha='center', va='center', fontsize=10)
        ax.set_xlabel('Stimulus Rate (pps)')
        ax.set_ylabel(f'VE by {ve_type} (%)')
        ax.set_title(title)
        ax.set_ylim(0, 100)
        return
    
    n_rates = len(stiRate)
    valid_animal_ids = sorted(set(unit['animal'] for unit in units_data))  # Animals with significant units
    n_animals = len(valid_animal_ids)
    
    # Set up x-axis positions for violins
    bar_width = 0.15
    group_width = 1.0
    positions = []
    for i in range(n_rates):
        start = i * (n_animals * bar_width + group_width - bar_width)
        positions.extend([start + j * bar_width for j in range(n_animals)])
    
    # Collect VE values and medians
    rate_data = [[] for _ in range(n_rates)]  # Median VE per rate
    max_y = 0
    all_data = []  # VE values for all units
    for i in range(n_rates):
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
        rate_idx = idx // n_animals
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
    sig_pairs = [(0, 1, f"{stiRate[0]} vs. {stiRate[1]} pps"),
                 (0, 2, f"{stiRate[0]} vs. {stiRate[2]} pps"),
                 (1, 2, f"{stiRate[1]} vs. {stiRate[2]} pps")]
    bar_heights = [max_y + 5 + k * 10 for k in range(len(sig_pairs))]
    for idx, (i, j, _) in enumerate(sig_pairs):
        if (i, j) in p_values and p_values[(i, j)] < 0.0167:
            start_i = i * (n_animals * bar_width + group_width - bar_width) + (n_animals - 1) * bar_width / 2
            start_j = j * (n_animals * bar_width + group_width - bar_width) + (n_animals - 1) * bar_width / 2
            y_bar = bar_heights[idx]
            ax.plot([start_i, start_j], [y_bar, y_bar], color='black', linewidth=1.5)
            ax.text((start_i + start_j)/2, y_bar, '***', ha='center', va='bottom', fontsize=8)
    
    # Set axis properties
    ax.set_xticks([i * (n_animals * bar_width + group_width - bar_width) + (n_animals - 1) * bar_width / 2 for i in range(n_rates)])
    ax.set_xticklabels([str(rate) for rate in stiRate])
    ax.set_xlabel('Stimulus Rate (pps)')
    ax.set_ylabel(f'VE by {ve_type} (%)')
    ax.set_title(title)
    ax.set_ylim(0, max_y + 30)
    ax.grid(True, alpha=0.3)
    
    # Add legend
    from matplotlib.lines import Line2D
    legend_elements = [Line2D([0], [0], color=animal_colors[animal_id], lw=2, label=label) 
                       for animal_id, label in animal_labels.items() if animal_id in valid_animal_ids]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=8)

def print_summary_stats(units_data, ve_type, animal_ids, animal_labels):
    """
    Compute and print summary statistics and perform Friedman test for VE values.
    
    Parameters:
    - units_data: List of dictionaries with VE values and metadata.
    - ve_type: 'ILD' or 'ITD'.
    - animal_ids: List of animal IDs with responsive units.
    - animal_labels: Dictionary mapping animal IDs to labels.
    """
    if not units_data:
        print(f"No {ve_type}-significant units found")
        return {}
    
    rate_data = [[] for _ in range(3)]  # Median VE per rate
    valid_animal_ids = sorted(set(unit['animal'] for unit in units_data))
    for animal_id in valid_animal_ids:
        animal_units = [unit for unit in units_data if unit['animal'] == animal_id]
        if not animal_units:
            continue
        animal_values = np.array([unit['values'] for unit in animal_units])
        median_values = np.median(animal_values, axis=0) if animal_values.size > 0 else np.zeros(3)
        for j in range(3):
            rate_data[j].append(median_values[j])
    
    print(f"\n=== {ve_type} Median VE Summary Statistics (Across Animals) ===")
    print(f"Number of animals: {len(valid_animal_ids)}")
    for j, rate in enumerate(stiRate):
        rate_data_arr = np.array(rate_data[j])
        if len(rate_data_arr) == 0:
            continue
        print(f"\nRate {rate} pps:")
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
    if all(len(rate_data[j]) > 0 for j in range(3)) and len(rate_data[0]) > 1:
        stat, p_value = stats.friedmanchisquare(rate_data[0], rate_data[1], rate_data[2])
        print(f"\nFriedman Test for {ve_type} across rates (within animals):")
        print(f"  Chi-squared statistic: {stat:.2f}")
        print(f"  p-value: {p_value:.4f}")
        if p_value < 0.05:
            print(f"\nPost-Hoc Wilcoxon Signed-Rank Tests for {ve_type} (Bonferroni-corrected α = 0.0167):")
            pairs = [(0, 1, f"{stiRate[0]} vs. {stiRate[1]} pps"),
                     (0, 2, f"{stiRate[0]} vs. {stiRate[2]} pps"),
                     (1, 2, f"{stiRate[1]} vs. {stiRate[2]} pps")]
            for i, j, label in pairs:
                stat, p_val = stats.wilcoxon(rate_data[i], rate_data[j], alternative='two-sided')
                p_values[(i, j)] = p_val
                print(f"  {label}:")
                print(f"    Statistic: {stat:.2f}")
                print(f"    p-value: {p_val:.4f}")
                print(f"    Significant: {'Yes' if p_val < 0.0167 else 'No'} (Bonferroni-corrected)")
    else:
        print(f"\nInsufficient data for Friedman test for {ve_type} (need at least 2 animals with data)")
    return p_values

# Collect significant units per animal
ild_units = []
itd_units = []
animal_ids = []

# Extract unique animal IDs
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
        responsive_channels = np.where(all_unitIndex[ff] == 1)[0]
        for cc in responsive_channels:
            if np.all(allunitVE_Stats[ff, :, cc, 1] == 1):
                has_ild = True
            if np.all(allunitVE_Stats[ff, :, cc, 0] == 1):
                has_itd = True
    if has_ild or has_itd:
        responsive_animals.append(animal_id)

# Assign colors and labels to responsive animals
animal_colors = {animal_id: plt.colormaps['tab20'](i % 20 / 20) for i, animal_id in enumerate(sorted(responsive_animals))}
animal_labels = {animal_id: f'Rat {i+1}' for i, animal_id in enumerate(sorted(responsive_animals))}

# Collect ILD and ITD significant units
for animal_id in responsive_animals:
    animal_files = [ff for ff, fname in enumerate(all_fnames) if fname.startswith(animal_id)]
    if not animal_files:
        continue
    for ff in animal_files:
        responsive_channels = np.where(all_unitIndex[ff] == 1)[0]
        for cc in responsive_channels:
            if np.all(allunitVE_Stats[ff, :, cc, 1] == 1):
                ild_units.append({'values': allunitVE_value[ff, :, cc, 1], 'animal': animal_id, 'file': ff, 'unit': cc})
            if np.all(allunitVE_Stats[ff, :, cc, 0] == 1):
                itd_units.append({'values': allunitVE_value[ff, :, cc, 0], 'animal': animal_id, 'file': ff, 'unit': cc})
    animal_ids.append(animal_id)

# Generate violin plots
fig, (ax_ild, ax_itd) = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
p_values_ild = print_summary_stats(ild_units, 'ILD', animal_ids, animal_labels)
plot_violins_per_animal(ild_units, ax_ild, 'ILD Significant Units', 'ILD', stiRate, animal_ids, p_values_ild, animal_colors, animal_labels)
p_values_itd = print_summary_stats(itd_units, 'ITD', animal_ids, animal_labels)
plot_violins_per_animal(itd_units, ax_itd, 'ITD Significant Units', 'ITD', stiRate, animal_ids, p_values_itd, animal_colors, animal_labels)
plt.suptitle(figname3)
plt.tight_layout()
plt.savefig(os.path.join(Figure_path, figname3+'.png'), dpi=300)
plt.show()






