#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 30 14:46:47 2023

Updated to process and analyze AMUA data with consolidated data matrix,
Wilcoxon-based responsiveness check, global responsive unit index,
per-file responsive unit counts, optional loading of saved data,
bar plot with global counts of significant units, violin plots for
ILD/ITD VE distributions across animals, and counts of units with
VE_ITD > VE_ILD and vice versa.

This script generates:
- Figure 5: Scatter plot of VE values (ITD vs. ILD) for each rate.
- Figure 5b: VE_ITD / (VE_ITD + VE_ILD) and VE_ITD - VE_ILD distributions.
- Figure 6: Bar plot of significant unit proportions across rates.
- Figure 7: Violin plots of VE distributions for ILD and ITD across animals
  and rates. All responsive units are plotted; nonsignificant VE values are
  shown as grey dots with black edges.

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

# Set paths based on operating system
if platform == "darwin":
    libPath = '/Volumes/Disk/ratCageProgramsV3/'
    Sig_path = '/Volumes/Disk/4_ITD_ILD_cueTrading/Data/AMUA/'
    results_path = '/Volumes/Disk/4_ITD_ILD_cueTrading/Data/AMUA_analysis_v2/'
    Figure_path = '/Volumes/Disk/4_ITD_ILD_cueTrading/Figures_paper/revision/'
elif platform == "linux":
    libPath = '/home/shiyi/PycharmProjects/ENVvsPS/'
    Sig_path = '/disks/CIdata/2023_02_23/'
    results_path = '/disks/CIdata/2023_02_23Results/'
    Figure_path = '/disks/CIdata/2_ITD_ILD_CueTrading/Figures/'
else:
    libPath = './'
    Sig_path = './Data/AMUA/'
    results_path = './Data/AMUA_analysis_v2/'
    Figure_path = './Figures/'

sys.path.append(libPath)
import RZ2ephys as ep

os.makedirs(Figure_path, exist_ok=True)

# Configuration parameters
LOAD_PREVIOUS_DATA = True
ResponsiveMethod = 2

t0 = 3
t1 = 30
ts1 = 3
ts2 = 30

figname1_pre = 'Fig5_VE_distribution'
figname2_pre = 'Fig6_VE_Count_of_significant_units'
figname3_pre = 'Fig7_VE_ILD_ITD_per_animal'

# Set file names based on time window
if t1 == 200 and ts1 == 3 and ts2 == 200:
    results_dataname = 'all_files_VE_data_v2_3to200ms.npz'
    figname1 = figname1_pre + '_3to200ms'
    figname2 = figname2_pre + '_3to200ms'
    figname3 = figname3_pre + '_3to200ms'

elif t1 == 30 and ts1 == 3 and ts2 == 200:
    results_dataname = 'all_files_VE_data_stiDur.npz'
    figname1 = figname1_pre + '_stiDur_v1'
    figname2 = figname2_pre + '_stiDur_v1'
    figname3 = figname3_pre + '_stiDur_v1'

elif t1 == 30 and ts1 == 3 and ts2 == 30:
    results_dataname = 'all_files_VE_data_3to30ms.npz'
    figname1 = figname1_pre + '_3to30ms'
    figname2 = figname2_pre + '_3to30ms'
    figname3 = figname3_pre + '_3to30ms'

elif t1 == 30 and ts1 == 30 and ts2 == 200:
    results_dataname = 'all_files_VE_data_30to200ms.npz'
    figname1 = figname1_pre + '_30to200ms'
    figname2 = figname2_pre + '_30to200ms'
    figname3 = figname3_pre + '_30to200ms'

#%% Get file list and initialize arrays
filelist = [
    f for f in os.listdir(Sig_path)
    if f.endswith('.npy') and 'AMUA' in f
]
filelist.sort()

nfiles = len(filelist)
stiRate = [1, 100, 900]
nrates = len(stiRate)
nchans = 32

allunitVE_value = np.zeros((nfiles, nrates, nchans, 2), dtype='float32')
allunitVE_Stats = np.zeros((nfiles, nrates, nchans, 2), dtype='float32')
all_Stats_perChan_pvalues = np.zeros((nfiles, nchans), dtype='float32')
all_unitIndex = np.zeros((nfiles, nchans), dtype='float32')
global_unitIndex = np.zeros(nchans, dtype='float32')
all_fnames = []
VE_sigunit = np.zeros((nfiles, nrates, 4), dtype='float32')
ResponsivePidx = np.zeros(nfiles, dtype='float32')
ResponsivePidxLen = np.zeros(nfiles, dtype='float32')

#%% Load or compute data
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

    print(f"allunitVE_Stats shape after loading: {allunitVE_Stats.shape}")

    expected_shape = (nfiles, nrates, nchans, 2)

    if (
        allunitVE_value.shape != expected_shape
        or allunitVE_Stats.shape != expected_shape
        or all_Stats_perChan_pvalues.shape != (nfiles, nchans)
        or all_unitIndex.shape != (nfiles, nchans)
        or global_unitIndex.shape != (nchans,)
        or len(all_fnames) != nfiles
    ):
        print("Warning: Loaded data dimensions or file list mismatch. Recomputing data.")
        LOAD_PREVIOUS_DATA = False
else:
    LOAD_PREVIOUS_DATA = False

if not LOAD_PREVIOUS_DATA:
    for ff, fname0 in enumerate(filelist):

        Sig_AMUA = np.load(os.path.join(Sig_path, fname0), allow_pickle=True)
        fname = fname0[:-8]
        all_fnames.append(fname)

        nITD = Sig_AMUA.shape[2]
        nILD = Sig_AMUA.shape[3]
        ntrials = Sig_AMUA.shape[4]
        fs = 2000

        xs = np.arange(0, Sig_AMUA.shape[5] / fs, 1 / fs) - 0.1
        xs = xs * 1000

        Sig_AMUA_avgTrial = np.mean(Sig_AMUA, 4)

        Stats_perChan_pvalues = np.zeros(nchans, dtype='float32')
        UnitIndex = np.zeros(nchans, dtype='float32')

        ind_baseline = np.where(xs < 0)[0]
        ind_onset = np.where((xs > 3) & (xs < 30))[0]
        ind_artifact = np.where((xs >= 0) & (xs <= 2))[0]

        for cc in range(nchans):

            temp_baseline = np.mean(
                Sig_AMUA[cc, :, :, :, :, ind_baseline[0]:ind_baseline[-1]],
                4
            )

            temp_onset = np.mean(
                Sig_AMUA[cc, :, :, :, :, ind_onset[0]:ind_onset[-1]],
                4
            )

            temp_artifact = np.max(
                Sig_AMUA_avgTrial[cc, :, :, :, ind_artifact[0]:ind_artifact[-1]],
                3
            )

            temp_baseline_avg = np.mean(
                Sig_AMUA_avgTrial[cc, :, :, :, ind_baseline[0]:ind_baseline[-1]],
                3
            )

            temp_baseline_SD = np.std(
                Sig_AMUA_avgTrial[cc, :, :, :, ind_baseline[0]:ind_baseline[-1]],
                3
            )

            artifact_mask = temp_artifact > (
                temp_baseline_avg + 10 * temp_baseline_SD
            )

            if np.any(artifact_mask):
                Stats_perChan_pvalues[cc] = 1.0
                continue

            temp_baseline_alldata = temp_baseline.flatten()
            temp_onset_alldata = temp_onset.flatten()

            Stats_baseline_onset = stats.wilcoxon(
                temp_baseline_alldata,
                temp_onset_alldata,
                alternative='less'
            )

            Stats_perChan_pvalues[cc] = Stats_baseline_onset.pvalue

            if Stats_perChan_pvalues[cc] < 0.05:
                UnitIndex[cc] = 1

        all_Stats_perChan_pvalues[ff, :] = Stats_perChan_pvalues
        all_unitIndex[ff, :] = UnitIndex

        VE_array = np.zeros((nrates, nchans, 2, 1001), dtype='float32')
        PermutationTest = np.zeros((nrates, nchans, 2), dtype='float32')

        for rr in range(nrates):

            if rr == 0:
                idx_sig = np.where((xs > t0) & (xs < t1))[0]
            else:
                idx_sig = np.where((xs > ts1) & (xs < ts2))[0]

            for cc in range(nchans):

                if UnitIndex[cc] == 0:
                    continue

                for ii in range(1001):

                    dat_temp = np.mean(
                        Sig_AMUA[cc, rr, :, :, :, idx_sig[0]:idx_sig[-1]],
                        3
                    )

                    if ii > 0:
                        dat_temp_flatten = dat_temp.flatten()
                        np.random.shuffle(dat_temp_flatten)
                        dat_temp = dat_temp_flatten.reshape(
                            nITD, nILD, ntrials
                        )

                    dat_grand_mean = np.mean(dat_temp)

                    dat_ITD = np.mean(dat_temp, 1)
                    dat_ITD_mean = np.mean(dat_ITD, 1)

                    dat_ILD = np.mean(dat_temp, 0)
                    dat_ILD_mean = np.mean(dat_ILD, 1)

                    dat_ITD_diff = dat_ITD_mean - np.repeat(
                        dat_grand_mean, nITD
                    )
                    SS_ITD = ntrials * nILD * sum(dat_ITD_diff ** 2)

                    dat_ILD_diff = dat_ILD_mean - np.repeat(
                        dat_grand_mean, nILD
                    )
                    SS_ILD = ntrials * nITD * sum(dat_ILD_diff ** 2)

                    dat_flatten = dat_temp.flatten()
                    dat_diff = dat_flatten - np.repeat(
                        dat_grand_mean, len(dat_flatten)
                    )
                    SS_total = sum(dat_diff ** 2)

                    VE_array[rr, cc, 0, ii] = (SS_ITD / SS_total) * 100
                    VE_array[rr, cc, 1, ii] = (SS_ILD / SS_total) * 100

                threshold_ITD = np.percentile(
                    VE_array[rr, cc, 0, 1:], 95
                )
                threshold_ILD = np.percentile(
                    VE_array[rr, cc, 1, 1:], 95
                )

                PermutationTest[rr, cc, 0] = (
                    1 if VE_array[rr, cc, 0, 0] > threshold_ITD else 0
                )
                PermutationTest[rr, cc, 1] = (
                    1 if VE_array[rr, cc, 1, 0] > threshold_ILD else 0
                )

        allunitVE_value[ff, :, :, :] = VE_array[:, :, :, 0]
        allunitVE_Stats[ff, :, :, :] = PermutationTest

    global_unitIndex = np.min(all_unitIndex, axis=0)

    np.savez(
        os.path.join(results_path, results_dataname),
        VE_value=allunitVE_value,
        VE_stats=allunitVE_Stats,
        pvalues=all_Stats_perChan_pvalues,
        unit_index=all_unitIndex,
        global_unit_index=global_unitIndex,
        fnames=np.array(all_fnames)
    )

#%% Count responsive units
allunitIndex = [
    np.where(all_unitIndex[ff] == 1)[0]
    for ff in range(nfiles)
]

NumResponsiveUnit = [len(idx) for idx in allunitIndex]

ResponsivePenetration = [
    ff for ff, count in enumerate(NumResponsiveUnit)
    if count > 0
]

NgoodPenetration = len(ResponsivePenetration)

print(f"Total Per-File Responsive Units: {sum(NumResponsiveUnit)}")
print(
    f"Globally Responsive Units (responsive in all files): "
    f"{np.sum(global_unitIndex)}"
)

for ff, fname in enumerate(all_fnames):
    print(f"File {fname}: {NumResponsiveUnit[ff]} responsive units")

#%% Figure 5: Scatter plot of VE_ITD vs. VE_ILD
plt.figure(figsize=[12, 5])

for idx_rate in range(nrates):

    alldata_VE_sig_all = []
    colors_all = ['grey', 'blue', 'magenta', 'red']

    itd_greater = 0
    ild_greater = 0
    total_compared_units = 0

    plt.subplot(1, 3, idx_rate + 1)

    for jj, (ITD_test, ILD_test) in enumerate(
        [(0, 0), (1, 0), (0, 1), (1, 1)]
    ):
        alldata_VE_sig = []

        for ii in range(NgoodPenetration):

            IdxP = ResponsivePenetration[ii]
            IdxGoodUnit = allunitIndex[IdxP]

            if len(IdxGoodUnit) == 0:
                continue

            VE_value_SingleP = allunitVE_value[
                IdxP, :, IdxGoodUnit, :
            ]

            VE_stats_SingleP = allunitVE_Stats[
                IdxP, :, IdxGoodUnit, :
            ]

            Idx_sigUnit = np.where(
                (VE_stats_SingleP[:, idx_rate, 0] == ITD_test)
                & (VE_stats_SingleP[:, idx_rate, 1] == ILD_test)
            )[0]

            if Idx_sigUnit.shape[0] > 0:
                alldata_VE_sig.append(
                    VE_value_SingleP[Idx_sigUnit, idx_rate, :]
                )

                if jj in [3]:
                    for unit_idx in Idx_sigUnit:
                        ve_itd = VE_value_SingleP[
                            unit_idx, idx_rate, 0
                        ]
                        ve_ild = VE_value_SingleP[
                            unit_idx, idx_rate, 1
                        ]

                        if ve_itd > ve_ild:
                            itd_greater += 1
                        elif ve_ild > ve_itd:
                            ild_greater += 1

                        total_compared_units += 1

        alldata_VE_sig_temp = (
            np.concatenate(alldata_VE_sig)
            if alldata_VE_sig
            else np.array([])
        )

        alldata_VE_sig_all.append(alldata_VE_sig_temp)

        if alldata_VE_sig_temp.size > 0:
            plt.scatter(
                alldata_VE_sig_temp[:, 0],
                alldata_VE_sig_temp[:, 1],
                color=colors_all[jj],
                s=12,
                edgecolors='black',
                linewidth=0.2
            )

    plt.xlim([0, 95])
    plt.ylim([0, 95])
    plt.title(f'{stiRate[idx_rate]} pps')
    plt.xlabel('VE by ITD (%)')

    if idx_rate == 0:
        plt.ylabel('VE by ILD (%)')

    plt.gca().set_aspect('equal', adjustable='box')

    print(f"\nRate {stiRate[idx_rate]} pps VE Comparison:")
    print(f"  Units with VE_ITD > VE_ILD: {itd_greater}")
    print(f"  Units with VE_ILD > VE_ITD: {ild_greater}")
    print(f"  Total units compared: {total_compared_units}")

    data_range = max(
        [
            np.max(data)
            for data in alldata_VE_sig_all
            if data.size > 0
        ],
        default=95
    )

    plt.axline(
        (0, 0),
        (data_range, data_range),
        color="black",
        linestyle='--'
    )

    ax_inset = inset_axes(
        plt.gca(),
        width="30%",
        height="30%",
        loc='upper right'
    )

    for data, color in zip(alldata_VE_sig_all, colors_all):
        if data.size > 0:
            ax_inset.scatter(
                data[:, 0],
                data[:, 1],
                color=color,
                s=12,
                edgecolors='black',
                linewidth=0.2
            )

    ax_inset.set_xlim(0, 5)
    ax_inset.set_ylim(0, 5)
    ax_inset.set_xticks([0, 5])
    ax_inset.set_yticks([0, 5])
    ax_inset.tick_params(labelsize=8)
    ax_inset.set_aspect('equal', adjustable='box')
    ax_inset.patch.set_edgecolor('black')
    ax_inset.patch.set_linewidth(1)

    if idx_rate == 2:
        plt.legend(
            [
                'Neither significant',
                'ITD significant only',
                'ILD significant only',
                'Both significant'
            ],
            bbox_to_anchor=(0.0001, 0.99)
        )

plt.subplots_adjust(
    left=0.08,
    bottom=0.1,
    right=0.95,
    top=0.9,
    wspace=0.15
)

plt.suptitle(figname1)
plt.tight_layout()
plt.savefig(os.path.join(Figure_path, figname1 + '.png'), dpi=300)
plt.show()

#%% Figure 5b: VE_ITD / (VE_ITD + VE_ILD)
plt.figure(figsize=[12, 4])

for idx_rate in range(nrates):

    plt.subplot(1, nrates, idx_rate + 1)

    ratio_values_all = []

    for ii in range(NgoodPenetration):

        IdxP = ResponsivePenetration[ii]
        IdxGoodUnit = allunitIndex[IdxP]

        if len(IdxGoodUnit) == 0:
            continue

        VE_value_SingleP = allunitVE_value[
            IdxP, :, IdxGoodUnit, :
        ]

        VE_stats_SingleP = allunitVE_Stats[
            IdxP, :, IdxGoodUnit, :
        ]

        Idx_sigUnit = np.where(
            (VE_stats_SingleP[:, idx_rate, 0] == 1)
            & (VE_stats_SingleP[:, idx_rate, 1] == 1)
        )[0]

        if Idx_sigUnit.shape[0] == 0:
            continue

        ve_itd = VE_value_SingleP[Idx_sigUnit, idx_rate, 0]
        ve_ild = VE_value_SingleP[Idx_sigUnit, idx_rate, 1]

        total_VE = ve_itd + ve_ild
        valid_units = total_VE > 0

        ratio_values_all.append(
            ve_itd[valid_units] / total_VE[valid_units]
        )

    if len(ratio_values_all) > 0:
        ratio_values_all = np.concatenate(ratio_values_all)

        plt.hist(
            ratio_values_all,
            bins=np.arange(0, 1.0001, 0.05),
            color='steelblue',
            alpha=0.8,
            edgecolor='black',
            linewidth=0.4
        )

    plt.axvline(0.5, color='black', linestyle='--', linewidth=1)
    plt.xlim([0, 1])
    plt.ylim(bottom=0)
    plt.title(f'{stiRate[idx_rate]} pps')
    plt.xlabel('VE_ITD / (VE_ITD + VE_ILD)')

    if idx_rate == 0:
        plt.ylabel('Number of units')

plt.subplots_adjust(
    left=0.08,
    bottom=0.12,
    right=0.97,
    top=0.88,
    wspace=0.25
)

plt.suptitle(figname1 + ' (both ITD and ILD significant)')

plt.savefig(
    os.path.join(Figure_path, figname1 + '_VEratio_bothSig.png'),
    dpi=300
)

plt.show()

#%% Figure 5c: VE_ITD - VE_ILD
plt.figure(figsize=[12, 4])

for idx_rate in range(nrates):

    plt.subplot(1, nrates, idx_rate + 1)

    ve_difference_all = []

    for ii in range(NgoodPenetration):

        IdxP = ResponsivePenetration[ii]
        IdxGoodUnit = allunitIndex[IdxP]

        if len(IdxGoodUnit) == 0:
            continue

        VE_value_SingleP = allunitVE_value[
            IdxP, :, IdxGoodUnit, :
        ]

        VE_stats_SingleP = allunitVE_Stats[
            IdxP, :, IdxGoodUnit, :
        ]

        Idx_sigUnit = np.where(
            (VE_stats_SingleP[:, idx_rate, 0] == 1)
            & (VE_stats_SingleP[:, idx_rate, 1] == 1)
        )[0]

        if Idx_sigUnit.shape[0] == 0:
            continue

        ve_itd = VE_value_SingleP[Idx_sigUnit, idx_rate, 0]
        ve_ild = VE_value_SingleP[Idx_sigUnit, idx_rate, 1]

        ve_difference_all.append(ve_itd - ve_ild)

    if len(ve_difference_all) > 0:
        ve_difference_all = np.concatenate(ve_difference_all)

        plt.hist(
            ve_difference_all,
            bins=np.arange(-100, 101, 5),
            color='steelblue',
            alpha=0.8,
            edgecolor='black',
            linewidth=0.4
        )

    plt.axvline(0, color='black', linestyle='--', linewidth=1)
    plt.xlim([-100, 100])
    plt.ylim(bottom=0)
    plt.title(f'{stiRate[idx_rate]} pps')
    plt.xlabel('VE_ITD - VE_ILD (%)')

    if idx_rate == 0:
        plt.ylabel('Number of units')

plt.subplots_adjust(
    left=0.08,
    bottom=0.12,
    right=0.97,
    top=0.88,
    wspace=0.25
)

plt.suptitle(figname1 + ' (both ITD and ILD significant)')

plt.savefig(
    os.path.join(Figure_path, figname1 + '_VEdifference_bothSig.png'),
    dpi=300
)

plt.show()

#%% Figure 6: Bar plot of significant-unit proportions
VE_counts = np.zeros((nrates, 4), dtype='float32')
total_units = 0

print(f"allunitVE_Stats shape: {allunitVE_Stats.shape}")

if (
    allunitVE_Stats.shape[1] != nrates
    or allunitVE_Stats.shape[3] != 2
):
    print(
        f"Error: allunitVE_Stats has incorrect shape "
        f"{allunitVE_Stats.shape}. Expected "
        f"(nfiles, {nrates}, {nchans}, 2)."
    )
    VE_counts[:] = 0

else:
    for ff in range(nfiles):

        UnitIndex = np.where(all_unitIndex[ff] == 1)[0]

        if len(UnitIndex) == 0:
            continue

        total_units += len(UnitIndex)

        print(
            f"File {ff} ({all_fnames[ff]}): "
            f"{len(UnitIndex)} responsive units"
        )

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
    print("Warning: No responsive units found across all files.")

else:
    fig, ax = plt.subplots(figsize=(6, 4.5))

    barWidth = 0.15
    br1 = np.arange(nrates)
    br2 = [x + barWidth for x in br1]
    br3 = [x + barWidth for x in br2]
    br4 = [x + barWidth for x in br3]

    bars1 = plt.bar(
        br1,
        VE_counts[:, 0] / total_units * 100,
        color='grey',
        width=barWidth,
        label='Neither significant'
    )

    bars2 = plt.bar(
        br2,
        VE_counts[:, 1] / total_units * 100,
        color='blue',
        width=barWidth,
        label='ITD significant only'
    )

    bars3 = plt.bar(
        br3,
        VE_counts[:, 2] / total_units * 100,
        color='magenta',
        width=barWidth,
        label='ILD significant only'
    )

    bars4 = plt.bar(
        br4,
        VE_counts[:, 3] / total_units * 100,
        color='red',
        width=barWidth,
        label='Both significant'
    )

    def add_value_labels(bars):
        for bar in bars:
            height = bar.get_height()
            plt.text(
                bar.get_x() + bar.get_width() / 2.,
                height,
                f'{height:.1f}',
                ha='center',
                va='bottom',
                fontsize=6,
                fontweight='bold'
            )

    add_value_labels(bars1)
    add_value_labels(bars2)
    add_value_labels(bars3)
    add_value_labels(bars4)

    plt.legend()
    plt.xticks(
        [r + barWidth * 1.5 for r in range(nrates)],
        ['1 pps', '100 pps', '900 pps']
    )
    plt.xlabel('Stimulus rate')
    plt.ylabel('% of multiunits')
    plt.ylim([0, 100])

    print(f"Total responsive units: {total_units}")
    print(f"Counts for Neither significant: {VE_counts[:, 0]}")
    print(f"Counts for ITD significant only: {VE_counts[:, 1]}")
    print(f"Counts for ILD significant only: {VE_counts[:, 2]}")
    print(f"Counts for Both significant: {VE_counts[:, 3]}")

    plt.suptitle(figname2)
    plt.tight_layout()
    plt.savefig(os.path.join(Figure_path, figname2 + '.png'), dpi=300)
    plt.show()

#%% Figure 7: Violin plots of VE across pulse rates
# All responsive units are included.
# Significant VE values are coloured by animal.
# Nonsignificant VE values are grey dots with black edges.

def print_summary_stats(units_data, ve_type, animal_ids, animal_labels):
    """
    Compute per-animal median VE values and perform:
    1. Friedman test across the three pulse rates.
    2. Post hoc paired Wilcoxon signed-rank tests with Bonferroni correction.
    """

    if not units_data:
        print(f"No {ve_type} units found")
        return {}

    rate_data = [[] for _ in range(3)]
    valid_animal_ids = sorted(set(unit['animal'] for unit in units_data))

    for animal_id in valid_animal_ids:
        animal_units = [
            unit for unit in units_data
            if unit['animal'] == animal_id
        ]

        if not animal_units:
            continue

        animal_values = np.array([
            unit['values'] for unit in animal_units
        ])

        if animal_values.size == 0:
            continue

        median_values = np.median(animal_values, axis=0)

        for j in range(3):
            rate_data[j].append(median_values[j])

    print(
        f"\n=== {ve_type} Median VE Summary Statistics "
        f"(Across Animals) ==="
    )
    print(f"Number of animals: {len(valid_animal_ids)}")

    for j, rate in enumerate(stiRate):
        rate_data_arr = np.array(rate_data[j])

        if len(rate_data_arr) == 0:
            continue

        print(f"\nRate {rate} pps:")
        print(f"  Median: {np.median(rate_data_arr):.2f}%")
        print(f"  Mean: {np.mean(rate_data_arr):.2f}%")
        print(
            f"  Q1 (25th %ile): "
            f"{np.percentile(rate_data_arr, 25):.2f}%"
        )
        print(
            f"  Q3 (75th %ile): "
            f"{np.percentile(rate_data_arr, 75):.2f}%"
        )
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

    if all(len(rate_data[j]) > 0 for j in range(3)) and len(rate_data[0]) > 1:

        stat, p_value = stats.friedmanchisquare(
            rate_data[0],
            rate_data[1],
            rate_data[2]
        )

        print(
            f"\nFriedman Test for {ve_type} across rates "
            f"(within animals):"
        )
        print(f"  Chi-squared statistic: {stat:.2f}")
        print(f"  p-value: {p_value:.4f}")

        if p_value < 0.05:
            print(
                f"\nPost-Hoc Paired Wilcoxon Signed-Rank Tests for {ve_type} "
                f"(Bonferroni-corrected alpha = 0.0167):"
            )

            pairs = [
                (0, 1, f"{stiRate[0]} vs. {stiRate[1]} pps"),
                (0, 2, f"{stiRate[0]} vs. {stiRate[2]} pps"),
                (1, 2, f"{stiRate[1]} vs. {stiRate[2]} pps"),
            ]

            for i, j, label in pairs:
                stat, p_val = stats.wilcoxon(
                    rate_data[i],
                    rate_data[j],
                    alternative='two-sided'
                )

                p_adj = min(p_val * 3, 1.0)
                p_values[(i, j)] = p_adj

                print(f"  {label}:")
                print(f"    Statistic: {stat:.2f}")
                print(f"    Raw p-value: {p_val:.4f}")
                print(
                    f"    Bonferroni-adjusted p-value: {p_adj:.4f}"
                )
                print(
                    f"    Significant: "
                    f"{'Yes' if p_adj < 0.05 else 'No'}"
                )

    else:
        print(
            f"\nInsufficient data for Friedman test for {ve_type} "
            f"(need at least 2 animals with data)"
        )

    return p_values


def plot_violins_per_animal(
    units_data,
    ax,
    title,
    ve_type,
    stiRate,
    animal_ids,
    p_values,
    animal_colors,
    animal_labels
):
    """
    Plot VE distributions for all responsive units across animals and rates.

    Significant VE values are coloured by animal.
    Nonsignificant VE values are grey dots with black edges.
    No legend is created inside the axes.
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
        ax.set_xlabel('Stimulus Rate (pps)')
        ax.set_ylabel(f'VE_{ve_type} (%)')
        ax.set_title(title)
        ax.set_ylim(0, 100)
        return

    n_rates = len(stiRate)
    valid_animal_ids = sorted(set(unit['animal'] for unit in units_data))
    n_animals = len(valid_animal_ids)

    bar_width = 0.15
    group_width = 1.0

    positions = []
    for i in range(n_rates):
        start = i * (n_animals * bar_width + group_width - bar_width)
        positions.extend([
            start + j * bar_width
            for j in range(n_animals)
        ])

    rate_data = [[] for _ in range(n_rates)]
    max_y = 0
    all_data = []
    all_sig = []

    for i in range(n_rates):
        for animal_id in valid_animal_ids:
            animal_units = [
                unit for unit in units_data
                if unit['animal'] == animal_id
            ]

            data_array = np.array([
                unit['values'][i] for unit in animal_units
            ])

            sig_array = np.array([
                unit['sig_flags'][i] for unit in animal_units
            ], dtype=bool)

            all_data.append(data_array)
            all_sig.append(sig_array)

            rate_data[i].append(
                np.median(data_array) if data_array.size > 0 else np.nan
            )

            if data_array.size > 0:
                max_y = max(max_y, np.max(data_array))

    violin_parts = ax.violinplot(
        [data for data in all_data if data.size > 0],
        positions=[
            pos for pos, data in zip(positions, all_data)
            if data.size > 0
        ],
        widths=bar_width * 0.8,
        showmedians=True
    )

    plotted_indices = [
        idx for idx, data in enumerate(all_data)
        if data.size > 0
    ]

    for violin_idx, idx in enumerate(plotted_indices):
        pc = violin_parts['bodies'][violin_idx]

        animal_idx = idx % n_animals
        animal_id = valid_animal_ids[animal_idx]

        pc.set_facecolor(animal_colors[animal_id])
        pc.set_alpha(0.3)
        pc.set_edgecolor('black')
        pc.set_linewidth(1)

    violin_parts['cmedians'].set_color('black')
    violin_parts['cmedians'].set_linewidth(2)

    for part in ['cmaxes', 'cmins', 'cbars']:
        if part in violin_parts:
            violin_parts[part].set_alpha(0)

    for idx, data_array in enumerate(all_data):
        if data_array.size == 0:
            continue

        pos = positions[idx]

        q1 = np.percentile(data_array, 25)
        q3 = np.percentile(data_array, 75)

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
            len(data_array)
        )

        animal_idx = idx % n_animals
        animal_id = valid_animal_ids[animal_idx]
        sig_array = all_sig[idx]

        for xp, val, sig in zip(
            x_positions,
            data_array,
            sig_array
        ):
            if sig:
                point_color = animal_colors[animal_id]
                point_alpha = 0.7
            else:
                point_color = 'grey'
                point_alpha = 0.6

            ax.scatter(
                xp,
                val,
                color=point_color,
                s=10,
                alpha=point_alpha,
                edgecolor='black',
                linewidth=0.5
            )

    sig_pairs = [
        (0, 1, f"{stiRate[0]} vs. {stiRate[1]} pps"),
        (0, 2, f"{stiRate[0]} vs. {stiRate[2]} pps"),
        (1, 2, f"{stiRate[1]} vs. {stiRate[2]} pps"),
    ]

    bar_heights = [
        max_y + 5 + k * 10
        for k in range(len(sig_pairs))
    ]

    for idx, (i, j, _) in enumerate(sig_pairs):
        if (i, j) in p_values and p_values[(i, j)] < 0.05:
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
                fontsize=10
            )

    ax.set_xticks([
        i * (n_animals * bar_width + group_width - bar_width)
        + (n_animals - 1) * bar_width / 2
        for i in range(n_rates)
    ])

    ax.set_xticklabels([
        str(rate) for rate in stiRate
    ])

    ax.set_xlabel('Stimulus Rate (pps)')
    ax.set_ylabel(f'VE_{ve_type} (%)')
    ax.set_title(title)
    ax.set_ylim(0, max_y + 30)
    ax.grid(True, alpha=0.3)

    # No ax.legend() here.
    # The shared figure-level legend is created after both panels are plotted.


#%% Collect all responsive units for Figure 7
ild_units = []
itd_units = []
animal_ids = []

unique_animals = sorted(set([
    fname.split('_')[0] for fname in all_fnames
]))

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
        if np.any(all_unitIndex[ff] == 1):
            has_responsive_unit = True
            break

    if has_responsive_unit:
        responsive_animals.append(animal_id)

animal_colors = {
    animal_id: plt.colormaps['tab20'](i % 20 / 20)
    for i, animal_id in enumerate(sorted(responsive_animals))
}

animal_labels = {
    animal_id: f'Rat {i + 1}'
    for i, animal_id in enumerate(sorted(responsive_animals))
}

for animal_id in responsive_animals:
    animal_files = [
        ff for ff, fname in enumerate(all_fnames)
        if fname.startswith(animal_id)
    ]

    for ff in animal_files:
        responsive_channels = np.where(all_unitIndex[ff] == 1)[0]

        for cc in responsive_channels:

            ild_sig_flags = allunitVE_Stats[ff, :, cc, 1] == 1
            itd_sig_flags = allunitVE_Stats[ff, :, cc, 0] == 1

            ild_units.append({
                'values': allunitVE_value[ff, :, cc, 1],
                'sig_flags': ild_sig_flags.astype(bool),
                'animal': animal_id,
                'file': ff,
                'unit': cc,
            })

            itd_units.append({
                'values': allunitVE_value[ff, :, cc, 0],
                'sig_flags': itd_sig_flags.astype(bool),
                'animal': animal_id,
                'file': ff,
                'unit': cc,
            })

    animal_ids.append(animal_id)

#%% Generate Figure 7: 2 rows, shared legend outside at upper right
fig, (ax_ild, ax_itd) = plt.subplots(
    2, 1,
    figsize=(9, 10),
    sharey=True
)

# ILD panel
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
    stiRate,
    animal_ids,
    p_values_ild,
    animal_colors,
    animal_labels
)

# ITD panel
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
    stiRate,
    animal_ids,
    p_values_itd,
    animal_colors,
    animal_labels
)


# Create one shared legend outside the axes, at the upper-right corner
legend_elements = [
    Line2D(
        [0], [0],
        color=animal_colors[animal_id],
        lw=2,
        label=label
    )
    for animal_id, label in animal_labels.items()
    if animal_id in responsive_animals
]

legend_elements.append(
    Line2D(
        [0], [0],
        color='grey',
        marker='o',
        markersize=6,
        markeredgecolor='black',
        lw=0,
        label='non-significant'
    )
)

fig.legend(
    handles=legend_elements,
    loc='right',
    bbox_to_anchor=(0.99, 0.99),
    fontsize=9,
    frameon=True
)

# Remove any existing legends inside the panels
for ax in (ax_ild, ax_itd):
    legend = ax.get_legend()
    if legend is not None:
        legend.remove()

# Remove any existing figure-level legends
for legend in list(fig.legends):
    legend.remove()

fig.suptitle(figname3)

# Reserve space on the right for the legend
fig.tight_layout(rect=[0, 0, 0.80, 0.95])

# One legend outside the top panel, aligned with its top edge
ax_ild.legend(
    handles=legend_elements,
    loc='upper left',
    bbox_to_anchor=(1.02, 1.0),
    borderaxespad=0,
    fontsize=9,
    frameon=True
)

plt.suptitle(figname3)

# Reserve space on the right for the outside legend
plt.tight_layout(rect=[0, 0, 0.82, 0.95])

fig.savefig(
    os.path.join(Figure_path, figname3 + '.png'),
    dpi=300,
    bbox_inches='tight'
)

plt.show()



# plt.savefig(
#     os.path.join(Figure_path, figname3 + '.png'),
#     dpi=300,
#     bbox_inches='tight'
# )

# plt.show()