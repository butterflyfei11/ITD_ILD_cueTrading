import numpy as np
import os
from sys import platform
import matplotlib.pyplot as plt
from scipy import interpolate
import json



# Configuration
use_interpolation = True  # Set to False to plot heatmaps without ILD interpolation (uses 5 ILD levels)
use_predefined = True

# Set paths based on platform
if platform == "darwin":
    Sig_path = '/Volumes/Disk/4_ITD_ILD_cueTrading/Data/AMUA/'
    results_path = '/Volumes/Disk/4_ITD_ILD_cueTrading/Data/AMUA_analysis_v2/'
    FigurePath = '/Volumes/Disk/4_ITD_ILD_cueTrading/Figures_paper/'
    data_path = '/Volumes/Disk/4_ITD_ILD_cueTrading/Data/AMUA_selected_units/'
elif platform == "linux":
    Sig_path = '/disks/CIdata/2023_02_23/'
    results_path = '/disks/CIdata/2023_02_23Results/'
    FigurePath = '/disks/CIdata/2_ITD_ILD_CueTrading/Figures/Fig3_ExampleUnits_AMUA/'
    data_path = '/disks/CIdata/2_ITD_ILD_CueTrading/Data/AMUA_selected_units/'
else:
    Sig_path = './Data/AMUA/'
    results_path = './Data/AMUA_analysis_v2/'
    FigurePath = './Figures/'
    data_path = './Data/AMUA_selected_units/'

# Ensure output directories exist
os.makedirs(FigurePath, exist_ok=True)
os.makedirs(data_path, exist_ok=True)

# Initialize parameters
stiRate = [1, 100, 900]  # Stimulus rates (pps)
nrates = len(stiRate)
stimILD = [-4, -1, 0, 1, 4]  # Original ILD levels (dB)
stimITD = np.round(np.arange(-0.12, 0.14, 0.04), 2)  # ITD levels (ms)
numITD = len(stimITD)  # 7
numILD_orig = len(stimILD)  # 5
stimILD_interp = np.arange(-4, 5, 1)  # Interpolated ILD levels: [-4, -3, -2, -1, 0, 1, 2, 3, 4]
numILD = len(stimILD_interp) if use_interpolation else numILD_orig  # 9 or 5
Fs_ds = 2000  # Sampling frequency (Hz)


# Define predefined units (IdxP, IdxChan) for each category
predefined_units = {
    'ITD-dominated': {'IdxP': 9, 'IdxChan': 14},
    'ILD-dominated': {'IdxP': 17, 'IdxChan': 1},
    'Balanced ITD-ILD': {'IdxP': 5, 'IdxChan': 1}
}

# Smoothing function for waveforms
def smooth(signal, WSZ):
    out0 = np.convolve(signal, np.ones(WSZ, dtype=int), 'valid') / WSZ
    r = np.arange(1, WSZ-1, 2)
    start = np.cumsum(signal[:WSZ-1])[::2] / r
    stop = (np.cumsum(signal[:-WSZ:-1])[::2] / r)[::-1]
    return np.concatenate((start, out0, stop))

# Load data(VE values computed) from .npz file (3-200 ms)
npz_file = os.path.join(results_path, 'all_files_VE_data_3to30ms.npz')    # Check whether the right file is loaded 
if os.path.exists(npz_file):
    print(f"Loading data from {npz_file}")
    data = np.load(npz_file, allow_pickle=True)
    ve_value = data['VE_value']  # Shape: (nfiles, nrates, nchans, 2)
    ve_stats = data['VE_stats']  # Shape: (nfiles, nrates, nchans, 2)
    fnames = data['fnames'].tolist()  # File names
    all_unitIndex = data['unit_index']  # Shape: (nfiles, nchans)
else:
    print(f"File {npz_file} not found. Please ensure the file exists.")
    exit()

nfiles, nrates, nchans, ntypes = ve_value.shape
print(f"Dataset dimensions: {nfiles} files, {nrates} rates, {nchans} channels, {ntypes} types (ITD, ILD)")

# Filter units significant for both ITD and ILD
sig_units = (ve_stats[..., 0] == 1) & (ve_stats[..., 1] == 1)
sig_indices = np.where(sig_units)
ve_itd_sig = ve_value[sig_indices[0], sig_indices[1], sig_indices[2], 0]
ve_ild_sig = ve_value[sig_indices[0], sig_indices[1], sig_indices[2], 1]
sig_file_indices = sig_indices[0]
sig_rate_indices = sig_indices[1]
sig_chan_indices = sig_indices[2]
print(f"Number of units significant for both ITD and ILD: {len(ve_itd_sig)}")

# Compute global percentiles and medians
if len(ve_itd_sig) == 0:
    print("No units significant for both ITD and ILD. Exiting.")
    exit()

itd_95 = np.percentile(ve_itd_sig, 90)
itd_5 = np.percentile(ve_itd_sig, 10)
ild_95 = np.percentile(ve_ild_sig, 90)
ild_5 = np.percentile(ve_ild_sig, 10)
median_itd = np.median(ve_itd_sig)
median_ild = np.median(ve_ild_sig)

print("\nGlobal Percentiles (significant units, across all files, rates, channels):")
print(f"  ITD: 5th={itd_5:.4f}, 95th={itd_95:.4f}, Median={median_itd:.4f}")
print(f"  ILD: 5th={ild_5:.4f}, 95th={ild_95:.4f}, Median={median_ild:.4f}")



# Validate and collect predefined units
predefined_candidates = {
    'ITD-dominated': [],
    'ILD-dominated': [],
    'Balanced ITD-ILD': []
}

print("\nValidating Predefined Units:")
selected_units = []

for unit_type, unit_info in predefined_units.items():
    idx_p = unit_info['IdxP']
    idx_chan = unit_info['IdxChan']
    
    # Validate indices
    if not (0 <= idx_p < nfiles):
        print(f"Error: {unit_type} IdxP={idx_p} is invalid (must be 0 to {nfiles-1}). Falling back to automatic selection.")
        use_predefined = False
        break
    if not (0 <= idx_chan < nchans):
        print(f"Error: {unit_type} IdxChan={idx_chan} is invalid (must be 0 to {nchans-1}). Falling back to automatic selection.")
        use_predefined = False
        break
    if all_unitIndex[idx_p, idx_chan] == 0:
        print(f"Error: {unit_type} unit at IdxP={idx_p}, IdxChan={idx_chan} is not responsive. Falling back to automatic selection.")
        use_predefined = False
        break
    
    # Check criteria for each rate
    valid_unit = False
    for rate_idx in range(nrates):
        ve_itd = ve_value[idx_p, rate_idx, idx_chan, 0]
        ve_ild = ve_value[idx_p, rate_idx, idx_chan, 1]
        unit_info = {
            'type': unit_type,
            'IdxP': idx_p,
            'idx_Rate': rate_idx,
            'idx_chan': idx_chan,
            'rate': stiRate[rate_idx],
            'file': fnames[idx_p],
            've_itd': float(ve_itd),
            've_ild': float(ve_ild)
        }
        
        # Validate criteria
        if unit_type == 'ITD-dominated' and ve_itd > itd_95:
            predefined_candidates['ITD-dominated'].append(unit_info)
            valid_unit = True
        elif unit_type == 'ILD-dominated' and ve_ild > ild_95:
            predefined_candidates['ILD-dominated'].append(unit_info)
            valid_unit = True
        elif unit_type == 'Balanced ITD-ILD':
        # and abs(ve_itd - ve_ild) < 5 and abs(ve_itd - 20) < 5 and abs(ve_ild - 20) < 5:
            predefined_candidates['Balanced ITD-ILD'].append(unit_info)
            valid_unit = True
    
    if valid_unit:
        print(f"{unit_type} unit at IdxP={idx_p}, IdxChan={idx_chan} is valid:")
        for unit in predefined_candidates[unit_type]:
            print(f"  Rate: {unit['rate']} pps, VE_ITD: {unit['ve_itd']:.4f}, VE_ILD: {unit['ve_ild']:.4f}")
        # Use the first valid rate for selected_units
        selected_units.append(predefined_candidates[unit_type][0])
    else:
        print(f"Error: {unit_type} unit at IdxP={idx_p}, IdxChan={idx_chan} does not meet criteria. Falling back to automatic selection.")
        use_predefined = False
        break

# Check if all predefined units are valid and unique
if use_predefined:
    keys = [(u['IdxP'], u['idx_chan']) for u in selected_units]
    if len(set(keys)) != 3:
        print("Error: Predefined units are not unique. Falling back to automatic selection.")
        use_predefined = False

# Automatic unit selection if predefined units are invalid or incomplete
if not use_predefined:
    print("\nPerforming automatic unit selection...")
    # Select units automatically
    candidates_unit1 = []  # High VE_ITD (>95th), prefer lower VE_ILD
    candidates_unit2 = []  # High VE_ILD (>95th), prefer lower VE_ITD
    candidates_unit3 = []  # Similar VE_ITD and VE_ILD around 20

    for idx in range(len(ve_itd_sig)):
        ve_itd = ve_itd_sig[idx]
        ve_ild = ve_ild_sig[idx]
        file_idx = sig_file_indices[idx]
        rate_idx = sig_rate_indices[idx]
        chan_idx = sig_chan_indices[idx]
        file_name = fnames[file_idx]
        unit_info = {
            'type': None,
            'IdxP': file_idx,
            'idx_Rate': rate_idx,
            'idx_chan': chan_idx,
            'rate': stiRate[rate_idx],
            'file': file_name,
            've_itd': float(ve_itd),
            've_ild': float(ve_ild)
        }

        # Unit 1: VE_ITD > 95th percentile
        if ve_itd > itd_95:
            candidates_unit1.append({**unit_info, 'type': 'ITD-dominated'})

        # Unit 2: VE_ILD > 95th percentile
        if ve_ild > ild_95:
            candidates_unit2.append({**unit_info, 'type': 'ILD-dominated'})

        # Unit 3: VE_ITD and VE_ILD close (difference < 5%), around 20 (±5%)
        if abs(ve_itd - ve_ild) < 5 and abs(ve_itd - 20) < 5 and abs(ve_ild - 20) < 5:
            candidates_unit3.append({**unit_info, 'type': 'Balanced ITD-ILD'})

    # Add predefined units to candidates (if any are valid)
    for unit_type in predefined_candidates:
        if predefined_candidates[unit_type]:
            if unit_type == 'ITD-dominated':
                candidates_unit1.append(predefined_candidates[unit_type][0])
            elif unit_type == 'ILD-dominated':
                candidates_unit2.append(predefined_candidates[unit_type][0])
            elif unit_type == 'Balanced ITD-ILD':
                candidates_unit3.append(predefined_candidates[unit_type][0])

    # Sort candidates
    candidates_unit1.sort(key=lambda x: x['ve_ild'])  # Sort by VE_ILD ascending
    candidates_unit2.sort(key=lambda x: x['ve_itd'])  # Sort by VE_ITD ascending
    candidates_unit3.sort(key=lambda x: abs(x['ve_itd'] - x['ve_ild']))  # Sort by VE difference

    # Check if candidates were found
    if not candidates_unit1:
        print("No unit found with VE_ITD > 95th percentile.")
    if not candidates_unit2:
        print("No unit found with VE_ILD > 95th percentile.")
    if not candidates_unit3:
        print("No unit found with VE_ITD and VE_ILD close to 20 (±5%).")
    if not (candidates_unit1 and candidates_unit2 and candidates_unit3):
        print("Insufficient units to form a set. Exiting.")
        exit()

    # Generate up to ten distinct sets with variety
    unit_sets = []
    max_sets = 10
    i = 0
    while len(unit_sets) < max_sets and i < max(len(candidates_unit1), len(candidates_unit2), len(candidates_unit3)):
        u1 = candidates_unit1[min(i, len(candidates_unit1)-1)]
        u2 = candidates_unit2[min(i, len(candidates_unit2)-1)]
        u3 = candidates_unit3[min(i, len(candidates_unit3)-1)]
        keys = [(u['IdxP'], u['idx_chan']) for u in [u1, u2, u3]]
        if len(set(keys)) == 3:  # all unique
            unit_sets.append([u1, u2, u3])
        i += 1

    if not unit_sets:
        print("No valid unit sets found. Exiting.")
        exit()

    # Print alternative unit sets
    print("\nAlternative Unit Sets:")
    for set_idx, unit_set in enumerate(unit_sets, 1):
        print(f"\nSet {set_idx}:")
        for i, unit in enumerate(unit_set, 1):
            print(f"  Unit {i} ({unit['type']}):")
            print(f"    File: {unit['file']}, Channel: {unit['idx_chan']}")
            print(f"    VE_ITD: {unit['ve_itd']:.4f}, VE_ILD: {unit['ve_ild']:.4f}")

    unit_sets_idx = 0
    selected_units = unit_sets[unit_sets_idx]

# Initialize unit data for JSON
unit_data = []

# Process each unit
for unit_idx, unit in enumerate(selected_units):
    IdxP = unit['IdxP']
    idx_chan = unit['idx_chan']
    fname = unit['file'] + 'AMUA.npy'
    ve_itd_round = round(unit['ve_itd'], 2)
    ve_ild_round = round(unit['ve_ild'], 2)

    # Load AMUA data once per unit
    try:
        Sig_AMUA = np.load(os.path.join(Sig_path, fname), allow_pickle=True)
        print(f"Loaded AMUA data for {fname}, shape: {Sig_AMUA.shape}")
    except FileNotFoundError:
        print(f"AMUA file {fname} not found. Skipping unit {unit['type']}.")
        continue

    # Verify AMUA shape
    expected_shape = (nchans, nrates, numITD, numILD_orig, 20)
    if len(Sig_AMUA.shape) < 6 or Sig_AMUA.shape[:5] != expected_shape:
        print(f"Warning: AMUA shape {Sig_AMUA.shape} does not match expected {expected_shape + (Sig_AMUA.shape[5],)}. Skipping.")
        continue

    # Time axis
    xs = np.arange(0, Sig_AMUA.shape[5] / Fs_ds, 1 / Fs_ds) - 0.1  # Time in seconds
    idx_wave = np.where((xs >= -0.1) & (xs <= 0.38))[0]  # Time range: -0.1 to 0.38 s
    print(f"Waveform time indices: {len(idx_wave)} samples")  # Expected: ~960

    # Initialize data storage
    unit_waveforms = []
    unit_heatmaps = []
    rate_data = {}

    # Compute VE per rate for JSON
    ve_per_rate = []
    for rate_idx, rate in enumerate(stiRate):
        ve_itd = ve_value[IdxP, rate_idx, idx_chan, 0]
        ve_ild = ve_value[IdxP, rate_idx, idx_chan, 1]
        ve_per_rate.append({'rate': int(rate), 've_itd': float(ve_itd), 've_ild': float(ve_ild)})

    # Extract data for all rates
    for rate_idx, rate in enumerate(stiRate):
     
        # Time indices for heatmaps
        # idx_heat = np.where((xs > 0.003) & (xs < (0.030 if rate_idx == 0 else 0.2)))[0]  # 3–200 ms for 1 pps, 3–30 ms for 100/900 pps
        idx_heat = np.where((xs > 0.003) & (xs < 0.03))[0]  # 3 to 30 ms for all rates


        print(f"Heatmap time indices (rate={rate}): {len(idx_heat)} samples")  # Expected: ~394 or ~54

        # Extract AMUA for waveforms and heatmaps
        sig_temp_wave = Sig_AMUA[idx_chan, rate_idx, :, :, :, np.min(idx_wave):np.max(idx_wave)+1]  # Shape: (7, 5, 20, len(idx_wave))
        sig_temp_heat = Sig_AMUA[idx_chan, rate_idx, :, :, :, np.min(idx_heat):np.max(idx_heat)+1]  # Shape: (7, 5, 20, len(idx_heat))
        print(f"sig_temp_wave shape (rate={rate}): {sig_temp_wave.shape}")
        print(f"sig_temp_heat shape (rate={rate}): {sig_temp_heat.shape}")

        # Average over trials
        sig_temp_avg_wave = np.mean(sig_temp_wave, axis=2)  # Shape: (7, 5, len(idx_wave))
        sig_temp_avg_heat = np.mean(sig_temp_heat, axis=2)  # Shape: (7, 5, len(idx_heat))
        sig_temp_mean_heat = np.mean(sig_temp_avg_heat, axis=2)  # Shape: (7, 5)
       
        # Store data for this rate
        rate_data[rate] = {
            'sig_temp_avg_wave': sig_temp_avg_wave,
            'sig_temp_mean_heat': sig_temp_mean_heat
        }
        
        # Store heatmap mean data
        unit_heatmaps.append({
            'rate': int(rate),
            'sig_AMUA_mean': sig_temp_mean_heat.tolist(),
            'sig_AMUA_interp': None
        })

    # Compute global min/max for waveform scaling
    sig_temp_all = Sig_AMUA[idx_chan, :, :, :, :, :]  # Shape: (nrates, numITD, numILD, ntrials, nsamples)
    sig_temp_avg_all = np.mean(sig_temp_all, axis=3)  # Average over trials
    sig_temp_max = np.max(sig_temp_avg_all)
    sig_temp_min = np.min(sig_temp_avg_all)

    # Plot time-series waveforms (3x7 grid)
    fig, axes = plt.subplots(3, numITD, figsize=(15, 6), sharex=True, sharey=True)
    plt.subplots_adjust(left=0.09, bottom=0.1, right=0.95, top=0.90, wspace=0.15, hspace=0.3)

    for row, rate in enumerate(stiRate):
        sig_temp_avg_wave = rate_data[rate]['sig_temp_avg_wave']
        waveforms = []

        for kk, itd in enumerate(stimITD):
            ax = axes[row, kk] if numITD > 1 else axes[row]
            rate_waveforms = []

            for jj, ild in enumerate(stimILD):
                # Extract and smooth waveform
                Res_temp = sig_temp_avg_wave[kk, jj, :]  # Shape: (nsamples,)
                Res_temp_smooth = smooth(Res_temp, 5)

                # Store waveform
                rate_waveforms.append({
                    'ITD': float(itd),
                    'ILD': float(ild),
                    # 'waveform': Res_temp_smooth.tolist()
                    'waveform': Res_temp.tolist()
                    
                })

                # Plot waveform with vertical offset
                offset = (jj - 1) * sig_temp_max * 0.2
                ax.plot(xs[idx_wave], Res_temp_smooth + offset, linewidth=1)

                # Annotations
                if row == 0 and kk == numITD - 1:  # Top-right plot (rate=1, ITD=120 μs)
                    ax.text(0.1, offset + 0.5 * np.min(Res_temp_smooth), f'ILD {ild} dB', fontsize=10)
                if kk == 0 and jj == 3:  # Leftmost plot, ILD=1 dB
                    ax.text(-0.35, offset, f'{rate} pps', fontsize=10)

            # Store waveforms for this ITD/rate
            waveforms.append({
                'ITD': float(itd),
                'waveforms': rate_waveforms
            })

            # Plot settings
            ax.axvline(x=0, color='k', linestyle=':')
            ax.axvspan(0, 0.2, color='k', alpha=0.03)
            ax.set_xlim([-0.1, 0.38])
            ax.set_yticks([])

            if row == 2:
                ax.set_xlabel('Time (s)', fontsize=12)
            if kk == 0:
                ax.set_ylabel('AMUA', fontsize=12)
            if row == 0:
                ax.set_title(f'ITD at {int(itd * 1000)} μs', fontsize=10)

        unit_waveforms.append({
            'rate': int(rate),
            'waveforms': waveforms
        })

    # Finalize waveform figure
    plt.suptitle(f"{unit['type']} P{IdxP} Chan{idx_chan} at {unit['rate']} pps\nVE_ITD={ve_itd_round:.2f}, VE_ILD={ve_ild_round:.2f}", y=0.98, fontsize=14)
    plt.tight_layout()

    # Save waveform figure
    figname_wave = f"ExampleUnit_P{IdxP}_Chan{idx_chan}_Waveforms"
    plt.savefig(os.path.join(FigurePath, f'{figname_wave}.png'), dpi=300)
    plt.show()
    plt.close('all')

    # Store unit data (waveforms only, heatmaps added later)
    unit_data.append({
        'idx_unit': unit_idx,
        'idx_P': int(IdxP),
        'idx_chan': int(idx_chan),
        'idx_rate': int(unit['idx_Rate']),
        'selected_rate': int(unit['rate']),
        'type': unit['type'],
        'file': unit['file'],
        'selected_ve_itd': float(ve_itd_round),
        'selected_ve_ild': float(ve_ild_round),
        've_per_rate': ve_per_rate,
        'waveforms': unit_waveforms,
        'heatmaps': unit_heatmaps
    })
#%%
# Plot heatmaps (3x3 grid for all units and rates)
fig, axes = plt.subplots(3, 3, figsize=(8, 8), sharex=True, sharey=True)
plt.subplots_adjust(left=0.2, bottom=0.15, right=0.8, top=0.90, wspace=0.25, hspace=0.5)

#
# Meshgrid for heatmaps (define corners for shading='flat')
x_value = np.linspace(0, numITD, numITD + 1)  # Corners: [0, 1, ..., 7] for 7 ITD bins
y_value = np.linspace(0, numILD, numILD + 1)  # Corners: [0, 1, ..., 9] or [0, 1, ..., 5]
x_plot, y_plot = np.meshgrid(x_value, y_value, indexing='ij')  # Shape: (numITD+1, numILD+1)
xlabels_heatmap = [str(int(itd * 1000)) for itd in stimITD]  # ITD in μs: [-120, -80, -40, 0, 40, 80, 120]
ylabels_heatmap = [str(ild) for ild in (stimILD_interp if use_interpolation else stimILD)]  # ILD in dB
x_value_ticks = np.arange(0, numITD, 1)  # Centers for tick labels
y_value_ticks = np.arange(0, numILD, 1)  # Centers for tick labels

for row, rate in enumerate(stiRate):
    for col, unit in enumerate(selected_units):
        ax = axes[row, col]
        # Add 'A', 'B', 'C' labels on first row
        if row == 0:
            ax.text(-0.2, 1.2, chr(65 + col), transform=ax.transAxes,
                    fontsize=16, fontweight='bold', va='top', ha='left')
        IdxP = unit['IdxP']
        idx_chan = unit['idx_chan']
        unit_type = unit['type']

        # Get VE for this rate
        ve_itd = ve_value[IdxP, row, idx_chan, 0]
        ve_ild = ve_value[IdxP, row, idx_chan, 1]
        ve_itd_round = round(ve_itd, 2)
        ve_ild_round = round(ve_ild, 2)

        # Retrieve pre-computed heatmap data
        for unit_entry in unit_data:
            if unit_entry['idx_P'] == IdxP and unit_entry['idx_chan'] == idx_chan:
                for h in unit_entry['heatmaps']:
                    if h['rate'] == int(rate):
                        sig_temp_mean_heat = np.array(h['sig_AMUA_mean'])
                        break
                else:
                    print(f"No heatmap data found for unit P{IdxP} Chan{idx_chan} at rate {rate}. Skipping.")
                    continue
                break
        else:
            print(f"No data found for unit P{IdxP} Chan{idx_chan} at rate {rate}. Skipping.")
            continue

        # Quadratic interpolation for heatmaps (if enabled)
        if use_interpolation:
            sig_interp_ILD = np.zeros((numITD, numILD), dtype='float32')
            for jj in range(numITD):
                f_interp = interpolate.interp1d(stimILD, sig_temp_mean_heat[jj, :], kind='quadratic')
                sig_interp_ILD[jj, :] = f_interp(stimILD_interp)
            sig_temp_mean_norm = sig_interp_ILD  # Shape: (7, 9)
            h['sig_AMUA_interp'] = sig_interp_ILD.tolist()
        else:
            sig_temp_mean_norm = sig_temp_mean_heat  # Shape: (7, 5)

        # Verify shapes
        # print(f"Heatmap data shape (rate={rate}, unit={unit['type']}): {sig_temp_mean_norm.shape}, x_plot shape: {x_plot.shape}, y_plot shape: {y_plot.shape}")

        # Plot heatmap
        # c = ax.pcolormesh(x_plot[:-1,:-1], y_plot[:-1,:-1], sig_temp_mean_norm, cmap='plasma',shading='gouraud')
        c = ax.pcolormesh(x_plot[:-1,:-1], y_plot[:-1,:-1], sig_temp_mean_norm, cmap='gnuplot2',shading='gouraud')
        ax.set_xticks(x_value_ticks)
        ax.set_xticklabels(xlabels_heatmap, rotation=45, fontsize=12)
        ax.set_yticks(y_value_ticks)
        ax.set_yticklabels(ylabels_heatmap, fontsize=12)

        # # Set title
        # if row == 0:
        #     ax.set_title(f"{unit['type']} P{unit['IdxP']} Chan{unit['idx_chan']}\nUnit {col+1}\n{rate} pps\n"
        #                  f"VE_ITD={ve_itd_round:.2f}, VE_ILD={ve_ild_round:.2f}", fontsize=10)
        # else:
        #     ax.set_title(f"{unit['type']} P{unit['IdxP']} Chan{unit['idx_chan']}\nUnit {col+1}\n{rate} pps\n"
        #                  f"VE_ITD={ve_itd_round:.2f}, VE_ILD={ve_ild_round:.2f}", fontsize=10)
        
        # Set subplot title with unit info, rate, and VE values
        if row == 0:
            ax.set_title(f"{unit_type} Unit {col+1}\n{rate} pps\n"
                         f"VE_ITD={ve_itd_round:.2f}\n VE_ILD={ve_ild_round:.2f}", fontsize=10)
        else:
            ax.set_title(f"{rate} pps\n"
                         f"VE_ITD={ve_itd_round:.2f}\n VE_ILD={ve_ild_round:.2f}", fontsize=10)
            
        # Labels
        if col == 0:
            ax.set_ylabel('ILD (dB)', fontsize=12)
        if row == 2:
            ax.set_xlabel('ITD (μs)', fontsize=12)

        # Colorbar
        cbar = fig.colorbar(c, ax=ax)
        if col == 2:
            cbar.set_label('AMUA', fontsize=12)

# Finalize heatmap figure
plt.suptitle(f"AMUA Heatmaps for Selected Units Across Rates {'(Interpolated)' if use_interpolation else '(Original)'}", y=0.98, fontsize=16)
plt.tight_layout(pad=1.0)

# Save heatmap figure
figname_heat = f"Figure4_ExampleUnit_SelectedUnits_Heatmaps{'_interpolated' if use_interpolation else '_original'}"
plt.savefig(os.path.join(FigurePath, f'{figname_heat}Unitset{0 if use_predefined else unit_sets_idx}_v1.png'), dpi=300)
plt.show()
# plt.close('all')

# Save unit data to JSON
def convert_numpy_types(data):
    if isinstance(data, np.integer):
        return int(data)
    elif isinstance(data, np.floating):
        return float(data)
    elif isinstance(data, np.ndarray):
        return data.tolist()
    elif isinstance(data, list):
        return [convert_numpy_types(item) for item in data]
    elif isinstance(data, dict):
        return {key: convert_numpy_types(value) for key, value in data.items()}
    return data

unit_data_converted = convert_numpy_types(unit_data)
output_file = os.path.join(data_path, f"unit_data_combined{'_interpolated' if use_interpolation else '_original'}.json")
with open(output_file, 'w') as f:
    json.dump(unit_data_converted, f, indent=4)
print(f"Saved unit data to {output_file}")









