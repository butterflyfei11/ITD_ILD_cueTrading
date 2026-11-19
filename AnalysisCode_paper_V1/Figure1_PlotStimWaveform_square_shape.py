#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 12 12:12:41 2025

@author: feipeng
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 21 16:57:30 2023

@author: colliculus
"""
from sys import platform
import numpy as np
import os 
import sys
from matplotlib import pyplot as plt
from scipy.signal import resample
from scipy.signal import find_peaks

if platform == "darwin":
    libPath = '/Volumes/Disk/ratCageProgramsV3/'
    sys.path.append(libPath)
elif platform == "linux":
    libPath = '/home/colliculus/behaviourBoxes/software/ratCageProgramsV2/'
    sys.path.append(libPath)
if not os.path.exists(libPath):
    libPath = '/home/colliculus/behaviourBoxes/software/ratCageProgramsV3'
    sys.path.append(libPath)    
import RZ2ephys as ep

# Load stimulus waveform    
if platform == "darwin":
    stimPath = '/Volumes/Disk/4_ITD_ILD_cueTrading/stimData/'
    Figure_Path = '/Volumes/Disk/4_ITD_ILD_cueTrading/Figures_paper/'
elif platform == "linux":
    stimPath = '/ephys/5/ITD_ILD_Cue_Trading/Data_Analysis/stimData/'
    Figure_Path = '/ephys/5/ITD_ILD_Cue_Trading/Data_Analysis/Figures/' 
    
StimulusData = np.load(stimPath+'Stim_cueTrading_template.npy')
Fs0 = 24414
new_Fs = 48828 * 4  # Quadrupled sampling rate
ii = 0
stimRate = [1, 100, 900]
stimILD = [-4, -1, 0, 1, 4]
stimITD = np.arange(-0.12, 0.14, 0.04)
stimITD = np.round(stimITD, 2)
rate = stimRate[ii]

#%%
plt.figure(figsize=(8, 6))

for ii in range(3):
    for jj in range(2):
        if jj == 0:
            idx_ITD = 0
            idx_ILD = 0
        else:
            idx_ITD = 6
            idx_ILD = 0
       
        stimWave = StimulusData[ii, idx_ITD, idx_ILD, :, :]
        temp_padding = np.zeros((1, 2), dtype='float32')
        stimWave = np.concatenate((temp_padding, stimWave))
        
        if ii == 0:
            # Resample to new sampling rate
            num_samples = int(stimWave.shape[0] * new_Fs / Fs0)
            stimWave_resampled = np.zeros((num_samples, 2), dtype='float32')
            stimWave_resampled[:, 0] = resample(stimWave[:, 0], num_samples)
            stimWave_resampled[:, 1] = resample(stimWave[:, 1], num_samples)
            
            # Add zero phase before the signal (40 µs)
            zero_duration = int(new_Fs * 0.00004)  # 40 µs zero phase
            zero_padding = np.zeros((zero_duration, 2), dtype='float32')
            stimWave_resampled = np.concatenate((zero_padding, stimWave_resampled))
            
            # Update number of samples for rect_wave
            total_samples = stimWave_resampled.shape[0]
            
            # Convert to rectangular waveform
            phase_width = int(new_Fs * 0.00004)  # 40 µs phase width
            rect_wave = np.zeros_like(stimWave_resampled)
            
            for ch in range(2):  # Process both channels (left, right)
                # Find positive and negative peaks with stricter threshold
                pos_peaks, _ = find_peaks(stimWave_resampled[:, ch], height=0.2 * np.max(np.abs(stimWave_resampled[:, ch])))
                neg_peaks, _ = find_peaks(-stimWave_resampled[:, ch], height=0.2 * np.max(np.abs(stimWave_resampled[:, ch])))
                
                for p_idx, pos in enumerate(pos_peaks):
                    # Find the next negative peak after this positive peak
                    next_neg_candidates = neg_peaks[neg_peaks > pos]
                    if len(next_neg_candidates) > 0:
                        next_neg = next_neg_candidates[0]
                        pos_amp = stimWave_resampled[pos, ch]
                        neg_amp = stimWave_resampled[next_neg, ch]
                    else:
                        continue  # Skip if no corresponding negative peak
                    
                    # Set positive phase centered on pos_peak (adjusted to fixed width)
                    pos_start = max(0, pos - phase_width // 2)
                    pos_end = pos_start + phase_width
                    rect_wave[pos_start:pos_end, ch] = pos_amp
                    
                    # Set negative phase immediately after positive phase
                    neg_start = pos_end
                    neg_end = neg_start + phase_width
                    if neg_end <= total_samples:
                        rect_wave[neg_start:neg_end, ch] = neg_amp
            
            # Explicitly zero out small artifacts below threshold
            threshold = 0.05 * np.max(np.abs(rect_wave))  # Adjust threshold as needed
            rect_wave[np.abs(rect_wave) < threshold] = 0
        
        plt.subplot(3, 2, ii * 2 + jj + 1)
        plt.subplots_adjust(left=0.1, bottom=0.1, right=0.92, top=0.9, wspace=0.2, hspace=0.5)
        if ii == 0:
            xs = np.arange(0, rect_wave.shape[0] / new_Fs, 1 / new_Fs)
            xs = xs * 1000  # Convert to ms
            plt.plot(xs, rect_wave[:, 0], linewidth=1.2, color='b')
            plt.plot(xs, rect_wave[:, 1], linewidth=1, color='r')
        else:
            xs = np.arange(0, stimWave.shape[0] / Fs0, 1 / Fs0)
            xs = xs * 1000  # Convert to ms
            plt.plot(xs, stimWave[:, 0], linewidth=2, color='b')
            plt.plot(xs, stimWave[:, 1], linewidth=1, color='r')
        if ii == 0:
            plt.xlim([0, 2])
            plt.legend(['Left ear', 'Right ear'], loc='upper right')
        elif ii == 1:
            plt.xlim([0, 25])
        else:
            plt.xlim([0, 25])
        if ii == 0:
            plt.title('ITD at ' + str(stimITD[idx_ITD]*1000) + ' µs ILD at ' + str(stimILD[idx_ILD]) + ' dB\n' +str(stimRate[ii]) + ' pps' )
        else:
            plt.title(str(stimRate[ii]) + ' pps')
        if ii == 2:
            plt.xlabel('Time (ms)')
        if jj == 0:
            plt.ylabel('Current level (µA)')
 
#%%       
plt.savefig(Figure_Path + 'Figure1_stimWaveform_square.png',dpi=300)
# plt.savefig(Figure_Path + 'Figure1_stimWaveform_square.svg')