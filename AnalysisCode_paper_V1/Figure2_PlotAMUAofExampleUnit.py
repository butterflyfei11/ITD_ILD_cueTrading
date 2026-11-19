#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 10 10:48:02 2023
Plot the AMUA of example unit 

@author: fei
"""

from sys import platform
import numpy as np
import numpy.matlib
import os 
import sys
from matplotlib import pyplot as plt
from numpy.lib.stride_tricks import as_strided
from scipy.signal import butter, filtfilt, iirnotch, resample_poly, csd, windows, welch, get_window, resample
import scipy
from scipy import signal
from scipy.io import savemat
import glob
from scipy import stats

if platform == "darwin":
    # libPath='/Users/fei/Documents/CI_projects/DataAnalysis/'
    libPath = '/Volumes/Disk/ratCageProgramsV3/'

    sys.path.append(libPath)
elif platform =="linux":
    libPath='/home/shiyi/PycharmProjects/ENVvsPS/'
    sys.path.append(libPath)
import RZ2ephys as ep


# give signal path
if platform == "darwin":  
    # Sig_path = '/Users/fei/Documents/CI_projects/4_ITD_ILD_cueTrading/AMUA/'
    # results_path = '/Users/fei/Documents/CI_projects/4_ITD_ILD_cueTrading/AMUA_analysis/'
    Sig_path = '/Volumes/Disk/4_ITD_ILD_cueTrading/Data/AMUA/'
    results_path = '/Volumes/Disk/4_ITD_ILD_cueTrading/Data/AMUA_analysis_v2/'
elif platform == "linux":
    Sig_path = '/disks/CIdata/2023_02_23/'
    results_path = '/disks/CIdata/2023_02_23Results/'
    fname = '20230223_P3_ITD_ILD_cue_trading_CI_1'
# load stimulus waveform    
if platform == "darwin":
    # stimPath = '/Users/fei/Documents/CI_projects/4_ITD_ILD_cueTrading/stimData/'
    stimPath = '/Volumes/Disk/4_ITD_ILD_cueTrading/stimData/'
elif platform == "linux":
    stimPath = '/disks/CIdata/2_ITD_ILD_cueTrading/stimData/'
StimulusData = np.load(stimPath+'Stim_cueTrading_template.npy')

# FigurePath = '/Users/fei/Documents/CI_projects/4_ITD_ILD_cueTrading/Figures/'
FigurePath = '/Volumes/Disk/4_ITD_ILD_cueTrading/Figures_paper/'


def smooth(signal,WSZ):
    # signal: NumPy 1-D array containing the data to be smoothed
    # WSZ: smoothing window size needs, which must be odd number,
    # as in the original MATLAB implementation
    out0 = np.convolve(signal,np.ones(WSZ,dtype=int),'valid')/WSZ    
    r = np.arange(1,WSZ-1,2)
    start = np.cumsum(signal[:WSZ-1])[::2]/r
    stop = (np.cumsum(signal[:-WSZ:-1])[::2]/r)[::-1]
    return np.concatenate((  start , out0, stop  ))

def trace_without_bad_rep(data_3d, max_sample_idx=300):
    """
    Remove repetitions whose maximum occurs too late.

    Parameters
    ----------
    data_3d : ndarray, shape (n_rep, n_time)
        All repetitions for a single stimulus condition.
    max_sample_idx : int, default 300
        Repetitions with peak at index >= max_sample_idx are discarded.

    Returns
    -------
    clean_mean : ndarray, shape (n_time,)
        Average of repetitions with early peaks.
    """
    if data_3d.ndim != 2:
        raise ValueError("data_3d must be 2D: (n_repetitions, n_time)")

    n_rep = data_3d.shape[0]
    if n_rep == 0:
        return np.zeros(data_3d.shape[1])
    if n_rep == 1:
        return data_3d[0, :]

    # Find index of maximum in each repetition
    peak_idx = np.argmax(data_3d, axis=1)          # shape: (n_rep,)

    # Keep only repetitions where peak is in first `max_sample_idx` samples
    # good_rep = peak_idx < max_sample_idx
    good_rep = (peak_idx < max_sample_idx) | (peak_idx > 600)
    good_idx = np.where(good_rep)[0]

    # Safety: if all are bad, keep the one with the *earliest* peak
    if len(good_idx) == 0:
        earliest_peak_rep = np.argmin(peak_idx)
        print(f"Warning: All reps have late peaks. Keeping rep {earliest_peak_rep} (peak at sample {peak_idx[earliest_peak_rep]})")
        good_idx = np.array([earliest_peak_rep])

    return data_3d[good_idx, :].mean(axis=0)


Index_responsive = np.load(results_path+'allP_responsive.npy')
filelist = os.listdir(Sig_path)
filelist.sort()

# Penetration = ['20230216_P04','20230221_P7','20230223_P4','20230303_P5','20230320_P2']
# IdxP = [2,6,10,15,30]

# 3 example units from Figure 4 (heatmap)
# ITD dominate unit 
IdxP = 9 
cc = 14
# ILD dominate unit
# IdxP = 17 
# cc = 1
# 
# tuned to both ITD and ILD 
# IdxP = 5
# cc=1

fname = filelist[IdxP]
print(fname)
Stats_significantCounts = np.load(results_path+fname[:-8]+'Stats_perSti_ChanIndex.npy')   # it saves the count of stimulus parame
UnitIndex2 = np.where(Stats_significantCounts>1)
UnitIndex = UnitIndex2[0]
print(UnitIndex)


Sig_AMUA = np.load(Sig_path+fname)
Sig_avg = np.mean(Sig_AMUA,4)
#%% want to plot the AMUA of example unit
#%% plot AMUA at each ITD and ILD stimulus parameter (group ILD and plot ITD separately)
stimRate = [1,100,900]
# stimILD = [4,1,0,-1,-4]
# stimILD = np.array(stimILD)
stimILD = [-4,-1,0,1,4]

stimITD = np.arange(-0.12,0.14,0.04)
stimITD = np.round(stimITD,2)
Fs_ds = 2000

numITD = len(stimITD)
numILD = len(stimILD)

#%


figname = 'Fig2_ExampleUnitAMUA'+fname[:-4]+'Chan'+str(cc)
plt.figure(figsize=(15,6))
for ii in range(len(stimRate)):
    
    if ii == 0:
        sig_temp = np.mean(Sig_AMUA[cc,ii,:,:,:,:int(Fs_ds*0.13)],2)
    else:
        sig_temp = np.mean(Sig_AMUA[cc,ii,:,:,:,:],2)
        
   
    # sig_temp = np.mean(Sig_AMUA[cc,ii,:,:,:,:],2)        
    sig_temp_max = np.max(sig_temp)
    sig_temp_min = np.min(sig_temp)
    #%
    
    
    for kk in range(len(stimITD)):
        for jj in range(len(stimILD)):
           
           # Res_temp = np.mean(Sig_AMUA[cc,ii,kk,jj,:,:],0)
           raw_reps = Sig_AMUA[cc, ii, kk, jj, :, :]                 # shape (n_rep, n_time)
           Res_temp = trace_without_bad_rep(raw_reps)
                           
           # Res_temp = np.mean(Sig_AMUA[cc,2,kk,jj,:,:],0)
           
           Res_temp_smooth = smooth(Res_temp,5)
           # window_size = int(0.01*Fs_ds)
           # moving_average = []
           # xx = 0
           # while xx<len(Res_temp) - window_size + 1:
           #     window_average = round(np.sum(Res_temp[xx:xx+window_size])/window_size,2)
               
           #     moving_average.append(window_average)
           #     xx+=1
           
           # Res_temp_movingAvg = np.array(moving_average)
           
           
           
           xs = np.arange(0,len(Res_temp)/Fs_ds,1/Fs_ds)
           xs = xs - 0.1
           
               
           idx_plots = ii*numITD+kk+1+ii
           # print(idx_plots)
           plt.subplot(3,numITD+1,idx_plots)
           plt.subplots_adjust(left=0.09,
               bottom=0.1,
               right=0.95,
               top=0.95,
               wspace=0.15,
               hspace=0.3)
           # plt.plot(xs,Res_temp)
           # plt.plot(xs,Res_temp+(jj-1)*(sig_temp_max)*0.5)

           plt.plot(xs,Res_temp_smooth+(jj-1)*(sig_temp_max)*0.2)
           if ii > 0:
               plt.axvspan(0,0.2,0,0.03,color='k')
           else:
               plt.axvspan(0,0.005,0,0.03,color = 'k')
           # if ii == 0 and kk == 6:
               
           #     plt.legend(['ILD of 4 dB','ILD of 1 dB','ILD of 0 dB','ILD of -1 dB','ILD of -4 dB'],bbox_to_anchor=(0.7,0.4))
           if ii == 0 and kk == 6:
               # plt.text(0.1,(jj)*(sig_temp_max)*0.2+0.5*np.min(Res_temp_smooth),'ILD  '+str(stimILD[jj])+' dB',fontSize = 10)
               plt.text(0.1,(jj)*(sig_temp_max)*0.2+0.5*np.min(Res_temp_smooth),'ILD  '+str(stimILD[jj])+' dB')

               
           if kk == 0 and jj == 3:
               # plt.text(-0.5,(jj)*(sig_temp_max)*0.2,'Rate '+str(stimRate[ii]) + 'pps',fontSize = 10 )
               plt.text(-0.35,(jj)*(sig_temp_max)*0.2,str(stimRate[ii]) + ' pps')

            
           plt.xlim([-0.1,0.38])
           if jj == len(stimILD)-1:
               plt.axvline(x=0,color='k',linestyle=':')
           
           
           if ii == 0 and kk == 1:
               # plt.title(fname[:-4]+'_C'+str(cc)+'\n'+'ITD at ' + str(stimITD[kk]*1000)+ ' μs')
               plt.title('ITD at ' + str(stimITD[kk]*1000)+ ' μs')

           elif ii == 0 :
               plt.title('ITD at ' + str(stimITD[kk]*1000)+ ' μs')
           # show x-ticks **only** on the bottom row (last rate of last unit)
          
           if ii < 2:
               plt.xticks([])
           
           plt.yticks([])
           # plt.xlabel('Time (s)')
           if ii == 2:
               plt.xlabel('Time (s)')
           if kk == 0:
               plt.ylabel('AMUA')
        plt.ylim([-0.4*sig_temp_min,1.4*sig_temp_max]) 
        # plt.xlim([-0.09,0.29])
        
plt.suptitle(fname[:-4]+'_C'+str(cc))
plt.tight_layout()
#%  

plt.savefig(FigurePath+figname+'v2.png',dpi=300)
# plt.savefig(FigurePath+figname+'v2.svg')
# #%% color map of the AMUA 

# nrates = len(stimRate)
# sig_norm_range = np.zeros((nrates,2),dtype='float32')
# # sig_norm_value = np.zeros((nrates,nchans,nITD,nILD),dtype='float32')
# #%

# xs = np.arange(0,Sig_AMUA.shape[5]/Fs_ds,1/Fs_ds)   # time axis in second
# xs = xs*1000    # change the time axis to ms
# xs = xs - 100   # subtracting the baseline 
# idx_heatMap = [8,16,24]


# for rr in range(len(stimRate)):
#     if rr == 0 or rr == 2:
#         idx = np.where((xs>3) & (xs<30))
#         idx_sig = idx[0]
#     else:
#         idx = np.where((xs>3) & (xs<200))
#         ixs_sig = idx[0]

#     idx_Rate = rr
        
#     sig_temp = Sig_AMUA[cc,idx_Rate,:,:,:,idx_sig[0]:idx_sig[-1]]
#     sig_temp_avg = np.mean(sig_temp,3)
#     sig_temp_mean = np.mean(sig_temp_avg,2)
    
#     sig_temp_mean_norm = stats.zscore(sig_temp_mean)
#     # sig_norm_value[rr,cc,:,:] = sig_temp_mean_norm
#     sig_temp_mean_norm_min = np.min(sig_temp_mean_norm)
#     sig_temp_mean_norm_max = np.max(sig_temp_mean_norm)
#     sig_norm_range[rr,0] = sig_temp_mean_norm_min
#     sig_norm_range[rr,1] = sig_temp_mean_norm_max
    
# for rr in range(nrates):
#     idx_Rate = rr
#     sig_temp = Sig_AMUA[cc,idx_Rate,:,:,:,idx_sig[0]:idx_sig[-1]]
#     sig_temp_avg = np.mean(sig_temp,3)
#     sig_temp_mean = np.mean(sig_temp_avg,2)
    
#     sig_temp_mean_norm = stats.zscore(sig_temp_mean)
#     sig_temp_mean_norm_min = np.min(sig_temp_mean_norm)
#     sig_temp_mean_norm_max = np.max(sig_temp_mean_norm)
    
#     plt.subplot(3,8,idx_heatMap[rr])
       
#     plt.subplots_adjust(left=0.08,
#         bottom=0.1,
#         right=0.95,
#         top=0.9,
#         wspace=0.35,
#         hspace=0.3)
#     x_value = np.arange(1,9,1)
#     xaxis = np.repeat(x_value,6)
#     x_plot = xaxis.reshape(8,6)
    
#     y_value = np.arange(1,7,1)
#     yaxis = np.repeat(y_value,8)
#     y_plot = yaxis.reshape(6,8)
#     temp_dat = sig_temp_mean_norm
#     xlabels_heatmap = [' ','-.12','-.08','-.04','0','.04','.08','.12']
#     ylabels_heatmap = [' ','-4','-1','0','1','4']
    
    
#     c = plt.pcolormesh(x_plot.T,y_plot,temp_dat.T,cmap='plasma',vmin=min(sig_norm_range[:,0]), vmax = max(sig_norm_range[:,1]))
    
#     # c = plt.pcolormesh(x_plot.T,y_plot,temp_dat.T,cmap='plasma',vmin=sig_temp_mean_norm_min, vmax = sig_temp_mean_norm_max)

#     plt.yticks(y_value,ylabels_heatmap)
    
#     plt.xticks(x_value,xlabels_heatmap,rotation=45)
    
   
    
#     plt.ylabel('ILD (dB)')
    
#     if rr == 2:
#         cbar = plt.colorbar(c)
#         cbar.set_label('z-score of AMUA')
#         plt.xlabel('ITD (ms)')
    
#     # plt.figure(figsize=(15,10))
   
    
# #%% group ITD and then plot ILD
# plt.figure(figsize=(12,10))

 
# for ii in range(len(stimRate)):
#     sig_temp = np.mean(Sig_AMUA[cc,ii,:,:,:,:],2)
#     sig_temp_max = np.max(sig_temp)
#     sig_temp_min = np.min(sig_temp)
#     #%
   
    
    
#     for jj in range(len(stimILD)):
#         for kk in range(len(stimITD)):
#            Res_temp = np.mean(Sig_AMUA[cc,ii,kk,jj,:,:],0)
           
#            Res_temp_smooth = smooth(Res_temp,5)
#            # window_size = int(0.01*Fs_ds)
#            # moving_average = []
#            # xx = 0
#            # while xx<len(Res_temp) - window_size + 1:
#            #     window_average = round(np.sum(Res_temp[xx:xx+window_size])/window_size,2)
               
#            #     moving_average.append(window_average)
#            #     xx+=1
           
#            # Res_temp_movingAvg = np.array(moving_average)
           
           
           
#            xs = np.arange(0,len(Res_temp)/Fs_ds,1/Fs_ds)
#            xs = xs - 0.1
           
#            if ii == 0:
#                idx_plots = jj+1
#            elif ii ==1:
#                idx_plots = ii*numILD+jj+1+1
#            elif ii == 2:
#                idx_plots = ii*numILD+jj+1+2
           
#            # print(idx_plots)
#            plt.subplot(3,numILD+1,idx_plots)
#            plt.subplots_adjust(left=0.05,
#                bottom=0.1,
#                right=0.95,
#                top=0.95,
#                wspace=0.15,
#                hspace=0.3)
#            # plt.plot(xs,Res_temp)
#            # plt.plot(xs,Res_temp+(jj-1)*(sig_temp_max)*0.5)

#            plt.plot(xs,Res_temp_smooth+(kk-1)*(sig_temp_max)*0.2)
#            # if ii == 0 and kk == 6:
               
#            #     plt.legend(['ILD of 4 dB','ILD of 1 dB','ILD of 0 dB','ILD of -1 dB','ILD of -4 dB'],bbox_to_anchor=(0.7,0.4))
#            if ii == 0 and jj == len(stimILD)-1:
#                plt.text(0.1,(kk)*(sig_temp_max)*0.2,'ITD of '+str(stimITD[kk])+' ms',fontSize = 10)
#            plt.xlim([-0.1,0.25])
#            if kk == len(stimITD)-1:
#                plt.axvline(x=0,color='k',linestyle=':')
#            # plt.ylim([sig_temp_min,sig_temp_max]) 
           
#            if ii == 0:
#                plt.title('ILD at' + ' '+ str(stimILD[jj]) +' dB')
           
                   
#            plt.yticks([])
#            if ii == 2:
#                plt.xlabel('Time (s)')


# #%%
# cc = 0
# for ii in range(len(stimRate)):
#     sig_temp = np.mean(Sig_AMUA[cc,ii,:,:,:,:],2)
#     sig_temp_max = np.max(sig_temp)
#     sig_temp_min = np.min(sig_temp)
#     #%
#     plt.figure(figsize=(20,12))
    
#     for jj in range(len(stimILD)):  # be careful here the jj represents ILD
#         #%
#         for kk in range(len(stimITD)):  # kk represents ITD
#             Res_temp = np.mean(Sig_AMUA[cc,ii,kk,jj,:,:],0)
#             xs = np.arange(0,len(Res_temp)/Fs_ds,1/Fs_ds)
#             xs = xs - 0.1
#             idx_plots = jj*7+kk
#             # print(idx_plots)
#             plt.subplot(5,7,idx_plots+1)
#             plt.subplots_adjust(left=0.05,
#                 bottom=0.05,
#                 right=0.95,
#                 top=0.95,
#                 wspace=0.15,
#                 hspace=0.4)
#             plt.plot(xs,Res_temp)
#             plt.xlim([-0.1,0.25])
#             plt.axvline(x=0,color='r',linestyle=':')
#             plt.ylim([sig_temp_min,sig_temp_max])
#             if jj == len(stimILD)-1:
#                 plt.xlabel('Time (s)')
#             ax = plt.gca()
#             if jj < len(stimILD)-1:
#                 ax.axes.xaxis.set_ticklabels([])
#             if jj ==0 and kk ==0:
#                 plt.title('Chan'+str(cc) + 'rate at' + str(stimRate[ii])+'\n'+'ILD at' + str(stimILD[jj])+ ' ITD at' + str(stimITD[kk]))
#             else:
#                 plt.title('ILD at' + str(stimILD[jj])+ ' ITD at' + str(stimITD[kk]))
#             if jj!=0 and kk>0:
#                 ax.axes.yaxis.set_ticklabels([])
#     plt.savefig(results_path+fname+'Chan'+str(cc)+'Rate'+str(stimRate[ii])+'.png')
#     # plt.close('all')
                    