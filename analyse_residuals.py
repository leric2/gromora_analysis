#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import datetime
import os
import matplotlib

import matplotlib.pyplot as plt
import netCDF4
import numpy as np
import pandas as pd
import scipy.io

from scipy import signal

import xarray as xr
import matplotlib.ticker as ticker

from matplotlib.ticker import (MultipleLocator, FormatStrFormatter, AutoMinorLocator)

from typhon.collocations import Collocator
# from level2_gromora import * 
#%matplotlib widget
import warnings
warnings.filterwarnings('ignore')
#from scipy.fft import fft, fftfreq

colormap = 'cividis'

# color_gromos = '#d95f02'
# color_somora = '#1b9e77'
MAX_PERIOD = 40

def compute_sine_baselines(da):
    N = len(da.f)
    T = np.median(np.diff(da.f))
    #fft_freq = fftfreq(N, T)[:N//2]
    freqs = np.fft.rfftfreq(N, T)
    idx = np.argsort(freqs)
    power_spectra = np.zeros((len(da.time),len(freqs)))
    for i in range(len(da.time)):
        #ps = np.abs(np.fft.rfft(da.isel(time=i).data))**2
        f, Pxx_spec = signal.periodogram(da.isel(time=i).data, window='flattop', scaling='spectrum')
        power_spectra[i,:] = Pxx_spec

    
    mean_ps = np.mean(power_spectra, axis=0)
    avgN = 100
    plt.plot(freqs[idx], mean_ps[idx])
    plt.plot(np.convolve(freqs[idx], np.ones(avgN)/avgN, mode='valid'), np.convolve(mean_ps[idx], np.ones(avgN)/avgN, mode='valid'))
    plt.ylim(0,0.005)
    #plt.ylim(0,100000)
    plt.xlim(1.5e-5,1.8e-5)


    # plt.plot(np.mean(np.square(np.abs(np.fft.rfft(ds.y.dropna(dim='f').data-ds.yf.dropna(dim='f').data))),axis=0))
    # plt.ylim(0,8000)
    #plt.plot(np.mean(da.data,axis=0))

    # plt.plot(np.square(np.abs(np.fft.rfft(np.mean(ds.y.dropna(dim='f').data-ds.yf.dropna(dim='f').data,axis=0)))))
    # plt.ylim(0,1000)
    # for i in range(len(ds.time)):
    #     y = ds.isel(time=i).y.dropna(dim='f').data
    #     yf = ds.isel(time=i).yf.dropna(dim='f').data
    #     res = y - yf
    #     
    #     #print('mean res', np.mean(res))

#compute_sine_baselines(level2_dataset['AC240'])


def compute_baselines(interpolated_residuals, display = True, quantile=.5):
    

    res = interpolated_residuals.data# .mean(dim='time').data

    N_channels = len(interpolated_residuals.f)
    freq = interpolated_residuals.f.data
    span = (interpolated_residuals.f[-1]  - interpolated_residuals.f[0]).data

    fft_channels = 8 * N_channels

    fft_periods = np.arange(0, MAX_PERIOD, 1/8)
    
    if  res.shape[0] != N_channels:
        fft_spectrum = np.zeros((len(interpolated_residuals.time), fft_channels))
        # for res in interpolated_residuals.data:
        for i, resi in enumerate(interpolated_residuals):
            fft_spectrum[i,:] = np.fft.fft( resi - np.mean(resi),  fft_channels)#[i,:] - np.mean(resi)
        res = np.median(res,0)
        fft_spectrum = np.mean(fft_spectrum,0)
    else:
         fft_spectrum = np.fft.fft( res  - np.mean(res),  fft_channels)
    # fft_spectrum = np.mean(fft_spectrum, axis=0)
    fft_spectrum  = fft_spectrum[0:len(fft_periods)] 
    # plt.plot(fft_periods, np.abs(fft_spectrum)**2)

    index, properties = scipy.signal.find_peaks(np.abs(fft_spectrum)**2, prominence=(10000,None))
    prominence = scipy.signal.peak_prominences(np.abs(fft_spectrum)**2, index)[0]

    index = index[prominence>np.quantile(prominence,quantile)] 
    prominence = prominence[prominence>np.quantile(prominence, quantile)] 
    

    sorted_ind = np.flip(np.argsort(prominence))

    sorted_index = index[sorted_ind]
    scale = np.std(res.data)*np.sqrt(2) / np.sum(np.abs(fft_spectrum[sorted_index])) 

    baseline = dict()
    # Frequenz der Baseline in [1/bandwidth] 
    baseline['periods'] = fft_periods[sorted_index]   # Vektor der Baseline Perioden [1/BW]
    # baseline_periodes = np.interp(np.arange(1,len(fft_periods)), fft_periods, center)                       
    # Vektor der Baseline Perioden [1/BW]
    baseline['frequencies']  =span/baseline['periods']   # Vektor der Baseline Perioden [Hz]
    baseline['amplitude']   = fft_spectrum[sorted_index] * scale # Vektor der Baseline Amplituden

    # Umrechnen in Entfernung [m]
    baseline['distance']  = 2.998e8/ baseline['frequencies']  / 2

    #  Fit mit 1:N Cosine Funktionen
    cosine = np.zeros((len(baseline['frequencies'] ), len(freq)))
    for i, bl_f in enumerate(baseline['frequencies'] ):
        cosine[i,:]   = np.cos( 2*np.pi * (freq-freq[0]) * (1/ bl_f) )+ 1j * np.sin(2*np.pi * (freq-freq[0]) *(1/ bl_f) )

    baseline['fit'] = np.real( np.multiply(cosine.T, baseline['amplitude']) )
    baseline['fft']     = [fft_periods, fft_spectrum]
   # baseline['fft'][0,:]  = np.nan

    # sorting_ind = np.flip(np.argsort(np.abs(baseline['amplitude'])))
    # sorting_ind = np.flip(np.sort_complex(baseline['amplitude']))
    # baseline['sorted_amplitudes'] = baseline['amplitude'][sorting_ind] 
    # baseline['sorted_frequencies'] = baseline['frequencies'][sorting_ind]
    # baseline['sorted_periods'] = baseline['periods'][sorting_ind] 
    # baseline['sorted_fit'] = baseline['fit'][:,sorting_ind] 

    if display:
        fig, axs = plt.subplots(2, 1, figsize=(15,10))
        axs[0].plot(fft_periods, np.abs(fft_spectrum)*scale)
        axs[0].plot(baseline['periods'], np.abs(baseline['amplitude']), 'x')

        axs[0].set_xlabel('FFT Period')
        axs[0].set_ylabel('FFT Amplitude')
        axs[0].set_title(pd.to_datetime(interpolated_residuals.time.data[0]).strftime('%Y-%m-%d'))

        axs[1].plot(freq,res, label='median of residuals')
        highest_amplitude = np.argwhere(np.abs(baseline['amplitude']) == max(np.abs(baseline['amplitude'])))[0][0] 
        # main_freq = baseline['fit'][:,highest_amplitude]
        axs[1].plot(freq, baseline['fit'][:,0] , label='first amplitude')
        axs[1].plot(freq, baseline['fit'][:,1] , label='second amplitude')

        axs[1].plot(freq, np.sum(baseline['fit'], axis=1), label='Sum of the fit')

        
        axs[1].set_ylabel('Residuals')
        axs[1].set_xlabel('Frequency')
        axs[1].legend() 
       
        
    return baseline


def read_residuals_all(basefolder, instrument_name, date_slice, years, prefix):
    counter = 0
    for i, y in enumerate(years):
        if isinstance(prefix, str):
            filename = basefolder+instrument_name+'_'+str(y)+'_12_31'+ prefix
        else:
            filename = basefolder+instrument_name+'_'+str(y)+'_12_31'+ prefix[i]
        gromora = xr.open_dataset(
            filename,
            decode_times=True,
            decode_coords=True,
            # use_cftime=True,
        )
        if counter == 0:
            gromora_ds=gromora
        else:
            gromora_ds=xr.concat([gromora_ds, gromora], dim='time')
        
        counter=counter+1
        print('Read : '+filename)

    pandas_time_gromos = pd.to_datetime(gromora_ds.time.data)

    gromora_ds = gromora_ds.sel(time=date_slice)
    return gromora_ds

def compute_all_baselines(instrument_name, interpolated_residuals, ind_v, outfolder, amp_thresh=0.05):
        
    bl_freq = np.zeros((len(ind_v), 8))
    bl_amp = np.zeros((len(ind_v), 8))
    time_vector = list()
    for c,i in enumerate(ind_v):
        # residuals = interpolated_residuals.sel(time=date_slice, drop = True) 
        residuals = interpolated_residuals.isel(time=i, drop = True) 
        baselines = compute_baselines(residuals, display=False, quantile=0.1) #.mean(dim='time')  

        # print('Main baseline frequencies identified:')
        # for j in range(2):
        #     f = baselines['frequencies'][i]/1e6
        #     amp = np.abs(baselines['amplitude'][i])
        #     per = baselines['periods'][i]
            # print(f'f = {f:.1f}, with amplitude = {amp:.3f} and periods = {per:.3f}')
        if len(baselines['frequencies'])>7:
            bl_freq[c,:] = baselines['frequencies'][0:8]/1e6
            bl_amp[c,:] = np.abs(baselines['amplitude'][0:8])
            time_vector.append(pd.to_datetime(interpolated_residuals.time[i].data))
        else:
            bl_freq[c,:] = np.ones(8)*np.nan
            bl_amp[c,:] = np.ones(8)*np.nan
            time_vector.append(pd.to_datetime(interpolated_residuals.time[i].data)) 

    baselines_frequencies = xr.Dataset(
        data_vars=dict(
            bl_frequencies = (['time', 'baseline'], np.array(bl_freq)),
            bl_amplitude = (['time', 'baseline'], np.array(bl_amp))
        ),
        coords=dict(
            time=time_vector,
            baseline=np.arange(0,8)
        )
    )

    datetime_vec = pd.to_datetime(baselines_frequencies.time.data)
    fig, axs = plt.subplots(1, 1, figsize=(12,10))
    for m in range(8):
        axs.plot(datetime_vec, baselines_frequencies.bl_frequencies[:,m].where(baselines_frequencies.bl_amplitude[:,m] >amp_thresh), '.', color='r')
        axs.plot(datetime_vec, baselines_frequencies.bl_frequencies[:,m].where(baselines_frequencies.bl_amplitude[:,m] <amp_thresh), '.', color='k')

    axs.set_ylabel('Baseline periods [MHz]')
    fig.savefig(outfolder + instrument_name + '_baseline_periods_' + datetime_vec[0].strftime('%Y')+'.pdf')

    return baselines_frequencies

def read_all_baseline(instrument, outfolder, date_slice):
    amp_thresh=0.1
    baseline = xr.open_dataset(
            outfolder+instrument+'_fitted_baselines_res.nc',
            decode_times=True,
            decode_coords=True,
            # use_cftime=True,
        )

    baseline = baseline.where(baseline.bl_amplitude > 0.05, drop=True).sel(time=date_slice)
    datetime_vec = pd.to_datetime(baseline.time.data)

    fig, axs = plt.subplots(1, 1, figsize=(12,10))
    for m in range(8):
        axs.plot(datetime_vec, baseline.bl_frequencies[:,m].where(baseline.bl_amplitude[:,m] > amp_thresh), '.', color='r')
        axs.plot(datetime_vec, baseline.bl_frequencies[:,m].where(baseline.bl_amplitude[:,m] < amp_thresh), '.', color='k')
    axs.set_ylim(0,1000)
    axs.set_ylabel('Baseline periods [MHz]')
    fig.savefig(outfolder + instrument + '_baseline_periods_all.pdf')

    return baseline


if __name__ == "__main__":

    yr = 2009
    date_slice=slice(str(yr)+'-01-01',str(yr)+'-12-31')
    date_slice=slice('2009-10-01','2009-12-31')
    outfolder = '/scratch/GROSOM/Level2/GROMOS_v3/'
    compute_new = True
    reprocessed = True

    instrument = 'GROMOS'
    if instrument == 'GROMOS':
        instNameGROMOS = 'GROMOS'
        pref = 'GROMOS_'+str(yr)+'_12_31'
        folder =  '/storage/tub/instruments/gromos/level2/GROMORA/v3/'
        # folder =  '/scratch/GROSOM/Level2/GROMORA_waccm/'
        freq_basename = '/scratch/GROSOM/Level1/frequency_grid_GROMOS.nc'

        prefix= '_v3_residuals.nc'
    elif instrument == 'SOMORA':
        pref = 'SOMORA_'+str(yr)+'_12_31'
        folder ='/storage/tub/instruments/somora/level2/v2/'
        # folder =  '/scratch/GROSOM/Level2/GROMORA_waccm/'
        freq_basename = '/scratch/GROSOM/Level1/frequency_grid_SOMORA.nc'
        prefix= '_res_residuals.nc'
       
    basefreq_ds = xr.open_dataset(
            freq_basename,
        )

    #basefolder=_waccm_low_alt_dx10_v2_SB_ozone
    # prefix= '_waccm_low_alt_all.nc'
    if reprocessed:
        level2_dataset = xr.open_dataset(
                folder+pref+prefix,
                decode_times=True,
                decode_coords=True,
                # use_cftime=True,
            )
        level2_dataset_res = level2_dataset['res']#.get(['y', 'yf', 'y_baseline'] )
        #level2_dataset_res = level2_dataset['y'].data - level2_dataset['yf'].data
    else:
        level2_dataset_res = xr.open_dataarray(
                folder+pref+prefix,
                decode_times=True,
                decode_coords=True,
                # use_cftime=True,
            )

        level2_dataset_res = level2_dataset_res.sel(time=date_slice, drop = True) 


    # level2_dataset = read_GROMORA_all(basefolder=folder, 
    # instrument_name=instrument,
    # date_slice=date_slice, 
    # years=[yr],#[2011, 2012,2013,2014,2015,2016,2017,2018,2019,2020],#[2011,2012,2013,2014,2015,2016,2017,2018,2019,],#[yr],#[],#
    # prefix= prefix+'.nc'
    # )

    # level2_residuals = read_residuals_all(basefolder=folder, 
    # instrument_name=instrument,
    # date_slice=date_slice, 
    # years=[yr],#[2011, 2012,2013,2014,2015,2016,2017,2018,2019,2020],#[2011,2012,2013,2014,2015,2016,2017,2018,2019,],#[yr],#[],#
    # prefix= prefix+'residuals.nc'
    # )

    if compute_new:
        baselines_frequencies = xr.DataArray()
        basefreq = basefreq_ds.frequencies.values
        IF_base = basefreq_ds.frequencies.values - basefreq_ds.frequencies.values[0]
        interpolated_residuals = level2_dataset_res.interp(f=basefreq)

        interpolated_residuals = interpolated_residuals.interpolate_na(dim='f', fill_value=0)
        # (interpolated_residuals.y - interpolated_residuals.yf).interpolate_na(dim='f', fill_value=0)
        date_slice_sel=slice(str(yr)+'-01-01 00:00:00',str(yr)+'-12-31 01:00:00')

        residuals = interpolated_residuals.sel(time=date_slice_sel, drop = True)
        if len(residuals.time)>0:
            baselines = compute_baselines(residuals, display=True, quantile=0.2)

        print('Main baseline frequencies identified:')
        if len(baselines['periods'])>6:
            max_per = 6
        else:
            max_per = len(baselines['periods'])
        for i in range(max_per):
            f = baselines['frequencies'][i]/1e6
            amp = np.abs(baselines['amplitude'][i])
            per = baselines['periods'][i]
            print(f'f = {f:.1f}, with amplitude = {amp:.3f} and periods = {per:.3f}')

        ind_v = np.arange(0, len(interpolated_residuals.time), 1)
        baseline_ds = compute_all_baselines(instrument, interpolated_residuals, ind_v, outfolder,amp_thresh=0.1)
        baseline_ds.to_netcdf(outfolder+instrument+'_fitted_baseline_periods_'+str(yr)+'.nc')
            # new_residuals = residuals.data + np.sum(baselines['fit'],1)
            # residuals.data = new_residuals
            # baselines = compute_baselines(residuals, display=True, quantile=0.5)

            # print('Main baseline frequencies identified:')
            # for i in range(5):
            #     f = baselines['frequencies'][i]/1e6
            #     amp = np.abs(baselines['amplitude'][i])
            #     per = baselines['periods'][i]
            #     print(f'f = {f:.1f}, with amplitude = {amp:.3f} and periods = {per:.3f}')
    else:
        baseline = read_all_baseline(instrument, outfolder, date_slice = date_slice)