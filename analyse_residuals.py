#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import datetime
import os

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
from level2_gromora import *

#from scipy.fft import fft, fftfreq

colormap = 'cividis'

# color_gromos = '#d95f02'
# color_somora = '#1b9e77'

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

if __name__ == "__main__":

    yr = 2016
    date_slice=slice(str(yr)+'-01-01',str(yr)+'-12-31')
    date_slice=slice('2016-01-01','2016-01-31')

    instrument = 'GROMOS'
    if instrument == 'GROMOS':
        instNameGROMOS = 'GROMOS'
        folder =  '/storage/tub/instruments/gromos/level2/GROMORA/v1/'
        freq_basename = '/scratch/GROSOM/Level1/frequency_grid_GROMOS.nc'
    elif instrument == 'SOMORA':
        folder ='/storage/tub/instruments/somora/level2/v1/'
        freq_basename = '/scratch/GROSOM/Level1/frequency_grid_SOMORA.nc'

    basefreq_ds = xr.open_dataset(
            freq_basename,
        )

    #basefolder=_waccm_low_alt_dx10_v2_SB_ozone

    prefix= '_waccm_low_alt_dx10_winCorr_'
    
    level2_dataset = read_GROMORA_all(basefolder=folder, 
    instrument_name=instrument,
    date_slice=date_slice, 
    years=[yr],#[2011, 2012,2013,2014,2015,2016,2017,2018,2019,2020],#[2011,2012,2013,2014,2015,2016,2017,2018,2019,],#[yr],#[],#
    prefix= prefix+'ozone.nc'
    )

    level2_residuals = read_residuals_all(basefolder=folder, 
    instrument_name=instrument,
    date_slice=date_slice, 
    years=[yr],#[2011, 2012,2013,2014,2015,2016,2017,2018,2019,2020],#[2011,2012,2013,2014,2015,2016,2017,2018,2019,],#[yr],#[],#
    prefix= prefix+'residuals.nc'
    )

    basefreq = basefreq_ds.frequencies.values
    IF_base = basefreq_ds.frequencies.values - basefreq_ds.frequencies.values[0]
    interpolated_residuals = level2_residuals.interp(f=basefreq)

    interpolated_residuals = (interpolated_residuals.y - interpolated_residuals.yf).interpolate_na(dim='f', fill_value=0)

    compute_sine_baselines(interpolated_residuals)
    