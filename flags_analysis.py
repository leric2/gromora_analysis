#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Fri Apr 10 11:37:52 2020

@author: eric

Integration script for IAP instruments

"""
from abc import ABC
import os

import numpy as np
import xarray as xr
import pandas as pd
import netCDF4
import matplotlib.pyplot as plt

from matplotlib.ticker import (
    MultipleLocator, FormatStrFormatter, AutoMinorLocator)

from base_tool import *

instrument_name = "SOMORA"

def read_level1_flags(basefolder= '/home/esauvageat/Documents/GROMORA/Analysis/ReprocessingLogs/', instrument='GROMOS', year=2016, suffixe=''):
    basename = os.path.join(basefolder,instrument)
    fn_lvl1a = basename+'_level1a_flags_'+str(year)+suffixe+'.nc'
    fn_lvl1b = basename+'_level1b_flags_'+str(year)+suffixe+'.nc'
    flags_level1a = xr.open_dataset(
        fn_lvl1a,
        group='flags',
        decode_times=True,
        decode_coords=True,
        #use_cftime=True,
    )
    flags_level1b = xr.open_dataset(
        fn_lvl1b,
        group='flags',
        decode_times=True,
        decode_coords=True,
        #use_cftime=True,
    )

    print('read: ',fn_lvl1a )
    print('read: ',fn_lvl1b )
    return flags_level1a,flags_level1b

if __name__ == "__main__":
    date = pd.date_range(start='2017-01-01', end='2017-12-31')
    #date = datetime.date(2019,1,15)

    int_time = 1

    plot_TN = True
    df_bins = 200e3

    plot_flag_ts = False
    plot_flag_ts_lvl1b = False
    plot_sum_flags = True

    integration_strategy = 'classic'
    classic = np.arange(1, 24)

    cycle = 14

    basename = "/home/esauvageat/Documents/GROMORA/Analysis/ReprocessingLogs/"

    basename_o3 = "/storage/tub/instruments/gromos/level2/GROMORA/v1"


    flags_filename = 'GROMOS_level1a_flags_2016.nc'
    flags_lvl1b_filename = 'GROMOS_level1b_flags_2016.nc'

    ozone_filename = 'GROMOS_2016_12_31_waccm_low_alt_ozone.nc'

    TN_filename = 'SOMORA_level1a_test_2012.nc'

    filename = os.path.join(basename, flags_filename)
    filename_flags_lvl1b = os.path.join(basename, flags_lvl1b_filename)

    o3_filename = os.path.join(basename_o3, ozone_filename)
    figures = []
    if plot_flag_ts:
        flags = xr.open_dataset(
            filename,
            group='flags',
            decode_times=True,
            decode_coords=True,
            #use_cftime=True,
        )
        cal_flags = flags.calibration_flags 
        ozone = xr.open_dataset(
            o3_filename,
            decode_times=True,
            decode_coords=True,
            #use_cftime=True,
        )

        time = cal_flags.time
        fig, axs= plt.subplots(2,1, sharex=True)
        cal_flags.sum(axis=1).resample(time='4H').mean().plot(ax=axs[1])
        #cal_flags.sum(axis=1).plot(ax=axs[1])
        axs[0].plot(time[np.where(cal_flags[:,0]==0)], 6+cal_flags[:,0][np.where(cal_flags[:,0]==0)], '.', label=cal_flags.attrs['errorCode_1'])
        axs[0].plot(time[np.where(cal_flags[:,1]==0)], 5+cal_flags[:,1][np.where(cal_flags[:,1]==0)], '.', label=cal_flags.attrs['errorCode_2'])
        axs[0].plot(time[np.where(cal_flags[:,2]==0)], 4+cal_flags[:,2][np.where(cal_flags[:,2]==0)], '.', label=cal_flags.attrs['errorCode_3'])
        axs[0].plot(time[np.where(cal_flags[:,3]==0)], 3+cal_flags[:,3][np.where(cal_flags[:,3]==0)], '.', label=cal_flags.attrs['errorCode_4'])
        axs[0].plot(time[np.where(cal_flags[:,4]==0)], 2+cal_flags[:,4][np.where(cal_flags[:,4]==0)], '.', label=cal_flags.attrs['errorCode_5'])
        axs[0].plot(time[np.where(cal_flags[:,5]==0)], 1+cal_flags[:,5][np.where(cal_flags[:,5]==0)], '.', label=cal_flags.attrs['errorCode_6'])
        axs[0].set_ylim(0,6.4)
        axs[0].legend(loc='lower right', fontsize=6)
        #ozone.o3_x.isel(o3_p=12).plot(ax=axs[1])
        plt.tight_layout()
        figures.append(fig)

        flags_lvl1b = xr.open_dataset(
            filename_flags_lvl1b,
            group='flags',
            decode_times=True,
            decode_coords=True,
            #use_cftime=True,
        )
        int_flags = flags_lvl1b.calibration_flags

        time = int_flags.time
        fig, axs= plt.subplots(2,1, sharex=True)
        int_flags.sum(axis=1).resample(time='16H').mean().plot(ax=axs[1])
        axs[0].plot(time[np.where(int_flags[:,0]==0)], 1+int_flags[:,0][np.where(int_flags[:,0]==0)], 'x', label=int_flags.attrs['errorCode_1'])
        axs[0].plot(time[np.where(int_flags[:,1]==0)], int_flags[:,1][np.where(int_flags[:,1]==0)], 'o', label=int_flags.attrs['errorCode_2'])

        axs[0].legend(loc='lower right', fontsize=6)
        plt.tight_layout()
        figures.append(fig)

        save_single_pdf(basename+'flags_level1_2016.pdf', figures)

    if plot_TN:
        data = xr.open_dataset(
            os.path.join(basename, TN_filename),
            group='spectrometer1',
            decode_times=True,
            decode_coords=True,
            #use_cftime=True,
            )

        data=data.sel(time=slice("2012-12-10", "2012-12-31"))
        #data = data.isel(time=np.arange(7000,len(data.time)))

        data['noise_level'] =  data['noise_level']
        fig, axs= plt.subplots(5,1, sharex=True, figsize=(12,12))

        data['noise_level'].resample(time='2H').mean().plot(ax=axs[0])
        axs[0].set_ylim(1,1.8)
        data.mean_std_Tb.resample(time='2H').mean().plot(ax=axs[1])
        data['noise_temperature'].resample(time='2H').mean().plot(ax=axs[2])
        #axs[2].set_ylim(2750,3050)
        data['mean_hot_counts'].resample(time='2H').mean().plot(ax=axs[3])
        data['number_of_hot_spectra'].resample(time='2H').mean().plot(ax=axs[4])

        axs[0].set_title('SOMORA, 10-31 december 2012')


        plt.tight_layout()
        figures.append(fig)
        save_single_pdf(basename+'TN_level1a_dec_2012.pdf', figures)
