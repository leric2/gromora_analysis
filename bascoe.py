#!/usr/bin/env python3

import datetime
import os
from time import time

import matplotlib.pyplot as plt
import netCDF4
import numpy as np
import pandas as pd
import scipy.io

import xarray as xr
import matplotlib.ticker as ticker

from matplotlib.ticker import (MultipleLocator, FormatStrFormatter, AutoMinorLocator)


def read_bascoe(filename, save_LST=False):

    bascoe = xr.open_dataset(filename)
    
    #bascoe = xr.open_dataset('/storage/tub/atmosphere/BASCOE/bascoe_bern_2010-2020_lst.nc')
    #bascoe = xr.open_dataset('/storage/tub/atmosphere/BASCOE/bascoe_bern_2010-2020_lst_NOAA.nc')
    # approx_lst = bascoe['UT_time'] + pd.Timedelta(hours=SWISS_LON*24/360)
    #bascoe = bascoe.sel(time=slice('2010-06-01','2015-07-31'))
    
    # Add LST:
    if save_LST:
        bascoe['UTC_time'] = bascoe['time'].data
    
        lst_list = list()
        sza_list = list()
        for i, t in enumerate(bascoe['time'].data):
            lst_bascoe, ha, sza, nigh, tc = utc2lst_NOAA(bascoe.isel(time=i).time.data, SWISS_LAT, SWISS_LON)
            lst_list.append(lst_bascoe)
            sza_list.append(sza)
    
        bascoe['sza_NOAA'] = ('time', sza_list)
        bascoe['time'] = pd.to_datetime(lst_list)
        bascoe = bascoe.sortby('time')
    
        bascoe.to_netcdf('/storage/tub/atmosphere/BASCOE/bascoe_bern_2010-2020_lst_NOAA.nc')
    return bascoe

def plot_basic_ts(bascoe):

    fig, axs = plt.subplots(5,1, sharex=True, sharey=True, figsize=(12, 14))
    bascoe.temperature.plot(ax=axs[0], y='lev')
    bascoe.o3_vmr.plot(ax=axs[1], y='lev')
    bascoe.h2o_vmr.plot(ax=axs[2], y='lev')
    bascoe.no2_vmr.plot(ax=axs[3], y='lev')
    bascoe.no_vmr.plot(ax=axs[4], y='lev')

    axs[0].invert_yaxis()
    axs[0].set_ylim(1000, 0.001)
    axs[0].set_yscale('log')

    for ax in axs:
        ax.set_xlabel('')
        ax.set_ylabel('hybrid layer')

    fig.tight_layout(rect=[0, 0.01, 0.95, 1])
    fig.savefig('/home/esauvageat/Documents/GROMORA/Data/BASCOE/'+'bascoe_example.pdf', dpi=500)

def plot_t_profile(bascoe):

    fig, axs = plt.subplots(1,1, sharex=True, sharey=True, figsize=(12, 14))
    bascoe.temperature.mean(dim='time').plot(ax=axs, y='lev')
    axs.invert_yaxis()
    axs.set_ylim(1000, 0.001)
    axs.set_yscale('log')

    axs.set_xlabel('')
    axs.set_ylabel('hybrid layer')

    fig.tight_layout(rect=[0, 0.01, 0.95, 1])
    fig.savefig('/home/esauvageat/Documents/GROMORA/Data/BASCOE/'+'bascoe_t_profile.pdf', dpi=500)

if __name__ == "__main__":
    time_period = slice("2006-01-01", "2009-12-31")
    yrs = [2006, 2007,2008]#,2019[2009,2010,2011,2012,2013,2014,2015,2016,2017,2018,2019,2020,2021,]
    
    filename='/home/esauvageat/Documents/GROMORA/Data/BASCOE/O3CYCLEa_at_BE_201001010030-201001030000.gbs.nc'
    bascoe  = read_bascoe(filename)

    plot_basic_ts(bascoe)
    plot_t_profile(bascoe)