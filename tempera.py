#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 06.01.22

@author: Eric Sauvageat

This is the main script to read and test TEMPERA data from Witali
"""
import datetime
import os

import matplotlib.pyplot as plt
import netCDF4
import numpy as np
import pandas as pd

import xarray as xr

colormap = 'cividis'

def read_tempera_level3(folder = '/storage/lake/level3_data/TEMPERA/',years=[2014,2015,2016,2017], date_slice=slice('2014-01-01', '2017-12-31')):
    prefix = 'TEMPERA_level3_6h_resolution_'
    all_ds = []
    counter = 0
    for yr in years:
        og=xr.open_dataset(folder+prefix+str(yr)+'.h5', group='data')
        og = og.isel(phony_dim_0=0)

        datenum = xr.open_dataset(folder+prefix+str(yr)+'.h5', group='info')
        datenum = datenum.time.data[:,0]
        datetime_diff = []
        for i, t in enumerate(datenum):
            datetime_diff.append(datetime.datetime.fromordinal(int(t)) + datetime.timedelta(days=datenum[i] % 1) - datetime.timedelta(days=366))

        
        og['phony_dim_2'] = datetime_diff
        og = og.rename({'phony_dim_2':'time','phony_dim_1':'altitude', 'tmp':'temperature'})
        og['altitude'] = 1e-3*og.alt.data


        # yearly_ds = xr.Dataset(
        #     data_vars=dict(
        #         o3_x=(['time', 'pressure'], np.flip(1e6*v2021['o3'], axis=1)),
        #         o3_xa=(['time', 'pressure'], np.flip(1e6*v2021['o3ap'], axis=1)),
        #         o3_e = (['time', 'pressure'], np.flip(1e6*v2021['o3e'], axis=1)),
        #         h = (['time', 'pressure'], np.flip(v2021['h'], axis=1)),
        #     ),
        #     coords=dict(
        #         time=datetime_diff,
        #         alt = og.alt
        #     ),
        #     attrs=dict(description='ozone time serie from old gromos routines, version 2021, Klemens Hocke')
        #)
        all_ds.append(og)    
    new_ds = xr.concat(all_ds, dim='time')
    return new_ds.sel(time=date_slice)

def read_tempera_level3_concat(folder = '/storage/nas/MW/projects/GROMOS_TEMPERA/Data/TEMPERA/level_2_v_2023/',file='TEMPERA_2014_2017_v23.nc', date_slice=slice('2014-01-01', '2017-12-31')):
    prefix = 'TEMPERA_level3_6h_resolution_'
    all_ds = []
    data=xr.open_dataset(folder+file, group='temperature')
    info=xr.open_dataset(folder+file, group='info')

    data['altitude'] = info.alt

    datenum = info.time.data
    datetime_diff = []
    for i, t in enumerate(info.time):
        datetime_diff.append(datetime.datetime.fromordinal(int(t)) + datetime.timedelta(days=datenum[i] % 1) - datetime.timedelta(days=366))

    
    data['time'] = datetime_diff
    # og = og.rename({'phony_dim_2':'time','phony_dim_1':'altitude', 'tmp':'temperature'})
    # og['altitude'] = 1e-3*og.alt.data

    #new_ds = xr.concat(all_ds, dim='time')
    return data.sel(time=date_slice)

def read_tempera_level2(filename = '/storage/lake/level2_data/TEMPERA/tempera_profiles_all.nc', date_slice=slice('2014-01-01', '2017-12-31')):
    tempera = xr.open_dataset(filename)
    tempera['pressure'] =  1e-2*tempera.pressure.data
    return tempera.sortby('time').sel(time=date_slice)

def plot_old_tempera(tempera, freq):
    fig, axs = plt.subplots(1, 1, sharex=True, figsize=(20,12))
    pl = tempera.resample(time=freq).mean().T.interpolate_na(dim='pressure').plot(x='time', y='pressure', yscale='log')
    pl.set_edgecolor('face')
    axs.set_title('Temperature')
    # ax.set_yscale('log')
    axs.invert_yaxis()
    axs.set_ylabel('Pressure [hPa]')

    plt.show()

def plot_tempera(tempera, freq):
    fig, axs = plt.subplots(1, 1, sharex=True, figsize=(20,12))
    pl = tempera.resample(time=freq).mean().temperature.plot(x='time', y='altitude')
    pl.set_edgecolor('face')
    axs.set_title('Temperature')
    # ax.set_yscale('log')
    #axs.invert_yaxis()
    axs.set_ylabel('Altititude [km]')

    plt.show()

if __name__ == "__main__":
    years = [2014,2015]

    tempera = read_tempera_level3()
    plot_tempera(tempera, freq='1M')