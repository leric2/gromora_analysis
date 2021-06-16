#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import datetime
import os

import matplotlib.pyplot as plt
import netCDF4
import numpy as np
import pandas as pd
import scipy.io

import xarray as xr
colormap = 'cividis'

def read_MLS(timerange):
    MLS_basename = '/home/esauvageat/Documents/AuraMLS/'
    filename_MLS = os.path.join(MLS_basename, 'aura-ozone-at-Bern.mat')
    mls_data = scipy.io.loadmat(filename_MLS)

    ozone_mls = mls_data['o3']
    p_mls = mls_data['p'][0]
    time_mls = mls_data['tm'][0]
    datetime_mls = []
    for i, t in enumerate(time_mls):
        datetime_mls.append(datetime.datetime.fromordinal(
            int(t)) + datetime.timedelta(days=time_mls[i] % 1) - datetime.timedelta(days=366))

    ds_mls = xr.Dataset(
        data_vars=dict(
            o3=(['time', 'p'], ozone_mls)
        ),
        coords=dict(
            lon=mls_data['longitude2'][0],
            lat=mls_data['latitude2'][0],
            time=datetime_mls,
            p=p_mls
        ),
        attrs=dict(description='ozone time series at bern')
    )
    #ds_mls = xr.decode_cf(ds_mls)
    #ds_mls.time.encoding['units'] = 'seconds since 1970-01-01 00:00:00'
    #ds_mls.time.encoding['calendar'] = "proleptic_gregorian"

    #ds_mls.to_netcdf('/home/esauvageat/Documents/AuraMLS/ozone_bern_ts.nc', format='NETCDF4')
    ds_mls= ds_mls.sel(time=timerange)
    return ds_mls

    
def plot_MLS(o3_mls):    
    fig, ax = plt.subplots(1, 1)
    #monthly_mls.plot(x='time', y='p', ax=ax ,vmin=0, vmax=9).resample(time='24H', skipna=True).mean()
    pl = o3_mls.plot(
        x='time',
        y='p',
        ax=ax,
        vmin=0,
        vmax=10,
        yscale='log',
        linewidth=0,
        rasterized=True,
        cmap=colormap
    )
   # pl.set_edgecolor('face')
    # ax.set_yscale('log')
    ax.set_ylim(0.01, 500)
    ax.invert_yaxis()
    ax.set_ylabel('P [hPa]')
    plt.tight_layout()
    # fig.savefig(instrument.level2_folder +
    #             '/ozone_mls_01-02-2019_mr'+'.pdf', dpi=500)

if __name__ == "__main__":
    ds_mls = read_MLS(timerange = slice("2019-01-30", "2019-06-22"))

    monthly_mls = ds_mls.o3.resample(time='1m', skipna=True).mean()
    plot_MLS(ds_mls.o3)