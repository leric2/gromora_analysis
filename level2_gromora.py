#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import datetime
import os

import matplotlib.pyplot as plt
import netCDF4
import numpy as np
from numpy.lib.shape_base import dsplit
import pandas as pd

import xarray as xr

from MLS import *

def read_GROMORA(filename, date_slice):

    gromora_ds = xr.open_dataset(
        filename,
        decode_times=True,
        decode_coords=True,
        # use_cftime=True,
    )
    gromora_ds['o3_p'] = gromora_ds['o3_p']/100

    pandas_time_gromos = pd.to_datetime(gromora_ds.time.data)

    gromora_ds = gromora_ds.sel(time=date_slice)
    return gromora_ds

def constant_altitude_gromora(gromora_ds, z_grid):
    o3 = gromora_ds.o3_x
    o3_alt = gromora_ds.o3_z

    o3_const_alt = np.ones((len(z_grid), len(o3.time)))*np.nan
    for i,t in enumerate(o3.time):
        o3_const_alt[:,i] = np.interp(z_grid, o3_alt.isel(time=i).values, o3.isel(time=i).values)

    ozone_ds = xr.Dataset(
        data_vars=dict(
            o3_x=(['altitude', 'time'], o3_const_alt),
        ),
        coords=dict(
            time=gromora_ds.time,
            altitude=z_grid
        ),
        attrs=dict(description='ozone interpolated at constant altitude grid')
    )
    return ozone_ds

def plot_ozone_ts(gromora, altitude=False):
    if altitude:
        fig, axs = plt.subplots(1, 1, sharex=True)
        pl = gromora.o3_x.plot(
            x='time',
            y='altitude',
            ax=axs, 
            vmin=0,
            vmax=10,
            linewidth=0,
            rasterized=True,
            cmap='cividis'
        )
        pl.set_edgecolor('face')
        axs.set_ylabel('Altitude [m]')
    else:
        fig, axs = plt.subplots(1, 1, sharex=True)
        pl = gromora.o3_x.plot(
            x='time',
            y='o3_p',
            ax=axs, 
            vmin=0,
            vmax=10,
            yscale='log',
            linewidth=0,
            rasterized=True,
            cmap='cividis'
        )
        pl.set_edgecolor('face')
        # ax.set_yscale('log')
        axs.invert_yaxis()
        axs.set_ylabel('P [hPa]')

def compare_ts(gromos, somora):
    fig, axs = plt.subplots(2, 1, sharex=True)
    pl = gromos.o3_x.where(gromos.o3_mr>0.75).plot(
        x='time',
        y='o3_p',
        ax=axs[0], 
        vmin=0,
        vmax=10,
        yscale='log',
        linewidth=0,
        rasterized=True,
        cmap='cividis'
    )
    pl.set_edgecolor('face')
    # ax.set_yscale('log')
    axs[0].invert_yaxis()
    axs[0].set_ylabel('P [hPa]')

    pl2 = somora.o3_x.where(somora.o3_mr>0.75).plot(
        x='time',
        y='o3_p',
        ax=axs[1], 
        vmin=0,
        vmax=10,
        yscale='log',
        linewidth=0,
        rasterized=True,
        cmap='cividis'
    )
    pl2.set_edgecolor('face')
    # ax.set_yscale('log')
    axs[1].invert_yaxis()
    axs[1].set_ylabel('P [hPa]')
    axs[1].set_title('SOMORA') 

    for ax in axs:
        ax.set_ylim(500, 1e-2)

    plt.tight_layout()

    fig.savefig('/scratch/GROSOM/Level2/GROMORA_retrievals_polyfit2/GROMOS_ozone_comparison_2018.pdf', dpi=500)
    
    rel_diff = 100*(gromos.o3_x.where(gromos.o3_mr>0.75).mean(dim='time') -
                        somora.o3_x.where(somora.o3_mr>0.75).mean(dim='time'))/somora.o3_x.mean(dim='time')
    fig, axs = plt.subplots(1, 2, sharey=True)
    pl1 = gromos.o3_x.where(gromos.o3_mr>0.75).mean(dim='time').plot(
        y='o3_p',
        ax=axs[0], 
        yscale='log',
    )
    pl2 = somora.o3_x.where(somora.o3_mr>0.75).mean(dim='time').plot(
        y='o3_p',
        ax=axs[0], 
       
    )
    axs[0].invert_yaxis()
    axs[0].set_ylabel('P [hPa]')
    axs[0].legend(('GROMOS', 'SOMORA'))

    pl3=rel_diff.plot(
        y='o3_p',
        ax=axs[1], 
    )
    axs[1].set_xlim(-20,20)

    for ax in axs:
        ax.grid()

    plt.tight_layout()

    fig.savefig('/scratch/GROSOM/Level2/GROMORA_retrievals_polyfit2/GROMOS_ozone_rel_diff_2018.pdf', dpi=500)

if __name__ == "__main__":

    filename_gromos = '/storage/tub/instruments/gromos/level2/GROMORA/v1/full_2018_waccm_cov_yearly_ozone.nc'
    
    filename_somora = '/storage/tub/instruments/somora/level2/v1/SOMORA2018_06_30_waccm_cov_yearly_ozone.nc'
    
    date_slice=slice("2018-01-01", "2018-10-31")
    gromos = read_GROMORA(filename_gromos, date_slice)
    somora = read_GROMORA(filename_somora, date_slice)

    compare_ts(gromos, somora)

    z_grid = np.arange(1e3, 90e3, 1e3)
    ozone_const_alt = constant_altitude_gromora(gromos, z_grid)

    plot_ozone_ts(gromos, altitude=False)
    plot_ozone_ts(ozone_const_alt, altitude=True)