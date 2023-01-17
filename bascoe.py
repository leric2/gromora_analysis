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

from base_tool import get_LST_from_UTC

BASCOE_LAT_BERN = 46.95
BASCOE_LON_BERN = 7.45

def read_bascoe(filename):

    bascoe = xr.open_dataset(filename)
        
    return bascoe

def read_all_bascoe(foldername, save=False):

    bascoe = xr.open_mfdataset(foldername)
    bascoe['pressure'] = (('time', 'lev'), 0.01*(bascoe.hyam + bascoe.hybm*bascoe.ps).data)
    bascoe['pressure_interface'] = (('time', 'ilev'), 0.01*(bascoe.hyai + bascoe.hybi*bascoe.ps).data)
    #bascoe['alt'] = (('time', 'lev'), np.flip(bascoe.hyam + bascoe.hybm*bascoe.ps))
    #bascoe['alt_interface'] = (('time', 'ilev'), np.flip(bascoe.hyai + bascoe.hybi*bascoe.ps))
    #bascoe['resolution'] = (('time', 'lev'), 1e-3*np.abs(np.diff(bascoe['alt_interface'])))

    #bascoe['time'] = bascoe['time'] + pd.Timedelta(hours=7.45*24/360)

    if save:
        bascoe['time'] = bascoe['time'] + pd.Timedelta(hours=7.45*24/360)
        # for t in bascoe.time.data:
        #     lst, ha, sza, nigh, tc = get_LST_from_UTC(t, BASCOE_LAT_BERN, BASCOE_LAT_BERN)

        bascoe.to_netcdf('/storage/tub/atmosphere/BASCOE/bascoe_NYA_2010-2020_lst_complete.nc') #.resample(time='1H').mean()
        
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

def plot_o3_profile(bascoe, ind=100):

    fig, axs = plt.subplots(1,1, sharex=True, sharey=True, figsize=(12, 14))
    bascoe.o3_vmr.isel(time=ind).plot(ax=axs, y='lev')
    axs.plot(bascoe.o3_vmr.isel(time=ind).data, bascoe.pressure.isel(time=ind).data)
    axs.invert_yaxis()
    axs.set_ylim(1000, 0.001)
    axs.set_yscale('log')

    axs.set_xlabel('')
    axs.set_ylabel('pressure')

    fig.tight_layout(rect=[0, 0.01, 0.95, 1])
    fig.savefig('/home/esauvageat/Documents/GROMORA/Data/BASCOE/'+'bascoe_ozone_profile.pdf', dpi=500)

def plot_p_profile(bascoe, ind=100):

    fig, axs = plt.subplots(1,2, sharey=True, figsize=(12, 14))
    bascoe.pressure.isel(time=ind).plot(ax=axs[0], y='lev')
    axs[0].plot(bascoe.lev.data, bascoe.lev.data)
    axs[0].invert_yaxis()
    axs[0].set_ylim(1000, 0.001)
    axs[0].set_xscale('log')
    axs[0].set_yscale('log')

    axs[0].set_xlabel('pressure')
    axs[0].set_ylabel('lev')

    axs[0].legend(['Pressure', 'lev'])

    deltaP = bascoe.pressure.isel(time=ind) - bascoe.lev


    deltaP.plot(ax=axs[1], y='lev')
    axs[1].set_xlabel('delta P [hPa]')
    axs[1].set_ylabel('')
    #xs[1].set_xlim(-2, 0.001)
    #axs[1].set_xscale('log')

    fig.tight_layout(rect=[0, 0.01, 0.95, 1])
    fig.savefig('/home/esauvageat/Documents/GROMORA/Data/BASCOE/'+'bascoe_pressure_profile.pdf', dpi=500)

def plot_resolution(bascoe):

    fig, axs = plt.subplots(1,1, sharex=True, sharey=True, figsize=(12, 14))
    monthly_p = bascoe.pressure.groupby('time.month').mean()
    monthly_p_i = bascoe.pressure_interface.groupby('time.month').mean()
    for m in monthly_p_i.month.data:
        monthly_resolution = 7*np.diff(np.log(monthly_p_i.sel(month=m)))
        axs.plot(monthly_resolution, monthly_p.lev , label=str(m))
    #bascoe.resolution.mean(dim='time').plot(ax=axs, y='lev')
    #bascoe.resolution.std(dim='time').plot(ax=axs, y='lev')
    #bascoe.alt.mean(dim='time').plot(ax=axs, y='lev')
    #bascoe.alt_interface.mean(dim='time').plot(ax=axs, y='ilev', ls='', marker='x')
    axs.invert_yaxis()
    axs.set_ylim(1000, 0.01)
    axs.set_yscale('log')

    axs.set_xlabel('Vertical resolution [km]')
    axs.set_ylabel('hybrid p level')

    fig.tight_layout(rect=[0, 0.01, 0.95, 1])
    fig.savefig('/home/esauvageat/Documents/GROMORA/Data/BASCOE/'+'bascoe_resolution.pdf', dpi=500)

if __name__ == "__main__":
    time_period = slice("2006-01-01", "2009-12-31")
    yrs = [2011]#,2019[2009,2010,2011,2012,2013,2014,2015,2016,2017,2018,2019,2020,2021,]
    
    folder='/storage/tub/atmosphere/BASCOE/NYA/*.nc'
    filename='/storage/tub/atmosphere/BASCOE/Bern/O3CYCLEa_at_BE_20091231.gbs.nc'
    #bascoe  = read_bascoe(filename)
    bascoe  = read_all_bascoe(folder, save=True)
    
    #bascoe = xr.open_dataset('/storage/tub/atmosphere/BASCOE/bascoe_bern_2010-2020.nc')
    
    #plot_basic_ts(bascoe.resample(time='1M').mean())
    #bascoe['pressure'] = (('time', 'lev'), 0.01*np.transpose((bascoe.hyam + bascoe.hybm*bascoe.ps).data))
    plot_resolution(bascoe)

    plot_p_profile(bascoe)
    plot_o3_profile(bascoe)