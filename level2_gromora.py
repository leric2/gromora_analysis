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

from GROMORA_harmo.scripts.retrieval import GROMORA_time

from MLS import *

MONTH_STR = ['Jan', 'Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']

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

def read_fshift_nc(basename, years=[2014,2015,2016,2017,2018,2019], date_slice=slice("2018-01-01", "2018-10-31")):
    fshit_tot = xr.Dataset()
    counter = 0
    for y in years:
        filename = basename+str(y)+'_fshift.nc'
        fshift = xr.open_dataset(
            filename,
            decode_times=True,
            decode_coords=True,
            # use_cftime=True,
        )
        if counter == 0:
            fshit_tot=fshift
        else:
            fshit_tot=xr.concat([fshit_tot, fshift], dim='time')
        
        counter=counter+1
        print('Read : '+filename)

    #fshit_tot.freq_shift_x.resample(time='24H', skipna=True).mean().plot()
    return fshit_tot

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

    year = pd.to_datetime(gromos.time.values[0]).year

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

    fig.savefig('/scratch/GROSOM/Level2/GROMORA_retrievals_polyfit2/GROMOS_ozone_rel_diff_'+str(year)+'.pdf', dpi=500)

def fshift_daily_cycle(ds_fshift, date_slice):
    fshift = ds_fshift.sel(time=date_slice)
    year = pd.to_datetime(fshift.time.values).year[0]

    fshift['tod'] = pd.to_datetime(fshift.time.values).hour
    fshift['month'] = pd.to_datetime(fshift.time.values).month
    daily_fshift = np.ones((12,23))*np.nan
    fig, axs = plt.subplots(1, 1)
    for m in range(0,12):
        for i in range(0,23):
            fshift_month = fshift.where(fshift['month'].values == m)
            daily_fshift[m,i] = np.nanmean(fshift_month.freq_shift_x.where(fshift_month['tod'].values == i)) 

        #axs.plot(np.arange(0,23), daily_fshift[m,:], label= MONTH_STR[m], lw=0.75)
    axs.plot(np.arange(0,23), np.nanmean(daily_fshift[[1,2,11],:], axis=0), label= 'DJF')
    axs.plot(np.arange(0,23), np.nanmean(daily_fshift[[3,4,5],:], axis=0), label= 'MAM')
    axs.plot(np.arange(0,23), np.nanmean(daily_fshift[[6,7,8],:], axis=0), label= 'JJA')
    axs.plot(np.arange(0,23), np.nanmean(daily_fshift[[9,10,11],:], axis=0), label= 'SON')
    axs.plot(np.arange(0,23), np.nanmean(daily_fshift, axis=0), 'k-x' ,label= 'Yearly')
    axs.grid()
    axs.set_xlabel('time of day')
    axs.set_ylabel('fshift [kHz]')
    #axs.set_ylim((-500, -100))
    axs.legend(bbox_to_anchor=(0.95, 0.95))
    fig.tight_layout(rect=[0, 0.01, 0.95, 1])
    fig.savefig('/scratch/GROSOM/Level2/GROMORA_retrievals_polyfit2/GROMOS_daily_fshift_'+str(year)+'.pdf', dpi=500)
    
def utc_to_lst(gromora):
    lsts = list()
    for i, t in enumerate(gromora.time.data):
        #print('from : ',t)
        lst, ha, sza, night = GROMORA_time.get_LST_from_UTC(t, gromora.obs_lat.data[i], gromora.obs_lon.data[i])
        #print('to :',lst)
        lsts.append(lst)

    gromora['time'] = lsts
    gromora['time'].attrs = {'description':'Local solar time'}
    return gromora

def compare_pressure(gromos, somora, pressure_level = [15,20,25]):
    year=pd.to_datetime(gromos.time.data[0]).year
    fig, axs = plt.subplots(len(pressure_level), 1, sharex=True, figsize=(15,10))
    for i, p in enumerate(pressure_level):
        gromos.o3_x.isel(o3_p=p).plot(ax=axs[i], color='b', lw=0.6)
        somora.o3_x.isel(o3_p=p).plot(ax=axs[i], color='r', lw=0.6)
        axs[i].set_xlabel('')
        axs[i].set_title(f'p = {gromos.o3_p.data[p]:.2f} hPa')
    axs[0].legend(['GROMOS','SOMORA'])
    for ax in axs:
        ax.grid()
    fig.tight_layout(rect=[0, 0.01, 0.95, 1])
    fig.savefig('/scratch/GROSOM/Level2/GROMORA_retrievals_polyfit2/ozone_comparison_pressure_level_'+str(year)+'.pdf', dpi=500)


def plot_fshift_ts(ds_fshift, date_slice, TRoom):
    fig, axs = plt.subplots(2, 1,sharex=True)
    ds_fshift.freq_shift_x.sel(time=date_slice).resample(time='2H').mean().plot(
        ax=axs[0]
    )
    axs[0].set_ylabel('fshift [kHz]')
    TRoom.TRoom.sel(time=date_slice).resample(time='2H').mean().plot(
        ax=axs[1]
    )
    axs[1].set_ylabel('TRoom [K]')
    fig.tight_layout(rect=[0, 0.01, 0.95, 1])

    fig.savefig('/scratch/GROSOM/Level2/GROMORA_retrievals_polyfit2/GROMOS_fshift_ts.pdf', dpi=500)

def compare_opacity(folder, year=2014, date_slice=slice("2014-01-01", "2014-01-31")):
    gromos = xr.open_dataset(
            folder+'GROMOS_opacity_'+str(year)+'.nc',
            group='spectrometer1',
            decode_times=True,
            decode_coords=True,
            # use_cftime=True,
        )
    somora = xr.open_dataset(
            folder+'SOMORA_opacity_'+str(year)+'.nc',
            group='spectrometer1',
            decode_times=True,
            decode_coords=True,
            # use_cftime=True,
        )

    gromos = gromos.sel(time=date_slice)
    somora = somora.sel(time=date_slice)

    fig, axs = plt.subplots(3, 1, sharex=True)
    gromos.tropospheric_opacity.resample(time='4H').mean().plot(
        ax=axs[0]
    )
    somora.tropospheric_opacity.resample(time='4H').mean().plot(
        ax=axs[0]
    )
    gromos.tropospheric_opacity_tc.plot(
        lw=0,
        marker='.',
        ms=0.5,
        ax=axs[0]
    )
    somora.tropospheric_opacity_tc.plot(
        lw=0,
        marker='.',
        ms=0.5,
        ax=axs[0]
    )
    axs[0].set_ylabel('opacity')
    axs[0].set_ylim((0,2))
    axs[0].legend(['GROMOS','SOMORA','GROMOS TC', 'SOMORA_TC'])
    gromos.tropospheric_transmittance.resample(time='4H').mean().plot(
        ax=axs[1]
    )
    somora.tropospheric_transmittance.resample(time='4H').mean().plot(
        ax=axs[1]
    )
    axs[1].set_ylabel('transmittance')

    rel_diff = 100*(gromos.tropospheric_opacity.resample(time='4H').mean() - somora.tropospheric_opacity.resample(time='4H').mean()) / gromos.tropospheric_opacity.resample(time='4H').mean()
    rel_diff.resample(time='4H').mean().plot(
        ax=axs[2]
    )
    axs[2].set_ylabel('opacity difference')
    axs[2].legend(['GRO - SOM'])
    fig.tight_layout(rect=[0, 0.01, 0.95, 1])
    fig.savefig('/scratch/GROSOM/Level2/GROMORA_retrievals_polyfit2/opactiy_comparison_'+str(year)+'.pdf', dpi=500)

if __name__ == "__main__":
    ds_fshift = read_fshift_nc(basename='/scratch/GROSOM/Level2/GROMORA_retrievals_polyfit2/GROMOS_',years=[2014, 2015, 2016, 2017, 2018])
    TRoom = xr.open_dataset(
            '/scratch/GROSOM/Level2/GROMORA_retrievals_polyfit2/'+'GROMOS_TRoom_'+str(2016)+'.nc',
            group='spectrometer1',
            decode_times=True,
            decode_coords=True,
            # use_cftime=True,
        )

    filename_gromos = '/storage/tub/instruments/gromos/level2/GROMORA/v1/GROMOS_2016_waccm_monthly_scaled_h2o_ozone.nc'
    #filename_gromos = '/storage/tub/instruments/gromos/level2/GROMORA/v1/2018_waccm_cov_yearly_ozone.nc'

    filename_somora = '/storage/tub/instruments/somora/level2/v1/SOMORA2018_06_30_waccm_cov_yearly_ozone.nc'
    #filename_gromos='/scratch/GROSOM/Level2/GROMORA_retrievals_polyfit2/GROMOS2014_03_31_waccm_monthly_scaled_h2o_ozone.nc'
    filename_somora='/scratch/GROSOM/Level2/GROMORA_retrievals_polyfit2/SOMORA2016_01_08_waccm_monthly_scaled_h2o_ozone.nc'
    
    date_slice=slice("2016-01-01", "2016-01-08")
    gromos = read_GROMORA(filename_gromos, date_slice)
    somora = read_GROMORA(filename_somora, date_slice)

    compare_ts(gromos, somora)

    compare_pressure(gromos,somora, pressure_level=[33 ,30, 23, 16,13, 8])

    gromos = utc_to_lst(gromos)
    
    plot_ozone_ts(gromos, altitude=False)
    # fshift_daily_cycle(ds_fshift, slice("2018-01-01", "2018-12-31"))
    # fshift_daily_cycle(ds_fshift, slice("2017-01-01", "2017-12-31"))
    # fshift_daily_cycle(ds_fshift, slice("2016-01-01", "2016-12-31"))
    # fshift_daily_cycle(ds_fshift, slice("2014-01-01", "2014-12-31"))
    # fshift_daily_cycle(ds_fshift, slice("2015-01-01", "2015-12-31"))

    #plot_fshift_ts(ds_fshift, date_slice = slice("2014-0-01", "2018-12-31"), TRoom=TRoom)
   # ds_fshift.freq_shift_x.sel(time=slice("2017-01-01", "2018-12-31")).resample(time='12H').mean().plot()
    # plt.matshow(gromos.o3_avkm.isel(time=0))
    # plt.colorbar()
    # 

    # z_grid = np.arange(1e3, 90e3, 1e3)
    # ozone_const_alt = constant_altitude_gromora(gromos, z_grid)

    compare_opacity(folder='/scratch/GROSOM/Level2/GROMORA_retrievals_polyfit2/',year=2014, date_slice=slice("2014-01-01", "2014-12-31"))

    #plot_ozone_ts(gromos, altitude=False)
    # plot_ozone_ts(ozone_const_alt, altitude=True)