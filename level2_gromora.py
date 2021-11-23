#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#%%
import datetime
import os

import matplotlib.pyplot as plt
import netCDF4
import numpy as np
from numpy.lib.shape_base import dsplit
import pandas as pd
from typhon import collocations

import xarray as xr

from GROMORA_harmo.scripts.retrieval import GROMORA_time
from flags_analysis import read_level1_flags

from matplotlib.ticker import (MultipleLocator, FormatStrFormatter, AutoMinorLocator)
from matplotlib.lines import Line2D
import matplotlib.ticker as ticker
import matplotlib
cmap = matplotlib.cm.get_cmap('plasma')

from MLS import *
from sbuv import *

plt.rcParams.update({
    "text.usetex": False,
    "font.family": "serif",
    "font.sans-serif": ["Free sans"]})

plt.rcParams['xtick.labelsize'] = 22
plt.rcParams['ytick.labelsize'] = 22
plt.rcParams['font.size'] = 22
plt.rcParams['axes.titlesize'] = 22

color_gromos= '#d7191c'# '#008837'# '#d95f02'
color_somora= '#2c7bb6' #7b3294' # '#1b9e77'

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

def read_GROMORA_all(basefolder, instrument_name, date_slice, years, prefix):
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

    gromora_ds['o3_p'] = gromora_ds['o3_p']/100

    pandas_time_gromos = pd.to_datetime(gromora_ds.time.data)

    gromora_ds = gromora_ds.sel(time=date_slice)
    return gromora_ds

def read_gromos_v2021(filename, date_slice):
    basename = '/home/esauvageat/Documents/GROMORA/Data/'
    full_name = os.path.join(basename, filename)
    v2021 = scipy.io.loadmat(full_name)

    datenum = v2021['time'][0,:]
   
    datetime_diff = []
    for i, t in enumerate(datenum):
        datetime_diff.append(datetime.datetime.fromordinal(
            int(t)) + datetime.timedelta(days=datenum[i] % 1) - datetime.timedelta(days=366))
            
    pressure = np.flip(1e-2*v2021['p'][0,:])

    ds_GROMORA = xr.Dataset(
        data_vars=dict(
            o3_x=(['time', 'pressure'], np.flip(1e6*v2021['o3'], axis=1)),
            o3_xa=(['time', 'pressure'], np.flip(1e6*v2021['o3ap'], axis=1)),
            o3_e = (['time', 'pressure'], np.flip(1e6*v2021['o3e'], axis=1)),
            h = (['time', 'pressure'], np.flip(v2021['h'], axis=1)),
        ),
        coords=dict(
            time=datetime_diff,
            pressure=pressure
        ),
        attrs=dict(description='ozone time serie from old gromos routines, version 2021, Klemens Hocke')
    )

    ds_GROMORA = ds_GROMORA.sel(time=date_slice)

    return ds_GROMORA

def read_old_GROMOA_diff(filename, date_slice):
    basename = '/home/esauvageat/Documents/GROMORA/Data/'
    full_name = os.path.join(basename, filename)
    diff = scipy.io.loadmat(full_name)

    datenum = diff['DIFF_G'][:,0]
    altitude = diff['DIFF_G'][0:30,3]
    datetime_diff = []
    for i, t in enumerate(datenum):
        datetime_diff.append(datetime.datetime.fromordinal(
            int(t)) + datetime.timedelta(days=datenum[i] % 1) - datetime.timedelta(days=366))
            

    o3_gromos = np.ones((192,len(altitude)))*np.nan
    o3_somora =np.ones((192,len(altitude)))*np.nan
    o3_diff = np.ones((192,len(altitude)))*np.nan
    o3_reldiff =np.ones((192,len(altitude)))*np.nan
    datetime_daily = list()

    ind0 = 0
    ind1 = 30
    for n in range(0,192):
        datetime_daily.append(datetime_diff[ind0])
        o3_gromos[n,:]= diff['DIFF_G'][ind0:ind1,5]
        o3_somora[n,:]= diff['DIFF_G'][ind0:ind1,6]
        o3_diff[n,:]= diff['DIFF_G'][ind0:ind1,8]
        o3_reldiff[n,:]= diff['DIFF_G'][ind0:ind1,9]
        ind0 = ind0+30
        ind1 = ind1+30

    ds_GROMORA = xr.Dataset(
        data_vars=dict(
            o3_gromos=(['time', 'altitude'], o3_gromos),
            o3_somora=(['time', 'altitude'], o3_somora),
            o3_diff = (['time', 'altitude'], o3_diff),
            o3_rel_diff = (['time', 'altitude'], o3_reldiff),
        ),
        coords=dict(
            time=datetime_daily,
            altitude=altitude
        ),
        attrs=dict(description='ozone time series diff from old GROMORA routines')
    )
    #ds_mls = xr.decode_cf(ds_mls)
    #ds_mls.time.encoding['units'] = 'seconds since 1970-01-01 00:00:00'
    #ds_mls.time.encoding['calendar'] = "proleptic_gregorian"

    #ds_mls.to_netcdf('/home/esauvageat/Documents/AuraMLS/ozone_bern_ts.nc', format='NETCDF4')

    ds_GROMORA = ds_GROMORA.sel(time=date_slice)
    return ds_GROMORA

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

def read_plot_fshift(basefolder, instrument_name, years):
    ds_fshift = read_fshift_nc(basename=basefolder+instrument_name+'_',years=years)
    for y in years:
        fshift_daily_cycle(ds_fshift, slice(str(y)+"-01-01", str(y)+"-12-31"), basename=basefolder+'/'+instrument_name)
    
    fig, ax = plt.subplots(1, 1, figsize=(10,8))
    #ax.set_ylim=(0,100)
    if instrument_name == 'GROMOS':
        fshift_lim =(-700,500)
    elif instrument_name == 'SOMORA':
        fshift_lim =(0,500)
    ds_fshift.freq_shift_x.resample(time='4H').mean().plot.line(ax=ax, ylim=fshift_lim, color='k')
    ax.grid()
    ax.set_ylabel('freq shift [kHz]')
    ax.set_xlabel('')
    ax.set_title('Frequency shift retrieval '+instrument_name)
    fig.savefig(basefolder+'/'+instrument_name+'_fshift.pdf', dpi=500)
    return ds_fshift

def constant_altitude_gromora(gromora_ds, z_grid):
    o3 = gromora_ds.o3_x
    try:
        o3_alt = 1e-3*gromora_ds.o3_z
    except:
        o3_alt = gromora_ds.h

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
        pl = gromora.o3_x.resample(time='1H').mean().plot(
            x='time',
            y='o3_p',
            ax=axs, 
            vmin=0,
            vmax=10,
            yscale='log',
            linewidth=0,
            rasterized=True,
            cmap='cividis',
        )
        pl.set_edgecolor('face')
        # ax.set_yscale('log')
        axs.invert_yaxis()
        axs.set_ylabel('P [hPa]')


def plot_ozone_flags(instrument, gromora, flags1a, flags1b, pressure_level=[27, 12], opacity=[], calib_version=1):
    year = pd.to_datetime(gromora.time.values[0]).year

    fig, axs = plt.subplots(4, 1, sharex=True, figsize=(15,10))
    pl = gromora.o3_x.resample(time='1H').mean().plot(
        x='time',
        y='o3_p',
        ax=axs[0], 
        vmin=0,
        vmax=10,
        yscale='log',
        linewidth=0,
        rasterized=True,
        cmap='cividis',
        add_colorbar=False
    )
    pl.set_edgecolor('face')
    # ax.set_yscale('log')
    axs[0].invert_yaxis()
    axs[0].set_ylabel('P [hPa]')

    for i, p in enumerate(pressure_level):
        color = ['r','b']
        gromora.o3_x.isel(o3_p=p).resample(time='1H').mean().plot(ax=axs[1], color=color[i], lw=1.5, label=f'p = {gromora.o3_p.data[p]:.3f} hPa' )
        axs[1].set_xlabel('')
        axs[1].legend()
        #axs[1].set_title(f'p = {gromora.o3_p.data[p]:.3f} hPa')

    cal_flags = flags1a.calibration_flags 
    sum_flags_level1a = cal_flags.sum(axis=1)
    sum_flags_level1a.resample(time='1H').mean().plot(ax=axs[2])
    time = cal_flags.time
    for i in range(len(cal_flags.flags)):
        cal_flags[:,i] = 0.5*(i+cal_flags[:,i])
    cal_flags[:,0].resample(time='1H').mean().plot(ax = axs[2], label=cal_flags.attrs['errorCode_1'])
    cal_flags[:,1].resample(time='1H').mean().plot(ax = axs[2], label=cal_flags.attrs['errorCode_2'])
    cal_flags[:,2].resample(time='1H').mean().plot(ax = axs[2], label=cal_flags.attrs['errorCode_3'])
    cal_flags[:,3].resample(time='1H').mean().plot(ax = axs[2], label=cal_flags.attrs['errorCode_4'])
    cal_flags[:,4].resample(time='1H').mean().plot(ax = axs[2], label=cal_flags.attrs['errorCode_5'])
    cal_flags[:,5].resample(time='1H').mean().plot(ax = axs[2], label=cal_flags.attrs['errorCode_6'])

    if calib_version==2:
        cal_flags[:,6].resample(time='1H').mean().plot(ax = axs[2], label=cal_flags.attrs['errorCode_7'])


    # axs[1+len(pressure_level)].plot(time[np.where(cal_flags[:,0]==0)], 6+cal_flags[:,0][np.where(cal_flags[:,0]==0)], '-', label=cal_flags.attrs['errorCode_1'])
    # axs[1+len(pressure_level)].plot(time[np.where(cal_flags[:,1]==0)], 5+cal_flags[:,1][np.where(cal_flags[:,1]==0)], '-', label=cal_flags.attrs['errorCode_2'])
    # axs[1+len(pressure_level)].plot(time[np.where(cal_flags[:,2]==0)], 4+cal_flags[:,2][np.where(cal_flags[:,2]==0)], '-', label=cal_flags.attrs['errorCode_3'])
    # axs[1+len(pressure_level)].plot(time[np.where(cal_flags[:,3]==0)], 3+cal_flags[:,3][np.where(cal_flags[:,3]==0)], '-', label=cal_flags.attrs['errorCode_4'])
    # axs[1+len(pressure_level)].plot(time[np.where(cal_flags[:,4]==0)], 2+cal_flags[:,4][np.where(cal_flags[:,4]==0)], '-', label=cal_flags.attrs['errorCode_5'])
    # axs[1+len(pressure_level)].plot(time[np.where(cal_flags[:,5]==0)], 1+cal_flags[:,5][np.where(cal_flags[:,5]==0)], '-', label=cal_flags.attrs['errorCode_6'])
    # axs[1+len(pressure_level)].plot(time[np.where(cal_flags[:,6]==0)], 1+cal_flags[:,6][np.where(cal_flags[:,6]==0)], '-', label=cal_flags.attrs['errorCode_7'])
    # axs[1+len(pressure_level)].set_ylim(0,7.4)
    axs[2].legend(loc='lower right', fontsize=6)
    # flags1a.calibration_flags.plot(
    #     x='time',
    #     y='o3_p',
    #     ax=axs[1+len(pressure_level)]
    # )
   
    # flags1b.calibration_flags[:,1].resample(time='1H').mean().plot(
    #     ax=axs[2+len(pressure_level)]
    #     )
    opacity.tropospheric_opacity.resample(time='1H').mean().plot(ax=axs[3])
    axs[3].set_ylabel('Opacity')
    plt.tight_layout(rect=[0, 0.01, 0.99, 1])
    fig.savefig('/scratch/GROSOM/Level2/GROMORA_waccm/'+instrument+'_'+str(year)+'_diagnostic.pdf', dpi=500)

def compare_ts(gromos, somora, freq, date_slice, basefolder):
    fs = 28
    year = pd.to_datetime(gromos.time.values[0]).year

    gromos['o3'] = gromos.o3_x
    somora['o3'] = somora.o3_x

    fig, axs = plt.subplots(2, 1, sharex=True, figsize=(38,12))
    pl = gromos.sel(time=date_slice).o3.resample(time=freq).mean().plot(
        x='time',
        y='o3_p',
        ax=axs[0], 
        vmin=0,
        vmax=10,
        yscale='log',
        linewidth=0,
        rasterized=True,
        cmap='cividis',
        add_colorbar=False
    )
    pl.set_edgecolor('face')
    axs[0].set_title('GROMOS', fontsize=fs+2)
    # ax.set_yscale('log')
    axs[0].invert_yaxis()
    axs[0].set_ylabel('Pressure [hPa]', fontsize=fs)

    pl2 = somora.sel(time=date_slice).o3.resample(time=freq).mean().plot(
        x='time',
        y='o3_p',
        ax=axs[1], 
        vmin=0,
        vmax=10,
        yscale='log',
        linewidth=0,
        rasterized=True,
        cmap='cividis',
        add_colorbar=False
    )
    pl2.set_edgecolor('face')
    # ax.set_yscale('log')
    axs[1].invert_yaxis()
    axs[1].set_ylabel('Pressure [hPa]', fontsize=fs)
    axs[1].set_title('SOMORA', fontsize=fs+2)

    cbaxes = fig.add_axes([0.92, 0.25, 0.02, 0.5]) 
    cb = plt.colorbar(pl, cax=cbaxes, orientation="vertical", pad=0.0)
    cb.set_label(label=r"O$_3$ [ppmv]", weight='bold', fontsize=fs)
    cb.ax.tick_params()

    for ax in axs:
        ax.set_ylim(100, 1e-2)
        ax.set_xlabel('')
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: '{:g}'.format(y)))


    plt.tight_layout(rect=[0, 0.01, 0.92, 1])

    fig.savefig(basefolder+'GROMOS_ozone_comparison_'+str(year)+'.pdf', dpi=500)
    
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

    plt.tight_layout(rect=[0, 0.01, 0.99, 1])
    fig.savefig(basefolder+'GROMOS_ozone_rel_diff_'+str(year)+'.pdf', dpi=500)



def compare_with_apriori(gromos, freq, date_slice, basefolder):
    fs = 28
    year = pd.to_datetime(gromos.time.values[0]).year

    gromos['o3'] = gromos.o3_x
    gromos['apriori'] = gromos.o3_xa*1e6

    diff = gromos['o3']- gromos['apriori']

    fig, axs = plt.subplots(2, 1, sharex=True, figsize=(20,12))
    pl = gromos.sel(time=date_slice).o3.resample(time=freq).mean().plot(
        x='time',
        y='o3_p',
        ax=axs[0], 
        vmin=0,
        vmax=10,
        yscale='log',
        linewidth=0,
        rasterized=True,
        cmap='cividis',
        add_colorbar=True
    )
    pl.set_edgecolor('face')
    axs[0].set_title('Retrieved Ozone', fontsize=fs+2)
    # ax.set_yscale('log')
    axs[0].invert_yaxis()
    axs[0].set_ylabel('Pressure [hPa]', fontsize=fs)

    pl2 = diff.sel(time=date_slice).resample(time=freq).mean().plot(
        x='time',
        y='o3_p',
        ax=axs[1], 
        yscale='log',
        linewidth=0,
        rasterized=True,
        cmap='coolwarm',
        add_colorbar=True
    )
    pl2.set_edgecolor('face')
    # ax.set_yscale('log')
    axs[1].invert_yaxis()
    axs[1].set_ylabel('Pressure [hPa]', fontsize=fs)
    axs[1].set_title('Difference with apriori', fontsize=fs+2)

    # #cbaxes = axs[0].add_axes([0.92, 0.25, 0.02, 0.5]) 
    # cb = axs[0].colorbar(pl, orientation="vertical", pad=0.0)
    # cb.set_label(label=r"O$_3$ [ppmv]", weight='bold', fontsize=fs)
    # cb.ax.tick_params()

    for ax in axs:
        #ax.set_ylim(100, 1e-2)
        ax.set_xlabel('')
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: '{:g}'.format(y)))


    plt.tight_layout(rect=[0, 0.01, 0.92, 1])

    fig.savefig(basefolder+'apriori_comparison'+str(year)+'.pdf', dpi=500)

def compare_ts_MLS(gromos, somora, date_slice, freq, basefolder, sbuv=None):
    fs = 16
    year = pd.to_datetime(gromos.time.values[0]).year

    ds_mls= read_MLS(date_slice)

    fig, axs = plt.subplots(4, 1, sharex=True, figsize=(25,12))
    pl = gromos.sel(time=date_slice).o3_x.where(gromos.o3_mr>0.75).resample(time=freq).mean().plot(
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
    axs[0].set_title('GROMOS', fontsize=fs+4) 
    # ax.set_yscale('log')
    axs[0].invert_yaxis()
    axs[0].set_ylabel('P [hPa]', fontsize=fs)

    pl2 = somora.sel(time=date_slice).o3_x.where(somora.o3_mr>0.75).resample(time=freq).mean().plot(
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
    axs[1].set_ylabel('P [hPa]', fontsize=fs)
    axs[1].set_title('SOMORA', fontsize=fs+4)

    pl3 = ds_mls.sel(time=date_slice).o3.resample(time=freq).mean().plot(
        x='time',
        y='p',
        ax=axs[2], 
        vmin=0,
        vmax=10,
        yscale='log',
        linewidth=0,
        rasterized=True,
        cmap='cividis'
    )
    pl3.set_edgecolor('face')
    # ax.set_yscale('log')
    axs[2].invert_yaxis()
    axs[2].set_ylabel('P [hPa]', fontsize=fs)
    axs[2].set_title('MLS', fontsize=fs+4)
    
    pl3 = sbuv.sel(time=date_slice).ozone.resample(time=freq).mean().plot(
        x='time',
        y='p',
        ax=axs[3], 
        vmin=0,
        vmax=10,
        yscale='log',
        linewidth=0,
        rasterized=True,
        cmap='cividis'
    )
    pl3.set_edgecolor('face')
    # ax.set_yscale('log')
    axs[3].invert_yaxis()
    axs[3].set_ylabel('P [hPa]', fontsize=fs)
    axs[3].set_title('SBUV', fontsize=fs+4)

    for ax in axs:
        ax.set_ylim(100, 1e-2)
        ax.set_xlabel('')
        ax.tick_params(axis='both', which='major', labelsize=fs-2)

    plt.tight_layout()
    fig.tight_layout(rect=[0, 0.01, 0.99, 1])
    fig.savefig(basefolder+'GROMORA_MLS_ozone_'+str(year)+'.pdf', dpi=500)
    
def fshift_daily_cycle(ds_fshift, date_slice, basename):
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
    fig.savefig(basename+'_daily_fshift_'+str(year)+'.pdf', dpi=500)
    
def utc_to_lst(gromora):
    lsts = list()
    sunrise = list()
    sunset = list()
    for i, t in enumerate(gromora.time.data):
        #print('from : ',t)
        lst, ha, sza, night = GROMORA_time.get_LST_from_UTC(t, gromora.obs_lat.data[i], gromora.obs_lon.data[i])
        #print('to :',lst)
        lsts.append(lst)

        sunr, suns = GROMORA_time.get_sunset_lst_from_lst(lst, gromora.obs_lat.data[i])
        sunrise.append(sunr)
        sunset.append(suns)

    gromora['time'] = lsts
    gromora['time'].attrs = {'description':'Local solar time'}

    sunrise_da = xr.DataArray(
        data = sunrise,
        dims=['time'],
        coords=dict(time=gromora['time']),
        attrs=dict(description='sunrise')
    )
    sunset_da = xr.DataArray(
        data = sunset,
        dims=['time'],
        coords=dict(time=gromora['time']),
        attrs=dict(description='sunset')
    )
    gromora['sunrise'] = sunrise_da
    gromora['sunset'] = sunset_da
    return gromora

def compare_pressure(gromos, somora, pressure_level = [15,20,25], add_sun=False, freq='1D',basefolder=''):
    fs=22
    year=pd.to_datetime(gromos.time.data[0]).year
    fig, axs = plt.subplots(len(pressure_level), 1, sharex=True, figsize=(18,12))
    for i, p in enumerate(pressure_level):
        gromos.o3_x.isel(o3_p=p).resample(time=freq).mean().plot(ax=axs[i], color=color_gromos, lw=1.5)
        somora.o3_x.isel(o3_p=p).resample(time=freq).mean().plot(ax=axs[i], color=color_somora, lw=1.5)
        axs[i].set_xlabel('')
        axs[i].set_title(f'p = {gromos.o3_p.data[p]:.3f} hPa', fontsize=fs)
        if add_sun:
            #for s, sr in enumerate(gromos.sunrise.data):
            #sr = gromos.sunrise.data
            sunrise = gromos.sunrise.resample(time='1D').max()
            sunset = gromos.sunset.resample(time='1D').max()
            
            for d in range(len(sunrise.data)):
                sr=sunrise.data[d]
                ss=sunset.data[d]
                axs[i].axvspan(sr,ss, color='orange', alpha=0.2)

            #     #axs[i].axvline(sunr, color='k', linestyle='-.')
            # for suns in sunset.data:
            #     axs[i].axvline(suns, color='k', linestyle='--')

    axs[0].legend(['GROMOS','SOMORA'], loc=1, fontsize=fs-2)
    #axs[0].legend(['OG','SB corr'], loc=1, fontsize=fs-2)

    for a in [0,1]:
        #axs[a].yaxis.set_major_locator(MultipleLocator(1))
        axs[a].set_ylim(0,4)

    for a in [2,3,4]:
        #axs[a].yaxis.set_major_locator(MultipleLocator(1))
        axs[a].set_ylim(0,10)

    for ax in axs:
        ax.grid()
        ax.set_ylabel('O$_3$ VMR [ppmv]', fontsize=fs-2)
        ax.yaxis.set_major_locator(MultipleLocator(2))
        ax.yaxis.set_minor_locator(MultipleLocator(0.5))
        ax.tick_params(axis='both', which='major', labelsize=fs)

    fig.tight_layout(rect=[0, 0.01, 0.99, 1])
    fig.savefig(basefolder+'ozone_comparison_pressure_level_'+str(year)+'.pdf', dpi=500)

def compare_pressure_mls_sbuv(gromos, somora, mls, sbuv, pressure_level = [15,20,25], add_sun=False, freq='1D',basefolder=''):
    fs=22
    year=pd.to_datetime(gromos.time.data[0]).year
    fig, axs = plt.subplots(len(pressure_level), 1, sharex=True, figsize=(25,12))
    for i, p_ind in enumerate(pressure_level):
        pressure =  gromos.o3_p.data[p_ind]
        mls.o3.sel(p=pressure, method='nearest', tolerance=0.4*pressure).resample(time=freq).mean().plot(ax=axs[i], color='k', lw=1.5)
        if pressure > 0.4:
            sbuv.ozone.sel(p=pressure, method='nearest', tolerance=0.4*pressure).resample(time=freq).mean().plot(ax=axs[i], color='#fdae61', lw=1.5)
        gromos.o3_x.isel(o3_p=p_ind).resample(time=freq).mean().plot(ax=axs[i], color=color_gromos, lw=1.5)
        somora.o3_x.isel(o3_p=p_ind).resample(time=freq).mean().plot(ax=axs[i], color=color_somora, lw=1.5)
        axs[i].set_xlabel('')
        axs[i].set_title(f'p = {pressure:.3f} hPa', fontsize=fs)

        if add_sun:
            #for s, sr in enumerate(gromos.sunrise.data):
            #sr = gromos.sunrise.data
            sunrise = gromos.sunrise.resample(time='1D').max()
            sunset = gromos.sunset.resample(time='1D').max()
            
            for d in range(len(sunrise.data)):
                sr=sunrise.data[d]
                ss=sunset.data[d]
                axs[i].axvspan(sr,ss, color='orange', alpha=0.2)

            #     #axs[i].axvline(sunr, color='k', linestyle='-.')
            # for suns in sunset.data:
            #     axs[i].axvline(suns, color='k', linestyle='--')

    axs[2].legend(['MLS', 'SBUV', 'GROMOS','SOMORA', ], loc=1, fontsize=fs-2)
    #axs[0].legend(['OG','SB corr'], loc=1, fontsize=fs-2)

    for a in [0,1]:
        #axs[a].yaxis.set_major_locator(MultipleLocator(1))
        axs[a].set_ylim(0,4)

    for a in [2,3,4]:
        #axs[a].yaxis.set_major_locator(MultipleLocator(1))
        axs[a].set_ylim(0,10)

    for ax in axs:
        ax.grid()
        ax.set_ylabel('O$_3$ VMR [ppmv]', fontsize=fs-2)
        ax.yaxis.set_major_locator(MultipleLocator(2))
        ax.yaxis.set_minor_locator(MultipleLocator(0.5))
        ax.tick_params(axis='both', which='major', labelsize=fs)

    fig.tight_layout(rect=[0, 0.01, 0.99, 1])
    fig.savefig(basefolder+'ozone_comparison_pressure_level_MLS_SBUV_'+str(year)+'.pdf', dpi=500)

def map_rel_diff(gromos, somora ,freq='12H', basefolder=''):
    fs=16
    year=pd.to_datetime(gromos.time.data[0]).year
    fig, axs = plt.subplots(1, 1, sharex=True, figsize=(16,8))
    
    rel_diff = 100*(gromos.o3_x.resample(time='1H').mean()- somora.o3_x.resample(time='1H').mean())/gromos.o3_x.resample(time='1H').mean()
    rel_diff.resample(time=freq).mean().plot(
        ax=axs,
        x='time',
        y='o3_p',
        vmin=-20,
        vmax=20,
        yscale='log',
        cmap='coolwarm'
    )
    axs.invert_yaxis()
    axs.set_xlabel('')
    axs.set_ylabel('Pressure [hPa]',fontsize=fs)

    axs.set_ylim(25,0.5)
    axs.set_title('Monthly relative diff GROMOS-SOMORA', fontsize=fs+4)
    axs.tick_params(axis='both', which='major', labelsize=fs-2)
    axs.grid()
    fig.tight_layout(rect=[0, 0.01, 0.95, 1])
    fig.savefig(basefolder+'ozone_rel_diff_'+str(year)+'.pdf', dpi=500)

def compute_correlation(gromos, somora, freq='1D', pressure_level = [15,20,25], basefolder=''):
    year=pd.to_datetime(gromos.time.data[0]).year
    fs=24
    
    fig, axs = plt.subplots(len(pressure_level),1,figsize=(6, 6*len(pressure_level)))

    #ds_o3_gromora=xr.merge(({'o3_gromos':gromos.o3_x.isel(o3_p=p).resample(time=freq).mean()},{'o3_somora':somora.o3_x.isel(o3_p=p).resample(time=freq).mean()}))
    ds_o3_gromora=xr.merge(({'o3_gromos':gromos.o3_x.resample(time=freq).mean()},{'o3_somora':somora.o3_x.resample(time=freq).mean()}))
    #ds_o3_gromora_plot = ds_o3_gromora.isel(o3_p=pressure_level)
    for i, p in enumerate(pressure_level): 
        pearson_corr = xr.corr(ds_o3_gromora.isel(o3_p=p).o3_gromos.interpolate_na(dim='time',fill_value="extrapolate"),ds_o3_gromora.isel(o3_p=p).o3_somora.interpolate_na(dim='time',fill_value="extrapolate"), dim='time')
        print('Pearson corr coef: ',pearson_corr)
        ds_o3_gromora.isel(o3_p=p).plot.scatter(
            ax=axs[i],
            x='o3_gromos', 
            y='o3_somora',
            color='k',
            marker='.'
        )
        axs[i].plot([min(ds_o3_gromora.isel(o3_p=p).o3_gromos.values),max(ds_o3_gromora.isel(o3_p=p).o3_gromos.values)],[min(ds_o3_gromora.isel(o3_p=p).o3_gromos.values), max(ds_o3_gromora.isel(o3_p=p).o3_gromos.values)],'k--')
        axs[i].set_xlabel(r'GROMOS O$_3$ [ppmv]', fontsize=fs-2)
        axs[i].set_ylabel(r'SOMORA O$_3$ [ppmv]', fontsize=fs-2)
        axs[i].set_title(r'O$_3$ VMR '+f'at p = {gromos.o3_p.data[p]:.3f} hPa')
        axs[i].set_xlim(min(ds_o3_gromora.isel(o3_p=p).o3_gromos.values),max(ds_o3_gromora.isel(o3_p=p).o3_gromos.values))
        axs[i].set_ylim(min(ds_o3_gromora.isel(o3_p=p).o3_gromos.values),max(ds_o3_gromora.isel(o3_p=p).o3_gromos.values))
        axs[i].xaxis.set_major_locator(MultipleLocator(1))
        axs[i].xaxis.set_minor_locator(MultipleLocator(0.5))
        axs[i].yaxis.set_major_locator(MultipleLocator(1))
        axs[i].yaxis.set_minor_locator(MultipleLocator(0.5))
        axs[i].text(
            0.65,
            0.1,
            r'$\rho = {:.2g}$'.format(pearson_corr.data),
            transform=axs[i].transAxes,
            verticalalignment="bottom",
            horizontalalignment="left",
            fontsize=fs
    )
    
    for ax in axs:
        ax.grid(which='both', axis='x')
        ax.grid(which='both', axis='y')
    fig.tight_layout(rect=[0, 0.01, 0.95, 1])
    fig.savefig(basefolder+'ozone_scatter_'+freq+'_'+str(year)+'.pdf', dpi=500)

def compute_corr_profile(gromos, somora, freq='1D', basefolder=''):
    year=pd.to_datetime(gromos.time.data[0]).year
    fs=24
    
    fig, axs = plt.subplots(1,1,figsize=(12,12))
    pearson_corr = xr.corr(gromos.o3_x.resample(time=freq).mean().interpolate_na(dim='time',fill_value="extrapolate"),somora.o3_x.resample(time=freq).mean().interpolate_na(dim='time',fill_value="extrapolate"), dim='time')
    pearson_corr.plot(
        ax=axs,
        y='o3_p',
        yscale='log',
    )
    axs.invert_yaxis()
    axs.set_xlim(0,1)
    axs.grid()

def compare_diff_daily(gromos, somora,gromora_old, pressure_level = [15,20,25], altitudes = [15,20,25]):
    year=pd.to_datetime(gromos.time.data[0]).year
    fig, axs = plt.subplots(len(pressure_level), 1, sharex=True, figsize=(15,10))
    for i, p in enumerate(pressure_level):
        daily_diff = 100*(somora.o3_x.isel(o3_p=p).resample(time='1D').mean() - gromos.o3_x.isel(o3_p=p).resample(time='1D').mean())/gromos.o3_x.isel(o3_p=p).resample(time='D').mean()
        daily_diff_old = (gromora_old.o3_somora.sel(altitude=altitudes[i]).resample(time='D').mean() - gromora_old.o3_gromos.sel(altitude=altitudes[i]).resample(time='D').mean())
        daily_diff.plot(ax=axs[i], color='b', marker ='.', lw=0.6, label='New routine')
        gromora_old.o3_rel_diff.sel(altitude=altitudes[i]).plot(ax=axs[i], color='r', marker ='.',lw=0.6, label='Old routine')
        #daily_diff_old.plot(ax=axs[i], color='k', lw=0.6, label='Old')
        axs[i].set_xlabel('')
        axs[i].set_ylabel(r'$\Delta$O$_3$ [%]')
        axs[i].set_title(f'p = {gromos.o3_p.data[p]:.3f}hPa, altitude = {altitudes[i]:.1f}km')

    #axs[0].legend(['Old routine','New routine'])
    axs[0].legend()
    for ax in axs:
        ax.grid()
        ax.axhline(y=0, ls='--', lw=0.7 , color='k')
        ax.set_ylim((-40,40))
       
        #ax.set_xlim("2018-01-01", "2018-05-31")
    axs[0].set_ylim((-100,100))
    fig.suptitle('Ozone relative difference SOMORA-GROMOS')
    fig.tight_layout(rect=[0, 0.01, 0.95, 1])
    fig.savefig('/scratch/GROSOM/Level2/GROMORA_retrievals_polyfit2/ozone_comparison_old_vs_new_AVK_smoothed'+str(year)+'.pdf', dpi=500)

def gromos_old_vs_new(gromos, gromos_v2021):
    year=pd.to_datetime(gromos.time.data[0]).year
    fig, axs = plt.subplots(2, 1, sharex=True, sharey=True, figsize=(10,10))

    # mean_diff = 100*(gromos.o3_x.mean(dim='time') - gromos_v2021.o3_x.mean(dim='time') )/gromos.o3_x.mean(dim='time')
    pl = gromos.o3_x.plot(
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
   
    pl2 = gromos_v2021.o3_x.plot(
        x='time',
        y='pressure',
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
    axs[1].set_title('old processing')

    #fig.suptitle('Ozone relative difference GROMOS new-old')
    fig.tight_layout(rect=[0, 0.01, 0.95, 1])
    fig.savefig('/scratch/GROSOM/Level2/GROMORA_retrievals_polyfit2/gromos_old_vs_new_'+str(year)+'.pdf', dpi=500)

def plot_o3_pressure_profile(gromos):
    fig, ax = plt.subplots(1, 1, figsize=(8,12))
    vmr = gromos.o3_x.mean(dim='time')*1e-6
    po3 = vmr*gromos.o3_p.data*100000
    po3.plot(
        y='o3_p', 
        ax=ax,
        color='k',
        yscale='log',
    )
    ax.invert_yaxis()
    ax.set_xlabel('Pressure [mPa]')
    ax.set_ylabel('Pressure [hPa]')
    ax.grid()

def compare_mean_diff(gromos, somora, sbuv=None):
    color_shading='grey'
    fs = 22
    year=pd.to_datetime(gromos.time.data[0]).year
    fig, axs = plt.subplots(1, 2, sharey=True, figsize=(8,12))

    mean_diff_new = 100*(gromos.o3_x.mean(dim='time') - somora.o3_x.mean(dim='time') )/gromos.o3_x.mean(dim='time')
    mean_altitude_gromos = 1e-3*gromos.o3_z.mean(dim='time')
    mean_altitude_somora = 1e-3*somora.o3_z.mean(dim='time')
    mr_somora = somora.o3_mr.data
    mr_gromos = gromos.o3_mr.data
    p_somora_mr = somora.o3_p.data[np.mean(mr_somora,0)>=0.8]
    p_gromos_mr = gromos.o3_p.data[np.mean(mr_gromos,0)>=0.8]

    somora.o3_x.mean(dim='time').plot(
        y='o3_p',
        ax=axs[0], 
        yscale='log',
        color=color_somora
    )
    gromos.o3_x.mean(dim='time').plot(
        y='o3_p',
        ax=axs[0], 
        color=color_gromos
    )
    if sbuv:
        sbuv.ozone.mean(dim='time').plot(
            y='p',
            ax=axs[0], 
            color='k'
        )

    axs[0].set_title(r'O$_3$ VMR', fontsize=fs+4)
    axs[0].set_xlabel('VMR [ppmv]', fontsize=fs)

    pl1 = mean_diff_new.plot(
        y='o3_p',
        ax=axs[1],
        yscale='log',
        color='k'
    )
    
    axs[1].set_title(r'GROMOS-SOMORA', fontsize=fs+2)
    #axs[1].set_title(r'OG-SB', fontsize=fs+2)
    # pl2 = mean_diff_old.plot(
    #     y='altitude',
    #     ax=axs[1], 
       
    # )
    axs[1].axvline(x=0,ls= '--', color='grey')
    if sbuv:
        axs[0].legend(('SOMORA','GROMOS'))
    else:
        axs[0].legend(('SOMORA','GROMOS','SBUV'))
    #axs[0].legend(('SB corr','OG'))
    axs[0].invert_yaxis()
    axs[0].set_xlim(-0.2, 9)
    axs[0].set_ylabel('Pressure [hPa]', fontsize=fs)
    axs[1].set_ylabel('', fontsize=fs)
    axs[1].set_xlim((-60,60))
    axs[1].set_xlabel(r'$\Delta$O$_3$ [\%]', fontsize=fs)
    axs[0].set_ylim(100, 0.01)
    axs[0].xaxis.set_major_locator(MultipleLocator(4))
    axs[0].xaxis.set_minor_locator(MultipleLocator(1))
    axs[1].xaxis.set_major_locator(MultipleLocator(30))
    axs[1].xaxis.set_minor_locator(MultipleLocator(10))
    #axs[1].set_ylim((somora.o3_z.mean(dim='time')[12]/1e3,somora.o3_z.mean(dim='time')[35]/1e3))


    for ax in axs:
        ax.grid(which='both', axis='y', linewidth=0.5)
        ax.grid(which='both', axis='x', linewidth=0.5)
        ax.tick_params(axis='both', which='major', labelsize=fs-2)
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: '{:g}'.format(y)))
        ax.fill_between(ax.get_xlim(),p_somora_mr[0],1e4, color=color_shading, alpha=0.2)
        ax.fill_between(ax.get_xlim(),p_somora_mr[-1],1e-4, color=color_shading, alpha=0.2)
        ax.fill_between(ax.get_xlim(),p_gromos_mr[0],1e4, color=color_shading, alpha=0.2)
        ax.fill_between(ax.get_xlim(),p_gromos_mr[-1],1e-4, color=color_shading, alpha=0.2)

    #fig.suptitle('Ozone relative difference GROMOS-SOMORA')
    fig.tight_layout(rect=[0, 0.01, 0.99, 1])
    fig.savefig('/scratch/GROSOM/Level2/GROMORA_waccm/rel_diff'+str(year)+'.pdf', dpi=500)

# def compare_mean_diff(gromos, somora, gromora_old):
#     year=pd.to_datetime(gromos.time.data[0]).year
#     fig, axs = plt.subplots(1, 2, sharex=True, figsize=(10,10))

#     mean_diff_new = 100*(somora.o3_x.mean(dim='time') - gromos.o3_x.mean(dim='time') )/gromos.o3_x.mean(dim='time')
#     mean_altitude_gromos = 1e-3*gromos.o3_z.mean(dim='time')
#     mean_altitude_somora = 1e-3*somora.o3_z.mean(dim='time')

#     mean_diff_old = 100*(gromora_old.o3_somora.mean(dim='time') - gromora_old.o3_gromos.mean(dim='time') )/gromora_old.o3_gromos.mean(dim='time')
    
#     pl1 = mean_diff_new.plot(
#         y='o3_p',
#         ax=axs[0], 
#         yscale='log',
#     )
#     axs[0].set_title('New Routine')
#     # pl2 = mean_diff_old.plot(
#     #     y='altitude',
#     #     ax=axs[1], 
       
#     # )
#     pl2 = gromora_old.o3_rel_diff.mean(dim='time').plot(
#         y='altitude',
#         ax=axs[1], 
       
#     )
#     axs[1].set_title('Old Routine')
#     axs[1].plot(mean_diff_new.data,mean_altitude_gromos.data)
#     axs[1].plot(mean_diff_new.data, mean_altitude_somora.data)

#     axs[0].invert_yaxis()
#     axs[0].set_ylabel('P [hPa]')
#     axs[1].set_ylabel('Altitude [km]')
#     axs[0].set_xlim((-25,25))
#     axs[0].set_ylim((somora.o3_p[12],somora.o3_p[35]))
#     axs[1].set_ylim((somora.o3_z.mean(dim='time')[12]/1e3,somora.o3_z.mean(dim='time')[35]/1e3))


#     for ax in axs:
#         ax.grid()
#     fig.suptitle('Ozone relative difference SOMORA-GROMOS')
#     fig.tight_layout(rect=[0, 0.01, 0.95, 1])
#     fig.savefig('/scratch/GROSOM/Level2/GROMORA_retrievals_polyfit2/rel_diff_old_vs_new_'+str(year)+'.pdf', dpi=500)

def compare_mean_diff_alt(ozone_const_alt_gromos, ozone_const_alt_somora, gromora_old, ozone_const_alt_gromos_v2021):
    year=pd.to_datetime(ozone_const_alt_gromos.time.data[0]).year
    
    fig, axs = plt.subplots(1, 2, sharey=True, figsize=(15,10))

    mean_diff_new = 100*(ozone_const_alt_somora.o3_x.mean(dim='time') - ozone_const_alt_gromos.o3_x.mean(dim='time') )/ozone_const_alt_gromos.o3_x.mean(dim='time')
    mean_diff_old = 100*(gromora_old.o3_somora.mean(dim='time') - gromora_old.o3_gromos.mean(dim='time') )/gromora_old.o3_gromos.mean(dim='time')
    
    ozone_const_alt_gromos.o3_x.mean(dim='time').plot(
        y='altitude',
        ax=axs[0], 
        label='GROMOS, new'
    )
    ozone_const_alt_somora.o3_x.mean(dim='time').plot(
        y='altitude',
        ax=axs[0], 
        label='SOMORA, new'
    )
    gromora_old.o3_gromos.mean(dim='time').plot(
        y='altitude',
        ax=axs[0], 
        label='GROMOS, old'
    )
    gromora_old.o3_somora.mean(dim='time').plot(
        y='altitude',
        ax=axs[0], 
        label='SOMORA, old'
    )
    ozone_const_alt_gromos_v2021.o3_x.mean(dim='time').plot(
        y='altitude',
        ax=axs[0], 
        label='GROMOS, v2021'
    )
    axs[0].set_xlim((0,9))
    axs[0].legend()

    pl1 = mean_diff_new.plot(
        y='altitude',
        ax=axs[1], 
    )
    pl2 = mean_diff_old.plot(
        y='altitude',
        ax=axs[1], 
       
    )
    # pl2 = gromora_old.o3_rel_diff.mean(dim='time').plot(
    #     y='altitude',
    #     ax=axs[1], 
       
    # )
    axs[1].legend(('New Routine','Old Routine'))

    axs[1].set_ylabel('Altitude [km]')
    axs[1].set_xlim((-25,25))
    axs[1].set_ylim((10,90))
    axs[1].xaxis.set_major_formatter(5)

    #axs[1].set_ylim((somora.o3_z.mean(dim='time')[12]/1e3,somora.o3_z.mean(dim='time')[35]/1e3))
    for ax in axs:
        ax.grid(which='both')
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: '{:g}'.format(y)))

    fig.suptitle('Ozone relative difference SOMORA-GROMOS')
    fig.tight_layout(rect=[0, 0.01, 0.95, 1])
    fig.savefig('/scratch/GROSOM/Level2/GROMORA_retrievals_polyfit2/rel_diff_old_vs_new_'+str(year)+'.pdf', dpi=500)

def compare_diff_daily_altitude(gromos, somora, gromora_old, altitudes = [15,20,25]):
    year=pd.to_datetime(gromos.time.data[0]).year
    fig, axs = plt.subplots(len(altitudes), 1, sharex=True, figsize=(15,10))
    for i, alt in enumerate(altitudes):
        daily_diff = 100*(somora.o3_x.sel(altitude=alt).resample(time='1D').mean() - gromos.o3_x.sel(altitude=alt).resample(time='1D').mean())/gromos.o3_x.sel(altitude=alt).resample(time='D').mean()
        daily_diff.plot(ax=axs[i], color='b', marker ='.', lw=0.6, label='New retrieval routine')

        gromora_old.o3_rel_diff.sel(altitude=alt).plot(ax=axs[i], color='r', marker ='.',lw=0.6, label='Old retrieval routine')

        #daily_diff_old.plot(ax=axs[i], color='k', lw=0.6, label='Old')
        axs[i].set_xlabel('')
        axs[i].set_ylabel(r'$\Delta$O$_3$ [%]')
        axs[i].set_title(f'altitude = {alt:.0f} km')

    #axs[0].legend(['Old routine','New routine'])
    axs[0].legend()
    for ax in axs:
        ax.grid()
        ax.set_ylim((-40,40))
        ax.axhline(y=0,  ls='--', lw=0.85 , color='k')
        #ax.set_xlim("2018-01-01", "2018-05-31")
    #axs[0].set_ylim((-100,100))
    fig.suptitle('Ozone daily relative difference SOMORA-GROMOS')
    fig.tight_layout(rect=[0, 0.01, 0.95, 1])
    fig.savefig('/scratch/GROSOM/Level2/GROMORA_retrievals_polyfit2/ozone_comparison_old_vs_new_alt_'+str(year)+'.pdf', dpi=500)


def compare_altitude_old(gromora_old, altitudes = [15,20,25]):
    year=pd.to_datetime(gromora_old.time.data[0]).year

    fig, axs = plt.subplots(len(altitudes), 1, sharex=True, figsize=(15,10))
    for i, z in enumerate(altitudes):
        gromora_old.o3_gromos.sel(altitude=z).plot(ax=axs[i], color='b', lw=0.6)
        gromora_old.o3_somora.sel(altitude=z).plot(ax=axs[i], color='r', lw=0.6)
        #gromora_old.o3_diff.sel(altitude=z).plot(ax=axs[i], color='k', lw=0.6)
        axs[i].set_xlabel('')
        axs[i].set_ylabel('ozone [VMR]')
        axs[i].set_title(f'z = {z:.1f} km')

    axs[0].legend(['GROMOS','SOMORA'])
    for ax in axs:
        ax.grid()
    fig.tight_layout(rect=[0, 0.01, 0.95, 1])
    fig.savefig('/scratch/GROSOM/Level2/GROMORA_retrievals_polyfit2/ozone_comparison_altitude_old_'+str(year)+'.pdf', dpi=500)


def plot_diagnostics(gromos, level1b, instrument_name, freq='1D'):
    year=pd.to_datetime(gromos.time.data[0]).year
    fig, axs = plt.subplots(7, 1, sharex=True, figsize=(15,20))
    
    gromos.median_noise.resample(time=freq).mean().plot(ax=axs[0],color='k')
    level1b.noise_level.resample(time=freq).mean().plot(ax=axs[0],color='r')
    axs[0].set_xlabel('')
    axs[0].set_ylabel('Noise Level')
    #axs[0].set_ylim((0,2))
    
    gromos.oem_diagnostics[:,2].resample(time=freq).mean().plot(ax=axs[1],color='k')
    axs[1].set_xlabel('')
    axs[1].set_ylabel('Total cost')
    axs[1].set_ylim((0.9,1.3))

    gromos.oem_diagnostics[:,4].resample(time=freq).mean().plot(ax=axs[2],color='k')
    axs[2].set_xlabel('')
    axs[2].set_ylabel('# iter')
    axs[2].set_ylim((0,10))

    #ax3bis = axs[3].twinx()    
    level1b.noise_temperature.resample(time=freq).mean().plot(ax=axs[3],color='k')
    axs[3].set_xlabel('')
    axs[3].set_ylabel(r'$T_N$')
    axs[3].set_ylim((2500,3100))
    
    level1b.tropospheric_opacity.resample(time=freq).mean().plot(ax=axs[4],color='k')
    axs[4].set_xlabel('')
    axs[4].set_ylabel('tau')
    axs[4].set_ylim((0,2.2))

    ax5bis = axs[5].twinx()    
    level1b.TRoom.resample(time=freq).mean().plot(ax=axs[5], color='r')
    level1b.TWindow.resample(time=freq).mean().plot(ax=ax5bis, color='k')
    axs[5].set_xlabel('')
    axs[5].set_ylabel(r'$T_{room}$')
    axs[5].set_ylim((280,310))
    ax5bis.set_xlabel('')
    ax5bis.set_ylabel(r'$T_{window}$', color='red')
    ax5bis.set_ylim((280,310))

    for p_ind in [21,15]:
        monthly_mean_o3 = gromos.o3_x.isel(o3_p=p_ind).groupby('time.month').mean()
        anomalies = gromos.o3_x.isel(o3_p=p_ind).groupby('time.month') - monthly_mean_o3
        anomalies.plot(ax=axs[6], label=f'p = {gromos.o3_p.data[p_ind]:.3f} hPa')
    
    level1b.tropospheric_opacity.where(level1b.tropospheric_transmittance<0.15).resample(time=freq).mean().plot(ax=axs[6],color='k')
    axs[6].set_xlabel('')
    axs[6].set_title('')
    axs[6].legend(loc=4, fontsize=16)
    axs[6].set_ylabel(r'$\Delta$O$_3$ VMR')
    #axs[6].set_ylim((0,10))
    
    for ax in axs:
        ax.grid()
    fig.tight_layout(rect=[0, 0.01, 0.99, 1])

    fig.savefig('/scratch/GROSOM/Level2/GROMORA_waccm/'+instrument_name+'_diagnostics_'+str(year)+'.pdf', dpi=500)

def plot_pressure(gromos, pressure_level = [15,20,25], add_sun=False):
    year=pd.to_datetime(gromos.time.data[0]).year
    fig, axs = plt.subplots(len(pressure_level), 1, sharex=True, figsize=(15,10))
    for i, p in enumerate(pressure_level):
        gromos.o3_x.isel(o3_p=p).plot(ax=axs[i], color='b', lw=0.6)
        axs[i].set_xlabel('')
        axs[i].set_title(f'p = {gromos.o3_p.data[p]:.3f} hPa')
        if add_sun:
            #for s, sr in enumerate(gromos.sunrise.data):
            #sr = gromos.sunrise.data
            sunrise = gromos.sunrise.resample(time='1D').max()
            sunset = gromos.sunset.resample(time='1D').max()
            
            for d in range(len(sunrise.data)):
                sr=sunrise.data[d]
                ss=sunset.data[d]
                axs[i].axvspan(sr,ss, color='orange', alpha=0.2)

            #     #axs[i].axvline(sunr, color='k', linestyle='-.')
            # for suns in sunset.data:
            #     axs[i].axvline(suns, color='k', linestyle='--')

    #axs[0].legend(['GROMOS','SOMORA'])
    for ax in axs:
        ax.grid()
    fig.tight_layout(rect=[0, 0.01, 0.95, 1])
    fig.savefig('/scratch/GROSOM/Level2/GROMORA_retrievals_polyfit2/GROMOS_ozone_pressure_level_'+str(year)+'.pdf', dpi=500)

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
    fig.tight_layout(rect=[0, 0.01, 0.99, 1])

    fig.savefig('/scratch/GROSOM/Level2/GROMORA_retrievals_polyfit2/GROMOS_fshift_ts.pdf', dpi=500)

def compare_avkm(gromos, somora, date_slice):
    fig, axs = plt.subplots(1, 2, sharey=True, figsize=(10,10))
    avkm_gromos = gromos.o3_avkm.sel(time=date_slice)
    mean_avks_gromos= avkm_gromos.mean(dim='time')
    avkm_somora = somora.o3_avkm.sel(time=date_slice)
    mean_avks_somora = avkm_somora.mean(dim='time')
    p = mean_avks_gromos.o3_p.data

    mean_MR_gromos=0.25*gromos.o3_mr.sel(time=date_slice).mean(dim='time')
    mean_MR_somora=0.25*somora.o3_mr.sel(time=date_slice).mean(dim='time')
    good_p_gromos = gromos.o3_p.where(mean_MR_gromos>0.2,drop=True)
    good_p_somora = somora.o3_p.where(mean_MR_somora>0.2,drop=True)

    # somora.o3_avkm.sel(time=slice(str(yr)+'-01-01',str(yr)+'-06-30')).mean(dim='time').plot(
    #     ax=axs[0],
    #     y='o3_p',
    #     yscale='log',
    # )
    # gromos.o3_avkm.sel(time=slice(str(yr)+'-01-01',str(yr)+'-06-30')).mean(dim='time').plot(
    #     ax=axs[1],
    #     y='o3_p',
    #     yscale='log',
    # )
    counter = 0
    color_count = 0
    for avk in mean_avks_gromos:
        #if 0.8 <= np.sum(avk) <= 1.2:
        counter=counter+1
        if np.mod(counter,8)==0:
            axs[0].plot(avk, p, color=cmap(color_count*0.25+0.01), lw=2)
            color_count = color_count +1            
        else:
            axs[0].plot(avk, p, color='k')
    mean_MR_gromos.plot(ax=axs[0],y='o3_p', color='red')
    axs[0].axhline(y=good_p_gromos[0], ls='--', color='red')
    axs[0].axhline(y=good_p_gromos[-1], ls='--', color='red')
    axs[0].set_title('AVKs GROMOS')
    counter = 0
    color_count = 0
    for avk in mean_avks_somora:
        #if 0.8 <= np.sum(avk) <= 1.2:
        counter=counter+1
        if np.mod(counter,8)==0:
            axs[1].plot(avk, p, color=cmap(color_count*0.25+0.01), lw=2)
            color_count = color_count +1  
        else:
            axs[1].plot(avk, p, color='k')
    mean_MR_somora.plot(ax=axs[1], y='o3_p', color='red')
    axs[1].axhline(y=good_p_somora[0], ls='--', color='red')
    axs[1].axhline(y=good_p_somora[-1], ls='--', color='red')
    axs[0].invert_yaxis()
    axs[0].set_yscale('log')
    axs[1].set_title('AVKs SOMORA')

    for ax in axs:
        ax.grid()
    fig.tight_layout(rect=[0, 0.01, 0.95, 1])
    fig.savefig('/scratch/GROSOM/Level2/GROMORA_waccm/AVKs_comparison_'+str(avkm_gromos.time.data[0])[0:10]+'.pdf', dpi=500)

def read_opacity(folder, year=2014):
    gromos_opacity = xr.open_dataset(
            folder+'GROMOS/GROMOS_opacity_'+str(year)+'.nc',
            group='spectrometer1',
            decode_times=True,
            decode_coords=True,
            # use_cftime=True,
        )
    somora_opacity = xr.open_dataset(
            folder+'SOMORA/SOMORA_opacity_'+str(year)+'.nc',
            group='spectrometer1',
            decode_times=True,
            decode_coords=True,
            # use_cftime=True,
        )
    return gromos_opacity, somora_opacity

def read_level1(folder, filename):
    level1 = xr.open_dataset(
        os.path.join(folder,filename),
        #group='spectrometer1',
        decode_times=True,
        decode_coords=True,
        # use_cftime=True,
    )
    return level1

def compare_opacity(folder, year=2014, date_slice=slice("2014-01-01", "2014-01-31")):
    gromos_opacity, somora_opacity = read_opacity(folder, year=year)

    gromos_opacity = gromos_opacity.sel(time=date_slice)
    somora_opacity = somora_opacity.sel(time=date_slice)

    fig, axs = plt.subplots(3, 1, sharex=True, figsize=(15,10))
    gromos_opacity.tropospheric_opacity.resample(time='4H').mean().plot(
        ax=axs[0]
    )
    somora_opacity.tropospheric_opacity.resample(time='4H').mean().plot(
        ax=axs[0]
    )
    if year>2013:
        gromos_opacity.tropospheric_opacity_tc.plot(
            lw=0,
            marker='.',
            ms=0.5,
            ax=axs[0]
        )
        somora_opacity.tropospheric_opacity_tc.plot(
            lw=0,
            marker='.',
            ms=0.5,
            ax=axs[0]
        )
    axs[0].set_ylabel('opacity')
    axs[0].set_ylim((0,2))
    axs[0].legend(['GROMOS','SOMORA','GROMOS TC', 'SOMORA_TC'])
    gromos_opacity.tropospheric_transmittance.resample(time='4H').mean().plot(
        ax=axs[1]
    )
    somora_opacity.tropospheric_transmittance.resample(time='4H').mean().plot(
        ax=axs[1]
    )
    axs[1].set_ylabel('transmittance')
    axs[1].set_ylim((-0.01,1))

    rel_diff = (gromos_opacity.tropospheric_opacity.resample(time='4H').mean() - somora_opacity.tropospheric_opacity.resample(time='4H').mean()) 
    rel_diff.resample(time='4H').mean().plot(
        ax=axs[2]
    )
    axs[2].axhline(y=0, ls='--', lw=0.8 , color='k')
    axs[2].set_ylabel('opacity difference')
    axs[2].legend(['GRO - SOM'])
    axs[2].set_ylim((-1,1))

    for ax in axs:
        ax.grid()
    fig.tight_layout(rect=[0, 0.01, 0.95, 1])
    fig.savefig('/scratch/GROSOM/Level2/GROMORA_retrievals_polyfit2/opactiy_comparison_'+str(year)+'.pdf', dpi=500)


if __name__ == "__main__":
    instrument = 'GROMOS'
   # ds_fshift = read_plot_fshift(basefolder='/scratch/GROSOM/Level2/GROMORA_waccm/',instrument_name=instrument,years=[2015, 2016, 2017, 2018, 2019])
    
    ds_fshift = read_fshift_nc(basename='/scratch/GROSOM/Level2/GROMORA_waccm/GROMOS_',years=[2016])


    TRoom = xr.open_dataset(
            '/scratch/GROSOM/Level2/GROMORA_retrievals_polyfit2/'+'GROMOS_TRoom_'+str(2016)+'.nc',
            group='spectrometer1',
            decode_times=True,
            decode_coords=True,
            # use_cftime=True,
        )

    filename_gromos = '/storage/tub/instruments/gromos/level2/GROMORA/v1/GROMOS_2016_waccm_monthly_scaled_h2o_ozone.nc'
    filename_gromos = '/storage/tub/instruments/gromos/level2/GROMORA/v1/2018_waccm_cov_yearly_ozone.nc'

    filename_somora = '/storage/tub/instruments/somora/level2/v1/SOMORA2018_06_30_waccm_cov_yearly_ozone.nc'
    
    filename_gromos='/scratch/GROSOM/Level2/GROMORA_waccm/GROMOS_2016_12_31_waccm_low_alt_ozone.nc'
    filename_somora='/scratch/GROSOM/Level2/GROMORA_waccm/SOMORA_2016_12_31_waccm_low_alt_ozone.nc'
    

    yr = 2016
    date_slice=slice(str(yr)+'-06-01',str(yr)+'-06-14')
    #date_slice=slice('2010-01-01','2020-12-31')
    #gromos = read_GROMORA(filename_gromos, date_slice)
    #somora = read_GROMORA(filename_somora, date_slice)
    prefix_gromos = [
        '_waccm_low_alt_dx10_ozone_ony.nc',
        '_waccm_low_alt_ozone.nc',
        '_waccm_low_alt_ozone.nc',
        '_waccm_low_alt_ozone.nc',
        '_waccm_low_alt_ozone.nc',
        '_waccm_low_alt_ozone.nc',
        '_waccm_low_alt_ozone.nc',
        #'_waccm_low_alt_dx10_v2_SB_ozone.nc',
        '_waccm_low_alt_ozone.nc',
        '_waccm_low_alt_ozone.nc',
        '_waccm_low_alt_ozone.nc',
        '_waccm_low_alt_ozone.nc'
    ]
    
    prefix_somora = [
        '_waccm_low_alt_ozone.nc',
        '_waccm_low_alt_ozone.nc',
        '_waccm_low_alt_ozone.nc',
        '_waccm_low_alt_ozone.nc',
        '_waccm_low_alt_ozone.nc',
        '_waccm_low_alt_ozone.nc',
        '_waccm_low_alt_dx10_sinefit_ozone.nc',
        '_waccm_low_alt_ozone.nc',
        '_waccm_low_alt_ozone.nc',
        '_waccm_low_alt_dx10_sinefit_ozone.nc',
        '_waccm_low_alt_ozone.nc'
    ]

    instrument = 'comp'
    if instrument == 'GROMOS':
        instNameGROMOS = 'GROMOS'
        instNameSOMORA = 'GROMOS'
        fold_somora = '/storage/tub/instruments/gromos/level2/GROMORA/v1/'
        fold_gromos =  '/storage/tub/instruments/gromos/level2/GROMORA/v1/'
    elif instrument == 'SOMORA':
        instNameGROMOS = 'SOMORA'
        instNameSOMORA = 'SOMORA'
        fold_somora ='/storage/tub/instruments/somora/level2/v1/'
        fold_gromos ='/storage/tub/instruments/somora/level2/v1/'
    else:
        instNameGROMOS = 'GROMOS'
        instNameSOMORA = 'SOMORA'
        fold_somora = '/storage/tub/instruments/somora/level2/v1/'
        fold_gromos = '/storage/tub/instruments/gromos/level2/GROMORA/v1/'
    #basefolder=_waccm_low_alt_dx10_v2_SB_ozone
    gromos = read_GROMORA_all(basefolder=fold_gromos, 
    instrument_name=instNameGROMOS,
    date_slice=date_slice, 
    years=[yr],#,#[yr],#[],#[2010, 2011, 2012,2013,2014,2015,2016,2017,2018,2019,2020]
    prefix='_waccm_low_alt_dx10_v2_SB_ozone.nc'
    )
    somora = read_GROMORA_all(basefolder=fold_somora, 
    instrument_name=instNameSOMORA,
    date_slice=date_slice, 
    years=[yr],#[2010, 2011, 2012,2013,2014,2015,2016,2017,2018,2019,2020],#[2011, 2012,2013,2014,2015,2016,2017,2018,2019,2020],
    prefix='_waccm_low_alt_ozone.nc'
    )

    bn = '/home/esauvageat/Documents/GROMORA/Data/SBUV/'
    sbuv = read_SBUV_dailyMean(date_slice, SBUV_basename = bn, specific_fname='sbuv_v87.mod_v2r1.vmr.payerne_156.txt')
    sbuv_arosa = read_SBUV_dailyMean(date_slice, SBUV_basename = bn, specific_fname='sbuv_v87.mod_v2r1.vmr.arosa_035.txt')

    mls= read_MLS(date_slice)
    gromos['o3_x'] = 1e6*gromos['o3_x']
    # flags1a, flags1b= read_level1_flags(instrument=instrument, year=yr, suffixe='')

    #compare_ts(gromos, somora, freq='1D', date_slice=slice('2021-01-01','2021-12-31'), basefolder='/scratch/GROSOM/Level2/GROMORA_waccm/')
    
    #plot_ozone_ts(gromos, altitude=False)

    #compare_ts_MLS(gromos, somora, date_slice=date_slice, freq='1H', basefolder='/scratch/GROSOM/Level2/GROMORA_waccm/', sbuv=sbuv)

    #gromos_opacity, somora_opacity = read_opacity(folder='/scratch/GROSOM/Level2/opacities/', year=yr)

    # plot_ozone_flags('SOMORA', somora, flags1a=flags1a, flags1b=flags1b, pressure_level=[27, 12], opacity = somora_opacity, calib_version=1)
    # plot_ozone_flags('GROMOS', gromos, flags1a=flags1a, flags1b=flags1b, pressure_level=[27, 12], opacity = gromos_opacity, calib_version=1)


    #map_rel_diff(gromos,somora, freq='1m', basefolder='/scratch/GROSOM/Level2/GROMORA_waccm/')

    #gromora_old = read_old_GROMOA_diff('DIFF_G_2017', date_slice)
    #gromos_v2021 = read_gromos_v2021('gromosplot_ffts_select_v2021', date_slice)
    
    #gromos_old_vs_new(gromos, gromos_v2021)
    
    #gromos = utc_to_lst(gromos)
    #somora = utc_to_lst(somora)

    gromos_clean = gromos.where(abs(1-gromos.oem_diagnostics[:,2])<0.2).where(gromos.o3_mr>0.8)
    somora_clean = somora.where(abs(1-somora.oem_diagnostics[:,2])<0.1).where(somora.o3_mr>0.8)

    compare_pressure(gromos_clean, somora_clean, pressure_level=[31, 25, 21, 15, 12], add_sun=False, freq='1H', basefolder='/scratch/GROSOM/Level2/GROMORA_waccm/')
    #compare_pressure_mls_sbuv(gromos_clean, somora_clean, mls, sbuv, pressure_level=[31, 25, 21, 15, 12], add_sun=False, freq='1H', basefolder='/scratch/GROSOM/Level2/GROMORA_waccm/')

   # # #compare_altitude_old(gromora_old, altitudes=[69, 63, 51, 42, 30, 24])
    #compare_avkm(gromos, somora, date_slice=slice('2016-03-01','2016-03-31'))
    #compare_avkm(gromos, somora, date_slice=slice('2016-10-01','2016-10-31'))
    # #compare_diff_daily(gromos ,somora, gromora_old, pressure_level=[34 ,31, 25, 21, 15, 12], altitudes=[69, 63, 51, 42, 30, 24])
    compare_mean_diff(gromos, somora, sbuv = None)

    #compare_with_apriori(gromos, freq='1H', date_slice=date_slice,basefolder='/scratch/GROSOM/Level2/GROMORA_waccm/GROMOS_')
    #compare_with_apriori(somora, freq='1H', date_slice=date_slice,basefolder='/scratch/GROSOM/Level2/GROMORA_waccm/SOMORA_')

    p_low=20
    p_high = 60

    plot_o3_pressure_profile(gromos)
    # altitude = gromos_clean.o3_z.mean(dim='time').where(gromos_clean.o3_p<p_high, drop=True).where(gromos_clean.o3_p>p_low, drop=True)
    # mean_diff_clean = (gromos_clean.o3_x.mean(dim='time') - somora_clean.o3_x.mean(dim='time') )#/gromos.o3_x.mean(dim='time')
    # mean_diff_prange = mean_diff_clean.where(gromos_clean.o3_p<p_high, drop=True).where(gromos_clean.o3_p>p_low, drop=True)
    # mean_gromos_clean_prange = gromos_clean.o3_x.mean(dim='time').where(gromos_clean.o3_p<p_high, drop=True).where(gromos_clean.o3_p>p_low, drop=True)
    
    # rel_diff_mean = np.mean(100*(mean_diff_prange/mean_gromos_clean_prange))
    # print(rel_diff_mean.values)

    #plot_pressure(gromos, pressure_level=[44, 33 ,30, 23, 16,13, 8], add_sun=False)
    #plot_pressure(somora, pressure_level=[44, 33 ,30, 23, 16,13, 8], add_sun=True)

    level1b_gromos = read_level1('/storage/tub/instruments/gromos/level1/GROMORA/v2','GROMOS_level1b_v2.nc')
    #level1b_gromos2 = read_level1('/scratch/GROSOM/Level1','GROMOS_level1b_2015.nc')
    #level1b_somora = read_level1('/scratch/GROSOM/Level1','SOMORA_level1b_v2.nc')
    #level1b_somora_all = xr.concat([level1b_gromos, level1b_gromos2], dim = 'time')
    #level1b_somora_all.to_netcdf('/scratch/GROSOM/Level1/GROMOS_level1b_v2_all.nc')
    
    plot_diagnostics(gromos, level1b_gromos.sel(time=slice('2009-07-01','2020-12-31')), instrument_name='GROMOS', freq='12H')
    #plot_diagnostics(somora, level1b_somora.sel(time=date_slice), instrument_name='SOMORA', freq='2H')
    #compute_correlation(gromos_clean, somora_clean, freq='1D', pressure_level=[25, 15], basefolder='/scratch/GROSOM/Level2/GROMORA_waccm/')
    #compute_corr_profile(gromos_clean, somora_clean, freq='7D', basefolder='/scratch/GROSOM/Level2/GROMORA_waccm/')


    #plot_fshift_ts(ds_fshift, date_slice = slice("2016-08-01", "2016-08-15"), TRoom=TRoom)
   # ds_fshift.freq_shift_x.sel(time=slice("2017-01-01", "2018-12-31")).resample(time='12H').mean().plot()
    # plt.matshow(gromos.o3_avkm.isel(time=0))
    # plt.colorbar()
    # 

    # #z_grid = np.arange(1e3, 90e3, 1e3)
    # z_grid = gromora_old.altitude.data
    # ozone_const_alt_gromos = constant_altitude_gromora(gromos, z_grid)
    
    # #ozone_const_alt_gromos_v2021 = constant_altitude_gromora(gromos_v2021, z_grid)

    # ozone_const_alt_somora = constant_altitude_gromora(somora, z_grid)

    # compare_mean_diff_alt(ozone_const_alt_gromos, ozone_const_alt_somora, gromora_old, ozone_const_alt_gromos_v2021)

    # compare_diff_daily_altitude(ozone_const_alt_gromos,ozone_const_alt_somora, gromora_old, altitudes=[63, 51, 42, 30, 24])
    
    compare_opacity(folder='/scratch/GROSOM/Level2/opacities/', year=2016, date_slice=slice("2016-06-01", "2016-06-30"))

    #plot_ozone_ts(gromos, altitude=False)
    # plot_ozone_ts(ozone_const_alt, altitude=True)

# %%
