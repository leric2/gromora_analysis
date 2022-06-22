#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on 17.03.22

@author: Eric Sauvageat

Main code for level2 diagnostics of the GROMORA v2 files.

This module contains only the code used to plots and analyse diagnostics quantities for the
GROMORA L2 files.

"""

#%%
from calendar import month_abbr, month_name
import os
from re import A
from typing import ValuesView
from matplotlib import units
import warnings

import matplotlib.pyplot as plt
import netCDF4
import numpy as np
from numpy.lib.shape_base import dsplit
import pandas as pd
from scipy.odr.odrpack import RealData
from scipy.stats.stats import RepeatedResults
from secretstorage import search_items
import typhon

import xarray as xr
from scipy import stats
from scipy.odr import *
from GROMORA_harmo.scripts.retrieval.gromora_time import get_LST_from_GROMORA,datetime64_2_datetime,mjd2k_date,gromora_tz

from flags_analysis import read_level1_flags
from base_tool import save_single_pdf, get_color

from matplotlib.ticker import (MultipleLocator, FormatStrFormatter, AutoMinorLocator)
from matplotlib.lines import Line2D
import matplotlib.ticker as ticker
import matplotlib
cmap = matplotlib.cm.get_cmap('plasma')

cmap_ts = 'cividis'


plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.sans-serif": ["Free sans"]})

plt.rcParams['xtick.labelsize'] = 22
plt.rcParams['ytick.labelsize'] = 22
plt.rcParams['font.size'] = 24
plt.rcParams['axes.titlesize'] = 24

MONTH_STR = ['Jan', 'Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=RuntimeWarning)

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

def read_GROMORA_all(basefolder, instrument_name, date_slice, years, prefix, flagged):
    counter = 0
    for i, y in enumerate(years):
        if flagged:
            filename = basefolder+instrument_name+'_level2_'+str(y)+'.nc'
        else:
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
    if not flagged:
        print('Changed pressure unit to hPa')
        gromora_ds['o3_p'] = gromora_ds['o3_p']/100
        gromora_ds['o3_p'].attrs['standard_name'] = 'pressure'
        gromora_ds['o3_p'].attrs['long_name'] = 'pressure'
        gromora_ds['o3_p'].attrs['units'] = 'hPa'
        gromora_ds['o3_p'].attrs['description'] = 'pressure of the ozone retrievals'

    pandas_time_gromos = pd.to_datetime(gromora_ds.time.data)

    gromora_ds = gromora_ds.sel(time=date_slice)
    return gromora_ds

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

def plot_ozone_ts(gromora, instrument_name, freq = '2H', altitude=False, add_MR=True, basefolder='/scratch/GROSOM/'):
    year = pd.to_datetime(gromora.time.values[0]).year

    fs = 25
    resampled_gromora = gromora.resample(time=freq).mean()
    
    if altitude:
        fig, axs = plt.subplots(1, 1, sharex=True)
        pl = resampled_gromora.o3_x.plot(
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
        fig, axs = plt.subplots(1, 1, sharex=True,  figsize=(16,8))
        pl = resampled_gromora.o3_x.plot(
            x='time',
            y='o3_p',
            ax=axs, 
            vmin=0,
            vmax=10,
            yscale='log',
            linewidth=0,
            rasterized=True,
            cmap=cmap_ts,
            add_colorbar=False
        )
        pl.set_edgecolor('face')
        # ax.set_yscale('log')
        axs.invert_yaxis()
        axs.set_ylabel('Pressure [hPa]', fontsize=fs)

        # axs[1].set_title('SOMORA', fontsize=fs+2)

        cbaxes = fig.add_axes([0.92, 0.25, 0.02, 0.5]) 
   # cb = plt.colorbar(pl, cax=cbaxes, orientation="vertical", pad=0.0)
        cb = fig.colorbar(pl, cax=cbaxes, orientation="vertical", pad=0.0)
        cb.set_label(label=r"O$_3$ [ppmv]", fontsize=fs)
        cb.ax.tick_params()
        if add_MR:
            resampled_gromora = resampled_gromora.resample(time='12H').mean()
            p_MR = np.ones((len(resampled_gromora.time), 2))
            for i, t in enumerate(resampled_gromora.time.data):
                profile = resampled_gromora.isel(time=i)
                mr = profile.o3_mr.data
                p = profile.o3_p.data
                try:
                    p_MR[i,:] = [p[np.where(mr>0.8)][0], p[np.where(mr>0.8)][-1]] 
                except:
                    p_MR[i,:] =[np.nan, np.nan]  
            axs.plot(resampled_gromora.time.data, p_MR,'-w', lw=0.75 )

        axs.set_ylim(100, 1e-2)
        axs.set_xlabel('')
        axs.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: '{:g}'.format(y)))
        plt.tight_layout(rect=[0, 0.01, 0.92, 1])

    fig.savefig(basefolder+instrument_name+'_ozone_time_series_'+str(year)+'.pdf', dpi=500)

def plot_ozone_flags(instrument, gromora, level1b, flags1a, flags1b, pressure_level=[27, 12], calib_version=2):
    year = pd.to_datetime(gromora.time.values[0]).year
    figures = list()
    # fig, axs = plt.subplots(4, 1, sharex=True, figsize=(10,15))
    # pl = gromora.o3_x.resample(time='1H').mean().plot(
    #     x='time',
    #     y='o3_p',
    #     ax=axs[0], 
    #     vmin=0,
    #     vmax=10,
    #     yscale='log',
    #     linewidth=0,
    #     rasterized=True,
    #     cmap='cividis',
    #     add_colorbar=False
    # )
    # pl.set_edgecolor('face')
    # # ax.set_yscale('log')
    # axs[0].invert_yaxis()
    # axs[0].set_ylabel('P [hPa]')

    # for i, p in enumerate(pressure_level):
    #     color = ['r','b']
    #     gromora.o3_x.isel(o3_p=p).resample(time='1H').mean().plot(ax=axs[1], color=color[i], lw=1.5, label=f'p = {gromora.o3_p.data[p]:.3f} hPa' )
    #     axs[1].set_xlabel('')
    #     axs[1].legend()
    
    # axs[1].set_title('')
    # axs[1].set_ylabel('o3')

    # cal_flags = flags1a.calibration_flags 
    # sum_flags_level1a = cal_flags.sum(axis=1)
    # sum_flags_level1a.resample(time='1H').mean().plot(ax=axs[2])
    # time = cal_flags.time
    # # for i in range(len(cal_flags.flags)):
    # #     cal_flags[:,i] = 0.5*(i+cal_flags[:,i])
    # # cal_flags[:,0].resample(time='1H').mean().plot(ax = axs[2], label=cal_flags.attrs['errorCode_1'])
    # # cal_flags[:,1].resample(time='1H').mean().plot(ax = axs[2], label=cal_flags.attrs['errorCode_2'])
    # # cal_flags[:,2].resample(time='1H').mean().plot(ax = axs[2], label=cal_flags.attrs['errorCode_3'])
    # # cal_flags[:,3].resample(time='1H').mean().plot(ax = axs[2], label=cal_flags.attrs['errorCode_4'])
    # # cal_flags[:,4].resample(time='1H').mean().plot(ax = axs[2], label=cal_flags.attrs['errorCode_5'])
    # # cal_flags[:,5].resample(time='1H').mean().plot(ax = axs[2], label=cal_flags.attrs['errorCode_6'])
    # sum_flags_level1a.plot(ax=axs[2] )
    # axs[2].set_ylabel('')
    # axs[2].set_title('Flag sum')
    # # if calib_version==2:
    # #     cal_flags[:,6].resample(time='1H').mean().plot(ax = axs[2], label=cal_flags.attrs['errorCode_7'])

    # # axs[1+len(pressure_level)].plot(time[np.where(cal_flags[:,0]==0)], 6+cal_flags[:,0][np.where(cal_flags[:,0]==0)], '-', label=cal_flags.attrs['errorCode_1'])
    # # axs[1+len(pressure_level)].plot(time[np.where(cal_flags[:,1]==0)], 5+cal_flags[:,1][np.where(cal_flags[:,1]==0)], '-', label=cal_flags.attrs['errorCode_2'])
    # # axs[1+len(pressure_level)].plot(time[np.where(cal_flags[:,2]==0)], 4+cal_flags[:,2][np.where(cal_flags[:,2]==0)], '-', label=cal_flags.attrs['errorCode_3'])
    # # axs[1+len(pressure_level)].plot(time[np.where(cal_flags[:,3]==0)], 3+cal_flags[:,3][np.where(cal_flags[:,3]==0)], '-', label=cal_flags.attrs['errorCode_4'])
    # # axs[1+len(pressure_level)].plot(time[np.where(cal_flags[:,4]==0)], 2+cal_flags[:,4][np.where(cal_flags[:,4]==0)], '-', label=cal_flags.attrs['errorCode_5'])
    # # axs[1+len(pressure_level)].plot(time[np.where(cal_flags[:,5]==0)], 1+cal_flags[:,5][np.where(cal_flags[:,5]==0)], '-', label=cal_flags.attrs['errorCode_6'])
    # # axs[1+len(pressure_level)].plot(time[np.where(cal_flags[:,6]==0)], 1+cal_flags[:,6][np.where(cal_flags[:,6]==0)], '-', label=cal_flags.attrs['errorCode_7'])
    # # axs[1+len(pressure_level)].set_ylim(0,7.4)
    # axs[2].legend(loc='lower right', fontsize=6)
    # # flags1a.calibration_flags.plot(
    # #     x='time',
    # #     y='o3_p',
    # #     ax=axs[1+len(pressure_level)]
    # # )
   
    # # flags1b.calibration_flags[:,1].resample(time='1H').mean().plot(
    # #     ax=axs[2+len(pressure_level)]
    # #     )
    # level1b.tropospheric_opacity.resample(time='1H').mean().plot(ax=axs[3])
    # axs[3].set_ylabel('Opacity')

    # for ax in axs:
    #     ax.set_xlabel('')
    #     ax.grid()
    # plt.tight_layout(rect=[0, 0.01, 0.99, 1])

    # figures.append(fig)
    # fig.savefig(outfolder+instrument+'_'+str(year)+'_calibration_flags.pdf', dpi=500)

    fig, axs = plt.subplots(2, 1, sharex=True, figsize=(15,10))
    ax = axs[0].twinx()
    sum_flags = flags1a.calibration_flags.sum(dim='flags')
    sum_flags.resample(time='1D').mean().plot(
        ax=ax,
        color='red',
        alpha=0.5
    )
    ax.set_ylabel('flags sum', color='red')
    ax.set_ylim(0,7.1)
    ax.yaxis.label.set_color('red')
    ax.tick_params(axis='y', colors='red')

    flags1a.calibration_flags.plot(
        ax=axs[0],
        x='time',
        rasterized=True,
        cmap='gist_gray',
        add_colorbar=False
    )

    axs[0].set_xlabel('')
    axs[0].set_ylabel('')
    axs[0].set_yticks([1, 2, 3, 4, 5, 6, 7])
    axs[0].set_yticklabels(['Cycles','TN','LN2 sensor','LN2 level','THot','Angle','FreqLock'] )
    axs[0].set_title('Flags level 1a') 
    # figures.append(fig)
   # fig.savefig(outfolder+instrument+'_'+str(year)+'_calibration_flags_contour.pdf', dpi=500)

    # fig, ax = plt.subplots(1, 1, sharex=True, figsize=(15,10))
    flags1b.calibration_flags.plot(
        ax=axs[1],
        x='time',
        rasterized=True,
        cmap='gist_gray',
        add_colorbar=False
    )
    axs[1].set_xlabel('')
    axs[1].set_ylabel('')
    axs[1].set_yticks([1, 2])
    axs[1].set_yticklabels(['Cycles', 'Opacity'] )
    axs[1].set_title('Flags level 1b') 
    plt.tight_layout(rect=[0, 0.01, 0.99, 1])
    figures.append(fig)

    save_single_pdf(outfolder+instrument+'_'+str(year)+'_calibration_flags.pdf', figures)

def compare_with_apriori(gromos, instrument_name, freq, date_slice, basefolder):
    '''
    Simple function to compare retrievals vs apriori
    '''
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
        vmin=-1,
        vmax=1,
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

    fig.savefig(basefolder+instrument_name+'_apriori_comparison'+str(year)+'.pdf', dpi=500)

def fshift_daily_cycle(gromos, instrument_name, date_slice, outfolder):
    '''
    Ugly ! Should be changed to use group_by method !!
    '''
    fshift = gromos.sel(time=date_slice).freq_shift_x.isel(f_shift_grid=0)
    year = pd.to_datetime(fshift.time.values).year[0]

    # gromos['tod'] = pd.to_datetime(gromos.time.values).hour
    gromos['month'] = pd.to_datetime(gromos.time.values).month
    daily_fshift = np.ones((12,23))*np.nan
    fig, axs = plt.subplots(1, 1)
    for m in range(0,12):
        for i in range(0,23):
            fshift_month = fshift.where(gromos['month'].values == m)
            daily_fshift[m,i] = 1e-3*np.nanmean(fshift_month.where(pd.to_datetime(fshift_month.time.values).hour == i)) 

        #axs.plot(np.arange(0,23), daily_fshift[m,:], label= MONTH_STR[m], lw=0.75)
    axs.plot(np.arange(0,23), np.nanmean(daily_fshift[[1,2,11],:], axis=0), label= 'DJF')
    axs.plot(np.arange(0,23), np.nanmean(daily_fshift[[3,4,5],:], axis=0), label= 'MAM')
    axs.plot(np.arange(0,23), np.nanmean(daily_fshift[[6,7,8],:], axis=0), label= 'JJA')
    axs.plot(np.arange(0,23), np.nanmean(daily_fshift[[9,10,11],:], axis=0), label= 'SON')
    axs.plot(np.arange(0,23), np.nanmean(daily_fshift, axis=0), 'k-x' ,label= 'Yearly')
    axs.grid()
    axs.set_xlabel('time of day')
    axs.set_ylabel('fshift [kHz]')
    axs.set_ylim((-500, 500))
    axs.legend(bbox_to_anchor=(0.95, 0.95))
    fig.tight_layout(rect=[0, 0.01, 0.95, 1])
    fig.savefig(outfolder+instrument_name+'_daily_fshift_'+str(year)+'.pdf', dpi=500)
    
def retrievals_diagnostics(gromos, level1b, instrument_name, freq='1D', outfolder='/scratch/GROSOM/Level2/GROMORA_retrievals_v2/'):
    year=pd.to_datetime(gromos.time.data[0]).year
    figure=list()
    fig, axs = plt.subplots(3, 1, sharex=True, figsize=(13,10))
    
    gromos.median_noise.resample(time=freq).mean().plot(ax=axs[0],color='k')
    #gromos.retrieval_quality[:,0].resample(time=freq).mean().plot(ax=axs[0], marker='x', color='k')
    #level1b.noise_level.resample(time=freq).mean().plot(ax=axs[0],color='r')
    axs[0].set_ylabel('Noise Level')
    axs[0].set_ylim((0,1))
    
    gromos.oem_diagnostics[:,2].resample(time=freq).mean().plot(ax=axs[1],color='k', label='cost')

    gromos.oem_diagnostics[:,4].resample(time=freq).mean().plot(ax=axs[2], marker='.', linewidth=0,color='k')
    axs[2].set_ylabel('Iterations')
    axs[2].set_ylim((0,10))

    # quality = xr.where(gromos.retrieval_quality.resample(time=freq).mean() == 1, x=np.nan, y=1)
    # quality.plot(ax=axs[1], marker='x', markersize=5, linewidth=0, color='r', label='L2 flags')
    axs[1].set_ylabel('Total cost')
    axs[1].set_ylim((0.9,1.3))
    # axs[1].legend()
    # axs[3].set_ylabel('L2 flags')
    # axs[3].set_title('Retrievals quality')

    for ax in axs:
        ax.grid()
        ax.set_xlabel('')
    fig.tight_layout(rect=[0, 0.01, 0.99, 1])

    figure.append(fig)
   # fig.savefig(outfolder+instrument_name+'_retrievals_diagnostics_'+str(year)+'.pdf', dpi=500)

    fig, axs = plt.subplots(3, 1, sharex=True, figsize=(13,10))
    #ax3bis = axs[3].twinx()    
    level1b.noise_temperature.resample(time=freq).mean().plot(ax=axs[0],color='k')
    axs[0].set_xlabel('')
    axs[0].set_ylabel(r'T$_{rec}$')
    #axs[3].set_ylim((2500,3100))
    
    level1b.tropospheric_opacity.resample(time=freq).mean().plot(ax=axs[1],color='k')
    axs[1].set_xlabel('')
    axs[1].set_ylabel(r'$\tau$ [-] ')
    axs[1].set_ylim((0,3))

    # ax5bis = axs[2].twinx() 
    level1b.TWindow.resample(time=freq).mean().plot(ax=axs[2], color=get_color('SOMORA'), label=r'T$_{window}$')
    level1b.TRoom.resample(time=freq).mean().plot(ax=axs[2], color= get_color('GROMOS'), label=r'T$_{room}$')
    axs[2].set_xlabel('')
    axs[2].set_ylabel('Temperature [K]', color= 'k')
    axs[2].set_ylim((280,310))
    axs[2].legend()

    # for p_ind in [21,15]:
    #     monthly_mean_o3 = gromos.o3_x.isel(o3_p=p_ind).groupby('time.month').mean()
    #     anomalies = gromos.o3_x.isel(o3_p=p_ind).groupby('time.month') - monthly_mean_o3
    #     anomalies.plot(ax=axs[6], label=f'p = {gromos.o3_p.data[p_ind]:.3f} hPa')
    
    # level1b.tropospheric_opacity.where(level1b.tropospheric_transmittance<0.15).resample(time=freq).mean().plot(ax=axs[6],color='k')
    # axs[6].set_xlabel('')
    # axs[6].set_title('')
    # axs[6].legend(loc=4, fontsize=16)
    # axs[6].set_ylabel(r'$\Delta$O$_3$ VMR')
    #axs[6].set_ylim((0,10))
    
    for ax in axs:
        ax.grid()
    fig.tight_layout(rect=[0, 0.01, 0.99, 1])
    figure.append(fig)
    save_single_pdf(outfolder+instrument_name+'_diagnostics_'+str(year)+'.pdf', figure)
    #  fig.savefig(outfolder+instrument_name+'_level1_diagnostics_'+str(year)+'.pdf', dpi=500)

def plot_pressure(gromos, instrument_name, pressure_level = [15,20,25], add_sun=False, basefolder = '/scratch/GROSOM/Level2/GROMORA_waccm/'):
    year=pd.to_datetime(gromos.time.data[0]).year
    fs = 22
    fig, axs = plt.subplots(len(pressure_level), 1, sharex=True, figsize=(18,12))
    for i, p in enumerate(pressure_level):
        gromos.o3_x.isel(o3_p=p).plot(ax=axs[i], color=get_color(instrument_name), lw=0.6)
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

    for ax in axs:
        ax.grid()
        ax.set_ylabel(r'O$_3$ [ppmv]', fontsize=fs-2)
        ax.yaxis.set_major_locator(MultipleLocator(1))
        ax.yaxis.set_minor_locator(MultipleLocator(0.5))
        ax.tick_params(axis='both', which='major', labelsize=fs)

    fig.tight_layout(rect=[0, 0.01, 0.95, 1])
    fig.savefig(basefolder+instrument_name+'_ozone_pressure_level_'+str(year)+'.pdf', dpi=500)

def plot_fshift_ts(gromos, instrument_name, level1b, flags1a, date_slice, outfolder):
    year=pd.to_datetime(gromos.time.data[0]).year
    fig, axs = plt.subplots(2, 1, sharex=True, figsize=(18,12))
    f_shift = 1e-3*gromos.freq_shift_x.sel(time=date_slice).resample(time='2H').mean()
    f_shift.plot(
        ax=axs[0],
        color='k'
    )
    axs[0].set_ylabel('fshift [kHz]')
    axs[0].set_title('Frequency shift')
    axs[0].set_ylim(f_shift.median()-300, f_shift.median()+300)
    level1b.TRoom.sel(time=date_slice).resample(time='2H').mean().plot(
        ax=axs[1],
        color='k'
    )
    axs[1].set_ylabel(r'T$_{room}$ [K]')
    # axs[1].set_ylim(285, 300)
    lims = axs[0].get_ylim()

    # flags1a.calibration_flags[:,6].resample(time='1H').mean().plot.(
    freqLock = flags1a.time.where(flags1a.calibration_flags[:,6]==0, drop=True).data
    axs[0].vlines( 
        x=freqLock,
        ymin=lims[0],
        ymax=lims[1],
        color='r'
    )
    # axs[2].set_title(flags1a.calibration_flags.attrs['errorCode_7']) 
    # axs[2].set_ylabel('Flag')

    for ax in axs:
        ax.set_xlabel('')

    fig.tight_layout(rect=[0, 0.01, 0.99, 1])

    fig.savefig(outfolder+instrument_name+'_fshift_ts_'+str(year)+'.pdf', dpi=500)
            
def plot_polyfit(gromos, level1b, instrument_name, outfolder):
    figures = list()

    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(15, 10), sharex=True, dpi=500)

    polyfit = gromos.poly_fit_x
    polyfit[:,0].plot(ax=axes[0], color='k')
    polyfit[:,1].plot(ax=axes[1], color='k')
    polyfit[:,2].plot(ax=axes[2], color='k')
    
    axes[0].set_ylabel('')
    axes[0].set_title('Poly order 0')
    axes[1].set_ylabel('')
    axes[1].set_title('Poly order 1')
    axes[2].set_ylabel('') 
    axes[2].set_title('Poly order 2')
    if instrument_name == 'SOMORA':
        axes[0].set_ylim((-0.01,0.15))
        axes[1].set_ylim((-2,0.5))
        axes[2].set_ylim((-0.5,0.5))
    else:
        axes[0].set_ylim((-0.01,0.2))

    for ax in axes:
        ax.grid()
        ax.set_xlabel('')

    fig.tight_layout(rect=[0, 0.03, 1, 0.99])
    
    figures.append(fig)
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(15, 10), sharex=True, dpi=500)
    gromos.h2o_continuum_x.plot(ax=axes[0], color='k')
    level1b.tropospheric_opacity.plot(ax=axes[1], color='k')
    #ds_opacity.tropospheric_opacity_tc.plot(marker='.',ax=axes[3])
    #axes[3].set_ylim(0,20)
    #axs[1].set_ylim(0,1)
    #  end_cost.plot(ax=axs, ylim=(0.75,8))
    axes[0].set_ylabel(r'H$_2$O PWR98')
    axes[0].set_title('Continuum (retrieved)')
    
    axes[1].set_ylim((-0.01,3))
    axes[1].set_ylabel(r'$\tau$ [-] ')
    axes[1].set_xlabel('')
    axes[1].set_title('Tropospheric opacity')

    if instrument_name == 'SOMORA':
        axes[0].set_ylim((-0.01,10))
    else:
        axes[0].set_ylim((-0.01,20))

    for ax in axes:
        ax.grid()

    fig.tight_layout(rect=[0, 0.03, 1, 0.99])
    
    figures.append(fig)
    save_single_pdf(outfolder+'/'+instrument_name+'_polyfit_'+str(gromos.time.data[0])[0:10]+'.pdf', figures)
   # fig.savefig(outfolder+'/'+instrument_name+'_polyfit_'+str(gromos.time.data[0])[0:10]+'.pdf', dpi=500)

def read_level1(folder, instrument_name, dateslice):
    level1 = xr.open_dataset(
        os.path.join(folder,instrument_name+'_level1b_v2_all.nc'),
        #group='spectrometer1',
        decode_times=True,
        decode_coords=True,
        use_cftime=False,
    )
    
    level1 =level1.sortby('time')
    level1['time'] = pd.to_datetime(level1.time.data)

    flags_1a = xr.open_dataset(
        os.path.join(folder,instrument_name+'_level1a_flags_v2_all.nc'),
        #group='spectrometer1',
        decode_times=True,
        decode_coords=True,
        use_cftime=False,
    )
    flags_1a =flags_1a.sortby('time')
    flags_1a['time'] = pd.to_datetime(flags_1a.time.data)

    flags_1b = xr.open_dataset(
        os.path.join(folder,instrument_name+'_level1b_flags_v2_all.nc'),
        #group='spectrometer1',
        decode_times=True,
        decode_coords=True,
        use_cftime=False,
    )
    flags_1b =flags_1b.sortby('time')
    flags_1b['time'] = pd.to_datetime(flags_1b.time.data)

    return level1.sel(time=dateslice), flags_1a.sel(time=dateslice), flags_1b.sel(time=dateslice)

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
        gromos_opacity.tropospheric_opacity_tc.resample(time='4H').mean().plot(
            lw=0.5,
            marker='.',
            ms=0.5,
            ax=axs[0]
        )
        somora_opacity.tropospheric_opacity_tc.resample(time='4H').mean().plot(
            lw=0.5,
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

def plot_sinefit(gromora, date_slice, instrument_name, outfolder, year):
    gromora = gromora.sel(time=date_slice)
    
    if (instrument_name == 'SOMORA') & (year > 2018):
        fig, axes = plt.subplots(nrows=4, ncols=1,sharex=True, figsize=(21, 15))
        gromora.sine_fit_3_x[:,0].plot(marker='.', ax=axes[3], color='k')
        gromora.sine_fit_3_x[:,1].plot(marker='.', ax=axes[3], color='r')
        axes[3].set_ylabel('sinefit')
        axes[3].set_title('Period ='+str(gromora.sine_fit_3_x.attrs['period MHz'])+' MHz')
        #axes[3].set_ylim(-0.2,0.2) 
    else:
        fig, axes = plt.subplots(nrows=3, ncols=1,sharex=True, figsize=(15, 10))
    
    gromora.sine_fit_0_x[:,0].plot(marker='.', ax=axes[0], color='k')
    gromora.sine_fit_0_x[:,1].plot(marker='.', ax=axes[0], color='r')
    gromora.sine_fit_1_x[:,0].plot(marker='.', ax=axes[1], color='k')
    gromora.sine_fit_1_x[:,1].plot(marker='.', ax=axes[1], color='r')
    gromora.sine_fit_2_x[:,0].plot(marker='.', ax=axes[2], color='k')
    gromora.sine_fit_2_x[:,1].plot(marker='.', ax=axes[2], color='r')

    axes[0].set_ylabel('sinefit')
    axes[0].legend(['sine', 'cosine'])
    axes[1].set_ylabel('sinefit')
    axes[2].set_ylabel('sinefit')

    axes[0].set_title('Period ='+str(gromora.sine_fit_0_x.attrs['period MHz'])+' MHz')
    axes[1].set_title('Period ='+str(gromora.sine_fit_1_x.attrs['period MHz'])+' MHz')
    axes[2].set_title('Period ='+str(gromora.sine_fit_2_x.attrs['period MHz'])+' MHz')

    axes[0].set_ylim(-0.2,0.2)
    axes[1].set_ylim(-0.1,0.1)
    axes[2].set_ylim(-0.2,0.2)        
    for a in axes:
        a.set_xlabel('')
        #axes[i].set_xticks([])
    #ds_opacity.tropospheric_opacity_tc.plot(marker='.',ax=axes[3])
    #axes[3].set_ylim(0,20)
    #axs[1].set_ylim(0,1)
    fig.tight_layout(rect=[0, 0.03, 1, 0.99])
    fig.savefig(outfolder+'/'+instrument_name+'_sinefit_' + str(gromora.time.data[0])[0:10]+'.pdf', dpi=500)   

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

def plot_o3_apriori_cov(filename, gromos, outfolder):
    pressure = gromos.o3_p.data
    o3_ap_cov = np.load(filename)
    figures = list()

    fig, ax = plt.subplots(1, 1, figsize=(8,14))
    ax.plot(1e12*np.diag(o3_ap_cov), pressure, color='k')
    ax.invert_yaxis()
    #ax.set_xlim(0,1.1)
    ax.set_yscale('log')
    ax.set_ylim(500,0.001)
    ax.set_xlabel(r'O$_3$ variance [ppmv$^2$]')
    ax.set_ylabel(r'P [hPa]')
    ax.grid()
    fig.tight_layout(rect=[0, 0.03, 1, 0.99])

    figures.append(fig)
    fig, ax = plt.subplots(1, 1, figsize=(12,12))
    c = ax.matshow(1e12*o3_ap_cov, cmap='Greys')
    ticks = [10, 20, 30, 40]
    ax.set_yticks(ticks)
    ax.set_xticks(ticks)
    ax.xaxis.tick_top()
    ax.set_ylabel(r'P [hPa]')
    ax.set_xlabel(r'P [hPa]')
    ax.xaxis.set_label_position('top') 
    lab = ['{:01.2f}'.format(p) for p in pressure[ticks]]
    ax.set_yticklabels(lab)
    ax.set_xticklabels(lab)
    ax.invert_xaxis()
    ax.invert_yaxis()
    cbar = fig.colorbar(c, ax = ax, shrink=0.8) 
    cbar.set_label(r'O$_3$ covariance [ppmv$^2$]')
    fig.tight_layout(rect=[0, 0.03, 1, 0.99])

    figures.append(fig)

    save_single_pdf(outfolder+'gromora_apriori_cov.pdf',figures)


def plot_o3_apriori_all(gromos, outfolder):
    
    fig, axs = plt.subplots(3, 4,  sharex=True, sharey=True, figsize=(18,22))
    o3_xa_daytime = gromos.o3_xa.where(gromos.time.dt.hour==11, drop =True)
    o3_xa_nightime = gromos.o3_xa.where(gromos.time.dt.hour==23, drop =True)
    
    o3_xa_daytime = o3_xa_daytime.groupby('time.month').mean(dim='time')*1e6
    o3_xa_nightime = o3_xa_nightime.groupby('time.month').mean(dim='time')*1e6
    
    axs[0,0].invert_yaxis()
    axs[0,0].set_xlim(0,9)
    
    axs[0,0].set_yscale('log')
    axs[0,0].set_ylim(500,0.001)
    
    axs[0,0].xaxis.set_major_locator(MultipleLocator(4))
    axs[0,0].xaxis.set_minor_locator(MultipleLocator(1))
    axs[0,0].tick_params(axis='both', which='major')

    for i in np.arange(0,4):
        o3_xa_daytime.sel(month=i+1).plot(
            y='o3_p', 
            ax=axs[0,i],
            color='k',
            label='daytime'
        )
        o3_xa_nightime.sel(month=i+1).plot(
            y='o3_p', 
             ax=axs[0,i],
            color='r',
            label='nighttime'
        )
        axs[0,i].set_xlabel(r'O$_3$ VMR [ppmv]')
        axs[0,i].set_ylabel(r'P [hPa]')
        axs[0,i].set_title(MONTH_STR[i])
        axs[0,i].grid(axis ='x', which='both')
        axs[0,i].grid(axis ='y', which='major')
    
    for i in np.arange(0,4):
        o3_xa_daytime.sel(month=5+i).plot(
            y='o3_p', 
            ax=axs[1,i],
            color='k',
        )
        o3_xa_nightime.sel(month=5+i).plot(
            y='o3_p', 
             ax=axs[1,i],
            color='r',
        )
        axs[1,i].set_xlabel(r'O$_3$ VMR [ppmv]')
        axs[1,i].set_ylabel(r'P [hPa]')
        axs[1,i].set_title(MONTH_STR[4+i])
        axs[1,i].grid(axis ='x', which='both')
        axs[1,i].grid(axis ='y', which='major')

    for i in np.arange(0,4):
        o3_xa_daytime.sel(month=9+i).plot(
            y='o3_p', 
            ax=axs[2,i],
            color='k',
        )
        o3_xa_nightime.sel(month=9+i).plot(
            y='o3_p', 
             ax=axs[2,i],
            color='r',
        )
        axs[2,i].set_xlabel(r'O$_3$ VMR [ppmv]')
        axs[2,i].set_ylabel(r'P [hPa]')
        axs[2,i].set_title(MONTH_STR[8+i])
        axs[2,i].grid(axis ='x', which='both')
        axs[2,i].grid(axis ='y', which='major')

    axs[0,0].legend()
    fig.tight_layout(rect=[0, 0.03, 1, 0.99])

    fig.savefig(outfolder+'o3_apriori.pdf', dpi=500)   
    #ax.set_xlabel('Pressure [mPa]')
    #ax.set_ylabel('Pressure [hPa]')
    #ax.grid()
    


def yearly_diagnostics(instrument_name, year, gromora, date_slice, level1_folder, outfolder, nice_ts=False, plots=True):
    print('###################################################################')

    print('Yearly diagnostics for: ',instrument_name,' ', str(year))
    print('Corrupted retrievals',instrument_name,': ',len(gromora['o3_x'].where((gromora['o3_x']<0), drop = True))+ len(gromora['o3_x'].where((gromora['o3_x']>1e-5), drop = True))) 
    # gromora = gromora.drop(['y', 'yf', 'bad_channels','y_baseline', 'f'] )
    # somora = somora.drop(['y', 'yf', 'bad_channels','y_baseline', 'f'] )
    #gromora['o3_x'] = 1e6*gromora['o3_x'].where((gromora['o3_x']>0)&(gromora['o3_x']<1e-5), drop = True)
    #somora['o3_x'] = 1e6*somora['o3_x'].where((somora['o3_x']>0)&(somora['o3_x']<1e-5), drop = True)
    gromora['o3_x'] = 1e6*gromora['o3_x'].where((gromora['o3_x']>gromora['o3_x'].valid_min)&(gromora['o3_x']<gromora['o3_x'].valid_max), drop = True)

    gromora_clean = gromora.where(gromora.retrieval_quality==1, drop=True)#.where(gromora.o3_mr>0.8)
    if plots:
        compare_with_apriori(gromora, instrument_name,  freq='6H', date_slice=date_slice, basefolder=outfolder)

    if nice_ts & plots:
        plot_ozone_ts(gromora_clean, instrument_name=instrument_name, freq='12H', altitude=False, basefolder=outfolder )

    level1b_gromos, gromos_flags_level1a, gromos_flags_level1b = read_level1(level1_folder, instrument_name, dateslice=slice('2009-01-01', '2021-12-31'))
    
    level1b_gromos=level1b_gromos.sel(time=date_slice)
    gromos_flags_level1a=gromos_flags_level1a.sel(time=date_slice)
    gromos_flags_level1b=gromos_flags_level1b.sel(time=date_slice)
    
    num_good_1a_gromos = len(gromos_flags_level1a.where(gromos_flags_level1a.calibration_flags.sum(dim='flags')>6, drop=True).time)
    num_good_1b_gromos = len(gromos_flags_level1b.where(gromos_flags_level1b.calibration_flags[:,0]==1, drop=True).time)
    print(instrument_name,' good quality level1a: ', 100*num_good_1a_gromos/len(pd.date_range(str(year)+'-01-01', str(year)+'-12-31 23:00:00', freq='10 min')))
    print(instrument_name,' good quality level1b: ', 100*num_good_1b_gromos/len(pd.date_range(str(year)+'-01-01', str(year)+'-12-31 23:00:00', freq='1H')))

    #gromora_opacity, somora_opacity = read_opacity(folder='/scratch/GROSOM/Level2/opacities/', year=yr)


    print(instrument_name,' good quality level2: ', 100*len(gromora_clean.time)/len(pd.date_range(str(year)+'-01-01', str(year)+'-12-31 23:00:00', freq='1H')) )
    #gromora = utc_to_lst(gromora)

   # test_plevel(10, slice('2017-01-02','2017-01-03'), gromos ,gromos_sel, mls, mls_gromos_colloc, mls_gromos_colloc_conv)


   # gromos_linear_fit = gromos_clean.o3_x.where((gromos_clean.o3_p<p_high) & (gromos_clean.o3_p>p_low), drop=True).mean(dim='o3_p').resample(time='1M').mean()#) .polyfit(dim='time', deg=1)
    # somora_linear_fit = somora_clean.o3_x.resample(time='1M').mean().polyfit(dim='time', deg=1)


    #  plot_o3_pressure_profile(gromos)
    # altitude = gromos_clean.o3_z.mean(dim='time').where(gromos_clean.o3_p<p_high, drop=True).where(gromos_clean.o3_p>p_low, drop=True)

    #plot_pressure(gromora, instrument_name='GROMOS' , pressure_level=[31, 25, 21, 15, 12], add_sun=False, basefolder=outfolder)
    if plots:
        retrievals_diagnostics(gromora_clean, level1b_gromos, instrument_name, freq='1H', outfolder=outfolder)

        plot_fshift_ts(gromora, instrument_name, level1b_gromos, gromos_flags_level1a,  date_slice, outfolder)
        fshift_daily_cycle(gromora_clean, instrument_name, date_slice, outfolder)
   # ds_fshift.freq_shift_x.sel(time=slice("2017-01-01", "2018-12-31")).resample(time='12H').mean().plot()
    # plt.matshow(gromora.o3_avkm.isel(time=0))
    # plt.colorbar()
    # 

        plot_polyfit(gromora, level1b_gromos,instrument_name, outfolder)

        plot_sinefit(gromora, date_slice, instrument_name, outfolder=outfolder, year=year)
        plot_ozone_flags(instrument_name, gromora, level1b_gromos, flags1a=gromos_flags_level1a, flags1b=gromos_flags_level1b, pressure_level=[27, 12], calib_version=2)

#     # plot_ozone_ts(ozone_const_alt, altitude=True)

    return gromora, gromora_clean, level1b_gromos, gromos_flags_level1a, gromos_flags_level1b

def add_flags_level2_gromora(gromos, instrument_name):
    if instrument_name == 'GROMOS':
        date2flag =  [
            slice('2015-08-26','2015-08-29')
        ]
        date2flag.append(slice('2012-07-24','2012-08-08'))
        date2flag.append(slice('2019-01-14','2019-02-12'))
    else:
        date2flag =  [
             slice('2012-04-24','2012-04-27')
        ]
        date2flag.append(slice('2016-09-29','2016-11-03'))
        date2flag.append(slice('2018-01-31','2018-02-11'))
        date2flag.append(slice('2018-07-25','2018-08-23'))
        #date2flag.append(slice('2019-09-27','2020-01-21'))
    #date2flag = np.array(date2flag)
    date_list = [pd.to_datetime(t).date() for t in gromos.time.data]
    #level2_flag = gromos.retrieval_quality
    gromos['level2_flag'] = ('time', np.zeros(len(gromos.time.data)))
    for flagD in date2flag:
        if len(gromos.time.sel(time=flagD))>0:     
            print(flagD)      
            gromos.level2_flag.loc[dict(time=flagD)] = 1
            #gromos.where(gromos.time.data==flagD)
    gromos['level2_flag'].attrs['standard_name'] = 'leve2_flags'
    gromos['level2_flag'].attrs['long_name'] = 'flags for spurious periods'
    gromos['level2_flag'].attrs['units'] = '1'
    gromos['level2_flag'].attrs['description'] = 'Manual level 2 flag of the retrievals for spurious periods: 0 is good periods, 1 is flagged'
    return gromos

def add_flags_save(instrument_name, year, gromora, date_slice, level1_folder, outfolder):
    print('###################################################################')

    print('Saving yearly netCDF for: ',instrument_name,' ', str(year))

    level1b, flags_level1a, flags_level1b = read_level1(level1_folder, instrument_name, dateslice=slice('2009-01-01', '2021-12-31'))
    level1b=level1b.sel(time=date_slice)

    # Removing some duplicated time values in GROMOS level 1 concatenated files.
    level1b=level1b.sel(time=~level1b.get_index("time").duplicated())

    lat = gromora.lat.mean(dim='time').data
    lon = gromora.lon.mean(dim='time').data

    # Julian dates from GEOMS https://avdc.gsfc.nasa.gov/PDF/GEOMS/geoms-1.0.pdf
    julian_date = mjd2k_date(pd.to_datetime(gromora.time.data))

    sza_list = list()
    for t in gromora.time.data:
        lst,ha,sza,night,tc = get_LST_from_GROMORA(datetime64_2_datetime(t).replace(tzinfo=gromora_tz), lat, lon, check_format=False)
        sza_list.append(sza)

    gromora['MJD2K'] =  ('time', julian_date)
    gromora.MJD2K.attrs['standard_name'] = 'MJD2K'
    gromora.MJD2K.attrs['long_name'] = 'Modified Julian Date 2000'
    gromora.MJD2K.attrs['units'] = 'MJD2K'
    gromora.MJD2K.attrs['description'] = 'MJD2K as defined by GEOMS: it is 0.000000 on January 1, 2000 at 00:00:00 UTC'

    gromora['solar_zenith_angle'] =  ('time', sza_list)
    gromora.solar_zenith_angle.attrs['standard_name'] = 'solar_zenith_angle'
    gromora.solar_zenith_angle.attrs['long_name'] = 'solar zenith angle'
    gromora.solar_zenith_angle.attrs['units'] = 'deg'
    gromora.solar_zenith_angle.attrs['description'] = 'angle between the sun rays and zenith, minimal at local solar noon'


    gromora['tropospheric_opacity'] = ('time',level1b.tropospheric_opacity.reindex_like(gromora, method='nearest', tolerance='1H'))
    gromora.tropospheric_opacity.attrs['standard_name'] = 'tropospheric_opacity'
    gromora.tropospheric_opacity.attrs['long_name'] = 'tropospheric_opacity computed with Ingold method during calibration'
    gromora.tropospheric_opacity.attrs['units'] = 'Np'

    gromora.to_netcdf(outfolder+'/'+instrument_name+'_level2_'+str(yr)+'.nc')

if __name__ == "__main__":
    yr = 2021
    date_slice=slice(str(yr)+'-01-01',str(yr)+'-12-31')

    instNameGROMOS = 'GROMOS'
    instNameSOMORA = 'SOMORA'
    fold_somora = '/storage/tub/instruments/somora/level2/v2/'
    level1_folder_somora = '/storage/tub/instruments/somora/level1/v2/'
    fold_gromos = '/storage/tub/instruments/gromos/level2/GROMORA/v2/'
    level1_folder_gromos = '/storage/tub/instruments/gromos/level1/GROMORA/v2/'
    prefix_all='_v2.nc'

    plot_yearly_diagnostics = False
    save = True
 
    gromos = read_GROMORA_all(basefolder=fold_gromos, 
    instrument_name=instNameGROMOS,
    date_slice=date_slice, 
    years=[yr], 
    prefix= prefix_all, #'_v2_noncorrangle.nc'
    flagged=False
    )
    somora = read_GROMORA_all(basefolder=fold_somora, 
    instrument_name=instNameSOMORA,
    date_slice=date_slice, 
    years= [yr],
    prefix=prefix_all,
    flagged=False
    )

    outfolder = '/scratch/GROSOM/Level2/Diagnostics_v2/'
    
    gromos = add_flags_level2_gromora(gromos, 'GROMOS')
    somora = add_flags_level2_gromora(somora, 'SOMORA')

    if plot_yearly_diagnostics:
        somora, somora_clean, level1b_somora, somora_flags_level1a, somora_flags_level1b = yearly_diagnostics('SOMORA', yr, somora, date_slice, level1_folder_somora, outfolder, nice_ts=False, plots=False)
        gromos, gromos_clean, level1b_gromos, gromos_flags_level1a, gromos_flags_level1b = yearly_diagnostics('GROMOS', yr, gromos, date_slice, level1_folder_gromos, outfolder, nice_ts=False, plots=False)

    if save:
        add_flags_save('GROMOS', yr, gromos, date_slice, level1_folder_gromos, outfolder='/scratch/GROSOM/Level2/GROMOS/v2/')
        add_flags_save('SOMORA', yr, somora, date_slice, level1_folder_somora, outfolder='/scratch/GROSOM/Level2/SOMORA/v2/')
        #gromos.to_netcdf('/scratch/GROSOM/Level2/GROMOS/GROMOS_level2_'+str(yr)+'.nc')
        #somora.to_netcdf('/scratch/GROSOM/Level2/SOMORA/SOMORA_level2_'+str(yr)+'.nc')
    #plot_o3_apriori_all(gromos, outfolder)

    #plot_o3_apriori_cov('/home/esauvageat/Documents/GROMORA/Data/apriori_cov.npy', gromos, outfolder)