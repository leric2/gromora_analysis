#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on 17.03.22

@author: Eric Sauvageat

Collection of function to compare the harmonized GROMORA L2 data

It includes the comparisons with satellites or old retrieval dataset.

To run this code, you need to use the main function located at the end of the file and define what kind of analysis you want to perform.
There is a set of flags that you can turn ON to do comparisons between GROMOS and SOMORA or with MLS for instance.

"""
#%%
import datetime
from datetime import date
from operator import truediv
import os

import matplotlib.pyplot as plt
import netCDF4
import numpy as np
import pandas as pd

import xarray as xr
from scipy import stats
from scipy.odr import *

from GROMORA_harmo.scripts.retrieval import gromora_time
from base_tool import get_color, save_single_pdf, regression_xy, linear, calcR2_wikipedia
from level2_gromora import read_GROMORA_all, read_GROMORA_concatenated, read_gromos_v2021, read_old_SOMORA, plot_ozone_ts, read_gromos_old_FB
from level2_gromora_diagnostics import read_level1, add_flags_level2_gromora
from compare_gromora_v2 import compare_avkm, compare_mean_diff_monthly, compare_pressure_mls_sbuv_paper, compare_pressure_mls_sbuv, compare_ts_gromora, compute_seasonal_correlation_paper, map_rel_diff
from ecmwf import read_ERA5, mmr_2_vmr
from tempera import read_tempera_level3, read_tempera_level2, read_tempera_level3_concat

from matplotlib.ticker import (MultipleLocator, FuncFormatter, FormatStrFormatter, AutoMinorLocator)
from matplotlib.lines import Line2D
import matplotlib.dates as mdates
import matplotlib.ticker as ticker
import matplotlib
cmap = matplotlib.cm.get_cmap('plasma')

cmap_ts = plt.get_cmap('density') # 'cividis' # plt.get_cmap('density') #'cividis' #

from MLS import *
from sbuv import *
from base_tool import *

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.sans-serif": ["Free sans"]})

plt.rcParams['xtick.labelsize'] = 28
plt.rcParams['ytick.labelsize'] = 28
plt.rcParams['font.size'] = 28
plt.rcParams['axes.titlesize'] = 30

color_gromos= get_color('GROMOS')
color_somora= get_color('SOMORA')
sbuv_color= get_color('SBUV')
color_shading='grey'

MONTH_STR = ['Jan', 'Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
PAPER_SYMBOL = ['a)','b)','c)', 'd)'] # ['a) lower mesosphere','b) upper stratosphere','c) lower stratosphere']#

def compare_ts_gromos_tempera(gromos, somora, tempera, date_slice, freq, basefolder):
    fs = 34
    year = pd.to_datetime(gromos.sel(time=date_slice).time.values[0]).year

    if somora is None:
        fig, axs = plt.subplots(2, 1, sharex=True, sharey=True, figsize=(22,12))
        axtemp =  axs[1]
    else:
        fig, axs = plt.subplots(3, 1, sharex=True, sharey=True, figsize=(24,16))
        axtemp =  axs[2]

    pl = gromos.sel(time=date_slice).o3_x.resample(time=freq).mean().plot(
        x='time',
        y='alt',
        ax=axs[0], 
        vmin=0,
        vmax=10,
        linewidth=0,
        rasterized=True,
        add_colorbar=True,
        cmap=cmap_ts,
        cbar_kwargs={'label':r'O$_3$ [ppmv]'}
    )
    pl.set_edgecolor('face')
    axs[0].set_title('GROMOS', fontsize=fs+4) 
    # ax.set_yscale('log')
    #axs[0].set_ylabel('Pressure [hPa]', fontsize=fs)
    if somora is not None:
        pl = somora.sel(time=date_slice).o3_x.resample(time=freq).mean().plot(
            x='time',
            y='alt',
            ax=axs[1], 
            vmin=0,
            vmax=10,
            linewidth=0,
            rasterized=True,
            add_colorbar=True,
            cmap=cmap_ts,
            cbar_kwargs={'label':r'O$_3$ [ppmv]'}
        )
        pl.set_edgecolor('face')
        axs[1].set_title('SOMORA', fontsize=fs+4) 
    # ax.set_yscale('log')
    #axs[1].set_ylabel('Pressure [hPa]', fontsize=fs)

    pl2 = tempera.sel(time=date_slice).tmp.resample(time=freq).mean().plot(
        x='time',
        y='altitude',
        ax=axtemp, 
        vmin=200,
        vmax=280,
        linewidth=0,
        rasterized=True,
        add_colorbar=True,
        cmap=plt.get_cmap('temperature'),
        cbar_kwargs={'label':r'T [K]'}
    )
    pl2.set_edgecolor('face')
    axtemp.set_ylabel('Altitude [km]', fontsize=fs)
    axtemp.set_title('TEMPERA', fontsize=fs+4)
    axtemp.set_ylim(20, 60)
    axtemp.set_xlabel('')

    for ax in axs:
        ax.set_xlabel('')
        ax.set_ylabel('Altitude [km]', fontsize=fs)
        ax.tick_params(axis='both', which='major', labelsize=fs)
        #ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: '{:g}'.format(y)))
        #ax.set_ylim(20, 80)

    plt.tight_layout(rect=[0, 0.01, 0.99, 1])
    fig.savefig(basefolder+'GROMORA_ozone_Temp_comparison_'+str(year)+'_'+freq+'.pdf', dpi=500)

def compare_pressure_gromos_tempera(gromos, somora, tempera, tempera_old, mls, mls_temp, ecmwf_ts, sbuv, p_min, p_max, freq='1D', basefolder=''):
    fs=32
    year=pd.to_datetime(gromos.time.data[0]).year
    fig, axs = plt.subplots(2, 1, sharex=True, figsize=(28,18))
    fig1, axs1 = plt.subplots(1, 1, sharex=True, figsize=(28,18))
    print('###########################################################')
    for i, p_ind in enumerate(p_min):
        midday_hours = [12,13,14]
        midnight_hours = [23,0,1]

        # Dealing with ozone time series:
        gromos_p = gromos.where(gromos.o3_p>p_min[i] , drop=True).where(gromos.o3_p<p_max[i], drop=True)
        somora_p = somora.where(somora.o3_p>p_min[i] , drop=True).where(somora.o3_p<p_max[i], drop=True)
        
        #corresponding altitude:
        alt_min = 1e-3*gromos_p.o3_z.data[0,0]
        alt_max = 1e-3*gromos_p.o3_z.data[0,-1]

        print('O3 and Temperature between '+f'{p_min[i]:.1f}'+ r' < p < '+f'{p_max[i]:.1f}'+' hPa'+' or '+f'{alt_min:.1f}'+ r' < z < '+f'{alt_max:.1f}'+' km')
        
        gromos_p = gromos_p.o3_x.mean(dim='o3_p')
        somora_p = somora_p.o3_x.mean(dim='o3_p')

        midday_o3_gromos = gromos_p.where(gromos_p.time.dt.hour.isin(midday_hours), drop=True).resample(time='1D').mean()
        midnight_o3_gromos = gromos_p.where(gromos_p.time.dt.hour.isin(midnight_hours), drop=True).resample(time='1D').mean()

        delta_o3 = (midday_o3_gromos - midnight_o3_gromos).resample(time='5D').median()

        gromos_p = gromos_p.resample(time=freq).median()
        somora_p = somora_p.resample(time=freq).median()

        # ERA5
        ecmwf_T = ecmwf_ts.temperature.where(ecmwf_ts.lev>p_min , drop=True).where(ecmwf_ts.lev<p_max, drop=True).mean(dim='lev').resample(time=freq).mean()

        # New tempera
        tempera_p = tempera.temperature.where(tempera.altitude>alt_min , drop=True).where(tempera.altitude<alt_max, drop=True).mean(dim='altitude')

        midday_T = tempera_p.where(tempera_p.time.dt.hour.isin(midday_hours), drop=True).resample(time='1D').mean()
        midnight_T = tempera_p.where(tempera_p.time.dt.hour.isin(midnight_hours), drop=True).resample(time='1D').mean()

        delta_T = (midday_T - midnight_T).resample(time='5D').median()

        tempera_p = tempera_p.resample(time=freq).median()

        # Old tempera
        tempera_old_p = tempera_old.T.where(tempera_old.pressure>p_min[i] , drop=True).where(tempera_old.pressure<p_max[i], drop=True).mean(dim='pressure').resample(time=freq).median()

        ds_merged=xr.merge((
            {'ozone':gromos_p},
            {'ozone_somora':somora_p},
            {'temperature':tempera_p},
            # {'tropospheric_opacity':gromos_p.tropospheric_opacity.resample(time=freq).mean()},
        ))

        ds_Delta_merged=xr.merge((
            {'ozone':delta_o3},
            {'temperature':delta_T},
            # {'tropospheric_opacity':gromos_p.tropospheric_opacity.resample(time=freq).mean()},
        ))

        # Clean the ds to remove any Nan entries:
        ds_merged_gromos = ds_merged.where(ds_merged.ozone.notnull(), drop=True).where(ds_merged.temperature.notnull(), drop=True)
        ds_merged_somora = ds_merged.where(ds_merged.ozone_somora.notnull(), drop=True).where(ds_merged.temperature.notnull(), drop=True)
        pearson_corr = xr.corr(ds_merged_gromos.ozone,ds_merged_gromos.temperature, dim='time')
        pearson_corr_somora = xr.corr(ds_merged_somora.ozone_somora,ds_merged_somora.temperature, dim='time')
        print('Pearson corr Temp-O3 GROMOS: ',f'{pearson_corr.values:.2f}')
        print('Pearson corr Temp-O3 SOMORA: ',f'{pearson_corr_somora.values:.2f}')
        print('##########################')
        ds_Delta_merged = ds_Delta_merged.where(ds_Delta_merged.ozone.notnull(), drop=True).where(ds_Delta_merged.temperature.notnull(), drop=True)
        pearson_corr_Delta = xr.corr(ds_Delta_merged.ozone,ds_Delta_merged.temperature, dim='time')
        print('Pearson corr DO3 vs DT: ',f'{pearson_corr_Delta.values:.2f}')

        mls_p = mls.where(mls.p>p_min[i] , drop=True).where(mls.p<p_max[i], drop=True).mean(dim='p').resample(time=freq).mean()

        mls_t_p = mls_temp.where(mls_temp.p>p_min[i] , drop=True).where(mls_temp.p<p_max[i], drop=True).mean(dim='p').resample(time=freq).mean()

        pressure =  p_min[i]

        #mls_anomalies = (mls_p.o3.mean(dim='p').groupby('time.dayofyear') - mls_p.o3.mean(dim='p').groupby('time.dayofyear').mean()).resample(time=freq).mean()
        mls_p.o3.plot(ax=axs[i], color=get_color('MLS'), lw=2, label='MLS')
        if (pressure > 0.4) & (pressure<50) & (sbuv is not None):
            sbuv_p = sbuv.where(sbuv.p>p_min[i] , drop=True).where(sbuv.p<p_max[i], drop=True).resample(time=freq).mean()
            #sbuv_anomalies = (sbuv_p.ozone.mean(dim='p').groupby('time.dayofyear') - sbuv_p.ozone.mean(dim='p').groupby('time.dayofyear').mean()).resample(time=freq).mean()
            sbuv_p.ozone.mean(dim='p').plot(ax=axs[i], color=sbuv_color, lw=2, label='SBUV')
        #gromos_anomalies = (gromos_p.o3_x.mean(dim='o3_p').groupby('time.dayofyear') - gromos_p.o3_x.mean(dim='o3_p').groupby('time.dayofyear').mean()).resample(time=freq).mean()
        #tempera_anomalies = (tempera_p.temperature.mean(dim='altitude').groupby('time.dayofyear') - tempera_p.o3_x.mean(dim='o3_p').groupby('time.dayofyear').mean()).resample(time=freq).mean()
        gromos_p.plot(ax=axs[i], color=color_gromos, lw=2, label='GROMOS')
        somora_p.plot(ax=axs[i], color=color_somora, lw=2, label='SOMORA')
        
        # tempera_old_p.plot(ax=axs[i+1], color='g', lw=2, label='TEMPERA Old')
        mls_t_p.o3.plot(ax=axs[i+1], color=get_color('MLS'), lw=2, label='MLS')
        ecmwf_T.plot(ax=axs[i+1], color=get_color('ECMWF'), lw=2, label='ERA5')
        tempera_p.plot(ax=axs[i+1], color=get_color('GDOC'), lw=2, label='TEMPERA')
        #axs[i].set_title(f'p = {pressure:.3f} hPa', fontsize=fs)
        axs[i].set_title(r'Mean O$_3$ VMR at $'+ str(p_min[i])+ ' < p < '+str(p_max[i])+'$ hPa',fontsize=fs+2)

        axs[i+1].set_title(r'Mean Temperature VMR at '+ f'{alt_min:.1f}'+ r' $< z <$ '+f'{alt_max:.1f}'+' km',fontsize=fs+2)
        if pressure > 10:
            axs[i+1].set_ylim(180,240)
        else:
            axs[i+1].set_ylim(200,280)
        axs[i+1].set_ylabel('T [K]', fontsize=fs)

        #ax.set_xlim(pd.to_datetime('2009-09-23'), pd.to_datetime('2022-01-01'))
        axs[i].set_ylabel('O$_3$ [ppmv]', fontsize=fs)
        axs[i].yaxis.set_major_locator(MultipleLocator(1))
        axs[i].yaxis.set_minor_locator(MultipleLocator(0.5))
        #axs[i].xaxis.set_major_locator(mdates.YearLocator())
        #axs[i]x.xaxis.set_major_locator(mdates.MonthLocator(bymonth=(1, 7)))
        axs[i].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        axs[i].tick_params(axis='both', which='major', labelsize=fs-2)

        mean_alt = (alt_min + alt_max)/2

        for ax in axs:
            ax.legend()
            ax.grid()
            ax.set_xlabel('')

    fig.tight_layout(rect=[0, 0.01, 0.99, 1])
    fig.savefig(basefolder+'ozone_temperature_comparison_pressure_level_MLS_SBUV_'+f'{mean_alt:.1f}km_'+str(year)+'.pdf', dpi=500)

    axs1.plot(delta_o3, color=get_color('GROMOS'))
    axs1.set_ylabel(r'DO3')
    axs2 = axs1.twinx()
    axs2.plot(delta_T, color=get_color('SOMORA'))
    axs2.set_ylabel(r'DT')
    fig1.tight_layout(rect=[0, 0.01, 0.99, 1])
    fig1.savefig(basefolder+'detlaO3_vs_deltaT'+str(year)+'.pdf', dpi=500)

def compare_altitude_gromos_tempera(gromos, somora, tempera, mls, mls_o3_convolved, mls_temp, ecmwf_ts, alt_min, alt_max, freq='1D', basefolder=''):
    fs=32
    year=pd.to_datetime(gromos.time.data[0]).year
    
    fig1, axs1 = plt.subplots(1, 1, sharex=True, figsize=(28,18))
    print('###########################################################')
    for i, p_ind in enumerate(alt_min):
        fig, axs = plt.subplots(2, 1, sharex=True, figsize=(28,18))
        midday_hours = [12,13,14]
        midnight_hours = [23,0,1]

        # Dealing with ozone time series:
        gromos_p = gromos.where(gromos.alt>alt_min[i] , drop=True).where(gromos.alt<alt_max[i], drop=True)
        somora_p = somora.where(somora.alt>alt_min[i] , drop=True).where(somora.alt<alt_max[i], drop=True)
        
        gromos_p = gromos_p.o3_x.mean(dim='alt')
        somora_p = somora_p.o3_x.mean(dim='alt')

        # New tempera
        tempera_temperature = tempera.tmp.where(tempera.altitude>alt_min[i] , drop=True).where(tempera.altitude<alt_max[i], drop=True)
        tempera_pressure = tempera.p.where(tempera.altitude>alt_min[i] , drop=True).where(tempera.altitude<alt_max[i], drop=True)

        # #corresponding altitude:
        p_min =1e-2*tempera_pressure.mean(dim='time').data[-1]
        p_max =1e-2*tempera_pressure.mean(dim='time').data[0]

        print('O3 and Temperature between '+f'{alt_min[i]:.1f}'+ r' < z < '+f'{alt_max[i]:.1f}'+' km'+' or '+f'{p_min:.1f}'+ r' < p < '+f'{p_max:.1f}'+' hPa')

        tempera_p=tempera_temperature.mean(dim='altitude')
        dT = tempera_p - tempera_p.mean(dim='time')

        midday_o3_gromos = gromos_p.where(gromos_p.time.dt.hour.isin(midday_hours), drop=True).resample(time='1D').mean()
        midnight_o3_gromos = gromos_p.where(gromos_p.time.dt.hour.isin(midnight_hours), drop=True).resample(time='1D').mean()

        o3_theoric = -0.0215*dT + midday_o3_gromos.mean(dim='time')

        delta_o3 = (midday_o3_gromos - midnight_o3_gromos).resample(time='5D').median()
        midday_T = tempera_p.where(tempera_p.time.dt.hour.isin(midday_hours), drop=True).resample(time='1D').mean()
        midnight_T = tempera_p.where(tempera_p.time.dt.hour.isin(midnight_hours), drop=True).resample(time='1D').mean()

        delta_T = (midday_T - midnight_T).resample(time='5D').median()
        gromos_p = gromos_p.resample(time=freq).median()
        somora_p = somora_p.resample(time=freq).median()
        tempera_p = tempera_p.resample(time=freq).median()

        ds_merged=xr.merge((
            {'ozone':gromos_p},
            {'ozone_somora':somora_p},
            {'temperature':tempera_p},
            # {'tropospheric_opacity':gromos_p.tropospheric_opacity.resample(time=freq).mean()},
        ))

        ds_Delta_merged=xr.merge((
            {'ozone':delta_o3},
            {'temperature':delta_T},
            # {'tropospheric_opacity':gromos_p.tropospheric_opacity.resample(time=freq).mean()},
        ))

        # Clean the ds to remove any Nan entries:
        ds_merged_gromora = ds_merged.where(ds_merged.ozone.notnull(), drop=True).where(ds_merged.ozone_somora.notnull(), drop=True)

        ds_merged_gromos = ds_merged.where(ds_merged.ozone.notnull(), drop=True).where(ds_merged.temperature.notnull(), drop=True)
        ds_merged_somora = ds_merged.where(ds_merged.ozone_somora.notnull(), drop=True).where(ds_merged.temperature.notnull(), drop=True)

        pearson_corr_gromora = xr.corr(ds_merged_gromora.ozone,ds_merged_gromora.ozone_somora, dim='time')
        pearson_corr = xr.corr(ds_merged_gromos.ozone,ds_merged_gromos.temperature, dim='time')
        pearson_corr_somora = xr.corr(ds_merged_somora.ozone_somora,ds_merged_somora.temperature, dim='time')
        print('####################################################')
        print('Pearson corr O3 GROMOS - SOMORA: ',f'{pearson_corr_gromora.values:.3f}')
        print('Pearson corr Temp-O3 GROMOS: ',f'{pearson_corr.values:.3f}')
        print('Pearson corr Temp-O3 SOMORA: ',f'{pearson_corr_somora.values:.3f}')
        print('##########################')
        ds_Delta_merged = ds_Delta_merged.where(ds_Delta_merged.ozone.notnull(), drop=True).where(ds_Delta_merged.temperature.notnull(), drop=True)
        pearson_corr_Delta = xr.corr(ds_Delta_merged.ozone,ds_Delta_merged.temperature, dim='time')
        print('Pearson corr DO3 vs DT: ',f'{pearson_corr_Delta.values:.2f}')

        if mls_o3_convolved:
            mls_p = mls.where(mls.o3_p>p_min , drop=True).where(mls.o3_p<p_max, drop=True).mean(dim='o3_p').resample(time=freq).mean()
            mls_p.o3_x.plot(ax=axs[0], color=get_color('MLS'), lw=2, label='MLS, convolved')
        else:
            mls_p = mls.where(mls.p>p_min , drop=True).where(mls.p<p_max, drop=True).mean(dim='p').resample(time=freq).mean()
            #mls_anomalies = (mls_p.o3.mean(dim='p').groupby('time.dayofyear') - mls_p.o3.mean(dim='p').groupby('time.dayofyear').mean()).resample(time=freq).mean()
            mls_p.o3.plot(ax=axs[0], color=get_color('MLS'), lw=2, label='MLS')

        mls_t_p = mls_temp.where(mls_temp.p>p_min , drop=True).where(mls_temp.p<p_max, drop=True).mean(dim='p').resample(time=freq).mean()

        #gromos_anomalies = (gromos_p.o3_x.mean(dim='o3_p').groupby('time.dayofyear') - gromos_p.o3_x.mean(dim='o3_p').groupby('time.dayofyear').mean()).resample(time=freq).mean()
        #tempera_anomalies = (tempera_p.temperature.mean(dim='altitude').groupby('time.dayofyear') - tempera_p.o3_x.mean(dim='o3_p').groupby('time.dayofyear').mean()).resample(time=freq).mean()
        gromos_p.plot(ax=axs[0], color=color_gromos, lw=2, label='GROMOS')
        somora_p.plot(ax=axs[0], color=color_somora, lw=2, label='SOMORA')
        # o3_theoric.plot(ax=axs[0], color='k', lw=2, label='THEORY')
        
        # tempera_old_p.plot(ax=axs[i+1], color='g', lw=2, label='TEMPERA Old')
        mls_t_p.o3.plot(ax=axs[1], color=get_color('MLS'), lw=2, label='MLS')

        # ERA5
        if ecmwf_ts is not None:
            ecmwf_T = ecmwf_ts.temperature.where(ecmwf_ts.lev>p_min , drop=True).where(ecmwf_ts.lev<p_max, drop=True).mean(dim='lev').resample(time=freq).mean()
            ecmwf_T.plot(ax=axs[i+1], color=get_color('ECMWF'), lw=2, label='ERA5')
        
        tempera_p.resample(time=freq).median().plot(ax=axs[1], color=get_color('GDOC'), lw=2, label='TEMPERA')
        #axs[i].set_title(f'p = {pressure:.3f} hPa', fontsize=fs)
        axs[0].set_title(r'Mean O$_3$ VMR at $'+ str(alt_min[i])+ ' < z < '+str(alt_max[i])+'$ km',fontsize=fs+2)

        axs[1].set_title(r'Mean Temperature VMR at '+ f'{alt_min[i]:.1f}'+ r' $< z <$ '+f'{alt_max[i]:.1f}'+' km',fontsize=fs+2)

        mean_alt = (alt_min[i] + alt_max[i])/2

        if mean_alt > 40:
            axs[0].set_ylim(2,6)
            axs[1].set_ylim(230,290)
        elif ((mean_alt > 30) & (mean_alt < 40)):
            axs[0].set_ylim(4,10)
            axs[1].set_ylim(200,260)
        else:
            axs[0].set_ylim(3,8)
            axs[1].set_ylim(190,240)

        if mean_alt > 50:
            axs[0].set_ylim(1,3)
        axs[1].set_ylabel('T [K]', fontsize=fs)

        #ax.set_xlim(pd.to_datetime('2009-09-23'), pd.to_datetime('2022-01-01'))
        axs[0].set_ylabel('O$_3$ [ppmv]', fontsize=fs)
        axs[0].yaxis.set_major_locator(MultipleLocator(1))
        axs[0].yaxis.set_minor_locator(MultipleLocator(0.5))
        #axs[i].xaxis.set_major_locator(mdates.YearLocator())
        #axs[i]x.xaxis.set_major_locator(mdates.MonthLocator(bymonth=(1, 7)))
        axs[1].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        axs[1].tick_params(axis='both', which='major', labelsize=fs-2)

        for ax in axs:
            ax.legend()
            ax.grid()
            ax.set_xlabel('')
            ax.set_xlim(pd.to_datetime(gromos.time.data[0])-pd.Timedelta(hours=12),pd.to_datetime(gromos.time.data[-1])+pd.Timedelta(hours=12))


        fig.tight_layout(rect=[0, 0.01, 0.99, 1])
        fig.savefig(basefolder+'ozone_temperature_comparison_altitude_level_MLS_'+f'{mean_alt:.1f}km_'+str(year)+'.pdf', dpi=500)

    # axs1.plot(delta_o3, color=get_color('GROMOS'))
    # axs1.set_ylabel(r'DO3')
    # axs2 = axs1.twinx()
    # axs2.plot(delta_T, color=get_color('SOMORA'))
    # axs2.set_ylabel(r'DT')
    # fig1.tight_layout(rect=[0, 0.01, 0.99, 1])
    # fig1.savefig(basefolder+'detlaO3_vs_deltaT'+str(year)+'.pdf', dpi=500)
def consecutive(data, stepsize=1):
    # https://stackoverflow.com/a/7353335/9940782
    return np.split(data, np.where(np.diff(data) != stepsize)[0] + 1)

def compare_altitude_anomalies_gromos_tempera(gromos, somora, tempera, alt_min, alt_max, freq='1D', basefolder=''):
    fs=32
    year=pd.to_datetime(gromos.time.data[0]).year
    
    fig, axs = plt.subplots(len(alt_min), 1, sharex=True, figsize=(28,18))
    print('###########################################################')
    for i, p_ind in enumerate(alt_min):
        axT = axs[i].twinx()
        
        midday_hours = [12,13,14]
        midnight_hours = [23,0,1]

        # Dealing with ozone time series:
        gromos_p = gromos.where(gromos.alt>alt_min[i] , drop=True).where(gromos.alt<alt_max[i], drop=True)
        somora_p = somora.where(somora.alt>alt_min[i] , drop=True).where(somora.alt<alt_max[i], drop=True)
        
        gromos_p = gromos_p.o3_x.mean(dim='alt').resample(time=freq).median()
        somora_p = somora_p.o3_x.mean(dim='alt').resample(time=freq).median()

        # anomalies
        anomalies_gromos_p = gromos_p - gromos_p.rolling(time=30, center=True, min_periods=1).mean()
        anomalies_somora_p = somora_p - somora_p.rolling(time=30, center=True, min_periods=1).mean()
        anomalies_gromos_p=100*anomalies_gromos_p/gromos_p.median(dim='time')
        anomalies_somora_p=100*anomalies_somora_p/somora_p.median(dim='time')
        
        # New tempera
        tempera_temperature = tempera.tmp.where(tempera.altitude>alt_min[i] , drop=True).where(tempera.altitude<alt_max[i], drop=True)
        tempera_pressure = tempera.p.where(tempera.altitude>alt_min[i] , drop=True).where(tempera.altitude<alt_max[i], drop=True)

        # #corresponding altitude:
        p_min =1e-2*tempera_pressure.mean(dim='time').data[-1]
        p_max =1e-2*tempera_pressure.mean(dim='time').data[0]

        print('O3 and Temperature between '+f'{alt_min[i]:.1f}'+ r' < z < '+f'{alt_max[i]:.1f}'+' km'+' or '+f'{p_min:.1f}'+ r' < p < '+f'{p_max:.1f}'+' hPa')

        tempera_p=tempera_temperature.mean(dim='altitude').resample(time=freq).median()
        #dT = tempera_p - tempera_p.mean(dim='time')
        anomalies_tempera = tempera_p - tempera_p.rolling(time=30, center=True, min_periods=1).mean()
        anomalies_tempera = 100*anomalies_tempera/tempera_p.median(dim='time')

        gromos_p = gromos_p.resample(time=freq).median()
        somora_p = somora_p.resample(time=freq).median()
        tempera_p = tempera_p.resample(time=freq).median()
        
        Do3_theoric = -2.15*anomalies_tempera/gromos_p # + midday_o3_gromos.mean(dim='time')

        ds_merged_anomalies=xr.merge((
            {'ozone':anomalies_gromos_p},
            {'ozone_somora':anomalies_somora_p},
            {'temperature':anomalies_tempera},
            # {'tropospheric_opacity':gromos_p.tropospheric_opacity.resample(time=freq).mean()},
        ))

        # Clean the ds to remove any Nan entries:
        ds_merged_gromora = ds_merged_anomalies.where(ds_merged_anomalies.ozone.notnull(), drop=True).where(ds_merged_anomalies.ozone_somora.notnull(), drop=True)

        ds_merged_gromos = ds_merged_anomalies.where(ds_merged_anomalies.ozone.notnull(), drop=True).where(ds_merged_anomalies.temperature.notnull(), drop=True)
        ds_merged_somora = ds_merged_anomalies.where(ds_merged_anomalies.ozone_somora.notnull(), drop=True).where(ds_merged_anomalies.temperature.notnull(), drop=True)

        pearson_corr_gromora = xr.corr(ds_merged_gromora.ozone,ds_merged_gromora.ozone_somora, dim='time')
        pearson_corr = xr.corr(ds_merged_gromos.ozone,ds_merged_gromos.temperature, dim='time')
        pearson_corr_somora = xr.corr(ds_merged_somora.ozone_somora,ds_merged_somora.temperature, dim='time')
        print('####################################################')
        print('Pearson corr O3 anomalies GROMOS - SOMORA: ',f'{pearson_corr_gromora.values:.3f}')
        print('Pearson corr Temp-O3 anomalies GROMOS: ',f'{pearson_corr.values:.3f}')
        print('Pearson corr Temp-O3 anomalies SOMORA: ',f'{pearson_corr_somora.values:.3f}')
        print('##########################')


        #gromos_anomalies = (gromos_p.o3_x.mean(dim='o3_p').groupby('time.dayofyear') - gromos_p.o3_x.mean(dim='o3_p').groupby('time.dayofyear').mean()).resample(time=freq).mean()
        #tempera_anomalies = (tempera_p.temperature.mean(dim='altitude').groupby('time.dayofyear') - tempera_p.o3_x.mean(dim='o3_p').groupby('time.dayofyear').mean()).resample(time=freq).mean()
        anomalies_gromos_p.plot(ax=axs[i], color=color_gromos, lw=2, label='GROMOS')
        anomalies_somora_p.plot(ax=axs[i], color=color_somora, lw=2, label='SOMORA')
        
        # if ( alt_min[i] > 35) & (alt_min[i] < 50):
        #     Do3_theoric.plot(ax=axs[i], color='k', lw=2, label='THEORY')

        anomalies_tempera.resample(time=freq).median().plot(ax=axT, color=get_color('GDOC'), lw=2, label='TEMPERA')
        axs[i].set_title(r'O$_3$ and T anomalies between '+f'{alt_min[i]:.1f}'+' and '+f'{alt_max[i]:.1f}'+' km', fontsize=fs+2)
        #axs[0].set_title(r'Mean O$_3$ VMR at $'+ str(alt_min[i])+ ' < z < '+str(alt_max[i])+'$ km',fontsize=fs+2)

        #axs[1].set_title(r'Mean Temperature VMR at '+ f'{alt_min[i]:.1f}'+ r' $< z <$ '+f'{alt_max[i]:.1f}'+' km',fontsize=fs+2)
        axT.set_ylabel(r'$\Delta$T [\%]', color=get_color('GDOC'))
        axT.set_title('')
        axT.tick_params(axis='y', colors=get_color('GDOC'))
        axT.yaxis.label.set_color(get_color('GDOC'))

        axs[i].axhline(0,0, 1, color='k', lw=2, ls='--')

        axs[i].set_ylim(-25,25)
        axT.set_ylim(-7,7)
        axT.legend(loc=1)
        # if mean_alt > 40:
        #     axs[0].set_ylim(3,7)
        #     axs[1].set_ylim(220,280)
        # elif ((mean_alt > 30) & (mean_alt < 40)):
        #     axs[0].set_ylim(4,10)
        #     axs[1].set_ylim(200,260)
        # else:
        #     axs[0].set_ylim(3,8)
        #     axs[1].set_ylim(190,240)

        # if mean_alt > 50:
        #     axs[0].set_ylim(1,3)
        
        #ax.set_xlim(pd.to_datetime('2009-09-23'), pd.to_datetime('2022-01-01'))
        axs[i].set_ylabel(r'$\Delta$O$_3$ [\%]', fontsize=fs)
        # axs[0].yaxis.set_major_locator(MultipleLocator(1))
        # axs[0].yaxis.set_minor_locator(MultipleLocator(0.5))
        #axs[i].xaxis.set_major_locator(mdates.YearLocator())
        #axs[i]x.xaxis.set_major_locator(mdates.MonthLocator(bymonth=(1, 7)))
        axs[i].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        axs[i].tick_params(axis='both', which='major', labelsize=fs-2)


        # plot winter spans in background
        # calculate days that are in the winter season (DJF)
        # 0 = DJF, 1 = MAM, 2 = JJA, 3 = SON
        is_winter = (gromos.time.dt.month % 12 // 3) == 0
        winter_time_indexes = np.argwhere(is_winter.values).flatten()
        # calculate each consecutive index as a winter period
        winters = consecutive(winter_time_indexes)
        for winter in winters:
            time_min, time_max = gromos.time.values[winter.min()], gromos.time.values[winter.max()]
            # axs[i].axvspan(time_min, time_max, facecolor='grey', alpha=0.2)

        axs[i].legend(loc=2)
        axs[i].grid()
        axs[i].set_xlabel('')
        axs[i].set_xlim(pd.to_datetime(gromos.time.data[0])-pd.Timedelta(hours=12),pd.to_datetime(gromos.time.data[-1])+pd.Timedelta(hours=12))



    fig.tight_layout(rect=[0, 0.01, 0.99, 1])
    fig.savefig(basefolder+'ozone_temperature_anomalies_comparison_altitude_level_'+str(year)+'.pdf', dpi=500)

    # axs1.plot(delta_o3, color=get_color('GROMOS'))
    # axs1.set_ylabel(r'DO3')
    # axs2 = axs1.twinx()
    # axs2.plot(delta_T, color=get_color('SOMORA'))
    # axs2.set_ylabel(r'DT')
    # fig1.tight_layout(rect=[0, 0.01, 0.99, 1])
    # fig1.savefig(basefolder+'detlaO3_vs_deltaT'+str(year)+'.pdf', dpi=500)

def correlation_profile_gromos_tempera(gromos, tempera, alt_grid, freq='1D', basefolder=''):
    fs=32
    year=pd.to_datetime(gromos.time.data[0]).year
    
    print('###########################################################')
    fig, axs = plt.subplots(1, 3, sharey=True, figsize=(24,14))

    # Dealing with ozone :
    gromos_interp = gromos.o3_x.interp(alt=alt_grid).resample(time=freq).mean()
    
    # New tempera
    tempera_interp = tempera.tmp.interp(altitude=alt_grid).resample(time=freq).mean().rename({'altitude':'alt'})

    ds_merged=xr.merge((
        {'ozone':gromos_interp},
        {'temperature':tempera_interp},
        # {'tropospheric_opacity':gromos_p.tropospheric_opacity.resample(time=freq).mean()},
    ))

    #season = ['DJF','MAM', 'JJA', 'SON']
    color_season = ['r', 'b', 'y', 'g']
    color_season = ['#377eb8', '#e41a1c', '#4daf4a', '#ff7f00']
    marker_season = ['s', 'o', 'D', 'X']
    fill_styles=['none','none', 'full', 'full']
    ms = 9

    # Clean the ds to remove any Nan entries:
    ds_merged_o3_T = ds_merged.where(ds_merged.ozone.notnull(), drop=True).where(ds_merged.temperature.notnull(), drop=True)

    ds_o3_gromora_groups = ds_merged_o3_T.groupby('time.season').groups
    #ds_o3_gromora_plot = ds_o3_gromora.isel(o3_p=pressure_level)

    for j, s in enumerate(ds_o3_gromora_groups):
        print("#################################################################################################################### ")
        print('Processing season ', s)
        ds = ds_merged_o3_T.isel(time=ds_o3_gromora_groups[s]).interpolate_na(dim='time',fill_value="extrapolate")
        pearson_corr = xr.corr(ds.ozone,ds.temperature, dim='time')
    
        mean_o3 = ds.ozone.mean(dim='time')
        std_o3 = ds.ozone.std(dim='time')
        mean_T = ds.temperature.mean(dim='time')
        std_T = ds.temperature.std(dim='time')

        mean_o3.plot(ax=axs[0], y='alt', color=color_season[j], lw=2, label=s)
        mean_T.plot(ax=axs[1], y='alt', color=color_season[j], lw=2,  label=s)

        axs[0].fill_betweenx(ds.alt.data,mean_o3.data-0.5*std_o3.data,mean_o3.data+0.5*std_o3.data, color=color_season[j], alpha=0.2)
        axs[1].fill_betweenx(ds.alt.data,mean_T.data-0.5*std_T.data,mean_T.data+0.5*std_T.data, color=color_season[j], alpha=0.2)

        pearson_corr.plot(ax=axs[2], y='alt', color=color_season[j], lw=2, label=s)

    axs[0].set_title(r'Ozone',fontsize=fs+2)
    axs[1].set_title(r'Temperature',fontsize=fs+2)
    axs[2].set_title(r'Pearson correlation: '+freq,fontsize=fs+2)

    #     if mean_alt > 40:
    #         axs[0].set_ylim(3,7)
    #         axs[1].set_ylim(220,280)
    #     elif ((mean_alt > 30) & (mean_alt < 40)):
    #         axs[0].set_ylim(4,10)
    #         axs[1].set_ylim(200,260)
    #     else:
    #         axs[0].set_ylim(3,8)
    #         axs[1].set_ylim(190,240)

    #     if mean_alt > 50:
    #         axs[0].set_ylim(1,3)
    #     axs[1].set_ylabel('T [K]', fontsize=fs)

    axs[0].set_xlim(0,9)
    axs[0].set_ylim(25,60)
    axs[0].set_ylabel('Altitude [km]', fontsize=fs)
    axs[0].set_xlabel(r'O$_3$ VMR [ppmv]', fontsize=fs)

    axs[1].set_xlim(200,280)
    axs[1].set_xlabel(r'T [K]', fontsize=fs)
    axs[1].set_ylabel('')

    axs[2].set_xlim(-1,1)
    axs[2].set_xlabel(r'R', fontsize=fs)
    axs[2].set_ylabel('')

    #     axs[0].yaxis.set_major_locator(MultipleLocator(1))
    #     axs[0].yaxis.set_minor_locator(MultipleLocator(0.5))
    #     #axs[i].xaxis.set_major_locator(mdates.YearLocator())
    #     #axs[i]x.xaxis.set_major_locator(mdates.MonthLocator(bymonth=(1, 7)))
    # axs[1].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    # axs[1].tick_params(axis='both', which='major', labelsize=fs-2)

    for ax in axs:
        ax.legend()
        ax.grid()

    fig.tight_layout(rect=[0, 0.01, 0.99, 1])
    fig.savefig(basefolder+'ozone_temperature_correlation_profile_'+freq+'.pdf', dpi=500)

#########################################################################################################
# Main function
#########################################################################################################
if __name__ == "__main__":
    yr = 2010
    # The full range:
    date_slice=slice('2014-01-01','2017-12-31')

    # date_slice=slice('2014-12-25','2015-01-15')
    # date_slice=slice('2014-02-15','2014-04-15')

    years = [2014, 2015, 2016, 2017] #[2014, 2015, 2016, 2017]
    
    instNameGROMOS = 'GROMOS'

    # By default, we use the latest version with L2 flags
    v2 = True
    flagged_L2 = False
    
    fold_gromos = '/storage/tub/instruments/gromos/level2/GROMORA/v3/'# #'/scratch/GROSOM/Level2/GROMOS/v2/'
    fold_gromos2 = '/storage/tub/instruments/gromos/level2/GROMORA/v3/' # '/scratch/GROSOM/Level2/GROMOS/v3/'
    prefix_FFT='_AC240_v3'

    ########################################################################################################
    # Different strategies can be chosen for the analysis:
    # 'read': default option which reads the full level 2 doing the desired analysis
    # 'read_save': To save new level 3 data from the full hourly level 2
    # 'plot_all': the option to reproduce the figures from the manuscript
    # 'anything else': option to read the level 3 data before doing the desired analysis

    strategy = 'zgrid'
    if strategy[0:4]=='read':
        read_gromos=True
        read_somora=False
        read_both=False

        # if len(years)>4 and read_both:
        #     raise ValueError('This will take too much space sorry !')

        if read_gromos or read_both:
            gromos = read_GROMORA_all(
                basefolder=fold_gromos, 
                instrument_name=instNameGROMOS,
                date_slice=date_slice,#slice('2010-01-01','2021-12-31'), 
                years=years,#, [2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021]
                prefix=prefix_FFT,
                flagged=flagged_L2,
                decode_time=False
            )
            gromos['o3_x'] = 1e6*gromos['o3_x'].where((gromos['o3_x']>gromos['o3_x'].valid_min)&(gromos['o3_x']<gromos['o3_x'].valid_max), drop = True)
            gromos_clean = gromos.where(gromos.retrieval_quality==1, drop=True)#.where(gromos.level2_flag==0, drop=True)#.where(gromos.o3_mr>0.8)
            print('GROMOS FFT good quality level2: ', 100*len(gromos_clean.time)/len(pd.date_range('2009-07-01', '2021-12-31 23:00:00', freq='1H')) )
        
        if read_somora or read_both:
            somora = read_GROMORA_all(
                basefolder=fold_gromos2, 
                instrument_name=instNameGROMOS,
                date_slice=date_slice, 
                years=years, #[1995, 1996, 1997, 1998, 1999, 2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011],
                prefix=prefix_FB,  # '_v2_all.nc'#
                flagged=flagged_L2,
                decode_time=False
            )
            somora['o3_x'] = 1e6*somora['o3_x'].where((somora['o3_x']>somora['o3_x'].valid_min)&(somora['o3_x']<somora['o3_x'].valid_max), drop = True)
            somora_clean = somora.where(somora.retrieval_quality==1, drop=True)#.where(somora.level2_flag==0, drop=True)#.where(gromos.o3_mr>0.8)
            print('GROMOS FB good quality level2: ', 100*len(somora_clean.time)/len(pd.date_range('2009-09-23', '2021-12-31 23:00:00', freq='1H')) )
            
        if not flagged_L2:
            if read_gromos or read_both:
                gromos = add_flags_level2_gromora(gromos, 'GROMOS')
                gromos_clean = gromos.where(gromos.retrieval_quality==1, drop=True).where(gromos.level2_flag==0, drop=True)#.where(gromos.o3_mr>0.8)

            if read_somora or read_both:
                somora = add_flags_level2_gromora(somora, 'FB')
                somora_clean = somora.where(somora.retrieval_quality==1, drop=True).where(somora.level2_flag==0, drop=True)#.where(gromos.o3_mr>0.8)
        
        #print('GROMOS good quality level2: ', 100*len(gromos_clean.time)/len(pd.date_range('2020-01-01', '2020-12-31 23:00:00', freq='1H')) )
        #print('SOMORA good quality level2: ', 100*len(somora_clean.time)/len(pd.date_range('2020-01-01', '2020-12-31 23:00:00', freq='1H')) )
        if strategy=='read_save':
            # Saving the level 3:
            if read_gromos:
                gromos_clean.resample(time='6H').mean().to_netcdf('/scratch/GROSOM/Level2/GROMOS_level3_6H_v3.nc')
            elif read_somora:
                somora_clean.resample(time='6H').mean().to_netcdf('/scratch/GROSOM/Level2/GROMOS_FB_level3_6H_v3.nc')
            exit()
    elif strategy == 'zgrid':
        #gromos_clean = xr.open_dataset('/storage/nas/MW/projects/GROMOS_TEMPERA/Data/GROMOS_v3/GROMOS_level3_v3_regridded_2014-2017.nc')
        gromos_clean = xr.open_dataset('/storage/nas/MW/projects/GROMOS_TEMPERA/Data/GROMORA_v2/GROMOS_level2_v2_regridded_2014-2017.nc')
        gromos_clean = gromos_clean.sel(time=date_slice)
        somora_clean = xr.open_dataset('/storage/nas/MW/projects/GROMOS_TEMPERA/Data/GROMORA_v2/SOMORA_level2_v2_regridded_2014-2017.nc')
        somora_clean = somora_clean.sel(time=date_slice)
    else:
        gromos_clean = read_GROMORA_concatenated('/scratch/GROSOM/Level2/GROMOS_level3_6H_v3.nc', date_slice)
        somora_clean = read_GROMORA_concatenated('/scratch/GROSOM/Level2/SOMORA_level3_6H_v2.nc', date_slice)
        
        # gromos_clean = gromos.where(gromos.level2_flag==0, drop=True)#.where(gromos.o3_mr>0.8)
        # somora_clean = somora.where(somora.level2_flag==0, drop=True)#.where(gromos.o3_mr>0.8)
        
        print('GROMOS good quality level2: ', 100*len(gromos_clean.time)/len(pd.date_range('2009-07-01', '2021-12-31 23:00:00', freq='6H')) )

    #####################################################################
    # Read SBUV and MLS
    bn = '/storage/tub/atmosphere/SBUV/O3/daily_mean_overpasses/'
    sbuv = read_SBUV_dailyMean(timerange=date_slice, SBUV_basename = bn, specific_fname='sbuv_v87.mod_v2r1.vmr.payerne_156.txt')
    sbuv_arosa = read_SBUV_dailyMean(date_slice, SBUV_basename = bn, specific_fname='sbuv_v87.mod_v2r1.vmr.arosa_035.txt')

    outfolder = '/storage/nas/MW/projects/GROMOS_TEMPERA/Results/comparisons/' #'/scratch/GROSOM/GROMORA_TEMPERA/'

    mls= read_MLS(timerange=date_slice, vers=5, filename_MLS='AuraMLS_L2GP-O3_v5_400-800_BERN_2004-2022.nc')#slice('2003-01-01','2021-12-31')

    #gromos_colloc, gromos_conv, mls_gromos_colloc, mls_gromos_conv, somora_colloc, somora_conv, mls_somora_colloc, mls_somora_conv = read_all_MLS(yrs = years)
    gromos_colloc, gromos_conv, mls_gromos_colloc, mls_gromos_conv = read_MLS_convolved(
        instrument_name='GROMOS', 
        folder='/scratch/GROSOM/Level2/MLS/', 
        years=years
        )
    # mls_temp = read_MLS_Temperature(date_slice, vers=5, filename_MLS='AuraMLS_L2GP-TEMP_v5_400-800_BERN.nc')

    # mls = read_MLS(date_slice, vers=5, filename_MLS='AuraMLS_L2GP-O3_v5_400-800_BERN_lst.nc', save_LST=False)
    mls_temp = read_MLS_Temperature(date_slice, vers=5, filename_MLS='AuraMLS_L2GP-TEMP_v5_400-800_BERN.nc')

    #####################################################################
    # Read TEMPERA
    if strategy != 'zgrid':
        tempera = read_tempera_level3(date_slice=date_slice)
        tempera = tempera.where(tempera.nr>2)
    else:
        tempera = read_tempera_level3_concat(folder = '/storage/nas/MW/projects/GROMOS_TEMPERA/Data/TEMPERA/level_2_v_2023/',file='TEMPERA_level3_2014_2017_v3_filtered.nc', date_slice=date_slice)

    #tempera_old = read_tempera_level2(date_slice=date_slice)
     
    #####################################################################
    # Convert altitude to km
    tempera['altitude'] = 1e-3*tempera.altitude
    gromos_clean['alt'] = 1e-3*gromos_clean.alt
    somora_clean['alt'] = 1e-3*somora_clean.alt

    #####################################################################
    # Read ERA5
    ERA5 = False
    if ERA5:
        #Reading ECMWF data:
        range_ecmwf = date_slice #slice('2018-01-01','2018-12-31')
        date = pd.date_range(start=range_ecmwf.start, end=range_ecmwf.stop)

        ecmwf_ts = read_ERA5(date, years, location='SwissPlateau', daybyday=False, save=False)

        ecmwf_ts = ecmwf_ts.sel(time=date_slice)
        ecmwf_ts['level'] = ecmwf_ts.pressure.mean(dim='time').data
        
        # Conversion to LST and o3 to VMR
        ecmwf_ts['O3'] =mmr_2_vmr (ecmwf_ts.ozone_mass_mixing_ratio)#.resample(time='1D').mean()
        ecmwf_ts = ecmwf_ts.rename({'level':'lev'})
    else:
        ecmwf_ts = None

    #####################################################################
    # Compare at P level
    plot_plev = False
    if plot_plev:
        compare_pressure_gromos_tempera(gromos_clean, somora_clean, tempera, tempera_old, mls, mls_temp, ecmwf_ts, sbuv, p_min=[20] , p_max=[30], freq='1D', basefolder=outfolder)
        compare_pressure_gromos_tempera(gromos_clean, somora_clean, tempera, tempera_old, mls, mls_temp, ecmwf_ts, sbuv, p_min=[0.5] , p_max=[1], freq='1D', basefolder=outfolder)
        # compare_pressure_gromos_tempera(gromos_clean, somora_clean, tempera, tempera_old, mls, mls_temp, ecmwf_ts, sbuv, p_min=[3.5] , p_max=[6.5], freq='1D', basefolder=outfolder)
    
    #####################################################################
    # Compare at altitude level
    plot_alt_lev = False
    if plot_alt_lev:
        compare_altitude_gromos_tempera(gromos_clean, somora_clean, tempera, mls, mls_o3_convolved=False, mls_temp=mls_temp, ecmwf_ts=None, alt_min=[25,45] , alt_max=[30,50], freq='2D', basefolder=outfolder)
        #compare_altitude_gromos_tempera(gromos_clean, somora_clean, tempera, mls, mls_o3_convolved=False, mls_temp=mls_temp, ecmwf_ts=None, alt_min=[20,30,40,50] , alt_max=[30,40,50,60], freq='1D', basefolder=outfolder)
        # compare_altitude_gromos_tempera(gromos_clean, somora_clean, tempera, tempera_old, mls, mls_temp, ecmwf_ts, alt_min=[30] , alt_max=[40], freq='1D', basefolder=outfolder)
        # compare_altitude_gromos_tempera(gromos_clean, somora_clean, tempera, tempera_old, mls, mls_temp, ecmwf_ts, alt_min=[40] , alt_max=[50], freq='1D', basefolder=outfolder)
        compare_altitude_anomalies_gromos_tempera(gromos_clean, somora_clean, tempera, alt_min=[45, 25] , alt_max=[50,30], freq='2D', basefolder=outfolder)


    #####################################################################
    # Time series 2D plots
    plot_2D = False
    if plot_2D:
        compare_ts_gromos_tempera(gromos_clean, somora=None, tempera=tempera, freq='1D', date_slice=date_slice, basefolder=outfolder)
    
    #####################################################################
    # Correlation plots
    plot_corr = True
    if plot_corr:
        correlation_profile_gromos_tempera(gromos_clean,tempera,alt_grid=np.arange(15,75,4), freq='10D', basefolder=outfolder)

    #####################################################################
    # Comparison with old retrievals
    plot_old_new = False
    if plot_old_new:

        mean_bias = xr.open_dataarray('/scratch/GROSOM/Level2/GROMOS_v3/mean_bias_FB-FFT_v3.nc')

        date_split = '2011-01-01'
        date_split2 = '2011-01-01'
        freq =  '1M'
        gromos_old_FB = gromos_old_FB.sel(time=slice(date_slice.start,date_split)).resample(time=freq).mean()
        gromos_v2021 = gromos_v2021.sel(time=slice(date_split2,date_slice.stop)).resample(time=freq).mean()
        somora_clean = somora_clean.sel(time=slice(date_slice.start,date_split)).resample(time=freq).mean()
        gromos_clean = gromos_clean.sel(time=slice(date_split2,date_slice.stop)).resample(time=freq).mean()

        somora_clean['o3_corr'] = somora_clean.o3_x - mean_bias

        gromos_all = xr.concat([somora_clean.o3_corr, gromos_clean.o3_x], dim='time')
        gromos_old_all = xr.concat([gromos_old_FB.o3_x, gromos_v2021.o3_x],  dim='time')

        # gromos_old_vs_new(gromos_all, gromos_old_all, freq='1M', basefolder=outfolder)
        #gromos_old_vs_new(gromos_clean, gromos_v2021, mls, seasonal=False)
        #gromos_old_vs_new(somora_clean, somora_old, mls, seasonal=False)
        # trends_diff_old_new(gromos_clean.o3_x.sel(time=slice('2010-01-01', '2021-12-31')), gromos_v2021.o3_x.sel(time=slice('2010-01-01', '2021-12-31')), p_min=[0.1, 1, 10] , p_max=[0.9, 5, 50], freq='7D', freq_avg='1M',  outfolder=outfolder)
        # trends_diff_old_new(somora_clean.o3_x.sel(time=slice('1998-01-01', '2010-12-31')), gromos_old_FB.o3_x.sel(time=slice('1998-01-01', '2010-12-31')), p_min=[0.1, 1, 10] , p_max=[0.9, 5, 50], freq='7D', freq_avg='1M',  outfolder=outfolder)

        trends_diff_old_new(gromos_all.sel(time=slice('2007-01-01', '2021-12-31')), gromos_old_all.sel(time=slice('2007-01-01', '2021-12-31')), p_min=[0.02, 0.1, 1, 10] , p_max=[0.08, 0.9, 5, 50], freq='7D', freq_avg='1M',  outfolder=outfolder)
        trends_diff_old_new(gromos_all, gromos_old_all, p_min=[0.02, 0.1, 1, 10] , p_max=[0.08,0.9, 5, 50], freq='7D', freq_avg='1M',  outfolder=outfolder)

        #trends_simplified_new(gromos_clean, somora_clean, mls, p_min=[0.1, 1, 10] , p_max=[0.9, 5, 50], freq='7D', freq_avg='1M',  outfolder=outfolder)


    #gromos = utc_to_lst(gromos)
    #somora = utc_to_lst(somora)

    #compute_compare_climatology(somora, slice_clim=slice("2010-01-01", "2020-12-31"), slice_plot=slice("2022-01-01", "2022-01-31"), percentile=[0.1, 0.9], pressure_level = [25, 21, 15], basefolder=outfolder)

    #compare_pressure(gromos_clean, somora_clean, pressure_level=[31, 25, 21, 15, 12], add_sun=False, freq='6H', basefolder='/scratch/GROSOM/Level2/GROMORA_retrievals_v2/')
    
    #####################################################################
    # Relative difference GROMOS vs SOMORA
    compare_gromora = False
    if compare_gromora:
        map_rel_diff(gromos_clean, somora_clean, freq='1M', basefolder=outfolder, FB=True)
        compare_ts_gromora(gromos_clean, somora_clean, date_slice=date_slice, freq='1D', basefolder=outfolder, paper=True, FB=True)

        # compute_corr_profile(somora_sel,mls_somora_colloc,freq='7D',basefolder='/scratch/GROSOM/Level2/MLS/')
        # #compare_diff_daily(gromos ,somora, gromora_old, pressure_level=[34 ,31, 25, 21, 15, 12], altitudes=[69, 63, 51, 42, 30, 24])
        compare_mean_diff(gromos_clean, somora_clean, sbuv = sbuv, mls=mls, basefolder=outfolder, corr_FFT=False)

        compare_mean_diff_monthly(gromos_clean, somora_clean, mls, sbuv, outfolder=outfolder)

    # gromos_linear_fit = gromos_clean.o3_x.where((gromos_clean.o3_p<p_high) & (gromos_clean.o3_p>p_low), drop=True).mean(dim='o3_p').resample(time='1M').mean()#) .polyfit(dim='time', deg=1)
    # somora_linear_fit = somora_clean.o3_x.resample(time='1M').mean().polyfit(dim='time', deg=1)

    

    #####################################################################
    # MLS comparisons
    compare_with_MLS = False
    if compare_with_MLS:
        gromos_colloc, gromos_conv, mls_gromos_colloc, mls_gromos_conv, somora_colloc, somora_conv, mls_somora_colloc, mls_somora_conv = read_all_MLS(yrs = years)

        # compute_correlation_MLS(gromos_sel, mls_gromos_colloc_conv, freq='1M', pressure_level=[ 15,20,25], basefolder='/scratch/GROSOM/Level2/GROMORA_waccm/GROMOS')
        # compute_correlation_MLS(somora_sel, mls_somora_colloc_conv, freq='1M', pressure_level=[15,20,24], basefolder='/scratch/GROSOM/Level2/GROMORA_waccm/SOMORA')
        # compare_GROMORA_MLS_profiles(gromos_colloc, gromos_conv, mls_gromos_colloc, mls_gromos_conv, somora_colloc, somora_conv, mls_somora_colloc, mls_somora_conv, mls, freq='7D', basefolder='/scratch/GROSOM/Level2/        Comparison_MLS/', convolved=False)
        #compare_GROMORA_MLS_profiles(gromos_sel, somora_sel, mls_gromos_colloc_conv, mls_somora_colloc_conv, mls,basefolder='/scratch/GROSOM/Level2/GROMORA_retrievals_v2/', convolved=True)
        # compare_seasonal_GROMORA_MLS_profiles(gromos_colloc, gromos_conv, mls_gromos_colloc, mls_gromos_conv, somora_colloc, somora_conv, mls_somora_colloc, mls_somora_conv, mls, sbuv, basefolder='/scratch/GROSOM/Level2/Comparison_MLS/', convolved=True, split_night=True)
        # compare_seasonal_GROMORA_MLS_profiles(gromos_colloc, gromos_conv, mls_gromos_colloc, mls_gromos_conv, somora_colloc, somora_conv, mls_somora_colloc, mls_somora_conv, mls, sbuv, basefolder='/scratch/GROSOM/Level2/Comparison_MLS/', convolved=False, split_night=True)
    
        compare_seasonal_GROMORA_MLS_profiles(gromos_colloc, gromos_conv, mls_gromos_colloc, mls_gromos_conv, somora_colloc, somora_conv, mls_somora_colloc, mls_somora_conv, mls, sbuv, p_min=[0.1, 1, 10] , p_max=[0.9, 5, 50], basefolder='/scratch/GROSOM/Level2/Comparison_MLS/', convolved=False, split_night=False)
        compare_GROMORA_MLS_profiles_egu(gromos_colloc, gromos_conv, mls_gromos_colloc, mls_gromos_conv, somora_colloc, somora_conv, mls_somora_colloc, mls_somora_conv, mls, sbuv, p_min=[0.1, 1, 10] , p_max=[0.9, 5, 50], basefolder='/scratch/GROSOM/Level2/Comparison_MLS/', convolved=False, split_night=False)


    #####################################################################
    # apriori comparisons
    # compare_with_apriori(gromos, freq='1H', date_slice=date_slice,basefolder='/scratch/GROSOM/Level2/GROMORA_retrievals_v2/GROMOS_')
    # compare_with_apriori(somora, freq='1H', date_slice=date_slice,basefolder='/scratch/GROSOM/Level2/GROMORA_retrievals_v2/SOMORA_')

    #  plot_o3_pressure_profile(gromos)

    #####################################################################
    # Correlations GROMOS-SOMORA
    compute_corr_GROMORA = False
    if compute_corr_GROMORA:
        # compute_seasonal_correlation(gromos_clean, somora_clean, freq='6H', pressure_level=[25, 20, 15], basefolder=outfolder) p_min=[0.1, 1, 11] , p_max=[0.9, 10, 50]
        compute_seasonal_correlation_FB_FFT(gromos_clean, somora_clean, freq='6H', p_min=[0.1, 1, 10] , p_max=[0.9, 5, 50], basefolder=outfolder)

        #compute_correlation(gromos_clean, somora_clean, freq='6H', pressure_level=[36, 31, 21, 12], basefolder=outfolder)
    
        #compute_corr_profile(gromos_clean, somora_clean, freq='6H', date_slice=slice('2009-10-01','2021-12-31'), basefolder=outfolder)

