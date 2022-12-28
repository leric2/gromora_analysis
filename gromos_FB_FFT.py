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
from compare_gromora_v2 import compare_avkm, compare_mean_diff_monthly, compare_pressure_mls_sbuv_paper, compare_pressure_mls_sbuv, compare_ts_gromora, compute_seasonal_correlation_paper, map_rel_diff, trends_diff_old_new

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
PAPER_SYMBOL = ['a)','b)','c)'] # ['a) lower mesosphere','b) upper stratosphere','c) lower stratosphere']#


def gromos_old_vs_new(gromos, gromos_fb, gromos_v2021, gromos_fb_old, freq='7D', basefolder='/scratch/GROSOM/Level2/GROMOS_FB_VS_FFT'):
    year=pd.to_datetime(gromos.time.data[0]).year
    fig, axs = plt.subplots(2, 2, sharey=True, figsize=(24,12))

    # mean_diff = 100*(gromos.o3_x.mean(dim='time') - gromos_v2021.o3_x.mean(dim='time') )/gromos.o3_x.mean(dim='time')
    pl = gromos.o3_x.resample(time=freq).mean().plot(
        x='time',
        y='o3_p',
        ax=axs[0,1], 
        vmin=0,
        vmax=10,
        yscale='log',
        linewidth=0,
        rasterized=True,
        cmap='cividis'
    )
    pl.set_edgecolor('face')
    # ax.set_yscale('log')
    axs[0,0].invert_yaxis()
    axs[0,1].set_ylabel('P [hPa]')
    axs[0,1].set_title('FFT new')
   
    pl2 = gromos_v2021.o3_x.resample(time=freq).mean().plot(
        x='time',
        y='pressure',
        ax=axs[1,1], 
        vmin=0,
        vmax=10,
        yscale='log',
        linewidth=0,
        rasterized=True,
        cmap='cividis'
    )
    pl2.set_edgecolor('face')
    # ax.set_yscale('log')
    axs[1,1].set_ylabel('P [hPa]')
    axs[1,1].set_title('FFT old')

    pl3 = gromos_fb.o3_x.resample(time=freq).mean().plot(
        x='time',
        y='o3_p',
        ax=axs[0,0], 
        vmin=0,
        vmax=10,
        yscale='log',
        linewidth=0,
        rasterized=True,
        cmap='cividis'
    )
    pl3.set_edgecolor('face')
    axs[0,0].set_title('FB new')
   
    pl4 = gromos_fb_old.o3_x.resample(time=freq).mean().plot(
        x='time',
        y='pressure',
        ax=axs[1,0], 
        vmin=0,
        vmax=10,
        yscale='log',
        linewidth=0,
        rasterized=True,
        cmap='cividis'
    )
    pl4.set_edgecolor('face')
    # ax.set_yscale('log')
    axs[1,0].set_ylabel('P [hPa]')
    axs[1,0].set_title('FB old')

    axs[0,0].set_ylim(500,5e-3)

    #fig.suptitle('Ozone relative difference GROMOS new-old')
    fig.tight_layout(rect=[0, 0.01, 0.99, 1])
    fig.savefig(basefolder+'gromos_old_vs_new_'+str(year)+'.pdf', dpi=500)

def diff_2_mls_FB(somora, gromos_old, fb_old, mls, sbuv, p_min, p_max, freq='1D',basefolder=''):
    fs=32
    year=pd.to_datetime(somora.time.data[0]).year
    fig, axs = plt.subplots(len(p_min), 1, sharex=True, figsize=(28,18))
    for i, p_ind in enumerate(p_min):
        somora_p = somora.where(somora.o3_p>p_min[i] , drop=True).where(somora.o3_p<p_max[i], drop=True)
        gromos_old_p = gromos_old.where(gromos_old.pressure>p_min[i] , drop=True).where(gromos_old.pressure<p_max[i], drop=True)
        fb_old_p = fb_old.where(fb_old.pressure>p_min[i] , drop=True).where(fb_old.pressure<p_max[i], drop=True)
        mls_p = mls.where(mls.p>p_min[i] , drop=True).where(mls.p<p_max[i], drop=True)
        mls_p = somora_p.o3_x.reindex_like(mls_p, method='nearest', tolerance='1D') - mls_p.o3
        pressure =  somora_p.o3_p.mean().values
        mls_p.mean(dim='p').resample(time=freq).mean().plot(ax=axs[i], color='k', lw=2)
        if (pressure > 0.4) & (pressure<50) & (sbuv is not None):
            sbuv_p = sbuv.where(sbuv.p>p_min[i] , drop=True).where(sbuv.p<p_max[i], drop=True)
            sbuv_p = somora_p.o3_x.reindex_like(sbuv_p, method='nearest', tolerance='1D') - sbuv_p.ozone
            sbuv_p.mean(dim='p').resample(time=freq).mean().plot(ax=axs[i], color=sbuv_color, lw=2)

        #somora_p.o3_x.mean(dim='o3_p').resample(time=freq).mean().plot(ax=axs[i], color=color_gromos, ls='--', lw=2)
        #gromos_old_p.o3_x.mean(dim='pressure').resample(time=freq).mean().plot(ax=axs[i], color=color_somora, lw=2)
        #fb_old_p.o3_x.mean(dim='pressure').resample(time=freq).mean().plot(ax=axs[i], color=color_somora, ls='--', lw=2)
        axs[i].set_xlabel('')
        #axs[i].set_title(f'p = {pressure:.3f} hPa', fontsize=fs)
        axs[i].set_title(r'Mean O$_3$ VMR at $'+ str(p_min[i])+ ' < p < '+str(p_max[i])+'$ hPa',fontsize=fs+2)

        axs[i].text(
            0.024,
            0.05,
            PAPER_SYMBOL[i],
            transform=axs[i].transAxes,
            verticalalignment="bottom",
            horizontalalignment="left",
            fontsize=fs
        )
    axs[0].legend(['FB - MLS', 'FB- SBUV',], ncol=1, loc=1, fontsize=fs-2)
    #axs[0].legend(['OG','SB corr'], loc=1, fontsize=fs-2)

    # axs[0].set_ylim(1,3)
    # axs[1].set_ylim(4,8)
    # axs[2].set_ylim(3,7)
    #axs[3].set_ylim(2,10)
    #axs[4].set_ylim(2,10)

    for ax in axs:
        ax.grid()
        #ax.set_xlim(pd.to_datetime('2009-09-23'), pd.to_datetime('2022-01-01'))
        #ax.set_xlim(pd.to_datetime('2003-01-01'), pd.to_datetime('2018-12-31'))
        ax.set_ylabel('O$_3$ [ppmv]', fontsize=fs)
        # ax.yaxis.set_major_locator(MultipleLocator(1))
        # ax.yaxis.set_minor_locator(MultipleLocator(0.5))
        # ax.xaxis.set_major_locator(mdates.YearLocator())
        # #ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=(1, 7)))
        # ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        ax.tick_params(axis='both', which='major', labelsize=fs-2)

    fig.tight_layout(rect=[0, 0.01, 0.99, 1])
    fig.savefig(basefolder+'diff_FB_MLS_SBUV_'+str(year)+'.pdf', dpi=500)


def compare_pressure_mls_sbuv_old(gromos, somora, gromos_old, fb_old, mls, sbuv, p_min, p_max, add_sun=False, freq='1D',basefolder=''):
    fs=32
    year=pd.to_datetime(somora.time.data[0]).year
    fig, axs = plt.subplots(len(p_min), 1, sharex=True, figsize=(28,18))
    for i, p_ind in enumerate(p_min):

        gromos_p = gromos.where(gromos.o3_p>p_min[i] , drop=True).where(gromos.o3_p<p_max[i], drop=True)
        somora_p = somora.where(somora.o3_p>p_min[i] , drop=True).where(somora.o3_p<p_max[i], drop=True)
        gromos_old_p = gromos_old.where(gromos_old.pressure>p_min[i] , drop=True).where(gromos_old.pressure<p_max[i], drop=True)
        fb_old_p = fb_old.where(fb_old.pressure>p_min[i] , drop=True).where(fb_old.pressure<p_max[i], drop=True)
        mls_p = mls.where(mls.p>p_min[i] , drop=True).where(mls.p<p_max[i], drop=True)
        pressure =  gromos_p.o3_p.mean().values
        mls_p.o3.mean(dim='p').resample(time=freq).mean().plot(ax=axs[i], color='k', lw=2)
        if (pressure > 0.4) & (pressure<50) & (sbuv is not None):
            sbuv_p = sbuv.where(sbuv.p>p_min[i] , drop=True).where(sbuv.p<p_max[i], drop=True)
            sbuv_p.ozone.mean(dim='p').resample(time=freq).mean().plot(ax=axs[i], color=sbuv_color, lw=2)
        gromos_p.o3_x.mean(dim='o3_p').resample(time=freq).mean().plot(ax=axs[i], color=color_gromos, lw=2)
        somora_p.o3_x.mean(dim='o3_p').resample(time=freq).mean().plot(ax=axs[i], color=color_gromos, ls='--', lw=2)
        gromos_old_p.o3_x.mean(dim='pressure').resample(time=freq).mean().plot(ax=axs[i], color=color_somora, lw=2)
        fb_old_p.o3_x.mean(dim='pressure').resample(time=freq).mean().plot(ax=axs[i], color=color_somora, ls='--', lw=2)
        axs[i].set_xlabel('')
        #axs[i].set_title(f'p = {pressure:.3f} hPa', fontsize=fs)
        axs[i].set_title(r'Mean O$_3$ VMR at $'+ str(p_min[i])+ ' < p < '+str(p_max[i])+'$ hPa',fontsize=fs+2)
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
        axs[i].text(
            0.024,
            0.05,
            PAPER_SYMBOL[i],
            transform=axs[i].transAxes,
            verticalalignment="bottom",
            horizontalalignment="left",
            fontsize=fs
        )
    axs[0].legend(['MLS', 'FFT','FB', 'FFT_OLD','FB_OLD'], ncol=6, loc=1, fontsize=fs-2)
    #axs[0].legend(['OG','SB corr'], loc=1, fontsize=fs-2)

    axs[0].set_ylim(1,3)
    axs[1].set_ylim(4,8)
    axs[2].set_ylim(3,7)
    #axs[3].set_ylim(2,10)
    #axs[4].set_ylim(2,10)

    for ax in axs:
        ax.grid()
        #ax.set_xlim(pd.to_datetime('2009-09-23'), pd.to_datetime('2022-01-01'))
        #ax.set_xlim(pd.to_datetime('2003-01-01'), pd.to_datetime('2018-12-31'))
        ax.set_ylabel('O$_3$ [ppmv]', fontsize=fs)
        # ax.yaxis.set_major_locator(MultipleLocator(1))
        # ax.yaxis.set_minor_locator(MultipleLocator(0.5))
        # ax.xaxis.set_major_locator(mdates.YearLocator())
        # #ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=(1, 7)))
        # ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        ax.tick_params(axis='both', which='major', labelsize=fs-2)

    fig.tight_layout(rect=[0, 0.01, 0.99, 1])
    fig.savefig(basefolder+'ozone_comparison_pressure_level_MLS_SBUV_'+str(year)+'.pdf', dpi=500)

def compare_mean_diff(gromos, somora, mls=None, sbuv=None, basefolder=None, corr_FFT=False):
    color_shading='grey'
    fs = 22
    year=pd.to_datetime(somora.time.data[0]).year
    fig, axs = plt.subplots(1, 2, sharey=True, figsize=(15,20))
    if corr_FFT:
        ozone_FFT = gromos.o3_x.mean(dim='time')*1.08
    else:
        ozone_FFT = gromos.o3_x.mean(dim='time')
    mean_diff = somora.o3_x.mean(dim='time') - ozone_FFT
    mean_diff_new = 100*(somora.o3_x.mean(dim='time') - ozone_FFT )/ozone_FFT
    # mean_diff.to_netcdf('/storage/tub/instruments/gromos/mean_bias_FB-FFT.nc')
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

    ozone_FFT.plot(
        y='o3_p',
        ax=axs[0], 
        color=color_gromos
    )
    if sbuv:
        sbuv.ozone.mean(dim='time').plot(
            y='p',
            ax=axs[0], 
            color=get_color('SBUV')
        )
    if mls:
        mls.o3.mean(dim='time').plot(
            y='p',
            ax=axs[0], 
            color=get_color('MLS')
        )


    pl1 = mean_diff_new.plot(
        y='o3_p',
        ax=axs[1],
        yscale='log',
        color='k'
    )
    xa = 1e6*gromos.o3_xa.where(gromos.time.dt.hour==12, drop=True).median(dim='time')
    xa_nt = 1e6*gromos.o3_xa.where(gromos.time.dt.hour==0, drop=True).median(dim='time')
    xa.plot(
        y='o3_p',
        ax=axs[0], 
        yscale='log',
        color='#b2abd2'
    )
    xa_nt.plot(
        y='o3_p',
        ax=axs[0], 
        yscale='log',
        color='#5e3c99'
    )
    axs[0].set_title(r'O$_3$ VMR', fontsize=fs+4)
    axs[0].set_xlabel('VMR [ppmv]', fontsize=fs)
    axs[1].set_title(r'corr - FFT', fontsize=fs+2)
    #axs[1].set_title(r'OG-SB', fontsize=fs+2)
    # pl2 = mean_diff_old.plot(
    #     y='altitude',
    #     ax=axs[1], 
       
    # )
    axs[1].axvline(x=0,ls= '--', color='grey')
    axs[0].legend(('GROMOS corr','GROMOS FFT','SBUV','MLS', 'AP daytime', 'AP nighttime'))
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
    axs[1].xaxis.set_major_locator(MultipleLocator(20))
    axs[1].xaxis.set_minor_locator(MultipleLocator(5))
    #axs[1].set_ylim((somora.o3_z.mean(dim='time')[12]/1e3,somora.o3_z.mean(dim='time')[35]/1e3))


    for ax in axs:
        ax.grid(which='both', axis='y', linewidth=0.5)
        ax.grid(which='both', axis='x', linewidth=0.5)
        ax.tick_params(axis='both', which='major', labelsize=fs-2)
        ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: '{:g}'.format(y)))
        ax.fill_between(ax.get_xlim(),p_somora_mr[0],1e4, color=color_shading, alpha=0.2)
        ax.fill_between(ax.get_xlim(),p_somora_mr[-1],1e-4, color=color_shading, alpha=0.2)
        ax.fill_between(ax.get_xlim(),p_gromos_mr[0],1e4, color=color_shading, alpha=0.2)
        ax.fill_between(ax.get_xlim(),p_gromos_mr[-1],1e-4, color=color_shading, alpha=0.2)

    #fig.suptitle('Ozone relative difference GROMOS-SOMORA')
    fig.tight_layout(rect=[0, 0.01, 0.99, 1])

    if corr_FFT:
        fig.savefig(basefolder+'FB_vs_FFT_corr_'+str(year)+'.pdf', dpi=500)
    else:
        fig.savefig(basefolder+'FB_vs_FFT_'+str(year)+'.pdf', dpi=500)




def compute_seasonal_correlation_FB_FFT(gromos, somora, freq='1M', p_min = [100], p_max = [0.01], basefolder='', MLS = False, split_by ='season'):
    year=pd.to_datetime(gromos.time.data[0]).year
    fs=34
    figure_list = list()
    fig1, axs1 = plt.subplots(1,3, sharey=True, figsize=(22, 16))
    fig, axs = plt.subplots(len(p_min),3,figsize=(28, 8*len(p_min)))
    error_gromos = np.sqrt( np.square(gromos.o3_eo) )#  + np.square(gromos.o3_es))
    if MLS:
        error_somora = 1e-6*0.1*somora.o3_x
    else:
        error_somora  = np.sqrt( np.square(somora.o3_eo))#  + np.square(somora.o3_es))

    #ds_o3_gromora=xr.merge(({'o3_gromos':gromos.o3_x.isel(o3_p=p).resample(time=freq).mean()},{'o3_somora':somora.o3_x.isel(o3_p=p).resample(time=freq).mean()}))
    ds_o3_gromora=xr.merge((
        {'o3_gromos':gromos.o3_x.resample(time=freq).mean()},
        {'o3_somora':somora.o3_x.resample(time=freq).mean()},
        {'error_gromos':error_gromos.resample(time=freq).mean()},
        {'error_somora':error_somora.resample(time=freq).mean()},
        {'tropospheric_opacity':somora.tropospheric_opacity.resample(time=freq).mean()},
        ))

    # Clean the ds to remove any Nan entries:
    ds_o3_gromora = ds_o3_gromora.where(ds_o3_gromora.o3_gromos.notnull(), drop=True).where(ds_o3_gromora.o3_somora.notnull(), drop=True)

    #season = ['DJF','MAM', 'JJA', 'SON']
    color_season = ['r', 'b', 'y', 'g']
    marker_season = ['s', 'o', 'D', 'X']
    fill_styles=['none','none', 'full', 'full']
    ms = 9
    ds_o3_gromora_groups = ds_o3_gromora.groupby('time.season').groups
    #ds_o3_gromora_plot = ds_o3_gromora.isel(o3_p=pressure_level)
    ds_all = ds_o3_gromora#.interpolate_na(dim='time',fill_value="extrapolate")

    for j, s in enumerate(ds_o3_gromora_groups):
        print("#################################################################################################################### ")
        print("#################################################################################################################### ")
        print('Processing season ', s)
        ds = ds_o3_gromora.isel(time=ds_o3_gromora_groups[s]).interpolate_na(dim='time',fill_value="extrapolate")
        pearson_corr_profile = xr.corr(ds.o3_gromos, ds.o3_somora, dim='time')

        ds.o3_gromos.mean(dim='time').plot(
            ax=axs1[0],
            y='o3_p',
            yscale='log',
            color=color_gromos,
            marker=marker_season[j],
            markersize=ms,
            fillstyle=fill_styles[j],
            linewidth=0.5,
            ) 
        ds.o3_somora.mean(dim='time').plot(
            ax=axs1[0],
            y='o3_p',
            color=color_somora,
            marker=marker_season[j],
            markersize=ms,
            fillstyle=fill_styles[j],
            linewidth=0.5,
            ) 
        
        rel_mean_diff = 100*(ds.o3_somora.mean(dim='time') - ds.o3_gromos.mean(dim='time'))/ ds.o3_gromos.mean(dim='time')
        rel_mean_diff.plot(
            ax=axs1[1],
            y='o3_p',
            color='k',
            marker=marker_season[j],
            markersize=ms,
            fillstyle=fill_styles[j],
            label=s,
            linewidth=0.5,
            )
        
        pearson_corr_profile.plot(
            ax=axs1[2],
            y='o3_p',
            color='k',
            marker=marker_season[j],
            markersize=ms,
            fillstyle=fill_styles[j],
            linewidth=0.5,
        )

        for i, p in enumerate(p_min):
            print("#################################################################################################################### ")

            ds_p = ds.where(ds.o3_p>p_min[i] , drop=True).where(ds.o3_p<p_max[i], drop=True) #.isel(o3_p=np.arange(p, pressure_level1[i]))
            ds_all_p = ds_all.where(ds_all.o3_p>p_min[i] , drop=True).where(ds_all.o3_p<p_max[i], drop=True) #.isel(o3_p=np.arange(p, pressure_level1[i]))

            x = ds_p.o3_gromos.mean(dim='o3_p')
            y = ds_p.o3_somora.mean(dim='o3_p')

            pearson_corr = xr.corr(x,y, dim='time')
            print('Pearson corr coef: ',pearson_corr.values)
            xerr = 1e6*ds_p.error_gromos.mean(dim='o3_p')
            yerr = 1e6*ds_p.error_somora.mean(dim='o3_p')

            #  Fit using Orthogonal distance regression
            #  uses the retrievals total errors 
            print('Orthogonal distance regression:')

            result = regression_xy(
                x.values, y.values, x_err = xerr.values, y_err=yerr.values
            )
            error_odr = result.beta[0]*x.values + result.beta[1] - y.values
            SE_odr = np.square(error_odr)
            MSE_odr = np.mean(SE_odr)
            
            df = len(x) - 2
            tinv = lambda p, df: abs(stats.t.ppf(p/2, df))
            ts = tinv(0.05, df)
            
            coeff_determination = 1 - (np.var(error_odr)/np.var(y.values))
            # coeff_determination = calcR2_wikipedia(y.values, result.beta[1] + result.beta[0] * x.values)

            print('Slope ODR (95%) ', result.beta[0], ' +- ', result.sd_beta[0]*ts)
            print('Intercept ODR (95%) ', result.beta[1], ' +- ', result.sd_beta[1]*ts )
            print('R2 odr ', coeff_determination)
            print('RMSE ', np.sqrt(MSE_odr))
            print("########################################################## ")
            
            # Least square fit   
            print('Linear square fit:')
            result_stats = stats.linregress(x.values, y.values)
            
            print('Slope LS (95%) ', result_stats.slope, ' +- ', result_stats.stderr*ts)
            print('Intercept LS (95%) ', result_stats.intercept, ' +- ', result_stats.intercept_stderr*ts )


            print('r2: ',pearson_corr.values**2)
            print('R2 stats: ', result_stats.rvalue**2)

            print("########################################################## ")
            ds_2plot = ds_p
            title = s
            do_plot = True
            if s == 'DJF':
                col = 1
            elif s == 'JJA':
                col =2
            elif s == 'MAM':
                col= 0
                ds_2plot = ds_all_p
                title = 'all'
            else:
                do_plot=False
            
            if do_plot:

                pl = ds_2plot.mean(dim='o3_p').plot.scatter(
                    ax=axs[i,col],
                    x='o3_gromos', 
                    y='o3_somora',
                    hue='tropospheric_opacity',
                    hue_style='continuous',
                    vmin=0,
                    vmax=2,
                    marker='.',
                    cmap=plt.get_cmap('temperature'),
                    add_guide=False
                    # cbar_kwargs={'label':r'$\\tau$'}
                    # levels=np.linspace(0, 3, 10)
                )
                axs[i,col].plot([np.nanmin(ds_2plot.o3_gromos.values),np.nanmax(ds_2plot.o3_gromos.values)],[np.nanmin(ds_2plot.o3_gromos.values), np.nanmax(ds_2plot.o3_gromos.values)],'k--', lw=0.8)
                #axs[i,j].errorbar(x, y, xerr=xerr, yerr=yerr, color=color_season[j], linestyle='None', marker='.') 
      #             axs[i].plot(x,y, '.', color='k') 
                axs[i,col].plot(np.arange(0,10), result.beta[1]  + result.beta[0] * np.arange(0,10), color='k') 
                # axs[i,col].plot(np.arange(0,10), result_stats.slope*np.arange(0,10)+ result_stats.intercept, color='g') 
                axs[i,col].set_xlabel(r'GROMOS FFT O$_3$ [ppmv]', fontsize=fs)
                axs[i,col].set_ylabel(r'GROMOS FB O$_3$ [ppmv]', fontsize=fs)
                axs[i,col].set_title(r'O$_3$ VMR, '+title+', p $= {:.1f}$ hPa'.format(ds_2plot.o3_p.mean().data), fontsize=fs) #str(p_min[i])+ ' hPa < p < '+str(p_max[i])+' hPa')
                #axs[i,col].set_xlim(np.nanmin(ds_2plot.o3_gromos.values),np.nanmax(ds_2plot.o3_gromos.values))
                #axs[i,col].set_ylim(np.nanmin(ds_2plot.o3_gromos.values),np.nanmax(ds_2plot.o3_gromos.values))
                axs[i,col].xaxis.set_major_locator(MultipleLocator(1))
                axs[i,col].xaxis.set_minor_locator(MultipleLocator(0.5))
                axs[i,col].yaxis.set_major_locator(MultipleLocator(1))
                axs[i,col].yaxis.set_minor_locator(MultipleLocator(0.5))
                axs[i,col].tick_params(axis='both', which='major', labelsize=fs)
                if result.beta[1] < 0:
                    sign = '-' 
                else: 
                    sign= '+'
                axs[i,col].text(
                    0.505,
                    0.02,
                    ' R$^2$ = {:.2f} \n y = {:.2f}x '.format(coeff_determination, result.beta[0])+sign+' {:.2f}'.format(np.abs(result.beta[1])),
                   # '$p={:.1f}$ hPa \n$R^2 = {:.3f}$, \n$m ={:.2f}$'.format(gromos.o3_p.data[p], coeff_determination, result.beta[0]),                    
                    transform=axs[i,col].transAxes,
                    verticalalignment="bottom",
                    horizontalalignment="left",
                    fontsize=fs
                    )
    for ax in axs[0,:]:
        ax.set_xlim(0.5, 3)
        ax.set_ylim(0.5, 3)
        ax.grid()
    for ax in axs[1,:]:
        ax.set_xlim(4, 8)
        ax.set_ylim(4, 8)
        ax.grid()
    for ax in axs[2,:]:
        ax.set_xlim(3, 7)
        ax.set_ylim(3, 7)
        ax.grid()
    
    cbaxes = fig.add_axes([0.92, 0.25, 0.02, 0.5]) 
    #    cb = plt.colorbar(pl, cax=cbaxes, orientation="vertical", pad=0.0)
    cb = fig.colorbar(pl, cax=cbaxes, orientation="vertical", extend='max', pad=0.0)
    cb.set_label(label=r"$\tau$ [-]", fontsize=fs)
    cb.ax.tick_params()

    axs1[1].axvline(x=0,ls= '--', color='grey')

    axs1[0].xaxis.set_major_locator(MultipleLocator(4))
    axs1[0].xaxis.set_minor_locator(MultipleLocator(1))
    axs1[1].xaxis.set_major_locator(MultipleLocator(20))
    axs1[1].xaxis.set_minor_locator(MultipleLocator(10))
    
    # adding altitude axis, thanks Leonie :)
    y1z=1e-3*gromos.o3_z.mean(dim='time').sel(o3_p=100 ,tolerance=20,method='nearest')
    y2z=1e-3*gromos.o3_z.mean(dim='time').sel(o3_p=1e-2 ,tolerance=1,method='nearest')
    ax2 = axs1[2].twinx()
    ax2.set_yticks(1e-3*gromos.o3_z.mean(dim='time')) #ax2.set_yticks(altitude)
    ax2.set_ylim(y1z,y2z)
    fmt = FormatStrFormatter("%.0f")
    loc=MultipleLocator(base=10)
    ax2.yaxis.set_major_formatter(fmt)
    ax2.yaxis.set_major_locator(loc)
    ax2.tick_params(axis='both', which='major', labelsize=fs)
    ax2.set_ylabel('Altitude [km] ', fontsize=fs)

    axs1[0].invert_yaxis()
    axs1[0].set_ylim(50,0.1)
    axs1[0].set_xlim(-0.2, 9)
    axs1[0].set_xlabel(r'O$_3$ [ppmv]', fontsize=fs)
    axs1[0].yaxis.set_major_formatter(FuncFormatter(lambda y, _: '{:g}'.format(y)))

    axs1[0].set_ylabel('Pressure [hPa]', fontsize=fs)
    axs1[1].set_ylabel('', fontsize=fs)
    axs1[2].set_ylabel('', fontsize=fs)
    #axs1[0].legend(['GROMOS','SOMORA'], fontsize=fs)
    axs1[1].set_xlim(-25,25)
    axs1[1].set_xlabel(r'$\Delta$O$_3$ [\%]', fontsize=fs) 
    axs1[1].set_title(r'FB - FFT', fontsize=fs+2)
    axs1[1].legend(fontsize=fs-2)

    legend_elements = [
        Line2D([0], [0], color=get_color('GROMOS'), label='GROMOS'),
        Line2D([0], [0], color=get_color('SOMORA'), label='SOMORA'),
        ]
    axs1[0].legend(handles=legend_elements,fontsize=fs-2)

    axs1[1].fill_betweenx([1e-4,1000], -10,10, color='grey', alpha=0.2)
    axs1[2].set_xlim(0,1)
    axs1[2].set_title('Pearson correlation', fontsize=fs)
    axs1[2].set_xlabel('R', fontsize=fs)

    for ax in axs1:
        ax.grid(which='both', axis='x')
        ax.grid(which='both', axis='y')
        ax.tick_params(axis='both', which='major', labelsize=fs)
        # ax.fill_between(ax.get_xlim(),p_max[0],p_min[0], color='green', alpha=0.2)
        # ax.fill_between(ax.get_xlim(),p_max[1],p_min[1], color='green', alpha=0.2)
        # ax.fill_between(ax.get_xlim(),p_max[2],p_min[2], color='green', alpha=0.2)
        
    fig.tight_layout(rect=[0, 0.01, 0.92, 1])
    fig1.tight_layout(rect=[0, 0.01, 0.95, 1])
    figure_list.append(fig1)
    figure_list.append(fig)
    save_single_pdf(basefolder+'seasonal_ozone_regression_FB_FFT_'+freq+'_'+str(year)+'.pdf',figure_list)
    #fig.savefig(basefolder+'ozone_scatter_'+freq+'_'+str(year)+'.pdf', dpi=500)

#########################################################################################################
# Main function
#########################################################################################################
if __name__ == "__main__":
    yr = 2010
    # The full range:
    date_slice=slice('2010-01-01','2021-12-31')
    
    #The GROMOS full series:
    #date_slice=slice('2011-05-01','2011-12-31')
    
    #date_slice=slice('1995-01-01','2021-12-31')
    date_slice=slice('2010-01-01','2010-10-31')
 
    years = [2010] #[2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012,]
    
    instNameGROMOS = 'GROMOS'

    # By default, we use the latest version with L2 flags
    v2 = True
    flagged_L2 = False
    
    fold_gromos = '/storage/tub/instruments/gromos/level2/GROMORA/v2/' #'/scratch/GROSOM/Level2/GROMOS/v2/'
    fold_gromos2 = '/storage/tub/instruments/gromos/level2/GROMORA/v3/' #'/scratch/GROSOM/Level2/GROMOS/v2/' #
    prefix_FFT='_v2'
    prefix_FB= '_v3'#'_FB_SB'


    ########################################################################################################
    # Different strategies can be chosen for the analysis:
    # 'read': default option which reads the full level 2 doing the desired analysis
    # 'read_save': To save new level 3 data from the full hourly level 2
    # 'plot_all': the option to reproduce the figures from the manuscript
    # 'anything else': option to read the level 3 data before doing the desired analysis

    strategy = 'read'
    if strategy[0:4]=='read':
        read_gromos=True
        read_somora=True
        read_both=True

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
            )
            gromos['o3_x'] = 1e6*gromos['o3_x'].where((gromos['o3_x']>gromos['o3_x'].valid_min)&(gromos['o3_x']<gromos['o3_x'].valid_max), drop = True)
            gromos_clean = gromos.where(gromos.retrieval_quality==1, drop=True)#.where(gromos.level2_flag==0, drop=True)#.where(gromos.o3_mr>0.8)
            print('GROMOS FFT good quality level2: ', 100*len(gromos_clean.time)/len(pd.date_range('2009-07-01', '2021-12-31 23:00:00', freq='1H')) )
        
        if read_somora or read_both:
            somora = read_GROMORA_all(
                basefolder=fold_gromos2, 
                instrument_name=instNameGROMOS,
                date_slice=date_slice, 
                years=years, #[1995, 1996, 1997, 1998, 1999, 2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011],#[1995, 1996, 1997, 2006, 2007, 2008, 2009, 2010, 2011]
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
                gromos_clean = gromos#.where(gromos.retrieval_quality==1, drop=True).where(gromos.level2_flag==0, drop=True)#.where(gromos.o3_mr>0.8)

            if read_somora or read_both:
                somora = add_flags_level2_gromora(somora, 'FB')
                somora_clean = somora#.where(somora.retrieval_quality==1, drop=True).where(somora.level2_flag==0, drop=True)#.where(gromos.o3_mr>0.8)
        
        #print('GROMOS good quality level2: ', 100*len(gromos_clean.time)/len(pd.date_range('2020-01-01', '2020-12-31 23:00:00', freq='1H')) )
        #print('SOMORA good quality level2: ', 100*len(somora_clean.time)/len(pd.date_range('2020-01-01', '2020-12-31 23:00:00', freq='1H')) )
        if strategy=='read_save':
            # Saving the level 3:
            if read_gromos:
                gromos_clean.resample(time='6H').mean().to_netcdf('/scratch/GROSOM/Level2/GROMOS_level3_6H_v2.nc')
            elif read_somora:
                somora_clean.resample(time='6H').mean().to_netcdf('/scratch/GROSOM/Level2/GROMOS_FB_level3_6H_v2.nc')
            exit()
    elif strategy=='plot_all':
        plot_figures_gromora_paper(do_sensitivity = False, do_L2=True, do_comp=False, do_old=False)
        exit()
    else:
        gromos = read_GROMORA_concatenated('/scratch/GROSOM/Level2/GROMOS_level3_6H_v2.nc', date_slice)
        somora = read_GROMORA_concatenated('/scratch/GROSOM/Level2/GROMOS_FB_level3_6H_v2.nc', date_slice)
        
        gromos_clean = gromos.where(gromos.level2_flag==0, drop=True)#.where(gromos.o3_mr>0.8)
        somora_clean = somora.where(somora.level2_flag==0, drop=True)#.where(gromos.o3_mr>0.8)
        
        print('GROMOS good quality level2: ', 100*len(gromos_clean.time)/len(pd.date_range('2009-07-01', '2021-12-31 23:00:00', freq='6H')) )
        print('FB good quality level2: ', 100*len(somora_clean.time)/len(pd.date_range('1995-01-01', '2011-12-31 23:00:00', freq='6H')) )

    #####################################################################
    # Read SBUV and MLS
    bn = '/storage/tub/atmosphere/SBUV/O3/daily_mean_overpasses/'
    sbuv = read_SBUV_dailyMean(timerange=date_slice, SBUV_basename = bn, specific_fname='sbuv_v87.mod_v2r1.vmr.payerne_156.txt')
    sbuv_arosa = read_SBUV_dailyMean(date_slice, SBUV_basename = bn, specific_fname='sbuv_v87.mod_v2r1.vmr.arosa_035.txt')

    mls= read_MLS(timerange=date_slice, vers=5, filename_MLS='AuraMLS_L2GP-O3_v5_400-800_BERN_2004-2022.nc')#slice('2003-01-01','2021-12-31')
    outfolder = '/scratch/GROSOM/Level2/GROMOS_FFT_antenna/'

    #####################################################################
    # Reading the old gromora datasets
    #gromora_old = read_old_GROMOA_diff('DIFF_G_2017', date_slice)
    gromos_v2021 = read_gromos_v2021('gromosplot_ffts_select_v2021', date_slice)
    gromos_old_FB = read_gromos_old_FB('gromosFB950', date_slice)#slice('1995-01-01','2011-12-31')
    somora_old = read_old_SOMORA('/scratch/GROSOM/Level2/SOMORA_old_all.nc', date_slice)
    #plot_ozone_ts(gromos, instrument_name='GROMOS', freq='1H', altitude=False, basefolder=outfolder )

    # gromos = read_GROMORA_concatenated('/scratch/GROSOM/Level2/GROMOS_level3_6H_v2.nc', slice('2010-01-01','2021-12-31'))

    # gromora = xr.merge([somora_clean[['o3_x', 'o3_mr']], gromos_clean[['o3_x', 'o3_mr']]])
    #plot_ozone_ts(gromora, instrument_name='FB', freq='1M', altitude=False, basefolder=outfolder )

    #gromos_opacity, somora_opacity = read_opacity(folder='/scratch/GROSOM/Level2/opacities/', year=yr)

    # plot_ozone_flags('SOMORA', somora, flags1a=flags1a, flags1b=flags1b, pressure_level=[27, 12], opacity = somora_opacity, calib_version=1)
    # plot_ozone_flags('GROMOS', gromos, flags1a=flags1a, flags1b=flags1b, pressure_level=[27, 12], opacity = gromos_opacity, calib_version=1)

    #####################################################################
    # Time series 2D plots
    plot_2D = False
    if plot_2D:
        diff_2_mls_FB(somora_clean, gromos_v2021, gromos_old_FB, mls, sbuv,  p_min=[0.1, 1, 10] , p_max=[0.9, 5, 50], freq='1W', basefolder=outfolder)
        #daily_median_save(gromos_clean, outfolder)
        #compare_ts_MLS(gromos, somora, date_slice=date_slice, freq='7D', basefolder='/scratch/GROSOM/Level2/GROMORA_retrievals_v2/', ds_mls=mls, sbuv=None)
        #compare_pressure_mls_sbuv(gromos_clean, somora_clean, mls, sbuv, pressure_level=[29, 25, 21, 18, 15], add_sun=False, freq='6H', basefolder=outfolder)
        #compare_pressure_mls_sbuv_paper(gromos_clean, somora_clean, mls, sbuv,  p_min=[0.1, 1, 10] , p_max=[0.9, 5, 50], add_sun=False, freq='1M', basefolder=outfolder)
        #compare_ts(gromos, somora, freq='7D', date_slice=date_slice, basefolder=outfolder)
        compare_pressure_mls_sbuv_old(gromos_clean, somora_clean, gromos_v2021, gromos_old_FB, mls, sbuv,  p_min=[0.1, 1, 10] , p_max=[0.9, 5, 50], add_sun=False, freq='1W', basefolder=outfolder)


    #####################################################################
    # Comparison with old retrievals
    plot_old_new = False
    if plot_old_new:
        gromos_old_vs_new(gromos_clean, somora_clean, gromos_v2021, gromos_old_FB, freq='7D', basefolder=outfolder)
        #gromos_old_vs_new(gromos_clean, gromos_v2021, mls, seasonal=False)
        #gromos_old_vs_new(somora_clean, somora_old, mls, seasonal=False)
    
        #trends_diff_old_new(gromos_clean, somora_clean, gromos_v2021, somora_old, mls, p_min=[0.1, 1, 10] , p_max=[0.9, 5, 50], freq='7D', freq_avg='1M',  outfolder=outfolder)
        #trends_simplified_new(gromos_clean, somora_clean, mls, p_min=[0.1, 1, 10] , p_max=[0.9, 5, 50], freq='7D', freq_avg='1M',  outfolder=outfolder)


    #gromos = utc_to_lst(gromos)
    #somora = utc_to_lst(somora)

    #compute_compare_climatology(somora, slice_clim=slice("2010-01-01", "2020-12-31"), slice_plot=slice("2022-01-01", "2022-01-31"), percentile=[0.1, 0.9], pressure_level = [25, 21, 15], basefolder=outfolder)

    #compare_pressure(gromos_clean, somora_clean, pressure_level=[31, 25, 21, 15, 12], add_sun=False, freq='6H', basefolder='/scratch/GROSOM/Level2/GROMORA_retrievals_v2/')

    #####################################################################
    # Relative difference GROMOS vs SOMORA
    compare_gromora = True
    if compare_gromora:
        map_rel_diff(gromos_clean, somora_clean, freq='6H', basefolder=outfolder, FB=True)
        compare_ts_gromora(gromos_clean, somora_clean, date_slice=date_slice, freq='6H', basefolder=outfolder, paper=True, FB=True)

        # compute_corr_profile(somora_sel,mls_somora_colloc,freq='7D',basefolder='/scratch/GROSOM/Level2/MLS/')
        # #compare_diff_daily(gromos ,somora, gromora_old, pressure_level=[34 ,31, 25, 21, 15, 12], altitudes=[69, 63, 51, 42, 30, 24])
        compare_mean_diff(gromos_clean, somora_clean, sbuv = sbuv, mls=mls, basefolder=outfolder, corr_FFT=False)

        #compare_mean_diff_monthly(gromos_clean, somora_clean, mls, sbuv, outfolder=outfolder)

    # gromos_linear_fit = gromos_clean.o3_x.where((gromos_clean.o3_p<p_high) & (gromos_clean.o3_p>p_low), drop=True).mean(dim='o3_p').resample(time='1M').mean()#) .polyfit(dim='time', deg=1)
    # somora_linear_fit = somora_clean.o3_x.resample(time='1M').mean().polyfit(dim='time', deg=1)

    #####################################################################
    # Averaging kernels
    plot_avk = True
    if plot_avk:
        #compare_avkm(gromos, somora, date_slice, outfolder)
        gromos_clean=gromos_clean.where(gromos_clean.o3_avkm.mean(dim='o3_p_avk')<10, drop=True).where(gromos_clean.o3_avkm.mean(dim='o3_p_avk')>-10, drop=True)
        somora_clean=somora_clean.where(somora_clean.o3_avkm.mean(dim='o3_p_avk')<10, drop=True).where(somora_clean.o3_avkm.mean(dim='o3_p_avk')>-10, drop=True)
        compare_avkm(gromos_clean, somora_clean, date_slice, outfolder, seasonal=False)

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
    # Adding Level 1 information

    level1b_gromos, gromos_flags_level1a, gromos_flags_level1b = read_level1('/storage/tub/instruments/gromos/level1/GROMORA/v2', 'GROMOS', dateslice=slice('2009-01-01', '2021-12-31'))
    
    level1b_gromos=level1b_gromos.sel(time=~level1b_gromos.get_index("time").duplicated())
    
    num_good_1a_gromos = len(gromos_flags_level1a.where(gromos_flags_level1a.calibration_flags.sum(dim='flags')>6, drop=True).time)
    num_good_1b_gromos = len(gromos_flags_level1b.where(gromos_flags_level1b.calibration_flags[:,0]==1, drop=True).time)
    
    print('GROMOS good quality level1a (2009-2021): ', 100*num_good_1a_gromos/len(pd.date_range('2009-07-01', '2021-12-31 23:00:00', freq='10 min')))
    print('GROMOS good quality level1b (2009-2021): ', 100*num_good_1b_gromos/len(pd.date_range('2009-07-01', '2021-12-31 23:00:00', freq='1H')))
    level1b_somora, somora_flags_level1a, somora_flags_level1b = read_level1('/storage/tub/instruments/somora/level1/v2','SOMORA', dateslice=slice('2009-01-01', '2021-12-31'))
    num_good_1b_somora = len(somora_flags_level1b.where(somora_flags_level1b.calibration_flags[:,0]==1, drop=True).time)
    num_good_1a_somora = len(somora_flags_level1a.where(somora_flags_level1a.calibration_flags.sum(dim='flags')>6, drop=True).time)
    print('SOMORA good quality level1a (2009-2021): ', 100*num_good_1a_somora/len(pd.date_range('2009-09-23', '2021-12-31 23:00:00', freq='10 min')) )
    print('SOMORA good quality level1b (2009-2021): ', 100*num_good_1b_somora/len(pd.date_range('2009-09-23', '2021-12-31 23:00:00', freq='1H')) )
    
    #####################################################################
    # Correlations GROMOS-SOMORA
    compute_corr_GROMORA = False
    if compute_corr_GROMORA:
        # compute_seasonal_correlation(gromos_clean, somora_clean, freq='6H', pressure_level=[25, 20, 15], basefolder=outfolder) p_min=[0.1, 1, 11] , p_max=[0.9, 10, 50]
        compute_seasonal_correlation_FB_FFT(gromos_clean, somora_clean, freq='6H', p_min=[0.1, 1, 10] , p_max=[0.9, 5, 50], basefolder=outfolder)

        #compute_correlation(gromos_clean, somora_clean, freq='6H', pressure_level=[36, 31, 21, 12], basefolder=outfolder)
    
        #compute_corr_profile(gromos_clean, somora_clean, freq='6H', date_slice=slice('2009-10-01','2021-12-31'), basefolder=outfolder)

    #####################################################################
    # Opacity GROMOS-SOMORA
    opacity = False
    if opacity:
        compare_opacity(level1b_gromos, level1b_somora, freq = '7D', tc = False, date_slice=date_slice)
