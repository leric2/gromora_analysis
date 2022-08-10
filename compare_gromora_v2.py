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
from level2_gromora import read_GROMORA_all, read_GROMORA_concatenated, read_gromos_v2021, read_old_SOMORA
from level2_gromora_diagnostics import read_level1, add_flags_level2_gromora

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
    "text.usetex": False,
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

def compare_ts(gromos, somora, freq, date_slice, basefolder):
    fs = 28
    year = pd.to_datetime(gromos.time.values[0]).year

    gromos['o3'] = gromos.o3_x
    somora['o3'] = somora.o3_x

    # plim_gromos = np.zeros(shape=(len(gromos.time.values),2))
    # for i in range(len(gromos.time.values)):
    #     gromos_p_lim = gromos.isel(time=i).o3_p.where(gromos.isel(time=i).o3_mr > 0.8, drop=True).values
    #     plim_gromos[i,0] = gromos_p_lim[0]
    # plim_gromos[i,1] = gromos_p_lim[-1]
    #good_p_somora = somora.o3_x.where(somora.o3_mr > 0.8, drop=True)

    # gromos['mr_lim'] =  plim_gromos

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
        cmap=cmap_ts,
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
        cmap=cmap_ts,
        add_colorbar=False
    )
    pl2.set_edgecolor('face')
    # ax.set_yscale('log')
    axs[1].invert_yaxis()
    axs[1].set_ylabel('Pressure [hPa]', fontsize=fs)
    axs[1].set_title('SOMORA', fontsize=fs+2)

    cbaxes = fig.add_axes([0.92, 0.25, 0.02, 0.5]) 
   # cb = plt.colorbar(pl, cax=cbaxes, orientation="vertical", pad=0.0)
    cb = fig.colorbar(pl, cax=cbaxes, orientation="vertical", pad=0.0)
    cb.set_label(label=r"O$_3$ [ppmv]", fontsize=fs)
    cb.ax.tick_params()

    for ax in axs:
        ax.set_ylim(100, 1e-2)
        ax.set_xlabel('')
        ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: '{:g}'.format(y)))
        ax.xaxis.set_major_locator(mdates.YearLocator())
        #ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=(1, 7)))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

    plt.tight_layout(rect=[0, 0.01, 0.92, 1])

    fig.savefig(basefolder+'GROMOS_ozone_comparison_'+str(year)+'.pdf', dpi=500)
    
def compare_ts_MLS(gromos, somora, date_slice, freq, basefolder, ds_mls=None, sbuv=None):
    fs = 34
    year = pd.to_datetime(gromos.time.values[0]).year

    if ds_mls is None:
        ds_mls= read_MLS(date_slice, vers=5)

    if sbuv is None:
        fig, axs = plt.subplots(3, 1, sharex=True, sharey=True, figsize=(26,16))
    else:
        fig, axs = plt.subplots(4, 1, sharex=True, sharey=True, figsize=(25,12))
    pl = gromos.sel(time=date_slice).o3_x.resample(time=freq).mean().plot(
        x='time',
        y='o3_p',
        ax=axs[0], 
        vmin=0,
        vmax=10,
        yscale='log',
        linewidth=0,
        rasterized=True,
        cmap=cmap_ts,
        add_colorbar=False
    )
    pl.set_edgecolor('face')
    axs[0].set_title('GROMOS', fontsize=fs+4) 
    # ax.set_yscale('log')
    axs[0].invert_yaxis()
    axs[0].set_ylabel('Pressure [hPa]', fontsize=fs)

    pl2 = somora.sel(time=date_slice).o3_x.resample(time=freq).mean().plot(
        x='time',
        y='o3_p',
        ax=axs[1], 
        vmin=0,
        vmax=10,
        yscale='log',
        linewidth=0,
        rasterized=True,
        cmap=cmap_ts,
        add_colorbar=False
    )
    pl2.set_edgecolor('face')
    axs[1].set_ylabel('Pressure [hPa]', fontsize=fs)
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
        cmap=cmap_ts,
        add_colorbar=False
    )
    pl3.set_edgecolor('face')
    # ax.set_yscale('log')
    axs[2].set_ylabel('Pressure [hPa]', fontsize=fs)
    axs[2].set_title('MLS', fontsize=fs+4)
    
    if sbuv is not None:
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
        axs[3].set_ylabel('P [hPa]', fontsize=fs)
        axs[3].set_title('SBUV', fontsize=fs+4)

    cbaxes = fig.add_axes([0.92, 0.25, 0.02, 0.5]) 
    #    cb = plt.colorbar(pl, cax=cbaxes, orientation="vertical", pad=0.0)
    cb = fig.colorbar(pl, cax=cbaxes, orientation="vertical", pad=0.0)
    cb.set_label(label=r"O$_3$ [ppmv]", fontsize=fs)
    cb.ax.tick_params()

    for ax in axs:
        ax.set_ylim(100, 1e-2)
        ax.set_xlabel('')
        ax.tick_params(axis='both', which='major', labelsize=fs)
        ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: '{:g}'.format(y)))
        ax.xaxis.set_major_locator(mdates.YearLocator())
        #ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=(1, 7)))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        #ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))

    plt.tight_layout(rect=[0, 0.01, 0.92, 1])
    fig.savefig(basefolder+'GROMORA_MLS_ozone_comparison_'+str(year)+'.pdf', dpi=800)
    
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

def compare_pressure_mls_sbuv(gromos, somora, mls, sbuv, pressure_level = [10, 15,20,25,30], add_sun=False, freq='1D',basefolder=''):
    fs=34
    year=pd.to_datetime(gromos.time.data[0]).year
    fig, axs = plt.subplots(len(pressure_level), 1, sharex=True, figsize=(26,22))
    for i, p_ind in enumerate(pressure_level):
        pressure =  gromos.o3_p.data[p_ind]
        mls.o3.sel(p=pressure, method='nearest', tolerance=0.4*pressure).resample(time=freq).mean().plot(ax=axs[i], color='k', lw=1.5)
        if (pressure > 0.4) & (pressure<50) & (sbuv is not None):
            sbuv.ozone.sel(p=pressure, method='nearest', tolerance=0.4*pressure).resample(time=freq).mean().plot(ax=axs[i], color=sbuv_color, lw=1.5)
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

    axs[2].legend(['MLS', 'SBUV', 'GROMOS','SOMORA', ], loc=1, fontsize=fs-4)
    #axs[0].legend(['OG','SB corr'], loc=1, fontsize=fs-2)

    axs[0].set_ylim(0,2)
    axs[1].set_ylim(0,4)

    axs[2].set_ylim(2,10)
    axs[3].set_ylim(2,10)
    axs[4].set_ylim(2,10)

    for ax in axs:
        ax.grid()
        #ax.set_xlim(pd.to_datetime('2009-07-01'), pd.to_datetime('2021-12-31'))
        ax.set_ylabel('O$_3$ [ppmv]', fontsize=fs-2)
        ax.yaxis.set_major_locator(MultipleLocator(2))
        ax.yaxis.set_minor_locator(MultipleLocator(0.5))
        #ax.xaxis.set_major_locator(mdates.YearLocator())
        #ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=(1, 7)))
        #ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        ax.tick_params(axis='both', which='major', labelsize=fs)

    fig.tight_layout(rect=[0, 0.01, 0.99, 1])
    fig.savefig(basefolder+'ozone_comparison_pressure_level_MLS_SBUV_'+str(year)+'.pdf', dpi=500)


def compare_pressure_mls_sbuv_paper(gromos, somora, mls, sbuv, p_min, p_max, add_sun=False, freq='1D',basefolder=''):
    fs=32
    year=pd.to_datetime(gromos.time.data[0]).year
    fig, axs = plt.subplots(len(p_min), 1, sharex=True, figsize=(28,18))
    for i, p_ind in enumerate(p_min):

        gromos_p = gromos.where(gromos.o3_p>p_min[i] , drop=True).where(gromos.o3_p<p_max[i], drop=True)
        somora_p = somora.where(somora.o3_p>p_min[i] , drop=True).where(somora.o3_p<p_max[i], drop=True)
        mls_p = mls.where(mls.p>p_min[i] , drop=True).where(mls.p<p_max[i], drop=True)
        pressure =  gromos_p.o3_p.mean().values
        mls_p.o3.mean(dim='p').resample(time=freq).mean().plot(ax=axs[i], color='k', lw=2)
        if (pressure > 0.4) & (pressure<50) & (sbuv is not None):
            sbuv_p = sbuv.where(sbuv.p>p_min[i] , drop=True).where(sbuv.p<p_max[i], drop=True)
            sbuv_p.ozone.mean(dim='p').resample(time=freq).mean().plot(ax=axs[i], color=sbuv_color, lw=2)
        gromos_p.o3_x.mean(dim='o3_p').resample(time=freq).mean().plot(ax=axs[i], color=color_gromos, lw=2)
        somora_p.o3_x.mean(dim='o3_p').resample(time=freq).mean().plot(ax=axs[i], color=color_somora, lw=2)
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
    axs[1].legend(['MLS', 'SBUV', 'GROMOS','SOMORA', ], ncol=4, loc=4, fontsize=fs-2)
    #axs[0].legend(['OG','SB corr'], loc=1, fontsize=fs-2)

    axs[0].set_ylim(0,4)
    axs[1].set_ylim(2,10)

    axs[2].set_ylim(2,8)
    #axs[3].set_ylim(2,10)
    #axs[4].set_ylim(2,10)
    

    for ax in axs:
        ax.grid()
        #ax.set_xlim(pd.to_datetime('2009-07-01'), pd.to_datetime('2021-12-31'))
        ax.set_ylabel('O$_3$ [ppmv]', fontsize=fs)
        ax.yaxis.set_major_locator(MultipleLocator(2))
        ax.yaxis.set_minor_locator(MultipleLocator(0.5))
        ax.xaxis.set_major_locator(mdates.YearLocator())
        #ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=(1, 7)))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        ax.tick_params(axis='both', which='major', labelsize=fs-2)

    fig.tight_layout(rect=[0, 0.01, 0.99, 1])
    fig.savefig(basefolder+'ozone_comparison_pressure_level_MLS_SBUV_paper_'+str(year)+'.pdf', dpi=500)

def compare_ts_gromora(gromos, somora, date_slice, freq, basefolder, paper=False):
    fs = 34
    year = pd.to_datetime(gromos.time.values[0]).year
    
    if paper:
        fig, axs = plt.subplots(2, 1, sharex=True, sharey=True, figsize=(24,16))
    else:
        fig, axs = plt.subplots(3, 1, sharex=True, sharey=True, figsize=(24,16))
    pl = gromos.sel(time=date_slice).o3_x.resample(time=freq).mean().plot(
        x='time',
        y='o3_p',
        ax=axs[0], 
        vmin=0,
        vmax=10,
        yscale='log',
        linewidth=0,
        rasterized=True,
        add_colorbar=False,
        cmap=cmap_ts,
       # cbar_kwargs={'label':r'O$_3$ [ppmv]'}
    )
    pl.set_edgecolor('face')
    axs[0].set_title('GROMOS', fontsize=fs+4) 
    # ax.set_yscale('log')
    axs[0].invert_yaxis()
    axs[0].set_ylabel('Pressure [hPa]', fontsize=fs)

    pl2 = somora.sel(time=date_slice).o3_x.resample(time=freq).mean().plot(
        x='time',
        y='o3_p',
        ax=axs[1], 
        vmin=0,
        vmax=10,
        yscale='log',
        linewidth=0,
        rasterized=True,
        add_colorbar=False,
        cmap=cmap_ts,
       # cbar_kwargs={'label':r'O$_3$ [ppmv]'}
    )
    pl2.set_edgecolor('face')
    axs[1].set_ylabel('Pressure [hPa]', fontsize=fs)
    axs[1].set_title('SOMORA', fontsize=fs+4)
    
    if not paper:
        rel_diff = 100*(gromos.o3_x.resample(time=freq).mean()- somora.o3_x.resample(time=freq).mean())/gromos.o3_x.resample(time=freq).mean()
        pl3 = rel_diff.sel(time=date_slice).plot(
            x='time',
            y='o3_p',
            ax=axs[2], 
            vmin=-40,
            vmax=40,
            yscale='log',
            linewidth=0,
            rasterized=True,
            cmap='coolwarm',
            cbar_kwargs={'label':r'$\Delta$O$_3$ [\%]'}
        )
        pl3.set_edgecolor('face')
        #ax.set_yscale('log')
        axs[2].set_ylabel('Pressure [hPa]', fontsize=fs)
        axs[2].set_title('(GRO-SOM)/GRO', fontsize=fs+4)
    
    cbaxes = fig.add_axes([0.92, 0.25, 0.02, 0.5]) 
    cb = fig.colorbar(pl, cax=cbaxes, cmap=cmap_ts, orientation="vertical", pad=0.0)
    cb.set_label(label=r"O$_3$ [ppmv]", fontsize=fs)
    cb.ax.tick_params()

    for ax in axs:  
        ax.set_ylim(250, 5e-3) 
        ax.set_xlabel('')
        ax.tick_params(axis='both', which='major', labelsize=fs)
        ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: '{:g}'.format(y)))
        if paper:
            ax.set_ylim(100, 1e-2)
            ax.xaxis.set_major_locator(mdates.YearLocator())
            #ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=(1, 7)))
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
            #ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))

    plt.tight_layout(rect=[0, 0.01, 0.92, 1])
    fig.savefig(basefolder+'GROMORA_ozone_comparison_rel_diff_'+str(year)+'_'+freq+'.pdf', dpi=500)

def map_rel_diff(gromos, somora ,freq='12H', basefolder=''):
    fs=24
    year=pd.to_datetime(gromos.time.data[0]).year
    fig, axs = plt.subplots(1, 1, sharex=True, figsize=(18,12))
    
    rel_diff = 100*(gromos.o3_x.resample(time=freq).mean()- somora.o3_x.resample(time=freq).mean())/gromos.o3_x.resample(time=freq).mean()
    rel_diff.plot(
        ax=axs,
        x='time',
        y='o3_p',
        vmin=-40,
        vmax=40,
        yscale='log',
        linewidth=0,
        rasterized=True,
        cmap='coolwarm',
        cbar_kwargs={'label':r'$\Delta$O$_3$ [\%]'}
    )
    axs.invert_yaxis()
    axs.set_xlabel('')
    axs.set_ylabel('Pressure [hPa]',fontsize=fs)
    axs.set_ylim(100,0.01)
    axs.yaxis.set_major_formatter(FuncFormatter(lambda y, _: '{:g}'.format(y)))
    # axs.set_title(freq+' relative diff GROMOS-SOMORA', fontsize=fs+4)
    axs.tick_params(axis='both', which='major', labelsize=fs-2)
    # axs.xaxis.set_major_locator(mdates.YearLocator())
    # axs.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    axs.set_title('GROMOS - SOMORA',fontsize=fs+4)
    fig.tight_layout(rect=[0, 0.01, 0.95, 1])
    fig.savefig(basefolder+'ozone_rel_diff_'+str(year)+'.pdf', dpi=500)

def compute_correlation(gromos, somora, freq='1M', pressure_level = [15,20,25], basefolder='', MLS = False):
    year=pd.to_datetime(gromos.time.data[0]).year
    fs=24
    
    fig, axs = plt.subplots(len(pressure_level),1,figsize=(6, 6*len(pressure_level)))
    error_gromos = np.sqrt( np.square(gromos.o3_eo) + np.square(gromos.o3_es))
    if MLS:
        error_somora = 1e-6*0.1*somora.o3_x
    else:
        error_somora  = np.sqrt( np.square(somora.o3_eo) + np.square(somora.o3_es))

    #ds_o3_gromora=xr.merge(({'o3_gromos':gromos.o3_x.isel(o3_p=p).resample(time=freq).mean()},{'o3_somora':somora.o3_x.isel(o3_p=p).resample(time=freq).mean()}))
    
    if freq == 'OG':
        ds_o3_gromora=xr.merge((
            {'o3_gromos':gromos.o3_x},
            {'o3_somora':somora.o3_x},
            {'error_gromos':error_gromos},
            {'error_somora':error_somora},
            {'opacity_somora':somora.opacity},
            ))
    else:
        ds_o3_gromora=xr.merge((
            {'o3_gromos':gromos.o3_x.resample(time=freq).mean()},
            {'o3_somora':somora.o3_x.resample(time=freq).mean()},
            {'error_gromos':error_gromos.resample(time=freq).mean()},
            {'error_somora':error_somora.resample(time=freq).mean()},
            {'opacity_somora':somora.opacity.resample(time=freq).mean()}
            ))
    for i, p in enumerate(pressure_level): 
        x = ds_o3_gromora.isel(o3_p=p).o3_gromos.interpolate_na(dim='time',fill_value="extrapolate")
        y = ds_o3_gromora.isel(o3_p=p).o3_somora.interpolate_na(dim='time',fill_value="extrapolate")
        pearson_corr = xr.corr(x,y, dim='time')
        print('Pearson corr coef: ',pearson_corr.values)
        xerr = 1e6*ds_o3_gromora.isel(o3_p=p).error_gromos.interpolate_na(dim='time',fill_value="extrapolate")
        yerr = 1e6*ds_o3_gromora.isel(o3_p=p).error_somora.interpolate_na(dim='time',fill_value="extrapolate")
        result, chi2 = regression_xy(#stats.linregress(
            x.values, y.values, x_err = xerr.values, y_err=yerr.values
        )
        print('Reduced chi2: ',chi2/(len(x.values)-2))
        print('RMSE ', np.sqrt(result.sum_square/len(x.values)))
        print('Slope ', result.beta[0], ' +- ', result.sd_beta[0] )
        print('Intercept ', result.beta[1], ' +- ', result.sd_beta[1] )

        coeff_determination = calcR2_wikipedia(y.values, result.beta[1] + result.beta[0] * x.values)
        #print('r2: ',result.rvalue**2)
        print('R2: ',coeff_determination)
        ds_o3_gromora.isel(o3_p=p).plot.scatter(
            ax=axs[i],
            x='o3_gromos', 
            y='o3_somora',
            hue='opacity_somora',
            hue_style='continuous',
            vmin=0,
            vmax=2,
            marker='.',
            cmap=plt.get_cmap('temperature'),
            cbar_kwargs={'label':r'opacity'}
        )
        axs[i].plot([np.nanmin(ds_o3_gromora.isel(o3_p=p).o3_gromos.values),np.nanmax(ds_o3_gromora.isel(o3_p=p).o3_gromos.values)],[np.nanmin(ds_o3_gromora.isel(o3_p=p).o3_gromos.values), np.nanmax(ds_o3_gromora.isel(o3_p=p).o3_gromos.values)],'k--')
     #   axs[i].errorbar(x, y, xerr=xerr, yerr=yerr, color='k', linestyle='None', marker='.') 
       # axs[i].plot(x,y, '.', color='k') 
        axs[i].plot(x, result.beta[1]  + result.beta[0] * x, color='red') 
        axs[i].set_xlabel(r'GROMOS O$_3$ [ppmv]', fontsize=fs-2)
        if MLS:
            axs[i].set_ylabel(r'MLS O$_3$ [ppmv]', fontsize=fs-2)
        else:
            axs[i].set_ylabel(r'SOMORA O$_3$ [ppmv]', fontsize=fs-2)
        axs[i].set_title(r'O$_3$ VMR '+f'at p = {gromos.o3_p.data[p]:.3f} hPa')
        axs[i].set_xlim(np.nanmin(ds_o3_gromora.isel(o3_p=p).o3_gromos.values),np.nanmax(ds_o3_gromora.isel(o3_p=p).o3_gromos.values))
        axs[i].set_ylim(np.nanmin(ds_o3_gromora.isel(o3_p=p).o3_gromos.values),np.nanmax(ds_o3_gromora.isel(o3_p=p).o3_gromos.values))
        axs[i].xaxis.set_major_locator(MultipleLocator(1))
        axs[i].xaxis.set_minor_locator(MultipleLocator(0.5))
        axs[i].yaxis.set_major_locator(MultipleLocator(1))
        axs[i].yaxis.set_minor_locator(MultipleLocator(0.5))
    #     axs[i].text(
    #         0.65,
    #         0.1,
    #         '$R^2 = {:.3f}$, \n $m ={:.2f} $'.format(result.rvalue**2, result.slope),
    #         transform=axs[i].transAxes,
    #         verticalalignment="bottom",
    #         horizontalalignment="left",
    #         fontsize=fs
    # )
    
    for ax in axs:
        ax.grid(which='both', axis='x')
        ax.grid(which='both', axis='y')
    fig.tight_layout(rect=[0, 0.01, 0.95, 1])
    fig.savefig(basefolder+'ozone_scatter_'+freq+'_'+str(year)+'.pdf', dpi=500)


def compute_compare_climatology(gromos, slice_clim, slice_plot, percentile=[0.05, 0.95], pressure_level = [15,20,25], basefolder='/scratch/GROSOM/Level2/'):
    year=pd.to_datetime(gromos.o3_x.sel(time=slice_plot).time.data[0])
    fs=18

    gromos_clim25 = gromos.o3_x.sel(time = slice_clim).groupby('time.dayofyear').quantile(percentile[0])
    gromos_clim75 = gromos.o3_x.sel(time = slice_clim).groupby('time.dayofyear').quantile(percentile[1])
    gromos_clim = gromos.o3_x.sel(time = slice_clim).groupby('time.dayofyear').median()
    ds_allp = gromos.o3_x.sel(time=slice_plot).groupby('time.dayofyear').median()
    fig, axs = plt.subplots(len(pressure_level),1,figsize=(20, 6*len(pressure_level)), sharex=True)
    
    for i,p in enumerate(pressure_level):
        gromos_clim_p = gromos_clim.isel(o3_p=p)
        gromos_clim25_p = gromos_clim25.isel(o3_p=p)
        gromos_clim75_p = gromos_clim75.isel(o3_p=p)
        ds = ds_allp.isel(o3_p=p)
        # ds = gromos.o3_x.isel(o3_p=p).sel(time=slice_plot).resample(time='1D').mean()
        axs[i].plot( ds.dayofyear, ds.data,color=color_gromos) 
        axs[i].plot( gromos_clim_p.dayofyear, gromos_clim_p.data,color='k')
        axs[i].fill_between(gromos_clim_p.dayofyear,gromos_clim25_p,gromos_clim75_p, color='k', alpha=0.2)
        axs[i].set_title(r'O$_3$ VMR '+f'at p = {ds.o3_p.data:.3f} hPa')

    axs[0].legend(['GROMOS 2020','GROMOS climatology'])

    d1 = gromos.sel(time=slice_plot).time.dt.dayofyear.data[0]
    d2 = gromos.sel(time=slice_plot).time.dt.dayofyear.data[-1]
    axs[0].set_xlim(d1,d2)
    axs[-1].set_xlabel('Day of year')
    #axs[-1].set_ylabel(r'O$_3$ VMR ') 
    for ax in axs:
        ax.grid()
        ax.set_ylabel(r'O$_3$ VMR ') 
    fig.tight_layout(rect=[0, 0.01, 0.95, 1])
    fig.savefig(basefolder + 'GROMOS_climatology_vs_'+str(year)+'.pdf', dpi=500)

def compute_seasonal_correlation(gromos, somora, freq='1M', pressure_level = [15,20,25], basefolder='', MLS = False, split_by ='season'):
    year=pd.to_datetime(gromos.time.data[0]).year
    fs=18
    figure_list = list()
    fig1, axs1 = plt.subplots(1,3, sharey=True, figsize=(15, 11.3))
    fig, axs = plt.subplots(len(pressure_level),4,figsize=(20, 6*len(pressure_level)))
    error_gromos = np.sqrt( np.square(gromos.o3_eo) + np.square(gromos.o3_es))
    if MLS:
        error_somora = 1e-6*0.1*somora.o3_x
    else:
        error_somora  = np.sqrt( np.square(somora.o3_eo) + np.square(somora.o3_es))

    #ds_o3_gromora=xr.merge(({'o3_gromos':gromos.o3_x.isel(o3_p=p).resample(time=freq).mean()},{'o3_somora':somora.o3_x.isel(o3_p=p).resample(time=freq).mean()}))
    ds_o3_gromora=xr.merge((
        {'o3_gromos':gromos.o3_x.resample(time=freq).mean()},
        {'o3_somora':somora.o3_x.resample(time=freq).mean()},
        {'error_gromos':error_gromos.resample(time=freq).mean()},
        {'error_somora':error_somora.resample(time=freq).mean()},
        {'opacity':somora.opacity.resample(time=freq).mean()},
        ))

    #season = ['DJF','MAM', 'JJA', 'SON']
    color_season = ['r', 'b', 'y', 'g']
    marker_season = ['x', 'o', 'd', '*']
    ds_o3_gromora_groups = ds_o3_gromora.groupby('time.season').groups
    #ds_o3_gromora_plot = ds_o3_gromora.isel(o3_p=pressure_level)
    
    for j, s in enumerate(ds_o3_gromora_groups):
        print('Processing season ', s)
        ds = ds_o3_gromora.isel(time=ds_o3_gromora_groups[s]).interpolate_na(dim='time',fill_value="extrapolate")
        pearson_corr_profile = xr.corr(ds.o3_gromos, ds.o3_somora, dim='time')

        ds.o3_gromos.mean(dim='time').plot(
            ax=axs1[0],
            y='o3_p',
            yscale='log',
            color=color_gromos,
            marker=marker_season[j],
            linewidth=0.5,
            ) 
        ds.o3_somora.mean(dim='time').plot(
            ax=axs1[0],
            y='o3_p',
            color=color_somora,
            marker=marker_season[j],
            linewidth=0.5,
            ) 
        
        rel_mean_diff = 100*(ds.o3_gromos.mean(dim='time')-ds.o3_somora.mean(dim='time') )/ ds.o3_gromos.mean(dim='time')
        rel_mean_diff.plot(
            ax=axs1[1],
            y='o3_p',
            color='k',
            marker=marker_season[j],
            label=s,
            linewidth=0.5,
            )

        pearson_corr_profile.plot(
            ax=axs1[2],
            y='o3_p',
            color='k',
            marker=marker_season[j],
            linewidth=0.5,
        )

        for i, p in enumerate(pressure_level):
            x = ds.isel(o3_p=p).o3_gromos
            y = ds.isel(o3_p=p).o3_somora
        
            pearson_corr = xr.corr(x,y, dim='time')
            print('Pearson corr coef: ',pearson_corr.values)
            xerr = 1e6*ds.isel(o3_p=p).error_gromos
            yerr = 1e6*ds.isel(o3_p=p).error_somora
            result, chi2 = regression_xy(#stats.linregress(
                x.values, y.values, x_err = xerr.values, y_err=yerr.values
            )
            print('Reduced chi2: ',chi2/(len(x.values)-2))
            print('RMSE ', np.sqrt(result.sum_square/len(x.values)))
            print('Slope ', result.beta[0], ' +- ', result.sd_beta[0] )
            print('Intercept ', result.beta[1], ' +- ', result.sd_beta[1] )

            coeff_determination = calcR2_wikipedia(y.values, result.beta[1] + result.beta[0] * x.values)
            #print('r2: ',result.rvalue**2)
            print('R2: ',coeff_determination)
            ds.isel(o3_p=p).plot.scatter(
                ax=axs[i,j],
                x='o3_gromos', 
                y='o3_somora',
                hue='opacity',
                hue_style='continuous',
                vmin=0,
                vmax=2,
                marker='.',
                cmap=plt.get_cmap('temperature'),
                # levels=np.linspace(0, 3, 10)
            )
            axs[i,j].plot([np.nanmin(ds.isel(o3_p=p).o3_gromos.values),np.nanmax(ds.isel(o3_p=p).o3_gromos.values)],[np.nanmin(ds.isel(o3_p=p).o3_gromos.values), np.nanmax(ds.isel(o3_p=p).o3_gromos.values)],'k--')
            #axs[i,j].errorbar(x, y, xerr=xerr, yerr=yerr, color=color_season[j], linestyle='None', marker='.') 
      #      axs[i].plot(x,y, '.', color='k') 
            axs[i,j].plot(x, result.beta[1]  + result.beta[0] * x, color='red') 
            axs[i,j].set_xlabel(r'GROMOS O$_3$ [ppmv]', fontsize=fs-2)
            axs[i,j].set_ylabel(r'SOMORA O$_3$ [ppmv]', fontsize=fs-2)
            axs[i,j].set_title(r'O$_3$ VMR, '+s)
            axs[i,j].set_xlim(np.nanmin(ds.isel(o3_p=p).o3_gromos.values),np.nanmax(ds.isel(o3_p=p).o3_gromos.values))
            axs[i,j].set_ylim(np.nanmin(ds.isel(o3_p=p).o3_gromos.values),np.nanmax(ds.isel(o3_p=p).o3_gromos.values))
            axs[i,j].xaxis.set_major_locator(MultipleLocator(1))
            axs[i,j].xaxis.set_minor_locator(MultipleLocator(0.5))
            axs[i,j].yaxis.set_major_locator(MultipleLocator(1))
            axs[i,j].yaxis.set_minor_locator(MultipleLocator(0.5))
            axs[i,j].text(
                0.65,
                0.1,
                '$p={:.1f}$ hPa \nm ={:.2f} \nR2 ={:.2f}'.format(gromos.o3_p.data[p], result.beta[0], coeff_determination),
               # '$p={:.1f}$ hPa \n$R^2 = {:.3f}$, \n$m ={:.2f}$'.format(gromos.o3_p.data[p], coeff_determination, result.beta[0]),
                transform=axs[i,j].transAxes,
                verticalalignment="bottom",
                horizontalalignment="left",
                fontsize=fs
                )
    
    axs1[1].axvline(x=0,ls= '--', color='grey')

    axs1[0].xaxis.set_major_locator(MultipleLocator(4))
    axs1[0].xaxis.set_minor_locator(MultipleLocator(1))
    axs1[1].xaxis.set_major_locator(MultipleLocator(20))
    axs1[1].xaxis.set_minor_locator(MultipleLocator(10))

    axs1[0].invert_yaxis()
    axs1[0].set_ylim(100,0.01)
    axs1[0].set_xlim(-0.2, 9)
    axs1[0].set_xlabel(r'O$_3$ [ppmv]', fontsize=fs)
    axs1[0].set_ylabel('', fontsize=fs)
    axs1[1].set_ylabel('', fontsize=fs)
    axs1[2].set_ylabel('', fontsize=fs)
    axs1[0].legend(['GROMOS','SOMORA'], fontsize=fs)
    axs1[1].set_xlim(-40,40)
    axs1[1].set_xlabel(r'$\Delta$O$_3$ [\%]', fontsize=fs) 
    axs1[1].set_title(r'GROMOS-SOMORA', fontsize=fs+2)
    axs1[1].legend(fontsize=fs)
    axs1[2].set_xlim(0,1)
    axs1[2].set_title('Pearson correlation', fontsize=fs)

    for ax in axs1:
        ax.grid(which='both', axis='x')
        ax.grid(which='both', axis='y')
        
    fig.tight_layout(rect=[0, 0.01, 0.95, 1])
    fig1.tight_layout(rect=[0, 0.01, 0.95, 1])
    figure_list.append(fig1)
    figure_list.append(fig)
    save_single_pdf(basefolder+'seasonal_ozone_regression_'+freq+'_'+str(year)+'.pdf',figure_list)
    #fig.savefig(basefolder+'ozone_scatter_'+freq+'_'+str(year)+'.pdf', dpi=500)

def compute_seasonal_correlation_paper(gromos, somora, freq='1M', p_min = [100], p_max = [0.01], basefolder='', MLS = False, split_by ='season'):
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
        
        rel_mean_diff = 100*(ds.o3_gromos.mean(dim='time')-ds.o3_somora.mean(dim='time') )/ ds.o3_gromos.mean(dim='time')
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
                axs[i,col].set_xlabel(r'GROMOS O$_3$ [ppmv]', fontsize=fs)
                axs[i,col].set_ylabel(r'SOMORA O$_3$ [ppmv]', fontsize=fs)
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
    axs1[0].set_ylim(100,0.01)
    axs1[0].set_xlim(-0.2, 9)
    axs1[0].set_xlabel(r'O$_3$ [ppmv]', fontsize=fs)
    axs1[0].yaxis.set_major_formatter(FuncFormatter(lambda y, _: '{:g}'.format(y)))

    axs1[0].set_ylabel('Pressure [hPa]', fontsize=fs)
    axs1[1].set_ylabel('', fontsize=fs)
    axs1[2].set_ylabel('', fontsize=fs)
    #axs1[0].legend(['GROMOS','SOMORA'], fontsize=fs)
    axs1[1].set_xlim(-40,40)
    axs1[1].set_xlabel(r'$\Delta$O$_3$ [\%]', fontsize=fs) 
    axs1[1].set_title(r'GROMOS-SOMORA', fontsize=fs+2)
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
    save_single_pdf(basefolder+'seasonal_ozone_regression_paper_'+freq+'_'+str(year)+'.pdf',figure_list)
    #fig.savefig(basefolder+'ozone_scatter_'+freq+'_'+str(year)+'.pdf', dpi=500)

def compute_corr_profile(gromos, somora, freq='1D', date_slice=slice('2009-07-01','2021-12-31'), basefolder=''):
    year=pd.to_datetime(gromos.time.data[0]).year
    fs=24
    x = gromos.sel(time=date_slice).o3_x.resample(time=freq).mean().interpolate_na(dim='time',fill_value="extrapolate")
    y = somora.sel(time=date_slice).o3_x.resample(time=freq).mean().interpolate_na(dim='time',fill_value="extrapolate")

    # error_gromos = np.sqrt( np.square(gromos.o3_eo.mean(dim='time')) + np.square(gromos.o3_es.mean(dim='time')))
    # error_somora = np.sqrt( np.square(somora.o3_eo.mean(dim='time')) + np.square(somora.o3_es.mean(dim='time')))

    slopes = np.zeros(len(gromos.o3_p.values))
    rsqared = np.zeros(len(gromos.o3_p.values))
    for i, p in enumerate(gromos.o3_p.values): 
        x_p = x.sel(o3_p=p).values
        y_p = y.sel(o3_p=p).values
        result = stats.linregress(
            x_p, y_p
        )
        slopes[i] = result.slope 
        rsqared[i] = result.rvalue**2
        

    fig, axs = plt.subplots(1,3,sharey=True,figsize=(12,12))
    pearson_corr = xr.corr(x,y, dim='time')
    x.mean(dim='time').plot(
        ax=axs[0],
        y='o3_p',
        yscale='log',
    )
    y.mean(dim='time').plot(
        ax=axs[0],
        y='o3_p',
        yscale='log',
    )
    # pearson_corr.plot(
    #     ax=axs[2],
    #     y='o3_p',
    #     yscale='log',
    # )
    axs[1].plot(slopes, x.o3_p.values) 
    axs[2].plot(rsqared, x.o3_p.values) 
    axs[1].invert_yaxis()
    axs[0].set_xlim(0,10)
    axs[0].set_ylim(100,1e-2)
    axs[0].set_title('ozone')
    axs[0].set_ylabel('Pressure [hPa]')
    axs[1].set_title('Slopes')
    axs[1].set_xlim(0,1.5)
    axs[1].set_ylabel('')
    axs[2].set_title('$R^2$')
    axs[2].set_xlim(0,1)
    axs[2].set_ylabel('')
    for ax in axs:
        ax.grid()
        ax.set_xlabel('')
        ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: '{:g}'.format(y)))

    fig.savefig(basefolder+'ozone_corr_profile_'+freq+'_'+str(year)+'.pdf', dpi=500) 

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
        axs[i].set_ylabel(r'$\Delta$O$_3$ [\%]')
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


def trends_diff_old_new(gromos,somora, gromos_v2021, somora_old, mls, p_min, p_max, freq='1H', freq_avg='1M', outfolder='/scratch/GROSOM/Level2/GROMORA_retrievals_v2/'):
    #from sklearn.linear_model import LinearRegression
    from numpy.polynomial import Polynomial
    import scipy.stats as stats
    year=pd.to_datetime(gromos.time.data[0]).year
    fs = 32
    fig2, axs2 = plt.subplots(len(p_min), 1, sharex=True, figsize=(26,18))

    for i in range(len(p_min)):
        o3_gromos = gromos.o3_x.where(gromos.o3_p>p_min[i] , drop=True).where(gromos.o3_p<p_max[i], drop=True).mean(dim='o3_p').resample(time=freq).mean()
        o3_gromos_old = gromos_v2021.o3_x.where(gromos_v2021.pressure>p_min[i] , drop=True).where(gromos_v2021.pressure<p_max[i], drop=True).mean(dim='pressure').resample(time=freq).mean()

        o3_somora = somora.o3_x.interp(o3_p=gromos.o3_p).where(gromos.o3_p>p_min[i] , drop=True).where(gromos.o3_p<p_max[i], drop=True).mean(dim='o3_p').resample(time=freq).mean()
        o3_somora_old = somora_old.ozone.interp(pressure=gromos_v2021.pressure).where(gromos_v2021.pressure>p_min[i] , drop=True).where(gromos_v2021.pressure<p_max[i], drop=True).mean(dim='pressure').resample(time=freq).mean()

        diff_new = o3_gromos-o3_somora # 100*(o3_gromos-o3_somora)/o3_somora #o3_gromos-o3_somora
        monthly_diff = 100*(o3_gromos.resample(time=freq_avg).mean()-o3_somora.resample(time=freq_avg).mean())/o3_somora
        diff_old = o3_gromos_old-o3_somora_old# 100*(o3_gromos_old-o3_somora_old)/o3_somora_old #
        monthly_diff_old = 100*(o3_gromos_old.resample(time=freq_avg).mean()-o3_somora_old.resample(time=freq_avg).mean())/o3_somora

        time_ordinal = [pd.to_datetime(x).toordinal()- datetime.date(2010, 1, 1).toordinal() for x in diff_new.time.values]
        diff_new['ordinal'] = ('time', time_ordinal)
        diff_new = diff_new.where(~diff_new.isnull(), drop=True)#.where(~diff_old.isnull(), drop=True)
        #ds_fit_new = diff_new.polyfit(dim='time', deg=1, full = False, cov=True)
        #fit_new, cov_new = Polynomial.fit(diff_new['ordinal'].values, diff_new.values, deg=1, full=True )
        fit_new, cov_new = np.polyfit(diff_new['ordinal'].values, diff_new.values, deg=1, cov=True)
        
        slope_SE_new = np.sqrt(cov_new[0][0])/np.sqrt(len(diff_new['ordinal']))

        time_ordinal_old = [pd.to_datetime(x).toordinal() - datetime.date(2010, 1, 1).toordinal() for x in diff_old.time.values]
        diff_old['ordinal'] = ('time', time_ordinal_old)
        diff_old = diff_old.where(~diff_old.isnull(), drop=True)#.where(~diff_new.isnull(), drop=True)
        
        #ds_fit_old = diff_old.polyfit(dim='time', deg=1, full = False, cov=True)
        fit_old, cov_old = np.polyfit(diff_old['ordinal'].values, diff_old.values, deg=1, cov=True)
        slope_SE_old = np.sqrt(cov_old[0][0])/np.sqrt(len(diff_old['ordinal']))

        #res = stats.linregress(diff_old['ordinal'].values, diff_new.values - diff_old.values)
        res_new = stats.linregress(diff_new['ordinal'].values, diff_new.values )
        res_old = stats.linregress(diff_old['ordinal'].values, diff_old.values)

        t_stats, p_stats = stats.ttest_ind(diff_new.values, diff_old.values, equal_var=False, nan_policy='omit')
        print(t_stats, p_stats)
        SE_diff = np.sqrt(slope_SE_new**2 + slope_SE_old**2)
        #t_stat = 3.15576e17*(ds_fit_new.polyfit_coefficients.values[0] - ds_fit_old.polyfit_coefficients.values[0])/np.sqrt((se_new**2+se_old**2))
        df = len(diff_new['ordinal']) + len(diff_old['ordinal']) - 2
        
        #critical value
        alpha = 0.05

        print('Using '+ str(100*(1-alpha))+' confidence interval !')

        cv = stats.t.ppf(1.0 - alpha, df)

        tinv = lambda p, df: abs(stats.t.ppf(p/2, df))
        ts_old = tinv(alpha, len(diff_old['ordinal'].values)-2)
        ts_new = tinv(alpha, len(diff_new['ordinal'].values)-2)
        print(f"New slope ({100*(1-alpha)}\%): {res_new.slope:.6f} +/- {ts_old*res_new.stderr:.6f}")
        print(f"Old slope ({100*(1-alpha)}\%): {res_old.slope:.6f} +/- {ts_new*res_old.stderr:.6f}")

        #print(f"New intercept (95%): {res_new.intercept:.6f}"
            # f" +/- {ts*res_new.intercept_stderr:.6f}")

        t_manual = (fit_new[0] - fit_old[0]) / SE_diff
        # p-value
        p = (1 - stats.t.cdf(abs(t_manual), df)) * 2
        print(t_manual, cv, p)
        
        # #ds_fit_new =  monthly_diff.polyfit(dim='time', deg=1, full = True)#np.polyfit(diff_new, diff_new.values, deg=1)#
        #ds_fit_old = monthly_diff_old.polyfit(dim='time', deg=1, full = True)
       # np.polyfit(np.arange(0, len(diff_new.time.data)), diff_new.data, deg=1)
        # pl = diff_new.plot(
        #     x='time',
        #     color='k',
        #    # marker='x',
        #     linewidth=0.25,
        #     ax=axs2[i],
        #     label='new'
        # )
        axs2[i].plot(diff_old['time'], diff_old, color='r',linewidth=0.25)#, label = 'old')
        #axs2[i].plot(diff_old['ordinal'], np.polyval(fit_old, diff_old['ordinal']), color='g',linewidth=1.5)
        axs2[i].plot(diff_old['time'], res_old.intercept + res_old.slope*diff_old['ordinal'], color='r',linewidth=1.5, label=f"Old slope: ${3652.5*res_old.slope:.3f} \pm {3652.5*ts_old*res_old.stderr:.3f}$ ppmv/dec")

        axs2[i].plot(diff_new['time'], diff_new, color='k',linewidth=0.25)#, label = 'new')
        #axs2[i].plot(diff_new['ordinal'], np.polyval(fit_new, diff_new['ordinal']), color='g',linewidth=1.5)
        axs2[i].plot(diff_new['time'], res_new.intercept + res_new.slope*diff_new['ordinal'], color='k',linewidth=1.5, label=f"New slope: ${3652.5*res_new.slope:.3f} \pm {3652.5*ts_new*res_new.stderr:.3f}$ ppmv/dec")

        # pl2 = monthly_diff.plot(
        #     x='time',
        #     color='k',
        #     #marker='x',
        #     linewidth=1.5,
        #     ax=axs2[i],
        # )
        # fit_new = xr.polyval(coord=diff_new['time'] , coeffs=ds_fit_new.polyfit_coefficients).plot(
        #     ax=axs2[i],
        #     color='k',
        #     label='fitted new' 
        #     )
              
        # # ax.set_yscale('log')   
        # pl2 = diff_old.plot(
        #     x='time',
        #     color='r',
        #      #marker='o',
        #     linewidth=0.25,
        #     ax=axs2[i],
        #     label='old' 
        # )
        # # pl4 = monthly_diff_old.plot(
        # #     x='time',
        # #     color='r',
        # #     #marker='x',
        # #     linewidth=1.5,
        # #     ax=axs2[i],
        # # )
        # fit_old = xr.polyval(coord=diff_old['time'] , coeffs=ds_fit_old.polyfit_coefficients).plot(
        #     ax=axs2[i],
        #     color='r',
        #     label='fitted old' 
        #     )
        axs2[i].set_ylim(-1,1)
        axs2[i].set_title(r'GROMOS-SOMORA, $'+ str(p_min[i])+ ' < p < '+str(p_max[i])+'$ hPa',fontsize=fs+4)
        axs2[i].text(
            0.024,
            0.05,
            PAPER_SYMBOL[i],
            transform=axs2[i].transAxes,
            verticalalignment="bottom",
            horizontalalignment="left",
            fontsize=fs
        )
    #     axs2[i].text(
    #         0.37,
    #         0.01,
    #         f"New slope: ${3652.5*res_new.slope:.3f} \pm {3652.5*ts_new*res_new.stderr:.3f}$ ppmv/dec \n Old slope: ${3652.5*res_old.slope:.3f} \pm {3652.5*ts_old*res_old.stderr:.3f}$ ppmv/dec",
    #         transform=axs2[i].transAxes,
    #         verticalalignment="bottom",
    #         horizontalalignment="left",
    #         fontsize=fs
    # )

    #'Slope new $={:.4f} \pm {:.4f}$ ppmv/dec \n Slope old $={:.4f} \pm {:.4f}$ ppmv/dec'.format(res_new.slope,np.sqrt(cov_new[0][0]), fit_old[0],np.sqrt(cov_old[0][0])),
    #'Slope new $={:.4f} \pm {:.4f}$ ppmv/dec \n Slope old $={:.4f} \pm {:.4f}$ ppmv/dec'.format(3.15576e17*ds_fit_new.polyfit_coefficients.values[0],3.15576e17*np.sqrt(ds_fit_new.polyfit_covariance.values[0][0]), 3.15576e17*ds_fit_old.polyfit_coefficients.values[0], 3.15576e17*np.sqrt(ds_fit_old.polyfit_covariance.values[0][0])),

    axs2[0].set_ylim(-0.8,0.8)
    axs2[1].set_ylim(-1.3,1.3)
    axs2[2].set_ylim(-1.3,1.3)
    
    axs2[2].xaxis.set_major_locator(mdates.YearLocator())
    axs2[2].xaxis.set_major_formatter(mdates.ConciseDateFormatter(axs2[2].xaxis.get_major_locator()))

    for ax in axs2:
        ax.legend(fontsize=fs)
        ax.set_xlabel('')
        ax.grid()
        ax.set_ylabel(r'$\Delta$ O$_3$ [ppmv]',fontsize=fs)
        ax.tick_params(axis='both', which='major', labelsize=fs)
        ax.xaxis.set_tick_params(rotation=45)
        #ax.xaxis.set_major_locator(mdates.YearLocator())
        #ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=(1, 7)))
        
        #ax.set_ylabel(r'RD [\%]')

    #ew_labels = [datetime.date.fromordinal(int(item + datetime.date(1970, 1, 1).toordinal())) for item in axs2[2].get_xticks()]
    #axs2[2].set_xticklabels(new_labels)


    fig2.tight_layout(rect=[0, 0.01, 0.99, 1])
    fig2.savefig(outfolder+'trends_old_new_v2_'+str(year)+'.pdf', dpi=500)


def trends_simplified_new(gromos,somora, mls, p_min, p_max, freq='1H', freq_avg='1M', outfolder='/scratch/GROSOM/Level2/GROMORA_retrievals_v2/'):
    #from sklearn.linear_model import LinearRegression
    
    year=pd.to_datetime(gromos.time.data[0]).year
    fs = 30
    fig2, axs2 = plt.subplots(len(p_min), 1, sharey=False, sharex=True, figsize=(26,16))

    for i in range(len(p_min)):
        o3_gromos = gromos.o3_x.where(gromos.o3_p>p_min[i] , drop=True).where(gromos.o3_p<p_max[i], drop=True).mean(dim='o3_p').resample(time=freq).mean()
        o3_mls = mls.o3.where(mls.p>p_min[i] , drop=True).where(mls.p<p_max[i], drop=True).mean(dim='p').resample(time=freq).mean()
        o3_somora = somora.o3_x.interp(o3_p=gromos.o3_p).where(gromos.o3_p>p_min[i] , drop=True).where(gromos.o3_p<p_max[i], drop=True).mean(dim='o3_p').resample(time=freq).mean()

        monthly_diff = 100*(o3_gromos.resample(time=freq_avg).mean()-o3_somora.resample(time=freq_avg).mean())/o3_somora

        #diff_new['ordinal'] = ('time', time_ordinal)
        gromos_fit = o3_gromos.polyfit(dim='time', deg=1, full = True)
        somora_fit = o3_somora.polyfit(dim='time', deg=1, full = True)
        mls_fit = o3_mls.polyfit(dim='time', deg=1, full = True)

        #ds_fit_new = monthly_diff.polyfit(dim='time', deg=1, full = True)#np.polyfit(diff_new, diff_new.values, deg=1)#
        #ds_fit_old = monthly_diff_old.polyfit(dim='time', deg=1, full = True)
       # np.polyfit(np.arange(0, len(diff_new.time.data)), diff_new.data, deg=1)
        pl = o3_gromos.plot(
            x='time',
            color=get_color('GROMOS'),
           # marker='x',
            linewidth=0.25,
            ax=axs2[i],
            label='GROMOS'
        )
        pl2 = o3_somora.plot(
            x='time',
            color=get_color('SOMORA'),
            #marker='x',
            linewidth=0.25,
            ax=axs2[i],
            label='SOMORA'
        )        
        pl_mls = o3_mls.plot(
            x='time',
            color=get_color('MLS'),
           # marker='x',
            linewidth=0.25,
            ax=axs2[i],
            label='MLS'
        )
        # pl2 = o3_somora.plot(
        #     x='time',
        #     color=get_color('SOMORA'),
        #     #marker='x',
        #     linewidth=1.5,
        #     ax=axs2[i],
        # )
        fit_gromos = xr.polyval(coord=o3_gromos['time'] , coeffs=gromos_fit.polyfit_coefficients).plot(
            ax=axs2[i],
            color=get_color('GROMOS'),
            linewidth=1.25,
            )
        fit_somora = xr.polyval(coord=o3_somora['time'] , coeffs=somora_fit.polyfit_coefficients).plot(
            ax=axs2[i],
            color=get_color('SOMORA'),
            linewidth=1.25,
            )
        fit_mls = xr.polyval(coord=o3_somora['time'] , coeffs=mls_fit.polyfit_coefficients).plot(
            ax=axs2[i],
            color=get_color('MLS'),
            linewidth=1.25,
            )
        #axs2[i].set_ylim(-1,1)
        axs2[i].set_title(r'Trends, $'+ str(p_min[i])+ ' < p < '+str(p_max[i])+'$ hPa',fontsize=fs+4)

        axs2[i].text(
            0.37,
            0.01,
            'Trends GROMOS $={:.4f} $ ppmv/dec \n Trend SOMORA $={:.4f} $ ppmv/dec \n Trend MLS $={:.4f} $ ppmv/dec'.format(3.15576e17*gromos_fit.polyfit_coefficients.values[0],3.15576e17*somora_fit.polyfit_coefficients.values[0],3.15576e17*mls_fit.polyfit_coefficients.values[0]),
            transform=axs2[i].transAxes,
            verticalalignment="bottom",
            horizontalalignment="left",
            fontsize=fs
    )
    axs2[0].set_ylim(0,3)
    axs2[1].set_ylim(4,7.5)
    axs2[2].set_ylim(3,6.5)
    axs2[0].legend(fontsize=fs)
    for ax in axs2:
        ax.set_xlabel('')
        ax.grid()
        ax.set_ylabel(r'O$_3$ [ppmv]',fontsize=fs)
        ax.xaxis.set_major_locator(mdates.YearLocator())
        #ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=(1, 7)))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        #ax.set_ylabel(r'RD [\%]')
    
    fig2.tight_layout(rect=[0, 0.01, 0.99, 1])
    fig2.savefig(outfolder+'trends_simplified_new_'+str(year)+'.pdf', dpi=500)

def compare_old_new(gromos,somora, gromos_v2021, somora_old, mls, freq='1M', outfolder='/scratch/GROSOM/Level2/GROMORA_retrievals_v2/'):
    figures=list()
    year=pd.to_datetime(gromos.time.data[0]).year
    lim = 40
    fs=30

    o3_gromos = gromos.o3_x.mean(dim='time')
    o3_gromos_old = gromos_v2021.o3_x.mean(dim='time')

    o3_somora = somora.o3_x.mean(dim='time').interp(o3_p=o3_gromos.o3_p)
    o3_somora_old = somora_old.ozone.mean(dim='time').interp(pressure=o3_gromos_old.pressure)

    diff_new = 100*(o3_gromos-o3_somora)/o3_somora
    diff_old = 100*(o3_gromos_old-o3_somora_old)/o3_somora_old
    fig, axs = plt.subplots(1, 2, sharey=True, figsize=(18,12))
    o3_gromos.plot(ax=axs[0] , y='o3_p', color=color_gromos, marker = 'x', label='new')
    o3_gromos_old.plot(ax=axs[0] , y='pressure', color=color_gromos,marker = 'o', label='old')
    o3_somora.plot(ax=axs[0] , y='o3_p', color=color_somora, marker = 'x', label='new')
    o3_somora_old.plot(ax=axs[0] , y='pressure', color=color_somora,marker = 'o', label='old')
    diff_new.plot(ax=axs[1] , y='o3_p', color='k', marker = 'x',label='new')
    diff_old.plot(ax=axs[1] , y='pressure', color='k',marker = 'o', label='old')
    axs[0].invert_yaxis()
    
    axs[0].set_ylim(200,0.005)
    axs[0].set_yscale('log')  
    axs[1].set_xlim(-40,40)
    axs[1].legend()
    for ax in axs:
        ax.set_xlabel('o3')
        ax.set_ylabel('P')
    # figures.append(fig)
    fig.savefig('/scratch/GROSOM/Level2/GROMORA_retrievals_v2/old_vs_new_mean_diff_'+str(year)+'.pdf', dpi=500)
    
    fig2, axs2 = plt.subplots(2, 1, sharey=True, sharex=True, figsize=(22,12))
    o3_gromos = gromos.o3_x.resample(time=freq).mean()
    o3_gromos_old = gromos_v2021.o3_x.resample(time=freq).mean()
    
    o3_somora = somora.o3_x.resample(time=freq).mean().interp(o3_p=o3_gromos.o3_p)
    o3_somora_old = somora_old.ozone.resample(time=freq).mean().interp(pressure=gromos_v2021.pressure)
    
    diff_new = 100*(o3_gromos-o3_somora)/o3_somora
    diff_old = 100*(o3_gromos_old-o3_somora_old)/o3_somora_old

    pl = diff_new.plot(
        x='time',
        y='o3_p',
        ax=axs2[0], 
        vmin=-lim,
        vmax=lim,
        yscale='log',
        linewidth=0,
        rasterized=True,
        cmap='coolwarm',
        add_colorbar=False
       # cbar_kwargs={'label':r'$\Delta$O$_3$ [\%]'}
    )
    axs2[0].set_title('GROMOS-SOMORA: harmonized retrievals')
    pl.set_edgecolor('face')
    # ax.set_yscale('log')   
    pl = diff_old.plot(
        x='time',
        y='pressure',
        ax=axs2[1], 
        vmin=-lim,
        vmax=lim,
        yscale='log',
        linewidth=0,
        rasterized=True,
        cmap='coolwarm',
        add_colorbar=False
       # cbar_kwargs={'label':r'$\Delta$O$_3$ [\%]'}
    )
    axs2[0].invert_yaxis()
    axs2[1].set_title('GROMOS-SOMORA: previous retrievals')
    cbaxes = fig2.add_axes([0.92, 0.25, 0.02, 0.5]) 
    #    cb = plt.colorbar(pl, cax=cbaxes, orientation="vertical", pad=0.0)
    cb = fig2.colorbar(pl, cax=cbaxes, orientation="vertical", extend='both', pad=0.0)
    cb.set_label(label=r"$\Delta$O$_3$ [\%]", fontsize=fs)
    cb.ax.tick_params()

    for ax in axs2:
        ax.set_ylim(100,0.01)
        ax.set_xlabel('')
        ax.set_ylabel('Pressure [hPa]')
        ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: '{:g}'.format(y)))
        ax.xaxis.set_major_locator(mdates.YearLocator())
        #ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=(1, 7)))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    fig2.tight_layout(rect=[0, 0.01, 0.92, 1])

    figures.append(fig2)

    # o3_mls_interp_gromos_old= mls.o3.resample(time=freq).mean().interp(p=o3_gromos_old.pressure)
    # o3_mls_interp_gromos = mls.o3.resample(time=freq).mean().interp(p=o3_gromos.o3_p)
    # diff_new_gromos_mls = 100*(o3_gromos-o3_mls_interp_gromos)/o3_gromos
    # diff_old_gromos_mls = 100*(o3_gromos_old - o3_mls_interp_gromos_old)/o3_gromos_old
    
    # o3_mls_interp_somora_old= mls.o3.resample(time=freq).mean().interp(p=o3_somora_old.pressure)
    # o3_mls_interp_somora = mls.o3.resample(time=freq).mean().interp(p=o3_somora.o3_p)
    # diff_new_somora_mls = 100*(o3_somora-o3_mls_interp_somora)/o3_gromos
    # diff_old_somora_mls = 100*(o3_somora_old - o3_mls_interp_somora_old)/o3_somora_old
    
    # fig2, axs2 = plt.subplots(2, 1, sharey=True, sharex=True, figsize=(10,10))
    # pl = diff_new_gromos_mls.plot(
    #     x='time',
    #     y='o3_p',
    #     ax=axs2[0], 
    #     vmin=-lim,
    #     vmax=lim,
    #     yscale='log',
    #     linewidth=0,
    #     rasterized=True,
    #     cmap='coolwarm',
    #     cbar_kwargs={'label':r'$\Delta$O$_3$ [\%]'}
    # )
    # axs2[0].set_title('GROMOS, new vs MLS')
    # pl.set_edgecolor('face')
    # # ax.set_yscale('log')   
    # pl = diff_old_gromos_mls.plot(
    #     x='time',
    #     y='pressure',
    #     ax=axs2[1], 
    #     vmin=-lim,
    #     vmax=lim,
    #     yscale='log',
    #     linewidth=0,
    #     rasterized=True,
    #     cmap='coolwarm',
    #     cbar_kwargs={'label':r'$\Delta$O$_3$ [\%]'}
    # )
    # axs2[0].invert_yaxis()
    # axs2[1].set_title('GROMOS, old vs MLS')
    # axs2[0].set_ylim(100,0.01)
    # for ax in axs2:
    #     ax.set_xlabel('')
    #     ax.set_ylabel('Pressure [hPa]')

    # figures.append(fig2)

    # fig2, axs2 = plt.subplots(2, 1, sharey=True, sharex=True, figsize=(10,10))
    # pl = diff_new_somora_mls.plot(
    #     x='time',
    #     y='o3_p',
    #     ax=axs2[0], 
    #     vmin=-lim,
    #     vmax=lim,
    #     yscale='log',
    #     linewidth=0,
    #     rasterized=True,
    #     cmap='coolwarm',
    #     cbar_kwargs={'label':r'$\Delta$O$_3$ [\%]'}
    # )
    # axs2[0].set_title('SOMORA, new vs MLS')
    # pl.set_edgecolor('face')
    # # ax.set_yscale('log')   
    # pl = diff_old_somora_mls.plot(
    #     x='time',
    #     y='pressure',
    #     ax=axs2[1], 
    #     vmin=-lim,
    #     vmax=lim,
    #     yscale='log',
    #     linewidth=0,
    #     rasterized=True,
    #     cmap='coolwarm',
    #     cbar_kwargs={'label':r'$\Delta$O$_3$ [\%]'}
    # )
    # axs2[0].invert_yaxis()
    # axs2[1].set_title('SOMORA, old vs MLS')
    # axs2[0].set_ylim(100,0.01)
    # for ax in axs2:
    #     ax.set_xlabel('')
    #     ax.set_ylabel('Pressure [hPa]')

    # figures.append(fig2)

    save_single_pdf(outfolder + 'GROMORA_old_vs_new_diff_'+str(year)+'.pdf', figures)
   # fig2.savefig('/scratch/GROSOM/Level2/GROMORA_retrievals_v2/GROMORA_old_vs_new_diff_'+str(year)+'.pdf', dpi=500)


def gromos_old_vs_new(gromos, gromos_v2021, mls, seasonal = True):
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
    fig.savefig('/scratch/GROSOM/Level2/GROMORA_retrievals_v2/gromos_old_vs_new_'+str(year)+'.pdf', dpi=500)

    if seasonal:
        gromora_groups = gromos.groupby('time.season').groups
        mls_groups = mls.groupby('time.season').groups
        gromos_v2021_groups = gromos_v2021.groupby('time.season').groups
        #ds_o3_gromora_plot = ds_o3_gromora.isel(o3_p=pressure_leel)
        
        color_season = ['r', 'b', 'y', 'g']
        fig, axs = plt.subplots(1, 4, sharex=True, sharey=True, figsize=(10,10))
        for j, s in enumerate(gromora_groups):
            print('Processing season ', s)
            gromos.isel(time=gromora_groups[s]).mean(dim='time').o3_x.plot(ax=axs[j] , y='o3_p', color=color_gromos, label='new')
            mls.isel(time=mls_groups[s]).mean(dim='time').o3.plot(ax=axs[j], y='p', color='k', label='MLS')
            gromos_v2021.isel(time=gromos_v2021_groups[s]).o3_x.mean(dim='time').plot(ax=axs[j], y='pressure', color=color_somora, label='old')
            axs[j].set_title(s)
        axs[0].invert_yaxis()
        axs[0].set_ylim(200,0.005)
        axs[0].set_yscale('log')   
        axs[3].legend() 
        for ax in axs:
            ax.grid()
        fig.savefig('/scratch/GROSOM/Level2/GROMORA_retrievals_v2/gromos_old_vs_new_seasonal_'+str(year)+'.pdf', dpi=500)
    else:
        o3_gromos = gromos.o3_x.mean(dim='time')
        o3_gromos_old = gromos_v2021.o3_x.mean(dim='time').interp(pressure=o3_gromos.o3_p)
        o3_mls = mls.o3.mean(dim='time').interp(p=o3_gromos.o3_p)
        
        diff_new = 100*(o3_gromos-o3_mls)/o3_mls
        diff_old = 100*(o3_gromos_old-o3_mls)/o3_mls
        fig, axs = plt.subplots(1, 2, sharey=True, figsize=(10,10))
        o3_mls.plot(ax=axs[0] , y='o3_p', color='k', label='MLS')
        o3_gromos.plot(ax=axs[0] , y='o3_p', color=color_gromos, label='new')
        o3_gromos_old.plot(ax=axs[0] , y='o3_p', color=color_somora, label='old')
        diff_new.plot(ax=axs[1] , y='o3_p', color=color_gromos, label='new')
        diff_old.plot(ax=axs[1] , y='o3_p', color=color_somora, label='old')
        axs[0].invert_yaxis()
        axs[0].set_ylim(200,0.005)
        axs[0].set_yscale('log')  
        axs[1].set_xlim(-50,50)
        axs[1].legend()
        fig.savefig('/scratch/GROSOM/Level2/GROMORA_retrievals_v2/gromos_old_vs_mean_diff_'+str(year)+'.pdf', dpi=500)

        freq='1M'
        fig2, axs2 = plt.subplots(2, 1, sharey=True, sharex=True, figsize=(10,10))
        o3_gromos = gromos.o3_x.resample(time=freq).mean()
        o3_gromos_old = gromos_v2021.o3_x.resample(time=freq).mean().interp(pressure=o3_gromos.o3_p)
        o3_gromos_old_2 = gromos_v2021.o3_x.resample(time=freq).mean()
        o3_mls_interp_old= mls.o3.resample(time=freq).mean().interp(p=o3_gromos_old_2.pressure)
        o3_mls = mls.o3.resample(time=freq).mean().interp(p=o3_gromos.o3_p)
        diff_new = (o3_gromos-o3_mls)
        diff_old = (o3_gromos_old_2-o3_mls_interp_old)
        pl = diff_new.plot(
            x='time',
            y='o3_p',
            ax=axs2[0], 
            vmin=-1,
            vmax=1,
            yscale='log',
            linewidth=0,
            rasterized=True,
            cmap='coolwarm'
        )
        axs2[0].set_title('New vs MLS')
        pl.set_edgecolor('face')
        # ax.set_yscale('log')   
        pl = diff_old.plot(
            x='time',
            y='pressure',
            ax=axs2[1], 
            vmin=-1,
            vmax=1,
            yscale='log',
            linewidth=0,
            rasterized=True,
            cmap='coolwarm'
        )
        axs2[0].invert_yaxis()
        axs2[1].set_title('Old vs MLS')
        axs2[0].set_ylim(100,0.01)

        for ax in axs2:
            ax.set_xlabel('')
            ax.set_ylabel('P [hPa]')
        fig2.savefig('/scratch/GROSOM/Level2/GROMORA_retrievals_v2/gromos_old_vs_diff_'+str(year)+'.pdf', dpi=500)

def compare_mean_diff(gromos, somora, sbuv=None, basefolder=None):
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
        ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: '{:g}'.format(y)))
        ax.fill_between(ax.get_xlim(),p_somora_mr[0],1e4, color=color_shading, alpha=0.2)
        ax.fill_between(ax.get_xlim(),p_somora_mr[-1],1e-4, color=color_shading, alpha=0.2)
        ax.fill_between(ax.get_xlim(),p_gromos_mr[0],1e4, color=color_shading, alpha=0.2)
        ax.fill_between(ax.get_xlim(),p_gromos_mr[-1],1e-4, color=color_shading, alpha=0.2)

    #fig.suptitle('Ozone relative difference GROMOS-SOMORA')
    fig.tight_layout(rect=[0, 0.01, 0.99, 1])
    fig.savefig(basefolder+'rel_diff'+str(year)+'.pdf', dpi=500)

def compare_mean_diff_monthly(gromos, somora, mls=None, sbuv=None, outfolder='/scratch/GROSOM/Level2/'):
    
    fs = 22
    year=pd.to_datetime(gromos.time.data[0]).year

    monthly_gromos = gromos.groupby('time.month').mean()
    monthly_somora = somora.groupby('time.month').mean()
    monthly_mls = mls.groupby('time.month').mean()
    monthly_sbuv = sbuv.groupby('time.month').mean()

    figure = list()
    for i in range(len(monthly_gromos.month)):
        fig, axs = plt.subplots(1, 1, sharey=True, figsize=(8,12))
        
        mr_somora = monthly_somora.isel(month=i).o3_mr.data
        mr_gromos = monthly_gromos.isel(month=i).o3_mr.data
        p_somora_mr = monthly_somora.isel(month=i).o3_p.data[mr_somora>=0.8]
        p_gromos_mr = monthly_gromos.isel(month=i).o3_p.data[mr_gromos>=0.8]

        monthly_somora.isel(month=i).o3_x.plot(
            y='o3_p',
            ax=axs, 
            yscale='log',
            color=color_somora
        )
        monthly_gromos.isel(month=i).o3_x.plot(
            y='o3_p',
            ax=axs, 
            color=color_gromos
        )
        
        monthly_sbuv.isel(month=i).ozone.plot(
            y='p',
            ax=axs, 
            color=sbuv_color
        )     
        monthly_mls.isel(month=i).o3.plot(
            y='p',
            ax=axs, 
            color='k'
        )

        axs.set_title(r'O$_3$ VMR, '+MONTH_STR[i], fontsize=fs+4)
        axs.set_xlabel('VMR [ppmv]', fontsize=fs)


        axs.legend(('SOMORA','GROMOS','SBUV','MLS'))
        #axs[0].legend(('SB corr','OG'))
        axs.invert_yaxis()
        axs.set_xlim(-0.2, 9)
        axs.set_ylabel('Pressure [hPa]', fontsize=fs)
        axs.set_ylim(100, 0.01)
        axs.xaxis.set_major_locator(MultipleLocator(4))
        axs.xaxis.set_minor_locator(MultipleLocator(1))
        #axs[1].set_ylim((somora.o3_z.mean(dim='time')[12]/1e3,somora.o3_z.mean(dim='time')[35]/1e3))

        axs.grid(which='both', axis='y', linewidth=0.5)
        axs.grid(which='both', axis='x', linewidth=0.5)
        axs.tick_params(axis='both', which='major', labelsize=fs-2)
        axs.yaxis.set_major_formatter(FuncFormatter(lambda y, _: '{:g}'.format(y)))
        axs.fill_between(axs.get_xlim(),p_somora_mr[0],1e4, color=color_shading, alpha=0.2)
        axs.fill_between(axs.get_xlim(),p_somora_mr[-1],1e-4, color=color_shading, alpha=0.2)
        axs.fill_between(axs.get_xlim(),p_gromos_mr[0],1e4, color=color_shading, alpha=0.2)
        axs.fill_between(axs.get_xlim(),p_gromos_mr[-1],1e-4, color=color_shading, alpha=0.2)
        fig.tight_layout(rect=[0, 0.01, 0.99, 1])
    
        figure.append(fig)
    save_single_pdf(outfolder+'monthly_mean_comparison_mls_sbuv_'+str(year)+'.pdf', figure)
    #fig.suptitle('Ozone relative difference GROMOS-SOMORA')
    
    #fig.savefig('/scratch/GROSOM/Level2/GROMORA_waccm/rel_diff'+str(year)+'.pdf', dpi=500)

def compare_mean_diff_alt(ozone_const_alt_gromos, ozone_const_alt_somora, gromora_old, ozone_const_alt_gromos_v2021, outfolder):
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
        ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: '{:g}'.format(y)))

    fig.suptitle('Ozone relative difference SOMORA-GROMOS')
    fig.tight_layout(rect=[0, 0.01, 0.95, 1])
    fig.savefig(outfolder+'rel_diff_old_vs_new_'+str(year)+'.pdf', dpi=500)

def compare_diff_daily_altitude(gromos, somora, gromora_old, altitudes = [15,20,25]):
    year=pd.to_datetime(gromos.time.data[0]).year
    fig, axs = plt.subplots(len(altitudes), 1, sharex=True, figsize=(15,10))
    for i, alt in enumerate(altitudes):
        daily_diff = 100*(somora.o3_x.sel(altitude=alt).resample(time='1D').mean() - gromos.o3_x.sel(altitude=alt).resample(time='1D').mean())/gromos.o3_x.sel(altitude=alt).resample(time='D').mean()
        daily_diff.plot(ax=axs[i], color='b', marker ='.', lw=0.6, label='New retrieval routine')

        gromora_old.o3_rel_diff.sel(altitude=alt).plot(ax=axs[i], color='r', marker ='.',lw=0.6, label='Old retrieval routine')

        #daily_diff_old.plot(ax=axs[i], color='k', lw=0.6, label='Old')
        axs[i].set_xlabel('')
        axs[i].set_ylabel(r'$\Delta$O$_3$ [\%]')
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

def compare_avkm(gromos, somora, date_slice, outfolder, seasonal=False):
    fs = 32
    fig, axs = plt.subplots(1, 2, sharey=True, figsize=(10,10))
    avkm_gromos = gromos.o3_avkm.sel(time=date_slice)
    mean_avks_gromos= avkm_gromos.mean(dim='time')
    mean_fwhm_gromos= gromos.o3_fwhm.mean(dim='time')
    avkm_somora = somora.o3_avkm.sel(time=date_slice)
    mean_avks_somora = avkm_somora.mean(dim='time')
    mean_fwhm_somora= somora.o3_fwhm.mean(dim='time')
    p = mean_avks_gromos.o3_p.data
    
    gromos_mr = gromos.o3_mr.sel(time=date_slice)
    somora_mr = somora.o3_mr.sel(time=date_slice)
    mean_MR_gromos= 0.25*gromos_mr.mean(dim='time')
    mean_MR_somora= 0.25*somora_mr.mean(dim='time')
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
    axs[0].fill_betweenx(mean_MR_gromos.o3_p.data,mean_MR_gromos.data-0.5*gromos_mr.std(dim='time').data,mean_MR_gromos.data+0.5*gromos_mr.std(dim='time').data, color='r', alpha=0.2)
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
    axs[1].fill_betweenx(mean_MR_somora.o3_p.data,mean_MR_somora.data-0.5*somora_mr.std(dim='time').data,mean_MR_somora.data+0.5*somora_mr.std(dim='time').data, color='r', alpha=0.2)
    axs[1].axhline(y=good_p_somora[0], ls='--', color='red')
    axs[1].axhline(y=good_p_somora[-1], ls='--', color='red')
    axs[0].invert_yaxis()
    axs[0].set_yscale('log')
    axs[1].set_title('AVKs SOMORA')

    for ax in axs:
        ax.grid()
        ax.set_xlim(-0.1, 0.35)
        ax.set_xlabel('AVKs')
        ax.set_ylabel('Pressure [hPa]')
        ax.xaxis.set_major_locator(MultipleLocator(0.25))
        ax.xaxis.set_minor_locator(MultipleLocator(0.125))
    fig.tight_layout(rect=[0, 0.01, 0.95, 1])
    fig.savefig(outfolder+'AVKs_comparison_'+str(avkm_gromos.time.data[0])[0:10]+'.pdf', dpi=500)
    
    
    season_id = ['a)', 'b)', 'c)', 'd)']
    if seasonal:
        figures = list()
        figures2 = list()
        figures3 = list()
        figures4 = list()
        gro_groups = gromos.groupby('time.season').groups
        som_groups = somora.groupby('time.season').groups
  
        for j, s in enumerate(gro_groups):
            fig, axs = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(10,14))
            fig2, axs2 = plt.subplots(1, 1, sharex=True, sharey=True, figsize=(10,14))
            fig3, axs3 = plt.subplots(1, 1, sharex=True, sharey=True, figsize=(10,14))
            fig4, axs4 = plt.subplots(1, 1, sharex=True, sharey=True, figsize=(10,14))

            print('Processing season ', s)
            p = avkm_gromos.o3_p.data
            o3_z_gromos = gromos.isel(time=gro_groups[s]).o3_z.mean(dim='time')
            o3_z_somora = somora.isel(time=som_groups[s]).o3_z.mean(dim='time')

            gromos_mr = gromos.o3_mr.isel(time=gro_groups[s])
            somora_mr = somora.o3_mr.isel(time=som_groups[s])
            mean_MR_gromos= gromos_mr.mean(dim='time')
            mean_MR_somora= somora_mr.mean(dim='time')
            good_p_gromos = gromos.o3_p.where(mean_MR_gromos>0.2,drop=True)
            good_p_somora = somora.o3_p.where(mean_MR_somora>0.2,drop=True)

            avkm_gromos = gromos.isel(time=gro_groups[s]).o3_avkm.mean(dim='time')
            axs2.plot(1e-3*gromos.isel(time=gro_groups[s]).o3_fwhm.mean(dim='time'), gromos.isel(time=gro_groups[s]).o3_p, color=get_color('GROMOS'), label='FWHM, GROMOS')
            axs2.plot(1e-3*somora.isel(time=som_groups[s]).o3_fwhm.mean(dim='time'), somora.isel(time=som_groups[s]).o3_p, color=get_color('SOMORA'), label='FWHM, SOMORA')

            axs2.plot(1e-3*gromos.isel(time=gro_groups[s]).o3_offset.mean(dim='time'), gromos.isel(time=gro_groups[s]).o3_p,color=get_color('GROMOS'), linestyle='--', label='AVKs offset, GROMOS')
           # mean_avks_gromos= avkm_gromos.mean(dim='time')

            avkm_somora = somora.isel(time=som_groups[s]).o3_avkm.mean(dim='time')
            axs2.plot(1e-3*somora.isel(time=som_groups[s]).o3_offset.mean(dim='time'),somora.isel(time=som_groups[s]).o3_p, color=get_color('SOMORA'), linestyle='--', label='AVKs offset, SOMORA')
           # mean_avks_somora = avkm_somora.mean(dim='time')

            axs4.plot(1e6*gromos.isel(time=gro_groups[s]).o3_eo.mean(dim='time'), gromos.isel(time=gro_groups[s]).o3_p,'-', color=get_color('GROMOS'), label='noise, GROMOS')
            axs4.plot(1e6*somora.isel(time=som_groups[s]).o3_eo.mean(dim='time'), somora.isel(time=som_groups[s]).o3_p,'-', color=get_color('SOMORA'), label='noise, SOMORA')
            axs4.plot(1e6*gromos.isel(time=gro_groups[s]).o3_es.mean(dim='time'), gromos.isel(time=gro_groups[s]).o3_p,'--' , color=get_color('GROMOS'), label='smoothing, GROMOS')
            axs4.plot(1e6*somora.isel(time=som_groups[s]).o3_es.mean(dim='time'), somora.isel(time=som_groups[s]).o3_p,'--', color=get_color('SOMORA'), label='smoothing, SOMORA')


            counter = 0
            color_count = 0
            for avk in avkm_gromos:
                #if 0.8 <= np.sum(avk) <= 1.2:
                counter=counter+1
                if np.mod(counter,8)==0:
                    axs[0].plot(avk, p, color=cmap(color_count*0.25+0.01), lw=2)
                    color_count = color_count +1            
                else:
                    axs[0].plot(avk, p, color='k')
            axs[0].plot(0.25*mean_MR_gromos.data,mean_MR_gromos.o3_p, color=get_color('GROMOS'))
            axs3.plot(mean_MR_gromos.data,mean_MR_gromos.o3_p, color=get_color('GROMOS'), label='GROMOS')
            axs[0].fill_betweenx(mean_MR_gromos.o3_p.data,0.25*mean_MR_gromos.data-0.5*gromos_mr.std(dim='time').data,0.25*mean_MR_gromos.data+0.5*gromos_mr.std(dim='time').data, color=get_color('GROMOS'), alpha=0.2)
            axs3.fill_betweenx(mean_MR_gromos.o3_p.data,mean_MR_gromos.data-0.5*gromos_mr.std(dim='time').data,mean_MR_gromos.data+0.5*gromos_mr.std(dim='time').data, color=get_color('GROMOS'), alpha=0.2)
            #axs[0].axhline(y=good_p_gromos[0], ls='--', color='red')
            #axs[0].axhline(y=good_p_gromos[-1], ls='--', color='red')
            axs[0].set_title('GROMOS: '+s)
            axs[0].set_ylabel('Pressure [hPa]', fontsize=fs)

            axs2.set_title('Resolution and vertical offset: '+s, fontsize=fs+4)
            axs2.set_xlabel("Resolution and offset [km]", fontsize=fs)
            axs2.set_xlim(-10, 25)
            axs4.set_title('Errors: '+s, fontsize=fs+4)
            axs2.set_xlabel("Errors [ppmv]", fontsize=fs)
            axs4.set_xlim(0, 1)
            for ax in[axs2, axs4]: 
                ax.set_yscale('log')
                ax.invert_yaxis()
                ax.set_ylim(500, 2e-3)
                ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: '{:g}'.format(y)))
                ax.set_ylabel('Pressure [hPa]', fontsize=fs)
                
                
                ax.grid()
                ax.legend(fontsize=fs-4)

                y1z=1e-3*o3_z_somora.sel(o3_p=500, tolerance=100,method='nearest')
                y2z=1e-3*o3_z_somora.sel(o3_p=2e-3, tolerance=1,method='nearest')
                ax2res = ax.twinx()
                #ax2.set_yticks(level2_data[spectro].isel(time=i).o3_z) #ax2.set_yticks(altitude)
                ax2res.set_ylim(y1z,y2z)
                fmt = FormatStrFormatter("%.0f")
                loc=MultipleLocator(base=10)
                ax2res.yaxis.set_major_formatter(fmt)
                ax2res.yaxis.set_major_locator(loc)
                ax2res.set_ylabel('Altitude [km] ')
                ax2res.set_xlabel('')
                ax2res.tick_params(axis='both', which='major')

            counter = 0
            color_count = 0
            for avk in avkm_somora:
                #if 0.8 <= np.sum(avk) <= 1.2:
                counter=counter+1
                if np.mod(counter,8)==0:
                    axs[1].plot(avk, p, color=cmap(color_count*0.25+0.01), lw=2)
                    color_count = color_count +1            
                else:
                    axs[1].plot(avk, p, color='k')
            axs[1].plot(0.25*mean_MR_somora.data,mean_MR_gromos.o3_p, color=get_color('SOMORA'))
            axs3.plot(mean_MR_somora.data,mean_MR_somora.o3_p, color=get_color('SOMORA'), label='SOMORA')
            axs[1].fill_betweenx(mean_MR_somora.o3_p.data,0.25*mean_MR_somora.data-0.5*somora_mr.std(dim='time').data,0.25*mean_MR_somora.data+0.5*somora_mr.std(dim='time').data, color=get_color('SOMORA'), alpha=0.2)
            axs3.fill_betweenx(mean_MR_somora.o3_p.data,mean_MR_somora.data-0.5*somora_mr.std(dim='time').data,mean_MR_somora.data+0.5*somora_mr.std(dim='time').data, color=get_color('SOMORA'), alpha=0.2)

            #axs[1].axhline(y=good_p_somora[0], ls='--', color='red')
            #axs[1].axhline(y=good_p_somora[-1], ls='--', color='red')
            axs[1].set_title('SOMORA: '+s)
            axs[0].invert_yaxis()
            axs[0].set_yscale('log')
            axs[1].set_ylabel('')

            axs3.set_yscale('log')
            axs3.invert_yaxis()
            axs3.set_ylim(500, 2e-3)
            axs3.set_xlim(0, 1.4)
            ax.xaxis.set_major_locator(MultipleLocator(0.25))
            ax.xaxis.set_minor_locator(MultipleLocator(0.125))
            axs3.yaxis.set_major_formatter(FuncFormatter(lambda y, _: '{:g}'.format(y)))
            axs3.set_ylabel('Pressure [hPa]', fontsize=fs)
            axs3.set_xlabel(r"MR [\%]", fontsize=fs)
            axs3.set_title('Measurement response: '+s, fontsize=fs+4)
            axs3.grid()
            axs3.legend(fontsize=fs-4)

            ax2 = axs[1].twinx()
            #ax2.set_yticks(level2_data[spectro].isel(time=i).o3_z) #ax2.set_yticks(altitude)
            ax2.set_ylim(y1z,y2z)
            fmt = FormatStrFormatter("%.0f")
            loc=MultipleLocator(base=10)
            ax2.yaxis.set_major_formatter(fmt)
            ax2.yaxis.set_major_locator(loc)
            ax2.set_ylabel('Altitude [km] ')
            ax2.set_xlabel('')
            ax2.tick_params(axis='both', which='major')
        
            # fig.text(
            #     0.5,
            #     0.02,
            #     season_id[j],
            #     verticalalignment="bottom",
            #     horizontalalignment="center",
            #     )

            for ax in axs:
                ax.set_xlabel('AVK')
                ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: '{:g}'.format(y)))
                ax.set_xlim(-0.05, 0.35)
                ax.set_ylim(500, 2e-3)
                ax.xaxis.set_major_locator(MultipleLocator(0.25))
                ax.xaxis.set_minor_locator(MultipleLocator(0.125))
                ax.grid(axis='x', which='both')
                ax.grid(axis='y', which='major')

            fig.tight_layout(rect=[0, 0.01, 0.99, 1])
            fig2.tight_layout(rect=[0, 0.01, 0.99, 1])
            fig3.tight_layout(rect=[0, 0.01, 0.99, 1])
            # fig4.tight_layout(rect=[0, 0.01, 0.99, 1])
            figures.append(fig)
            figures2.append(fig2)
            figures3.append(fig3)
            figures4.append(fig4)
            
        save_single_pdf(outfolder+'AVKs_seasonal_comparison_'+str(gromos.time.data[0])[0:10]+'.pdf', figures)
        save_single_pdf(outfolder+'diagnostics_seasonal_comparison_'+str(gromos.time.data[0])[0:10]+'.pdf', figures2)
        save_single_pdf(outfolder+'mc_seasonal_comparison_'+str(gromos.time.data[0])[0:10]+'.pdf', figures3)
        save_single_pdf(outfolder+'errors_seasonal_comparison_'+str(gromos.time.data[0])[0:10]+'.pdf', figures4)


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

def compare_opacity(gromos, somora, freq='6H', tc = False, outfolder='/scratch/GROSOM/Level2/opacities/', date_slice = slice('2010-01-04','2020-12-31')):
    #gromos_opacity, somora_opacity = read_opacity(folder, year=year)
    gromos = gromos.sel(time=date_slice)
    somora = somora.sel(time=date_slice)
    datestr = str(gromos.time[0].data)

    fig, axs = plt.subplots(3, 1, sharex=True, figsize=(15,10))
    gromos.tropospheric_opacity.resample(time=freq).mean().plot(
        ax=axs[0]
    )
    somora.tropospheric_opacity.resample(time=freq).mean().plot(
        ax=axs[0]
    )
    if tc:
        gromos.tropospheric_opacity_tc.resample(time=freq).mean().plot(
            lw=0.5,
            marker='.',
            ms=0.5,
            ax=axs[0]
        )
        somora.tropospheric_opacity_tc.resample(time=freq).mean().plot(
            lw=0.5,
            marker='.',
            ms=0.5,
            ax=axs[0]
        )
    axs[0].set_ylabel('opacity')
    axs[0].set_ylim((0,2))
    axs[0].legend(['GROMOS','SOMORA','GROMOS TC', 'SOMORA_TC'])
    gromos.tropospheric_transmittance.resample(time=freq).mean().plot(
        ax=axs[1]
    )
    somora.tropospheric_transmittance.resample(time=freq).mean().plot(
        ax=axs[1]
    )
    axs[1].set_ylabel('transmittance')
    axs[1].set_ylim((-0.01,1))

    rel_diff = (gromos.tropospheric_opacity.resample(time=freq).mean() - somora.tropospheric_opacity.resample(time=freq).mean()) 
    rel_diff.resample(time=freq).mean().plot(
        ax=axs[2]
    )
    axs[2].axhline(y=0, ls='--', lw=0.8 , color='k')
    axs[2].set_ylabel('opacity difference')
    axs[2].legend(['GRO - SOM'])
    axs[2].set_ylim((-1,1))

    for ax in axs:
        ax.grid()
    fig.tight_layout(rect=[0, 0.01, 0.95, 1])
    fig.savefig(outfolder+'opactiy_comparison_'+datestr+'.pdf', dpi=500)

    # Seasonal anaylsis:
    ds_gromora=xr.merge((
        {'gromos_opacity':gromos.tropospheric_opacity.resample(time=freq).mean()},
        {'somora_opacity':somora.tropospheric_opacity.resample(time=freq).mean()},
        ))
    # Clean the ds to remove any Nan entries:
    ds_gromora = ds_gromora.where(ds_gromora.gromos_opacity.notnull(), drop=True).where(ds_gromora.somora_opacity.notnull(), drop=True)

    #season = ['DJF','MAM', 'JJA', 'SON']
    color_season = ['r', 'b', 'y', 'g']
    marker_season = ['s', 'o', 'D', 'X']
    fill_styles=['none','none', 'full', 'full']
    ms = 9
    groups = ds_gromora.groupby('time.season').groups
    
    # pl = ds_gromora.plot.scatter(
    #     x='gromos_opacity', 
    #     y='somora_opacity',
    #     hue='gromos_opacity',
    #     hue_style='continuous',
    #     vmin=0,
    #     vmax=2,
    #     marker='.',
    # )
    # plt.show()
    
    fig, axs = plt.subplots(2, 2, sharex=True, figsize=(15,15))
    counter = 0
    fs=22
    for j, s in enumerate(groups):
        print("#################################################################################################################### ")
        print("#################################################################################################################### ")
        print('Processing season ', s)
        ds = ds_gromora.isel(time=groups[s])#.interpolate_na(dim='time',fill_value="extrapolate")

        x = ds.gromos_opacity
        y = ds.somora_opacity

        pearson_corr = xr.corr(x,y, dim='time')
        print('Pearson corr coef: ',pearson_corr.values)

        #  Fit using Orthogonal distance regression
        #  uses the retrievals total errors 
        print('Orthogonal distance regression:')

        # result = regression_xy(
        #     x.values, y.values
        # )
        # error_odr = result.beta[0]*x.values + result.beta[1] - y.values
        # SE_odr = np.square(error_odr)
        # MSE_odr = np.mean(SE_odr)
        
        df = len(x) - 2
        tinv = lambda p, df: abs(stats.t.ppf(p/2, df))
        ts = tinv(0.05, df)
        
        # coeff_determination = 1 - (np.var(error_odr)/np.var(y.values))
        #coeff_determination = calcR2_wikipedia(y.values, result.beta[1] + result.beta[0] * x.values)

        # print('Slope ODR (95%) ', result.beta[0], ' +- ', result.sd_beta[0]*ts)
        # print('Intercept ODR (95%) ', result.beta[1], ' +- ', result.sd_beta[1]*ts )
        # print('R2 odr ', coeff_determination)
        # print('RMSE ', np.sqrt(MSE_odr))
        # print("########################################################## ")
        
        # Least square fit   
        print('Linear square fit:')
        result_stats = stats.linregress(x.values, y.values)
        
        print('Slope LS (95%) ', result_stats.slope, ' +- ', result_stats.stderr*ts)
        print('Intercept LS (95%) ', result_stats.intercept, ' +- ', result_stats.intercept_stderr*ts )
        print('r2: ',pearson_corr.values**2)
        print('R2 stats: ', result_stats.rvalue**2)
        plot_ax = axs[int(counter>1),np.mod(counter,2)]
        ds.plot.scatter(
            ax=plot_ax,
            x='gromos_opacity', 
            y='somora_opacity',
            vmin=0,
            vmax=3,
            marker='.',
            color='k'
        )
        counter += 1
        #plot_ax.plot([np.nanmin(ds.gromos_opacity.values),np.nanmax(ds.gromos_opacity.values)],[np.nanmin(ds.gromos_opacity.values), np.nanmax(ds.gromos_opacity.values)],'k--', lw=0.8)
        plot_ax.plot([0,3],[0,3],'k--', lw=0.8)
        plot_ax.plot(np.arange(0,3), result_stats.slope*np.arange(0,3)+ result_stats.intercept, color=color_gromos) 
        plot_ax.set_xlabel(r'GROMOS $\tau$ [Np]', fontsize=fs)
        plot_ax.set_ylabel(r'SOMORA $\tau$ [Np]', fontsize=fs)
        plot_ax.set_title(r'Opacity, '+s, fontsize=fs) #str(p_min[i])+ ' hPa < p < '+str(p_max[i])+' hPa')
        plot_ax.set_xlim(0,3)
        plot_ax.set_ylim(0,3)

        plot_ax.xaxis.set_major_locator(MultipleLocator(1))
        plot_ax.xaxis.set_minor_locator(MultipleLocator(0.5))
        plot_ax.yaxis.set_major_locator(MultipleLocator(1))
        plot_ax.yaxis.set_minor_locator(MultipleLocator(0.5))
        plot_ax.tick_params(axis='both', which='major', labelsize=fs)
        if result_stats.intercept < 0:
            sign = '-' 
        else: 
            sign= '+'
        plot_ax.text(
            0.505,
            0.02,
            ' R$^2$ = {:.2f} \n y = {:.2f}x '.format(result_stats.rvalue**2, result_stats.slope)+sign+' {:.2f}'.format(np.abs(result_stats.intercept)),
           # '$p={:.1f}$ hPa \n$R^2 = {:.3f}$, \n$m ={:.2f}$'.format(gromos.o3_p.data[p], coeff_determination, result.beta[0]),                    
            transform=plot_ax.transAxes,
            verticalalignment="bottom",
            horizontalalignment="left",
            fontsize=fs
            )

    fig.tight_layout(rect=[0, 0.01, 0.95, 1])
    fig.savefig(outfolder+'opactiy_correlation_'+datestr+'.pdf', dpi=500)

def daily_median_save(gromos_clean, outfolder):
    fs = 22
    daily_median = gromos_clean.resample(time='D').median()

    fig, axs = plt.subplots(2, 1, sharex=True, sharey=True, figsize=(15,10))

    gromos_clean.o3_x.resample(time='12H').mean().plot( #.isel(o3_p=10)
        x='time',
        y='o3_p',
        ax=axs[0], 
        yscale='log',
        vmin=0,
        vmax=10,
        linewidth=0,
        rasterized=True,
        cmap=cmap_ts
    )
    daily_median.o3_x.plot(
        x='time',
        y='o3_p',
        ax=axs[1], 
        vmin=0,
        vmax=10,
        linewidth=0,
        rasterized=True,
        cmap=cmap_ts
    )
    axs[0].set_title(r'GROMOS OG', fontsize=fs+2)
    axs[1].set_title(r'Daily median', fontsize=fs+2)

    axs[0].invert_yaxis()
    for ax in axs:
        ax.set_ylabel('Pressure hPa')
        ax.set_xlabel('')

    fig.savefig(outfolder+'GROMOS_2015-2021_dailyMedian.pdf')
    daily_median.to_netcdf(outfolder+'GROMOS_2015-2021_dailyMedian.nc')

def plot_figures_gromora_paper(do_sensitivity = True, do_L2=True, do_comp=True, do_old=True):
    '''
    Just a convenient function which repoduces figures 2 to B2 from the paper.
    '''
    outfolder = '/scratch/GROSOM/Level2/GROMORA_paper_plots/'
    ############################################################
    # Sensitivity figures:
    if do_sensitivity:
        from sensitivity_analysis import plot_sensi_fig_gromora_paper
        # Fig. 5 & A1
        plot_sensi_fig_gromora_paper('GROMOS')
        # Fig. 6 & A2
        plot_sensi_fig_gromora_paper('SOMORA')

    ############################################################
    # Diagnostics figures:
    if do_L2:
        from plot_L2_paper import plot_L2
        # Fig. 2a & 3
        plot_L2('GROMOS', date = [datetime.date(2017,1,9)], cycles=[14])

        # Fig. 2b and 4
        plot_L2('SOMORA', date = [datetime.date(2017,1,9)], cycles=[13])

    ############################################################
    ############################################################
    # Full time series figures:
    if do_comp:
        date_slice=slice('2009-07-01','2021-12-31')
    
        gromos = read_GROMORA_concatenated('/scratch/GROSOM/Level2/GROMOS_level3_6H_v2.nc', date_slice)
        somora = read_GROMORA_concatenated('/scratch/GROSOM/Level2/SOMORA_level3_6H_v2.nc', date_slice)
        
        gromos_clean = gromos.where(gromos.level2_flag==0, drop=True)
        somora_clean = somora.where(somora.level2_flag==0, drop=True)
    
        # Fig. 7:
        compare_ts_gromora(gromos_clean, somora_clean, date_slice=date_slice, freq='7D', basefolder=outfolder, paper=True)
    
        # Fig. 8 & 9:
        #level1b_somora, somora_flags_level1a, somora_flags_level1b = read_level1('/storage/tub/instruments/somora/level1/v2','SOMORA', dateslice=slice('2009-01-01', '2021-12-31'))
        #somora_clean['opacity'] = ('time',level1b_somora.tropospheric_opacity.reindex_like(somora_clean, method='nearest', tolerance='1H'))
        compute_seasonal_correlation_paper(gromos_clean, somora_clean, freq='6H', p_min=[0.1, 1, 10] , p_max=[0.9, 5, 50], basefolder=outfolder)    
        
        # Fig. 12:
        sbuv = read_SBUV_dailyMean(date_slice, SBUV_basename = '/storage/tub/atmosphere/SBUV/O3/daily_mean_overpasses/', specific_fname='sbuv_v87.mod_v2r1.vmr.payerne_156.txt')
        mls= read_MLS(date_slice, vers=5, filename_MLS='AuraMLS_L2GP-O3_v5_400-800_BERN.nc')
        
        compare_pressure_mls_sbuv_paper(gromos_clean, somora_clean, mls, sbuv,  p_min=[0.1, 1, 10] , p_max=[0.9, 5, 50], add_sun=False, freq='2D', basefolder=outfolder)
    
        # Fig 13, 14, B1 & B2
        gromos_colloc, gromos_conv, mls_gromos_colloc, mls_gromos_conv, somora_colloc, somora_conv, mls_somora_colloc, mls_somora_conv = read_all_MLS(yrs = years)
        compare_seasonal_GROMORA_MLS_profiles(gromos_colloc, gromos_conv, mls_gromos_colloc, mls_gromos_conv, somora_colloc, somora_conv, mls_somora_colloc, mls_somora_conv, mls, sbuv, p_min=[0.1, 1, 10] , p_max=[0.9, 5, 50], basefolder=outfolder, convolved=False, split_night=False)
        
        # And data for table 5:
        compare_GROMORA_MLS_profiles_egu(gromos_colloc, gromos_conv, mls_gromos_colloc, mls_gromos_conv, somora_colloc, somora_conv, mls_somora_colloc, mls_somora_conv, mls, sbuv, p_min=[0.1, 1, 10] , p_max=[0.9, 5, 50], basefolder=outfolder, convolved=False, split_night=False)
    
    ############################################################
    ############################################################
    # For comparisons with old datasets: 
    if do_old:
        date_slice=slice('2010-01-04','2020-12-31')
        sbuv = read_SBUV_dailyMean(date_slice, SBUV_basename = '/storage/tub/atmosphere/SBUV/O3/daily_mean_overpasses/', specific_fname='sbuv_v87.mod_v2r1.vmr.payerne_156.txt')
        mls= read_MLS(date_slice, vers=5, filename_MLS='AuraMLS_L2GP-O3_v5_400-800_BERN.nc')

        gromos = read_GROMORA_concatenated('/scratch/GROSOM/Level2/GROMOS_level3_6H_v2.nc', date_slice)
        somora = read_GROMORA_concatenated('/scratch/GROSOM/Level2/SOMORA_level3_6H_v2.nc', date_slice)
        
        gromos_clean = gromos.where(gromos.level2_flag==0, drop=True)
        somora_clean = somora.where(somora.level2_flag==0, drop=True)
    
        # Fig. 10:
        gromos_v2021 = read_gromos_v2021('gromosplot_ffts_select_v2021', date_slice)
        somora_old = read_old_SOMORA('/scratch/GROSOM/Level2/SOMORA_old_all.nc', date_slice)
        compare_old_new(gromos_clean, somora_clean, gromos_v2021, somora_old, mls, freq='7D', outfolder=outfolder)
        
        # Fig. 11:
        trends_diff_old_new(gromos_clean, somora_clean, gromos_v2021, somora_old, mls, p_min=[0.1, 1, 10] , p_max=[0.9, 5, 50], freq='7D', freq_avg='1M',  outfolder=outfolder)

#########################################################################################################
# Main function
#########################################################################################################
if __name__ == "__main__":
    yr = 2021
    #date_slice=slice(str(yr)+'-01-01',str(yr)+'-12-31')
    # The full range:
    date_slice=slice('2010-01-01','2021-12-31')
    
    # The GROMOS full series:
    #date_slice=slice('2009-07-01','2021-12-31')
    
    # For comparisons with old retrievals (no old gromos data after)
    #date_slice=slice('2010-01-04','2021-04-30')

    #date_slice=slice('2010-01-04','2020-12-31')

    # Date range for tests:
    #date_slice=slice('2018-01-01','2018-12-31')

    years = [2010,2011] #[2009, 2010, 2011, 2012,2013,2014,2015,2016,2017,2018,2019,2020, 2021]
    
    instNameGROMOS = 'GROMOS'
    instNameSOMORA = 'SOMORA'

    # By default, we use the latest version with L2 flags
    v2 = True
    flagged_L2 = True
    if v2:
        fold_somora = '/scratch/GROSOM/Level2/SOMORA/v2/'
        fold_gromos = '/scratch/GROSOM/Level2/GROMOS/v2/'
        prefix_all='.nc'
    else:
        fold_somora = '/storage/tub/instruments/somora/level2/v1/'
        fold_gromos = '/storage/tub/instruments/gromos/level2/GROMORA/v1/'
        prefix_all='_waccm_low_alt_ozone.nc'

    ########################################################################################################
    # Different strategies can be chosen for the analysis:
    # 'read': default option which reads the full level 2 doing the desired analysis
    # 'read_save': To save new level 3 data from the full hourly level 2
    # 'plot_all': the option to reproduce the figures from the manuscript
    # 'anything else': option to read the level 3 data before doing the desired analysis

    strategy = 'plot'
    if strategy[0:4]=='read':
        read_gromos=False
        read_somora=False
        read_both=False

        if len(years)>4 and read_both:
            raise ValueError('This will take too much space sorry !')

        if read_gromos or read_both:
            gromos = read_GROMORA_all(
                basefolder=fold_gromos, 
                instrument_name=instNameGROMOS,
                date_slice=date_slice, 
                years=years,#
                prefix=prefix_all,
                flagged=flagged_L2,
            )
            gromos['o3_x'] = 1e6*gromos['o3_x'].where((gromos['o3_x']>gromos['o3_x'].valid_min)&(gromos['o3_x']<gromos['o3_x'].valid_max), drop = True)
            gromos_clean = gromos.where(gromos.retrieval_quality==1, drop=True)#.where(gromos.level2_flag==0, drop=True)#.where(gromos.o3_mr>0.8)
            print('GROMOS good quality level2: ', 100*len(gromos_clean.time)/len(pd.date_range('2009-07-01', '2021-12-31 23:00:00', freq='1H')) )
        
        if read_somora or read_both:
            somora = read_GROMORA_all(
                basefolder=fold_somora, 
                instrument_name=instNameSOMORA,
                date_slice=date_slice, 
                years=years, #[2010, 2011, 2012,2013,2014,2015,2016,2017,2018,2019,2020]
                prefix=prefix_all,  # '_v2_all.nc'#
                flagged=flagged_L2,
            )
            somora['o3_x'] = 1e6*somora['o3_x'].where((somora['o3_x']>somora['o3_x'].valid_min)&(somora['o3_x']<somora['o3_x'].valid_max), drop = True)
            somora_clean = somora.where(somora.retrieval_quality==1, drop=True)#.where(somora.level2_flag==0, drop=True)#.where(gromos.o3_mr>0.8)
            print('SOMORA good quality level2: ', 100*len(somora_clean.time)/len(pd.date_range('2009-09-23', '2021-12-31 23:00:00', freq='1H')) )
            
        if not flagged_L2:
            gromos = add_flags_level2_gromora(gromos, 'GROMOS')
            somora = add_flags_level2_gromora(somora, 'SOMORA')

        #print('GROMOS good quality level2: ', 100*len(gromos_clean.time)/len(pd.date_range('2020-01-01', '2020-12-31 23:00:00', freq='1H')) )
        #print('SOMORA good quality level2: ', 100*len(somora_clean.time)/len(pd.date_range('2020-01-01', '2020-12-31 23:00:00', freq='1H')) )
        if strategy=='read_save':
            # Saving the level 3:
            if read_gromos:
                gromos_clean.resample(time='6H').mean().to_netcdf('/scratch/GROSOM/Level2/GROMOS_level3_6H_v2.nc')
            elif read_somora:
                somora_clean.resample(time='6H').mean().to_netcdf('/scratch/GROSOM/Level2/SOMORA_level3_6H_v2.nc')

    elif strategy=='plot_all':
        plot_figures_gromora_paper(do_sensitivity = False, do_L2=True, do_comp=False, do_old=False)
        exit()
    else:
        gromos = read_GROMORA_concatenated('/scratch/GROSOM/Level2/GROMOS_level3_6H_v2.nc', date_slice)
        somora = read_GROMORA_concatenated('/scratch/GROSOM/Level2/SOMORA_level3_6H_v2.nc', date_slice)
        
        gromos_clean = gromos.where(gromos.level2_flag==0, drop=True)#.where(gromos.o3_mr>0.8)
        somora_clean = somora.where(somora.level2_flag==0, drop=True)#.where(gromos.o3_mr>0.8)
        
        print('GROMOS good quality level2: ', 100*len(gromos_clean.time)/len(pd.date_range('2009-07-01', '2021-12-31 23:00:00', freq='6H')) )
        print('SOMORA good quality level2: ', 100*len(somora_clean.time)/len(pd.date_range('2009-09-23', '2021-12-31 23:00:00', freq='6H')) )

    #####################################################################
    # Read SBUV and MLS
    bn = '/storage/tub/atmosphere/SBUV/O3/daily_mean_overpasses/'
    sbuv = read_SBUV_dailyMean(date_slice, SBUV_basename = bn, specific_fname='sbuv_v87.mod_v2r1.vmr.payerne_156.txt')
    sbuv_arosa = read_SBUV_dailyMean(date_slice, SBUV_basename = bn, specific_fname='sbuv_v87.mod_v2r1.vmr.arosa_035.txt')

    mls= read_MLS(date_slice, vers=5, filename_MLS='AuraMLS_L2GP-O3_v5_400-800_BERN.nc')
    outfolder = '/scratch/GROSOM/Level2/GROMORA_retrievals_v2/'

    #####################################################################
    # Reading the old gromora datasets
    #gromora_old = read_old_GROMOA_diff('DIFF_G_2017', date_slice)
    gromos_v2021 = read_gromos_v2021('gromosplot_ffts_select_v2021', date_slice)
    somora_old = read_old_SOMORA('/scratch/GROSOM/Level2/SOMORA_old_all.nc', date_slice)
    #plot_ozone_ts(gromos, instrument_name='GROMOS', freq='1H', altitude=False, basefolder=outfolder )

    #gromos_opacity, somora_opacity = read_opacity(folder='/scratch/GROSOM/Level2/opacities/', year=yr)

   #  plot_ozone_flags('SOMORA', somora, flags1a=flags1a, flags1b=flags1b, pressure_level=[27, 12], opacity = somora_opacity, calib_version=1)
    # plot_ozone_flags('GROMOS', gromos, flags1a=flags1a, flags1b=flags1b, pressure_level=[27, 12], opacity = gromos_opacity, calib_version=1)

    #####################################################################
    # Time series 2D plots
    plot_2D = False
    if plot_2D:
        #daily_median_save(gromos_clean, outfolder)
        #compare_ts_MLS(gromos, somora, date_slice=date_slice, freq='7D', basefolder='/scratch/GROSOM/Level2/GROMORA_retrievals_v2/', ds_mls=mls, sbuv=None)
        #compare_pressure_mls_sbuv(gromos_clean, somora_clean, mls, sbuv, pressure_level=[29, 25, 21, 18, 15], add_sun=False, freq='6H', basefolder=outfolder)
        compare_pressure_mls_sbuv_paper(gromos_clean, somora_clean, mls, sbuv,  p_min=[0.1, 1, 10] , p_max=[0.9, 5, 50], add_sun=False, freq='2D', basefolder=outfolder)
        #compare_ts(gromos, somora, freq='7D', date_slice=date_slice, basefolder=outfolder)

    #####################################################################
    # Comparison with old retrievals
    plot_old_new = False
    if plot_old_new:
        compare_old_new(gromos_clean, somora_clean, gromos_v2021, somora_old, mls, freq='7D', outfolder=outfolder)
        #gromos_old_vs_new(gromos_clean, gromos_v2021, mls, seasonal=False)
        #gromos_old_vs_new(somora_clean, somora_old, mls, seasonal=False)
    
        trends_diff_old_new(gromos_clean, somora_clean, gromos_v2021, somora_old, mls, p_min=[0.1, 1, 10] , p_max=[0.9, 5, 50], freq='7D', freq_avg='1M',  outfolder=outfolder)
        #trends_simplified_new(gromos_clean, somora_clean, mls, p_min=[0.1, 1, 10] , p_max=[0.9, 5, 50], freq='7D', freq_avg='1M',  outfolder=outfolder)


    #gromos = utc_to_lst(gromos)
    #somora = utc_to_lst(somora)

    #compute_compare_climatology(somora, slice_clim=slice("2010-01-01", "2020-12-31"), slice_plot=slice("2022-01-01", "2022-01-31"), percentile=[0.1, 0.9], pressure_level = [25, 21, 15], basefolder=outfolder)

    #compare_pressure(gromos_clean, somora_clean, pressure_level=[31, 25, 21, 15, 12], add_sun=False, freq='6H', basefolder='/scratch/GROSOM/Level2/GROMORA_retrievals_v2/')

    #####################################################################
    # Relative difference GROMOS vs SOMORA
    compare_gromora = False
    if compare_gromora:
        #map_rel_diff(gromos_clean, somora_clean, freq='6H', basefolder=outfolder)
        compare_ts_gromora(gromos_clean, somora_clean, date_slice=date_slice, freq='7D', basefolder=outfolder, paper=True)

        #compute_corr_profile(somora_sel,mls_somora_colloc,freq='7D',basefolder='/scratch/GROSOM/Level2/MLS/')
        # #compare_diff_daily(gromos ,somora, gromora_old, pressure_level=[34 ,31, 25, 21, 15, 12], altitudes=[69, 63, 51, 42, 30, 24])
        # compare_mean_diff(gromos, somora, sbuv = None, basefolder=outfolder)

        compare_mean_diff_monthly(gromos_clean, somora_clean, mls, sbuv, outfolder=outfolder)

    # gromos_linear_fit = gromos_clean.o3_x.where((gromos_clean.o3_p<p_high) & (gromos_clean.o3_p>p_low), drop=True).mean(dim='o3_p').resample(time='1M').mean()#) .polyfit(dim='time', deg=1)
    # somora_linear_fit = somora_clean.o3_x.resample(time='1M').mean().polyfit(dim='time', deg=1)

    #####################################################################
    # Averaging kernels
    plot_avk = False
    if plot_avk:
        #compare_avkm(gromos, somora, date_slice, outfolder)
        gromos_clean=gromos_clean.where(gromos_clean.o3_avkm.mean(dim='o3_p_avk')<10, drop=True).where(gromos_clean.o3_avkm.mean(dim='o3_p_avk')>-10, drop=True)
        somora_clean=somora_clean.where(somora_clean.o3_avkm.mean(dim='o3_p_avk')<10, drop=True).where(somora_clean.o3_avkm.mean(dim='o3_p_avk')>-10, drop=True)
        compare_avkm(gromos_clean, somora_clean, date_slice, outfolder, seasonal=True)

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
        compute_seasonal_correlation_paper(gromos_clean, somora_clean, freq='6H', p_min=[0.1, 1, 10] , p_max=[0.9, 5, 50], basefolder=outfolder)

        #compute_correlation(gromos_clean, somora_clean, freq='6H', pressure_level=[36, 31, 21, 12], basefolder=outfolder)
    
        #compute_corr_profile(gromos_clean, somora_clean, freq='6H', date_slice=slice('2009-10-01','2021-12-31'), basefolder=outfolder)

    #####################################################################
    # Opacity GROMOS-SOMORA
    opacity = True
    if opacity:
        compare_opacity(level1b_gromos, level1b_somora, freq = '7D', tc = False, date_slice=date_slice)