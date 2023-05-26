#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on 24.02.23

@author: Eric Sauvageat

Collection of function to plot figure for the Thesis
"""
#%%
import datetime
from datetime import date
from operator import truediv
import os, pickle

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
from tempera import read_tempera_level3, read_tempera_level2

from matplotlib.ticker import (MultipleLocator, FuncFormatter, FormatStrFormatter, AutoMinorLocator)
from matplotlib.lines import Line2D
import matplotlib.dates as mdates
import matplotlib.ticker as ticker
import matplotlib
cmap = matplotlib.cm.get_cmap('plasma')

cmap_ts = plt.get_cmap('density') # 'cividis' # plt.get_cmap('density') #'cividis' #
black_cmap = plt.get_cmap('plasma')
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

F0=142.17504e9

def plot_vertical_structure(gromos_clean, basefolder='/home/esauvageat/Documents/Thesis/ThesisES/Figures/'):
    fs = 32
    lw=4
    year = pd.to_datetime(gromos_clean.time.values[0]).year
    
    fig, axs = plt.subplots(1, 2, sharex=False, sharey=True, figsize=(20,16))

    temperature_seasonal = gromos_clean.temperature_profile.groupby('time.season').mean()
    temperature_seasonal.sel(season='DJF').plot(
        ax=axs[0],
        y='o3_p',
        yscale='log',
        color=color_somora,
        linewidth=lw,
        label='Winter'
    )
    temperature_seasonal.sel(season='JJA').plot(
        ax=axs[0],
        y='o3_p',
        yscale='log',
        color=color_gromos,
        linewidth=lw,
        label='Summer'
    )
    ozone_seasonal = gromos_clean.o3_x.groupby('time.season').mean()
    ozone_seasonal.sel(season='DJF').plot(
        ax=axs[1],
        y='o3_p',
        yscale='log',
        color=color_somora,
        linewidth=lw,
        label='Winter'
    )
    ozone_seasonal.sel(season='JJA').plot(
        ax=axs[1],
        y='o3_p',
        yscale='log',
        color=color_gromos,
        linewidth=lw,
        label='Summer'
    )

    y1z=1e-3*gromos_clean.o3_z.sel(o3_p=870, tolerance=100,method='nearest').mean(dim='time')
    y2z=1e-3*gromos_clean.o3_z.sel(o3_p=0.0001, tolerance=1,method='nearest').mean(dim='time')
    ax2 = axs[1].twinx()
    ax2.set_yticks(gromos_clean.o3_z.mean(dim='time')) #ax2.set_yticks(altitude)
    ax2.set_ylim(y1z,y2z)
    fmt = FormatStrFormatter("%.0f")
    loc=MultipleLocator(base=10)
    ax2.yaxis.set_major_formatter(fmt)
    ax2.yaxis.set_major_locator(loc)
    ax2.set_ylabel('Altitude [km] ', fontsize=fs)
    ax2.tick_params(axis='both', which='major', labelsize=fs)

    axs[0].set_title('Temperature', fontsize=fs+4) 
    
    # ax.set_yscale('log')
    axs[0].invert_yaxis()
    axs[0].set_ylabel('Pressure [hPa]', fontsize=fs)
    axs[0].set_xlabel('T [K]', fontsize=fs)
    axs[0].xaxis.set_major_locator(MultipleLocator(20))
    axs[0].xaxis.set_minor_locator(MultipleLocator(10))
    axs[0].set_xlim(180,280)
    axs[0].text(
    0.04,
    0.015,
    r'\textbf{a)}',
    transform=axs[0].transAxes,
    verticalalignment="bottom",
    horizontalalignment="left",
    fontsize=fs+2
    )

    axs[0].text(
    0.64,
    0.13,
    r'\textbf{Tropopause}',
    transform=axs[0].transAxes,
    verticalalignment="bottom",
    horizontalalignment="left",
    fontsize=fs-2
    )

    axs[0].text(
    0.02,
    0.538,
    r'\textbf{Stratopause}',
    transform=axs[0].transAxes,
    verticalalignment="bottom",
    horizontalalignment="left",
    fontsize=fs-2
    )

    axs[0].text(
    0.65,
    0.95,
    r'\textbf{Mesopause}',
    transform=axs[0].transAxes,
    verticalalignment="bottom",
    horizontalalignment="left",
    fontsize=fs-2
    )
    axs[1].set_title('Ozone', fontsize=fs+4) 
    axs[1].set_xlabel(r'O$_3$ [ppmv]', fontsize=fs)
    axs[1].set_ylabel('', fontsize=fs)
    axs[1].xaxis.set_major_locator(MultipleLocator(2))
    axs[1].xaxis.set_minor_locator(MultipleLocator(1))
    axs[1].set_xlim(-0.1,10)

    axs[1].text(
    0.04,
    0.015,
    r'\textbf{b)}',
    transform=axs[1].transAxes,
    verticalalignment="bottom",
    horizontalalignment="left",
    fontsize=fs+2
    )

    axs[1].text(
    0.55,
    0.06,
    r'\textbf{Troposphere}',
    transform=axs[1].transAxes,
    verticalalignment="bottom",
    horizontalalignment="left",
    fontsize=fs-2
    )
    axs[1].text(
    0.02,
    0.33,
    r'\textbf{Stratopshere}',
    transform=axs[1].transAxes,
    verticalalignment="bottom",
    horizontalalignment="left",
    fontsize=fs-2
    )
    axs[1].text(
    0.55,
    0.85,
    r'\textbf{Mesosphere}',
    transform=axs[1].transAxes,
    verticalalignment="bottom",
    horizontalalignment="left",
    fontsize=fs-2
    )

    axs[0].axhline(y=0.009, xmin=0, xmax=0.62, color='k', linestyle='-.', linewidth=lw-2)
    axs[0].axhline(1.2 , xmin=0.4, xmax=1,  color='k', linestyle='-.', linewidth=lw-2)
    axs[0].axhline(150, xmin=0, xmax=0.62, color='k', linestyle='-.', linewidth=lw-2)

    axs[1].axhline(0.009, color='k', linestyle='-.', linewidth=lw-2)
    axs[1].axhline(1.2, color='k', linestyle='-.', linewidth=lw-2)
    axs[1].axhline(150, color='k', linestyle='-.', linewidth=lw-2)
    
    for ax in axs:
        ax.set_ylim(800, 6e-3)
        ax.grid(which='both')
        ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: '{:g}'.format(y)))
    
    axs[1].legend(fontsize=fs-2, loc='lower left', bbox_to_anchor=(0.56, 0.64))

    plt.tight_layout(rect=[0, 0.01, 0.92, 1])
    fig.savefig(basefolder+'vertical_structure.pdf', dpi=500)


def plot_jacobian(basefolder='/home/esauvageat/Documents/Thesis/ThesisES/Figures/'):
    fs = 32
    lw=4
    year = pd.to_datetime(gromos_clean.time.values[0]).year
    
    f = open('/home/esauvageat/Documents/GROMORA/Data/GROMOS_jacobian.pkl', 'rb')
    jacobian = pickle.load(f)
    f.close()
    f = open('/home/esauvageat/Documents/GROMORA/Data/SOMORA_jacobian.pkl', 'rb')
    jacobian_somora = pickle.load(f)
    f.close()

    altitude = gromos_clean.o3_z[0].values/1000
    pressure = gromos_clean.o3_p.values
    level2 = xr.open_dataset(
        '/home/esauvageat/Documents/GROMORA/Data/GROMOS_level2_AC240_2017_01_10_v2.nc',
        decode_times=True,
        decode_coords=True,
        # use_cftime=True,
    )
    level2_somora = xr.open_dataset(
        '/home/esauvageat/Documents/GROMORA/Data/SOMORA_level2_AC240_2017_01_10_v2.nc',
        decode_times=True,
        decode_coords=True,
        # use_cftime=True,
    )
    central_gromos = 13119 #13107
    central_somora = 8191
    x = np.arange(0, 20, 1)
    y = np.arange(0, 20, 2)
    channels = np.concatenate((central_gromos-2*y, central_gromos-2*x**3))
    channels_somora = np.concatenate((central_somora-y, central_somora-2*x**3))
    col = np.arange(0,len(channels))/len(channels)
    colsom = np.arange(0,len(channels_somora))/len(channels_somora)
    #channels_symetric = central_channel + np.arange(-100, 100, 1)*100
        
    frequency = (level2.f.values-F0)*1e-6
    frequency_somora = (level2_somora.f.values-F0)*1e-6
    jacobian_o3 = jacobian[:,0:47]*1e-6 # np.flip(jacobian[:,0:47])
    jacobian_o3_somora = jacobian_somora[:,0:47]*1e-6 # np.flip(jacobian[:,0:47])

    fig, axs = plt.subplots(1, 2, sharex=False, sharey=True, figsize=(20,14))
    for j,i in enumerate(channels):
        axs[0].plot(jacobian_o3[i,:], pressure, color=black_cmap(col[j]), label=f'{(-frequency[i]):.2f} MHz')
        
    for j,i in enumerate(channels_somora):
        axs[1].plot(jacobian_o3_somora[i,:], pressure, color=black_cmap(colsom[j]), label=f'{(-frequency_somora[i]):.2f} MHz')
    axs[0].set_title('GROMOS', fontsize=fs+4) 
    axs[1].set_title('SOMORA', fontsize=fs+4) 
    axs[0].set_yscale('log')
    axs[0].invert_yaxis()
    axs[0].set_ylabel('Altitude [km]', fontsize=fs)
    axs[0].set_ylabel('Pressure [hPa]', fontsize=fs)
    #axs[0]
    # axs[0].xaxis.set_major_locator(MultipleLocator(20))
    # axs[0].xaxis.set_minor_locator(MultipleLocator(10))
    #axs[0].set_xlim(180,280)
    # sm =  plt.cm.ScalarMappable(cmap=black_cmap)
    # fig.colorbar(sm)
    for ax in axs:
        #ax.legend(loc=1)
        ax.set_ylim(800, 1e-2)
        ax.set_xlim(-0.005, 0.175)
        ax.grid(which='both')
        ax.set_xlabel('Weighting functions [K/ppmv]', fontsize=fs)
        ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: '{:g}'.format(y)))
    
    #axs[1].legend(fontsize=fs-2, loc='lower left', bbox_to_anchor=(0.56, 0.64))

    plt.tight_layout(rect=[0, 0.01, 0.92, 1])
    fig.savefig(basefolder+'Jacobians.pdf', dpi=500)

    plot_altitudes = np.arange(8,47,2)

    fig, axs = plt.subplots(2, 2, sharex=False, sharey=True, figsize=(20,16))
    for ax in axs[0,:]:
        ax.set_prop_cycle('color', list(black_cmap(np.arange(0,len(plot_altitudes))/len(plot_altitudes))))
    for ax in axs[1,:]:
        ax.set_prop_cycle('color', list(black_cmap(np.arange(0,len(plot_altitudes))/len(plot_altitudes))))
    axs[0,0].plot(frequency, jacobian_o3[:,plot_altitudes])
    
    axs[1,0].plot(frequency, jacobian_o3[:,plot_altitudes])   
    axs[0,1].plot(frequency_somora, jacobian_o3_somora[:,plot_altitudes])
    axs[1,1].plot(frequency_somora, jacobian_o3_somora[:,plot_altitudes])
    axs[0,0].set_title('GROMOS', fontsize=fs+4) 
    axs[0,1].set_title('SOMORA', fontsize=fs+4) 
    # axs[0].set_yscale('log')
    # axs[0].invert_yaxis()
    axs[0,0].set_ylabel('Jacobian [K/ppmv]', fontsize=fs)
    axs[1,0].set_ylabel('Jacobian [K/ppmv]', fontsize=fs)
    # axs[0].set_ylabel('Pressure [hPa]', fontsize=fs)
    axs[1,0].set_xlabel('Frequency offset [MHz] ', fontsize=fs)
    axs[1,1].set_xlabel('Frequency offset [MHz] ', fontsize=fs)
    # axs[0].xaxis.set_major_locator(MultipleLocator(20))
    # axs[0].xaxis.set_minor_locator(MultipleLocator(10))
    # axs[0,1].legend((np.rint(altitude[np.arange(6,47,3)])).tolist(), bbox_to_anchor=(1.04,0) )
    #axes = fig.add_axes([0.5, 1, 0.5, 1])
    #axs[0,1].legend((np.rint(altitude[np.arange(24,47)])).tolist(), bbox_to_anchor=(0.5,0), ncol=4 )
    for ax in axs[0,:]:
        #ax.set_ylim(1013, 1e-2)
        ax.set_xlim(-50,50)
        ax.grid(which='both')
        ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: '{:g}'.format(y)))
    for ax in axs[1,:]:
        #ax.set_ylim(1013, 1e-2)
        ax.set_xlim(-5,5)
        ax.grid(which='both')
        ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: '{:g}'.format(y)))
    #axs[1].legend(fontsize=fs-2, loc='lower left', bbox_to_anchor=(0.56, 0.64))

    plt.tight_layout(rect=[0, 0.01, 0.92, 1])
    fig.savefig(basefolder+'Jacobians2.pdf', dpi=500)

def plot_contribution(basefolder='/home/esauvageat/Documents/Thesis/ThesisES/Figures/'):
    fs = 32
    lw=4
    year = pd.to_datetime(gromos_clean.time.values[0]).year
    
    f = open('/home/esauvageat/Documents/GROMORA/Data/GROMOS_gain.pkl', 'rb')
    gain = pickle.load(f)
    f.close()
    f = open('/home/esauvageat/Documents/GROMORA/Data/SOMORA_gain.pkl', 'rb')
    gain_somora = pickle.load(f)
    f.close()

    altitude = gromos_clean.o3_z[0].values/1000
    pressure = gromos_clean.o3_p.values
    level2 = xr.open_dataset(
        '/home/esauvageat/Documents/GROMORA/Data/GROMOS_level2_AC240_2017_01_10_v2.nc',
        decode_times=True,
        decode_coords=True,
        # use_cftime=True,
    )
    level2_somora = xr.open_dataset(
        '/home/esauvageat/Documents/GROMORA/Data/SOMORA_level2_AC240_2017_01_10_v2.nc',
        decode_times=True,
        decode_coords=True,
        # use_cftime=True,
    )
    # central_gromos = 13107
    # central_somora = 8192
    # x = np.arange(0, 8000, 10)
    # channels = central_gromos - x
    # channels_somora = central_somora - x
    
    central_gromos = 13119 #13107
    central_somora = 8191
    x = np.arange(0, 20, 1)
    y = np.arange(0, 20, 2)
    channels = np.concatenate((central_gromos-2*y, central_gromos-2*x**3))
    channels_somora = np.concatenate((central_somora-y, central_somora-2*x**3))
    col = np.arange(0,len(channels))/len(channels)
    colsom = np.arange(0,len(channels_somora))/len(channels_somora)

    col = channels/channels[0]

    #channels_symetric = central_channel + np.arange(-100, 100, 1)*100
        
    frequency = level2.f.values*1e-9
    frequency_somora = level2_somora.f.values*1e-9
    gain_o3 = gain[0:47,:]*1e6  # np.flip(jacobian[:,0:47])
    gain_o3_somora = gain_somora[0:47,:]*1e6 # np.flip(jacobian[:,0:47])

    fig, axs = plt.subplots(1, 2, sharex=False, sharey=True, figsize=(20,16))
    for j,i in enumerate(channels):
        axs[0].plot(gain_o3[:,i], pressure, color=black_cmap(col[j]), label=f'{(-frequency[i]):.2f} MHz')
        
    for i in channels_somora:
        axs[1].plot(gain_o3_somora[:,i], pressure, label=f'{(-frequency_somora[i]):.2f} MHz')
    axs[0].set_title('GROMOS', fontsize=fs+4) 
    axs[1].set_title('SOMORA', fontsize=fs+4) 
    axs[0].set_yscale('log')
    axs[0].invert_yaxis()
    axs[0].set_ylabel('Altitude [km]', fontsize=fs)
    axs[0].set_ylabel('Pressure [hPa]', fontsize=fs)
    axs[0].set_xlabel('Contribution [ppmv/K]', fontsize=fs)
    # axs[0].xaxis.set_major_locator(MultipleLocator(20))
    # axs[0].xaxis.set_minor_locator(MultipleLocator(10))
    axs[0].set_xlim(-0.8,0.8)
   
    for ax in axs:
        ax.set_ylim(1013, 1e-2)
        ax.grid(which='both')
        ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: '{:g}'.format(y)))
    
    #axs[1].legend(fontsize=fs-2, loc='lower left', bbox_to_anchor=(0.56, 0.64))

    plt.tight_layout(rect=[0, 0.01, 0.92, 1])
    fig.savefig(basefolder+'gain_matrix.pdf', dpi=500)

def doppler_broadening(f, T):
    R = 8.314 # J/mol/K
    M = 48e-3 # kg/mol
    c = 2.99792458e8 # m/s
    gammaD = F0/c * np.sqrt(2*R*T/M)
    fD = (1/gammaD*np.sqrt(np.pi))*np.exp(-((f-F0)/gammaD)**2 )
    hwhm = gammaD*np.sqrt(np.log(2))
    return fD, hwhm

def pressure_broadening(f, T, p, o3_vmr):
    p0 = 1013.25
    T0 = 296
    pO3 = o3_vmr*p
    gamma_self = 3.17471e6 # for ozone in Hz per hPa
    gamma_air = 2.38473e6 # for air in Hz per hPa
    kappa = 0.76

    #gammaL = gamma_air * p/p0 * (T/T0)**kappa
    gammaL = (gamma_air*(p-pO3)+gamma_self*pO3)*(T0/T)**kappa
    gammaL = gamma_air*p*(T0/T)**kappa

    fL = (1/gammaL*np.pi)*gammaL / ((f-F0)**2 + gammaL**2 )
    return fL, gammaL

#########################################################################################################
# Main function
#########################################################################################################
if __name__ == "__main__":
    yr = 2010
    # The full range:
    date_slice=slice('2014-01-01','2017-12-31')

    # date_slice=slice('2016-02-01','2016-10-31')

    years = [2014, 2015, 2016, 2017] #[2014, 2015, 2016, 2017]
    
    instNameGROMOS = 'GROMOS'

    # By default, we use the latest version with L2 flags
    v2 = True
    flagged_L2 = False
    
    fold_gromos = '/storage/tub/instruments/gromos/level2/GROMORA/v3/'# #'/scratch/GROSOM/Level2/GROMOS/v2/'
    fold_gromos2 = '/storage/tub/instruments/gromos/level2/GROMORA/v3/' # '/scratch/GROSOM/Level2/GROMOS/v3/'
    prefix_FFT='_AC240_v3'
    basefolder='/home/esauvageat/Documents/Thesis/ThesisES/Figures/'
    ########################################################################################################
    # Different strategies can be chosen for the analysis:
    # 'read': default option which reads the full level 2 doing the desired analysis
    # 'read_save': To save new level 3 data from the full hourly level 2
    # 'plot_all': the option to reproduce the figures from the manuscript
    # 'anything else': option to read the level 3 data before doing the desired analysis

    strategy = 'pt'
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
    else:
        gromos_clean = read_GROMORA_concatenated('/scratch/GROSOM/Level2/GROMOS_level3_6H_v3.nc', date_slice)
        somora_clean = read_GROMORA_concatenated('/scratch/GROSOM/Level2/SOMORA_level3_6H_v2.nc', date_slice)


    plot_vertical_struct=False
    if plot_vertical_struct:
        plot_vertical_structure(gromos_clean)

    plot_jac=False
    if plot_jac:
        plot_jacobian()
        plot_contribution()

    plot_line_shape=True
    if plot_line_shape:
        fs = 18
        lw=2
        t=2200
        temperature = gromos_clean.temperature_profile.isel(time=t).values
        pressure = gromos_clean.o3_p.values
        o3_vmr = 1e-6*gromos_clean.o3_x.isel(time=t).values
        fD, hwhm_doppler = doppler_broadening(142e9, temperature)
        fL, hwhm_p = pressure_broadening(142e9, temperature, pressure, o3_vmr)
        
        fig, ax = plt.subplots(1, 1, sharex=False, sharey=True, figsize=(9,7))

        ax.loglog(1e-6*hwhm_doppler, pressure, color=color_gromos, linewidth=lw, label=r'Doppler')
        ax.loglog(1e-6*hwhm_p, pressure, color=color_somora, linewidth=lw, label=r'Pressure')
        ax.loglog(1e-6*(hwhm_doppler+hwhm_p), pressure, '--', color='k', linewidth=lw, label=r'Total')
        #ax.fill_betweenx(1e-3*gromos_clean.o3_z.mean(dim='time').values,33e-3,1e3, color='k', alpha=0.1)
        ax.invert_yaxis()
        ax.set_ylim(400,5e-3)
        ax.set_xlim(1e-2,1e3)
        y1z=1e-3*gromos_clean.isel(time=t).o3_z.sel(o3_p=400, tolerance=100,method='nearest')
        y2z=1e-3*gromos_clean.isel(time=t).o3_z.sel(o3_p=0.005, tolerance=0.002,method='nearest')
        ax2 = ax.twinx()
        ax2.set_yticks(gromos_clean.isel(time=t).o3_z) #ax2.set_yticks(altitude)
        ax2.set_ylim(y1z,y2z)
        fmt = FormatStrFormatter("%.0f")
        loc=MultipleLocator(base=10)
        ax2.yaxis.set_major_formatter(fmt)
        ax2.yaxis.set_major_locator(loc)
        ax2.set_ylabel('Altitude [km] ', fontsize=fs)
        ax2.tick_params(axis='both', which='major', labelsize=fs)

        ax.set_ylabel('Pressure [hPa]', fontsize=fs)
        ax.set_xlabel('Line broadening (HWHM) [MHz]', fontsize=fs)
        ax.tick_params(axis='both', which='major', labelsize=fs)
        ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: '{:g}'.format(y)))
        ax.legend(fontsize=fs-2)
        ax.grid(which='both')
        plt.tight_layout(rect=[0, 0.01, 0.92, 1])
        fig.savefig(basefolder+'broadening.pdf', dpi=500)

# %%
