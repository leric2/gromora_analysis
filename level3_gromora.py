#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on 31.03.23

@author: Eric Sauvageat

Collection of function to compare the harmonized GROMORA L2 data


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
from tempera import read_tempera_level3, read_tempera_level2

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

def plot_anomalies(gromos, date_slice, freq, basefolder):
    fs = 34
    year = pd.to_datetime(gromos.time.values[0]).year

    ozone_ts = gromos.sel(time=date_slice).o3_x
    
    fig, axs = plt.subplots(2, 1, sharex=True, sharey=False, figsize=(24,16))

    pl = ozone_ts.resample(time=freq).mean().plot(
        x='time',
        y='o3_p',
        ax=axs[0], 
        vmin=0,
        vmax=10,
        yscale='log',
        linewidth=0,
        rasterized=True,
        add_colorbar=True,
        cmap=cmap_ts,
        cbar_kwargs={'label':r'O$_3$ [ppmv]'}
    )
    pl.set_edgecolor('face')
    axs[0].set_title('GROMOS', fontsize=fs+4) 
    # ax.set_yscale('log')
    axs[0].invert_yaxis()
    axs[0].set_ylabel('Pressure [hPa]', fontsize=fs)

    anomalies = ozone_ts - ozone_ts.rolling(time=24*5, center=True, min_periods=1).mean()#.mean(dim='time'))#/ozone_ts.mean(dim='time') #.rolling(time=24*5, center=True, min_periods=1).mean()#

    pl = anomalies.resample(time=freq).mean().sel(time=date_slice).plot(
        x='time',
        y='o3_p',
        ax=axs[1], 
        vmin=-0.7,
        vmax=0.7,
        yscale='log',
        linewidth=0,
        rasterized=True,
        add_colorbar=True,
        cmap='coolwarm',
        cbar_kwargs={'label':r'O$_3$ [ppmv]'}
    )
    pl.set_edgecolor('face')
    axs[1].set_title('Anomalies', fontsize=fs+4) 
    # ax.set_yscale('log')
    axs[1].invert_yaxis()
    axs[1].set_ylabel('Pressure [hPa]', fontsize=fs)

    for ax in [axs[0], axs[1]]:
        ax.set_xlabel('')
        ax.tick_params(axis='both', which='major', labelsize=fs)
        ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: '{:g}'.format(y)))
        ax.set_ylim(100, 1e-2)

    plt.tight_layout(rect=[0, 0.01, 0.92, 1])
    fig.savefig(basefolder+'GROMOS_ozone_anomalies_'+str(year)+'_'+freq+'.pdf', dpi=500)

#########################################################################################################
def plot_climatology(gromos, outfolder='/scratch/GROSOM/Level2/GROMOS_v3/'):
    fs = 22
    gromos_climato = gromos.groupby('time.dayofyear').mean()
    fig, ax = plt.subplots(1, 1, sharex=True, sharey=True, figsize=(12,6))
    gromos_climato.plot(
        ax=ax,
        y='o3_p',
        vmin=0,
        vmax=10,
        yscale='log',
        linewidth=0,
        rasterized=True,
        cmap=cmap_ts,
        add_colorbar=True,
        cbar_kwargs={'label':r'O$_3$ [ppmv]'},
        # levels=np.arange(-90, 10, 5),
        # norm=colors.CenteredNorm(vcenter=0)    
    )

    ax.set_ylabel('Pressure [hPa]', fontsize=fs)
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: '{:g}'.format(y)))
    ax.xaxis.set_major_locator(ticker.MultipleLocator(60))
    
    ax.invert_yaxis()
    ax.set_ylim(100,0.01)
    ax.set_xlim(0,365)


    ax.set_xlabel('')
    ax.set_xlabel('Day of year')
    ax.set_title(r'GROMOS', fontsize=fs)

    fig.tight_layout(rect=[0, 0.01, 0.99, 1])
    #figures.append(fig)
    fig.savefig(outfolder+'GROMOS_ozone_climatology.pdf')

    percentile=[0.25, 0.75]
    pressure_level = [25,20,15]
    gromos_clim25 = gromos.groupby('time.dayofyear').quantile(percentile[0])
    gromos_clim75 = gromos.groupby('time.dayofyear').quantile(percentile[1])
    gromos_clim = gromos.groupby('time.dayofyear').median()
    ds_allp = gromos.groupby('time.dayofyear').median()
    
    fig, axs = plt.subplots(len(pressure_level),1,figsize=(20, 6*len(pressure_level)), sharex=True)
    for i,p in enumerate(pressure_level):
        gromos_clim_p = gromos_clim.isel(o3_p=p)
        gromos_clim25_p = gromos_clim25.isel(o3_p=p)
        gromos_clim75_p = gromos_clim75.isel(o3_p=p)
        ds = ds_allp.isel(o3_p=p)
        # ds = gromos.o3_x.isel(o3_p=p).sel(time=slice_plot).resample(time='1D').mean()
        #axs[i].plot( ds.dayofyear, ds.data,color=color_gromos) 
        axs[i].plot( gromos_clim_p.dayofyear, gromos_clim_p.data,color='k')
        axs[i].fill_between(gromos_clim_p.dayofyear,gromos_clim25_p,gromos_clim75_p, color='k', alpha=0.2)
        axs[i].set_title(r'O$_3$ VMR '+f'at p = {ds.o3_p.data:.3f} hPa')

    #axs[0].legend(['GROMOS 2020','GROMOS climatology'])

    d1 = 0
    d2 = 365
    axs[0].set_xlim(d1,d2)
    axs[-1].set_xlabel('Day of year')
    #axs[-1].set_ylabel(r'O$_3$ VMR ') 
    for ax in axs:
        ax.grid()
        ax.set_ylabel(r'O$_3$ VMR ') 
    fig.tight_layout(rect=[0, 0.01, 0.95, 1])
    fig.savefig(outfolder + 'GROMOS_climatology_plevel.pdf', dpi=500)

def regrid_ds(dataset, new_z, freq='1H'):
    new_z = np.int32(new_z)
    new_dataset = list()

    for t in dataset.time.data:
        ds = dataset.sel(time=t)

        ds.coords['alt'] = ('o3_p', ds.o3_z.data)

        ds = ds.swap_dims({'o3_p':'alt'})
        #ds.coords['o3_p_avk'] = ('alt', ds.o3_z.data)

        #ds = ds.coords['o3_p_avk'].rename('alt')
        #ds=ds.drop_dims(['o3_p_avk'])

        #ds = ds.swap_dims({'o3_p_avk':'alt'})

        #f = np.interp(new_z, ds.o3_z.data[1:], np.diff(ds.o3_z.data))/np.median(np.diff(new_z))

        interp_ds = ds.interp(alt=new_z)
        new_dataset.append(interp_ds)
    
    
    interpolated_dataset = xr.concat(new_dataset, dim='time')
    
    if freq is not None:
        interpolated_dataset = interpolated_dataset.resample(time=freq).mean()
    
    return interpolated_dataset


#########################################################################################################
# Main function
#########################################################################################################
if __name__ == "__main__":
    yr = 2023
    # The full range:
    date_slice=slice('1994-01-01','2023-12-31')

   # date_slice=slice('2009-07-01','2011-10-31')

    years = [2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023] #[2014, 2015, 2016, 2017]
    years =[1994, 1995, 1996, 1997, 1998, 1999, 2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011] 
    instNameGROMOS = 'GROMOS'

    # By default, we use the latest version with L2 flags
    v2 = True
    flagged_L2 = True
    
    fold_gromos = '/storage/tub/instruments/gromos/level2/GROMORA/v2/'# #'/scratch/GROSOM/Level2/GROMOS/v2/'
    fold_gromos2 = '/scratch/GROSOM/Level2/GROMOS/v21/'# '/storage/tub/instruments/gromos/level2/GROMORA/v3/' # '/scratch/GROSOM/Level2/GROMOS/v3/'
    prefix_FFT= '_FB_v21' # '_AC240_v21'#'_AC240_v3'

    ########################################################################################################
    # Different strategies can be chosen for the analysis:
    # 'read': default option which reads the full level 2 doing the desired analysis
    # 'read_save': To save new level 3 data from the full hourly level 2
    # 'plot_all': the option to reproduce the figures from the manuscript
    # 'anything else': option to read the level 3 data before doing the desired analysis

    strategy = 'plt'
    if strategy[0:4]=='read':
        read_gromos=False
        read_somora=True
        read_both=False

        # if len(years)>4 and read_both:
        #     raise ValueError('This will take too much space sorry !')

        if read_gromos or read_both:
            gromos = read_GROMORA_all(
                basefolder=fold_gromos, 
                instrument_name='GROMOS',
                date_slice=date_slice,#slice('2010-01-01','2021-12-31'), 
                years=years,#, [2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021]
                prefix=prefix_FFT+'.nc',
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
                prefix=prefix_FFT+'.nc',  # '_v2_all.nc'#
                flagged=flagged_L2,
                decode_time=True
            )
            # somora['o3_x'] = 1e6*somora['o3_x'].where((somora['o3_x']>somora['o3_x'].valid_min)&(somora['o3_x']<somora['o3_x'].valid_max), drop = True)
            somora['o3_x'] = 1e6*somora['o3_x'].where((somora['o3_x']>0)&(somora['o3_x']<5e-5), drop = True)
            somora_clean = somora.where(somora.retrieval_quality==1, drop=True)#.where(somora.level2_flag==0, drop=True)#.where(gromos.o3_mr>0.8)
            print('GROMOS FB good quality level2: ', 100*len(somora_clean.time)/len(pd.date_range('1994-01-01', '2011-12-31 23:00:00', freq='1H')) )
            
        if not flagged_L2:
            if read_gromos or read_both:
                gromos = add_flags_level2_gromora(gromos, 'GROMOS')
                gromos_clean = gromos.where(gromos.retrieval_quality==1, drop=True).where(gromos.level2_flag==0, drop=True)#.where(gromos.o3_mr>0.8)

            if read_somora or read_both:
                somora = add_flags_level2_gromora(somora, 'SOMORA')
                somora_clean = somora.where(somora.retrieval_quality==1, drop=True).where(somora.level2_flag==0, drop=True)#.where(gromos.o3_mr>0.8)
        
        if strategy=='read_regrid':
            new_alt = 1e3*np.arange(4, 92, 2)
            if read_gromos:
                gromos_regridded = regrid_ds(gromos_clean.drop_dims(['h2o_continuum_p','poly_order','f_shift_grid','sine_grid',]).drop_vars(['o3_avkm']), new_alt, freq=None) 
                gromos_regridded.time.encoding['units'] = 'seconds since 2000-01-01 00:00:00'
                gromos_regridded.time.encoding['calendar'] = 'proleptic_gregorian'
                gromos_regridded.to_netcdf('/scratch/GROSOM/Level2/GROMOS_level2_v2_regridded_2014-2017_og.nc', encoding={'time':{'units': 'seconds since 2000-01-01 00:00:00','calendar':'proleptic_gregorian'}})
            if read_somora:
                somora_regridded = regrid_ds(somora_clean.drop_dims(['h2o_continuum_p','poly_order','f_shift_grid','sine_grid',]).drop_vars(['o3_avkm']), new_alt, freq=None) 
                somora_regridded.time.encoding['units'] = 'seconds since 2000-01-01 00:00:00'
                somora_regridded.time.encoding['calendar'] = 'proleptic_gregorian'
                somora_regridded.to_netcdf('/scratch/GROSOM/Level2/SOMORA_level2_v2_regridded_2014-2017_og.nc', encoding={'time':{'units': 'seconds since 2000-01-01 00:00:00','calendar':'proleptic_gregorian'}})

            exit()
        if strategy=='read_save':
            # Saving the level 3:
            if read_gromos:
                gromos_clean.resample(time='6H').mean().to_netcdf('/scratch/GROSOM/Level2/GROMOS_level3_6H_v21.nc')
            elif read_somora:
                somora_clean.resample(time='6H').mean().to_netcdf('/scratch/GROSOM/Level2/GROMOS_FB_level3_6H_v21.nc')
            exit()
    else:
        gromos_clean = read_GROMORA_concatenated('/scratch/GROSOM/Level2/GROMOS_level3_6H_v21.nc', date_slice)
        somora_clean = read_GROMORA_concatenated('/scratch/GROSOM/Level2/GROMOS_FB_level3_6H_v3.nc', date_slice)
        
        # gromos_clean = gromos.where(gromos.level2_flag==0, drop=True)#.where(gromos.o3_mr>0.8)
        # somora_clean = somora.where(somora.level2_flag==0, drop=True)#.where(gromos.o3_mr>0.8)
        
        print('GROMOS good quality level2: ', 100*len(gromos_clean.time)/len(pd.date_range('2009-07-01', '2021-12-31 23:00:00', freq='6H')) )

    #####################################################################
    # Read SBUV and MLS
    bn = '/storage/tub/atmosphere/SBUV/O3/daily_mean_overpasses/'
    sbuv = read_SBUV_dailyMean(timerange=date_slice, SBUV_basename = bn, specific_fname='sbuv_v87.mod_v2r1.vmr.payerne_156.txt')
    sbuv_arosa = read_SBUV_dailyMean(date_slice, SBUV_basename = bn, specific_fname='sbuv_v87.mod_v2r1.vmr.arosa_035.txt')

    outfolder = '/scratch/GROSOM/Level2/GROMOS_v3/'

    mls= read_MLS(timerange=date_slice, vers=5, filename_MLS='AuraMLS_L2GP-O3_v5_400-800_BERN_2004-2022.nc')#slice('2003-01-01','2021-12-31')
    # mls_temp = read_MLS_Temperature(date_slice, vers=5, filename_MLS='AuraMLS_L2GP-TEMP_v5_400-800_BERN.nc')

    # mls = read_MLS(date_slice, vers=5, filename_MLS='AuraMLS_L2GP-O3_v5_400-800_BERN_lst.nc', save_LST=False)
    mls_temp = read_MLS_Temperature(date_slice, vers=5, filename_MLS='AuraMLS_L2GP-TEMP_v5_400-800_BERN.nc')

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
        ecmwf_ts['O3'] =mmr_2_vmr(ecmwf_ts.ozone_mass_mixing_ratio)#.resample(time='1D').mean()
        ecmwf_ts = ecmwf_ts.rename({'level':'lev'})
    else:
        ecmwf_ts = None

    #####################################################################
    # Homogenization of GROMOS FB and FFT
    mean_bias = xr.open_dataarray('/scratch/GROSOM/Level2/GROMOS_v21/mean_bias_FB-FFT_v21.nc')

    date_split = '2010-01-01'
    date_split2 = '2010-01-01'
    #freq =  '1M'
    somora_clean = somora_clean.sel(time=slice(date_slice.start,date_split))#.resample(time=freq).mean()
    gromos_clean = gromos_clean.sel(time=slice(date_split2,date_slice.stop))#.resample(time=freq).mean()


    # somora_clean['o3_corr'] = somora_clean.o3_x - mean_bias

    somora_clean['o3_x'] = somora_clean['o3_x'].copy(data= somora_clean['o3_x'] - mean_bias) 
    
    full_gromos = xr.concat([somora_clean, gromos_clean], dim='time')

    # gromos_all_corr = xr.concat([somora_clean.o3_corr, gromos_clean.o3_x], dim='time')

    # full_gromos['o3_x'] = gromos_all_corr.o3_corr

    full_gromos.to_netcdf('/scratch/GROSOM/Level2/GROMOS_v21/GROMOS_homogenized_v21.nc')

    #####################################################################
    # Plot anomalies during this period
    plot_old_new = False
    if plot_old_new:
        plot_anomalies(gromos, date_slice, freq='2H', basefolder=outfolder)
        
    #####################################################################
    # Plot climatology during this period
    plot_climato = False
    if plot_climato:
        plot_climatology(gromos=gromos_all_corr, outfolder=outfolder)
        

    #####################################################################
    # MLS comparisons
    compare_with_MLS = False
    if compare_with_MLS:
        gromos_colloc, gromos_conv, mls_gromos_colloc, mls_gromos_conv = read_MLS_convolved(instrument_name='GROMOS',folder='/scratch/GROSOM/Level2/MLS/v2/', years=years, prefix='FB_')
    
        #compare_seasonal_GROMORA_MLS_profiles(gromos_colloc, gromos_conv, mls_gromos_colloc, mls_gromos_conv, somora_colloc, somora_conv, mls_somora_colloc, mls_somora_conv, mls, sbuv, p_min=[0.1, 1, 10] , p_max=[0.9, 5, 50], basefolder='/scratch/GROSOM/Level2/Comparison_MLS/', convolved=False, split_night=False)
        compare_gromos_MLS_profiles(gromos_colloc, gromos_conv, mls_gromos_colloc, mls_gromos_conv, mls, sbuv, p_min=[0.1, 1, 10] , p_max=[0.9, 5, 50], basefolder='/scratch/GROSOM/Level2/Comparison_MLS/v2/', convolved=False, split_night=False)


