#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on 17.03.22

@author: Eric Sauvageat

This is the main script for the GROMORA MLS comparisons

This module contains the code to create the yearly MLS collocated and convolved file. 
It basically create 4 new netCDF files per instruments and per year:

1. instrument_name + '_collocation_MLS_'+str(yr)+'.nc': the collocated GROMORA profiles
2. instrument_name + '_convolved_MLS_'+str(yr)+'.nc': the collocated GROMORA profiles to the convolved MLS profiles (can be different from 1 if there is a problem during the convolution for instance)
3. 'MLS_collocation_'+instrument_name+'_'+str(yr)+'.nc': the MLS collocated profiles (should correspond to 1.)
4. 'MLS_convolved_'+instrument_name+'_' +str(yr)+'.nc': the MLS convolved profiles (should correspond to 2.)

To write new files, use the main function located at the end of this file.

"""
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

from typhon.collocations import Collocator
from base_tool import *
from level2_gromora_diagnostics import read_GROMORA_all, add_flags_level2_gromora
from flags_analysis import read_level1_flags

color_gromos= get_color('GROMOS')
color_somora= get_color('SOMORA')
sbuv_color= get_color('SBUV')
color_shading='grey'

colormap='cividis'

def read_MLS_convolved(instrument_name='GROMOS', folder='/scratch/GROSOM/Level2/MLS/', years=[2018]):
    ds_colloc=xr.Dataset()
    ds_mls_conv=xr.Dataset()
    gromora_sel=xr.Dataset()
    ds_gromora_conv=xr.Dataset()
    counter=0
    for yr in years:
        filename_gromora = instrument_name + '_collocation_MLS_'+str(yr)+'.nc'
        filename_gromora_convolved = instrument_name + '_convolved_MLS_'+str(yr)+'.nc'
        filename_colloc = 'MLS_collocation_'+instrument_name+'_'+str(yr)+'.nc'
        filename_convolved_mls = 'MLS_convolved_'+instrument_name+'_' +str(yr)+'.nc'

        gromora = xr.open_dataset(os.path.join(folder, filename_gromora))
        gromora_convolved = xr.open_dataset(os.path.join(folder, filename_gromora_convolved))
        ds_col = xr.open_dataset(os.path.join(folder, filename_colloc))
        ds_conv = xr.open_dataset(os.path.join(folder, filename_convolved_mls))
        if counter==0:
            ds_gromora_conv=xr.merge([ds_gromora_conv, gromora_convolved])
            ds_colloc = xr.merge([ds_colloc, ds_col])
            ds_mls_conv=xr.merge([ds_mls_conv, ds_conv] )
            gromora_sel=xr.merge([gromora_sel, gromora] )
            counter=counter+1
        else:
            ds_colloc=xr.concat([ds_colloc, ds_col], dim='time')
            ds_gromora_conv=xr.concat([ds_gromora_conv, gromora_convolved], dim='time')
            ds_mls_conv=xr.concat([ds_mls_conv, ds_conv] , dim='time')
            gromora_sel=xr.concat([gromora_sel, gromora] , dim='time'   )
        
    return gromora_sel, ds_gromora_conv, ds_colloc, ds_mls_conv


def read_MLS(timerange, vers, filename_MLS, save_LST=False):
    if vers == 5:
        MLS_basename = '/storage/tub/atmosphere/AuraMLS/Level2_v5/locations/'
        #filename_MLS = 'AuraMLS_L2GP-O3_v5_400-800.nc'
    
        ds_mls = xr.open_dataset(os.path.join(MLS_basename, filename_MLS))

        # Add LST:
        if save_LST:
            ds_mls['UT_time'] = ds_mls['time']
            lst_list = list()
            for i, t in enumerate(ds_mls['time'].data):
                mls_lst =  pd.to_datetime(t).replace(hour=0, minute=0,second=0, microsecond=0) + datetime.timedelta(hours=ds_mls.isel(time=i).local_solar_time.item())
                lst_list.append(mls_lst)

            ds_mls['time'] = pd.to_datetime(lst_list)
            ds_mls = ds_mls.sortby('time')
            ds_mls.to_netcdf(os.path.join(MLS_basename, filename_MLS[:-3]+'_lst.nc'))

        ds_mls.attrs['history']=''
        ds_mls = ds_mls.rename({'value':'o3', 'pressure':'p'})
        ds_mls['o3'] =  ds_mls['o3']*1e6
        ds_mls['p'] =  ds_mls['p']/100
    else:
        MLS_basename = '/home/esauvageat/Documents/AuraMLS/'
        #filename_MLS = os.path.join(MLS_basename, 'aura-ozone-at-Bern.mat')
        mls_data = scipy.io.loadmat(filename_MLS)

        ozone_mls = mls_data['o3']
        p_mls = mls_data['p'][0]
        time_mls = mls_data['tm'][0]
        datetime_mls = []
        for i, t in enumerate(time_mls):
            datetime_mls.append(datetime.datetime.fromordinal(
                int(t)) + datetime.timedelta(days=time_mls[i] % 1) - datetime.timedelta(days=366))

        ds_mls = xr.Dataset(
            data_vars=dict(
                o3=(['time', 'p'], ozone_mls)
            ),
            coords=dict(
                lon=mls_data['longitude2'][0],
                lat=mls_data['latitude2'][0],
                time=datetime_mls,
                p=p_mls
            ),
            attrs=dict(description='ozone time series at bern')
        )

    #ds_mls = xr.decode_cf(ds_mls)
    #ds_mls.time.encoding['units'] = 'seconds since 1970-01-01 00:00:00'
    #ds_mls.time.encoding['calendar'] = "proleptic_gregorian"
    print('Read MLS dataset file: ', filename_MLS)
    #ds_mls.to_netcdf('/home/esauvageat/Documents/AuraMLS/ozone_bern_ts.nc', format='NETCDF4')
    ds_mls= ds_mls.sel(time=timerange)
    return ds_mls

def read_MLS_Temperature(timerange, vers, filename_MLS):
    MLS_basename = '/storage/tub/atmosphere/AuraMLS/Level2_v5/locations/'
    #filename_MLS = 'AuraMLS_L2GP-O3_v5_400-800.nc'

    ds_mls = xr.open_dataset(os.path.join(MLS_basename, filename_MLS))
    ds_mls.attrs['history']=''
    ds_mls = ds_mls.rename({'value':'o3', 'pressure':'p'})
    ds_mls['p'] =  ds_mls['p']/100
    
    #ds_mls = xr.decode_cf(ds_mls)
    #ds_mls.time.encoding['units'] = 'seconds since 1970-01-01 00:00:00'
    #ds_mls.time.encoding['calendar'] = "proleptic_gregorian"
    print('Read MLS dataset file: ', filename_MLS)
    #ds_mls.to_netcdf('/home/esauvageat/Documents/AuraMLS/ozone_bern_ts.nc', format='NETCDF4')
    ds_mls= ds_mls.sel(time=timerange)
    return ds_mls

def select_gromora_corresponding_mls(gromora, instrument_name, ds_mls, time_period, save_ds=False, basename='/scratch/', convolved=True):
    time = pd.date_range(time_period.start + ' 01:30:00', time_period.stop, freq='3H')
    
    time_mls = pd.to_datetime(ds_mls.sel(time=time_period).time.data)
    time_gromora = pd.to_datetime(gromora.sel(time=time_period).time.data)
    new_ds = xr.Dataset()
    new_gromora = list()
    mls_new_time_list = list()
    mls_colloc_list = list()
    gromora_new_time_list = list()
    gromora_colloc_list = list()
    if convolved:
        p_name = 'o3_p'
        o3_name = 'o3_x'
    else:
        p_name = 'p'
        o3_name = 'o3'

    counter = 0
    for i, t in enumerate(time):
        range = slice(t-datetime.timedelta(minutes=90), t+datetime.timedelta(minutes=90))
        #try
        #mls_colloc = ds_mls.sel(time=t, method='nearest', tolerance='30M')
        gromora_coloc = gromora.sel(time=range)
        mls_colloc = ds_mls.sel(time=range)
        
        if (len(mls_colloc.time)>0) & (len(gromora_coloc.time) > 0):
            #print(i)
            #mls_colloc = mls_colloc.mean(dim='time')
            mls_colloc_list.append(mls_colloc[o3_name].mean(dim='time', skipna=True).values)
            mls_new_time_list.append(mls_colloc.time.mean(dim='time', skipna=True).values)

            gromora_colloc_list.append(gromora_coloc['o3_x'].mean(dim='time', skipna=True).values)
            gromora_new_time_list.append(gromora_coloc.time.mean(dim='time', skipna=True).values)

            # The full GROMORA dataset averaged on this time range.
            new_gromora.append(gromora_coloc.mean(dim='time', skipna=True))

            # gromora_sel = gromora.sel(time=t, method='nearest', tolerance='2H')
            #if counter == 0:
                #new_ds = mls_colloc
                #counter = counter+1
            #else:
                #new_ds=xr.concat([new_ds, mls_colloc], dim='time')
        # elif len(mls_colloc.time)==1:
        #     #print(i)
        #     mls_colloc_list.append(mls_colloc[o3_name].mean(dim='time').values)
        #     new_time_list.append(mls_colloc.time.values[0])
        else:
            pass
            #new_gromora = new_gromora.drop_sel(time=t)
        #except:
         #   new_gromora = new_gromora.drop_sel(time=t)

    new_ds = xr.Dataset(
        data_vars=dict(
            o3_x=(['time','o3_p'], mls_colloc_list)
        ),
        coords=dict(
            time=mls_new_time_list,
            o3_p=ds_mls[p_name].data
        ),
        attrs=dict(description='Collocated MLS ozone time series at bern')
    )
    # gromora_new_ds = xr.Dataset(
    #     data_vars=dict(
    #         o3_x=(['time','o3_p'], gromora_colloc_list)
    #     ),
    #     coords=dict(
    #         time=gromora_new_time_list,
    #         o3_p=gromora['o3_p'].data
    #     ),
    #     attrs=dict(description='Collocated ' + instrument_name+ ' ozone time series with MLS over Bern')
    # )

    gromora_new_ds = xr.concat(new_gromora, dim='time')
    gromora_new_ds['time'] = gromora_new_time_list

    print('Number of collocated profiles '+instrument_name+' : '+str(len(gromora_new_time_list)))
    # new_ds.o3_x.isel(o3_p=12).resample(time='1D').mean().plot()
    # new_mls.o3.isel(p=12).resample(time='1D').mean().plot()
    if save_ds:
        if convolved:
            new_ds.to_netcdf('/scratch/GROSOM/Level2/MLS/MLS_collocation_'+instrument_name+'_'+'MLS_'+t.strftime('%Y')+'.nc')
            gromora_new_ds.to_netcdf('/scratch/GROSOM/Level2/MLS/'+basename+'MLS_'+t.strftime('%Y')+'.nc')
           # new_gromora.to_netcdf('/scratch/GROSOM/Level2/MLS/'+basename+t.strftime('%Y')+'.nc')
        else:
            new_ds.to_netcdf('/scratch/GROSOM/Level2/MLS/MLS_collocation_'+instrument_name+'_'+t.strftime('%Y')+'.nc')
            gromora_new_ds.to_netcdf('/scratch/GROSOM/Level2/MLS/'+basename+'MLS_'+t.strftime('%Y')+'.nc')

           # new_gromora.to_netcdf('/scratch/GROSOM/Level2/MLS/'+basename+t.strftime('%Y')+'.nc')

    return gromora_new_ds, new_ds

def avk_smooth_mls_old(gromora, ds_mls, folder='/scratch/GROSOM/Level2/'):
    time_mls = pd.to_datetime(ds_mls.time.data)
    time_gromora = pd.to_datetime(gromora.time.data)
    new_ds = xr.Dataset()
    convolved_MLS_list = []
    time_list = []
    counter = 0
  #  ds_mls = ds_mls.resample(time='12H').mean()
    for t in time_mls:
        try:
            gromora_sel = gromora.sel(time=t, method='nearest', tolerance='2H')
            ds_mls_sel= ds_mls.sel(time=t)#) , method='nearest', tolerance='2H')
            avkm = gromora_sel.o3_avkm.data

            o3_p_mls = ds_mls_sel.p.data
            idx = np.argsort(o3_p_mls, kind='heapsort')
            o3_p_mls_sorted = o3_p_mls[idx]  
            o3_mls_sorted = ds_mls_sel.o3.data[idx]
    
            interpolated_mls = np.interp(np.log(gromora_sel.o3_p.data), np.log(o3_p_mls_sorted),o3_mls_sorted)
            conv = gromora_sel.o3_xa.values*1e6 + np.matmul(avkm,(interpolated_mls - gromora_sel.o3_x.values))

            convolved_MLS_list.append(conv)
            time_list.append(t)

            
            if counter == 0:
                new_ds = gromora_sel
                counter = counter+1
            else:
                new_ds=xr.concat([new_ds, gromora_sel], dim='time')
        except:
            pass
    
    convolved_MLS = xr.Dataset(
        data_vars=dict(
            o3_x=(['time', 'o3_p'], convolved_MLS_list)
        ),
        coords=dict(
            time=time_list,
            o3_p=gromora_sel.o3_p.data
        ),
        attrs=dict(description='ozone time series at bern')
    )
    convolved_MLS.to_netcdf(folder+'convolved_MLS_'+t.strftime('%Y')+'.nc')
    # new_ds.o3_x.isel(o3_p=12).resample(time='1D').mean().plot()
    # new_mls.o3.isel(p=12).resample(time='1D').mean().plot()
    return new_ds, convolved_MLS


def avk_smooth_mls_new(gromora, mls_gromora_colloc, instrument_name, basefolder='/scratch/GROSOM/Level2/', sel=True, save_ds=False):
    time_mls = pd.to_datetime(mls_gromora_colloc.time.data)
    time_gromora = pd.to_datetime(gromora.time.data)
    new_ds = xr.Dataset()
    convolved_MLS_list = list()
    gromora_ozone_list = list()
    time_list = list()
    counter = 0
    if sel:
        pname = 'o3_p'
        o3_name = 'o3_x'

  #  ds_mls = ds_mls.resample(time='12H').mean()
    for t in time_gromora:
        try:
            gromora_sel = gromora.sel(time=t, method='nearest', tolerance='2H')
            ds_mls_sel= mls_gromora_colloc.sel(time=t, method='nearest', tolerance='2H')#) , method='nearest', tolerance='2H')
            avkm = gromora_sel.o3_avkm.data

            o3_p_mls = ds_mls_sel[pname].data
            idx = np.argsort(o3_p_mls, kind='heapsort')
            o3_p_mls_sorted = o3_p_mls[idx]  
            o3_mls_sorted = ds_mls_sel[o3_name].data[idx]
    
            interpolated_mls = np.interp(np.log(gromora_sel.o3_p.data), np.log(o3_p_mls_sorted),o3_mls_sorted)
            conv = gromora_sel.o3_xa.values*1e6 + np.matmul(avkm,(interpolated_mls - gromora_sel.o3_x.values))

            if np.max(conv)>15:
                print('Potential problem with '+instrument_name+' at time: ',t )
            else:
                convolved_MLS_list.append(conv)
                gromora_ozone_list.append(gromora_sel.o3_x.data)
                time_list.append(t)

                if counter == 0:
                    new_ds = gromora_sel
                    counter = counter+1
                else:
                    new_ds=xr.concat([new_ds, gromora_sel], dim='time')
        except:
            pass
    
    convolved_MLS = xr.Dataset(
        data_vars=dict(
            o3_x=(['time', 'o3_p'], convolved_MLS_list)
        ),
        coords=dict(
            time=time_list,
            o3_p=gromora_sel['o3_p'].data
        ),
        attrs=dict(description='Convolved MLS ozone time series at bern')
    )

    gromora_ozone = xr.Dataset(
        data_vars=dict(
            o3_x=(['time','o3_p'], gromora_ozone_list)
        ),
        coords=dict(
            time=time_list,
            o3_p=gromora_sel['o3_p'].data
        ),
        attrs=dict(description='Collocated ozone ' + instrument_name+ ' ozone time series with MLS over Bern')
    )
    print('Number of convolved profiles '+instrument_name+' : '+str(len(convolved_MLS_list)))
    if save_ds:
        convolved_MLS.to_netcdf(basefolder+'MLS_convolved_'+instrument_name+'_'+t.strftime('%Y')+'.nc')
        gromora_ozone.to_netcdf(basefolder+instrument_name+'_ozone_convolved_MLS_'+t.strftime('%Y')+'.nc')
        new_ds.to_netcdf(basefolder+instrument_name+'_convolved_MLS_'+t.strftime('%Y')+'.nc')

    # new_ds.o3_x.isel(o3_p=12).resample(time='1D').mean().plot()
    # new_mls.o3.isel(p=12).resample(time='1D').mean().plot()
    return gromora_ozone, convolved_MLS


def compare_MLS(o3_mls, o3_mls_v5, freq='D'):  
    year=pd.to_datetime(o3_mls.time.data[0]).year  
    figure=list()
    fig, axs = plt.subplots(3, 1, sharex=True, figsize=(20,15))
    #monthly_mls.plot(x='time', y='p', ax=ax ,vmin=0, vmax=9).resample(time='24H', skipna=True).mean()
    pl = o3_mls.plot(
        x='time',
        y='p',
        ax=axs[0] ,
       # vmin=0,
      #  vmax=10,
        yscale='log',
        linewidth=0,
        rasterized=True,
        cmap=colormap
    )
   # pl.set_edgecolor('face')
    # ax.set_yscale('log')
    #ax.set_ylim(0.01, 500)
    axs[0].invert_yaxis()
    axs[0].set_ylabel('P [hPa]')
    pl2 = o3_mls_v5.plot(
        x='time',
        y='p',
        ax=axs[1] ,
        # vmin=0,
       # vmax=10,
        yscale='log',
        linewidth=0,
        rasterized=True,
        cmap=colormap
    )
   # pl.set_edgecolor('face')
    # ax.set_yscale('log')
    #ax.set_ylim(0.01, 500)
    axs[1].invert_yaxis()
    axs[1].set_ylabel('P [hPa]')

    o3_v5 = o3_mls_v5.where((o3_mls_v5.p>=min(o3_mls.p)) & (o3_mls_v5.p<=max(o3_mls.p)),drop=True)
   # o3_v5 = o3_v5.sel(time=o3_mls.time, method ='nearest').interp_like(o3_mls_v5, assume_sorted=False)
    
    o3_v5 = o3_v5.reindex_like(o3_mls, method='nearest')
   # diff_MLS =100*(o3_mls - o3_v5)/o3_v5
    diff_MLS = 100*(o3_mls.resample(time=freq).mean() - o3_v5.resample(time=freq).mean())/o3_mls.resample(time=freq).mean()
    diff_MLS.plot(
        x='time',
        y='p',
        ax=axs[2] ,
        vmin=-10,
        vmax=10,
        yscale='log',
        linewidth=0,
        rasterized=True,
        cmap='coolwarm'
    )
   # pl.set_edgecolor('face')
    # ax.set_yscale('log')
    #ax.set_ylim(0.01, 500)
    axs[2].invert_yaxis()
    axs[2].set_ylabel('P [hPa]')

    plt.tight_layout()
    figure.append(fig)
    # fig.savefig('/scratch/GROSOM/Level2/GROMORA_waccm/' +
    #             '/ozone_mls_'+str(year)+'.pdf', dpi=500)

    mean_o3_mls = o3_mls.mean(dim='time')
    mean_o3_mls_v5 = o3_mls_v5.mean(dim='time')

    mean_diff_profile = 100*(mean_o3_mls - mean_o3_mls_v5.reindex_like(mean_o3_mls, method='nearest'))/mean_o3_mls

    fig, axs = plt.subplots(1, 2, sharey=True, figsize=(10,15))
    mean_o3_mls.plot(
        y='p',
        ax=axs[0],
        yscale='log',
    )
    mean_o3_mls_v5.plot(
        y='p',
        ax=axs[0],
        yscale='log',
    )
    mean_diff_profile.plot(
        y='p',
        ax=axs[1],
        yscale='log',
    )
    axs[1].set_xlim(-5,5)
    
    for ax in axs:
        ax.set_ylim(0.01, 200)
        ax.invert_yaxis()
        ax.grid()
        ax.set_ylabel('P [hPa]')

    plt.tight_layout()
    figure.append(fig)
    save_single_pdf('/scratch/GROSOM/Level2/GROMORA_waccm/' +
            '/ozone_mls_'+str(year)+'.pdf',figure)
    # fig.savefig('/scratch/GROSOM/Level2/GROMORA_waccm/' +
    #         '/ozone_mls_'+str(year)+'.pdf', dpi=500)

def plot_gromora_and_corresponding_MLS(gromora_sel, mls_sel, mls_convolved, freq='1D', basename=''):
    fig, axs= plt.subplots(3, 1, figsize=(16,12))
    year=pd.to_datetime(gromora_sel.time.data[0]).year  
    pl = gromora_sel.o3_x.resample(time=freq).mean().plot(
        x='time',
        y='o3_p',
        ax=axs[0],
        vmin=0,
        vmax=10,
        yscale='log',
        linewidth=0,
        rasterized=True,
        cmap=colormap
    )
    axs[0].set_title('GROMORA Corresponding to MLS')
    pl_mls_conv = mls_convolved.o3_x.resample(time=freq).mean().plot(
        x='time',
        y='o3_p',
        ax=axs[1],
        vmin=0,
        vmax=10,
        yscale='log',
        linewidth=0,
        rasterized=True,
        cmap=colormap
    )
    pl_mls = mls_sel.o3.resample(time=freq).mean().plot(
        x='time',
        y='p',
        ax=axs[2],
        vmin=0,
        vmax=10,
        yscale='log',
        linewidth=0,
        rasterized=True,
        cmap=colormap
    )
    axs[1].set_title('MLS Convolved')
    axs[2].set_title('MLS OG')
    # pl.set_edgecolor('face')
    # ax.set_yscale('log')
    for ax in axs:
        ax.set_ylim(0.01, 200)
        ax.invert_yaxis()
        ax.set_ylabel('P [hPa]')
    plt.tight_layout()
    fig.savefig('/scratch/GROSOM/Level2/MLS/'+basename +'_comparison_MLS_'+str(year)+'.pdf', dpi=500)


def compare_seasonal_GROMORA_MLS_profiles(gromos_colloc, gromos_conv, mls_gromos_colloc, mls_gromos_conv, somora_colloc, somora_conv, mls_somora_colloc, mls_somora_conv, ds_mls, sbuv, p_min=[0.1, 1, 10] , p_max=[0.9, 5, 50], basefolder='', convolved=True, split_night=True):
    fs = 22
    color_shading = 'gray'
    seasons = ['DJF','MAM','JJA','SON']
    year=pd.to_datetime(gromos_colloc.time.data[0]).year

    sbuv_groups = sbuv.groupby('time.season').groups
    mls_groups = ds_mls.groupby('time.season').groups

    figures = [] 
    for j, s in enumerate(seasons):
        print('#################################################################')
        print('Season: ',s)
        if split_night:
            fig, axs = plt.subplots(1, 4, sharey=True, figsize=(24,14))
        else:
            fig, axs = plt.subplots(1, 3, sharey=True, figsize=(24,14))

        if convolved:
            gromos_groups = gromos_conv.groupby('time.season').groups
            somora_groups = somora_conv.groupby('time.season').groups
            gromos = gromos_conv.isel(time=gromos_groups[s])
            somora = somora_conv.isel(time=somora_groups[s])
            mls_gromos = mls_gromos_conv.isel(time=gromos_groups[s])
            mls_somora = mls_somora_conv.isel(time=somora_groups[s])
            # mls_gromos['time'] = gromos.time
            # mls_somora['time'] = somora.time
        else:
            gromos_groups = gromos_colloc.groupby('time.season').groups
            somora_groups = somora_colloc.groupby('time.season').groups
            mls_gromos_groups = mls_gromos_colloc.groupby('time.season').groups
            mls_somora_groups = mls_somora_colloc.groupby('time.season').groups
            gromos = gromos_colloc.isel(time=gromos_groups[s])
            somora = somora_colloc.isel(time=somora_groups[s])
            mls_gromos = mls_gromos_colloc.interp_like(gromos_colloc.o3_p)
            mls_somora = mls_somora_colloc.interp_like(somora_colloc.o3_p)
            # SBUV does only goes until 2020-12-31 so to do the full
            # comparisons, we just take all SBUV values and not the ones
            # corresponding only to GROMOS or SOMORA (otherwise the 2 following lines fail !)
            sbuv_gromos = sbuv.isel(time=sbuv_groups[s])#.sel(time=gromos.sel(time=slice('2009-07-01','2020-12-31')).time, method='nearest', tolerance='1D', drop=True)
            sbuv_somora = sbuv.isel(time=sbuv_groups[s])#.sel(time=somora.sel(time=slice('2009-07-01','2020-12-31')).time, method='nearest', tolerance='1D', drop=True)
            #.where(sbuv.time.day.isin(gromos.time.day),drop=True)
            mls_gromos = mls_gromos.isel(time=mls_gromos_groups[s])
            mls_somora = mls_somora.isel(time=mls_somora_groups[s])
            mls_gromos['time'] = gromos.time
            mls_somora['time'] = somora.time

        mls = ds_mls.isel(time=mls_groups[s]) 
        sbuv_season = sbuv.isel(time=sbuv_groups[s]) 
        sbuv_mls = sbuv_season.sel(time=mls.sel(time=slice('2009-07-01','2020-12-31')).time, method='nearest', tolerance='1D', drop=True)

        if split_night:
            gromos_dt = gromos.where((gromos.time.dt.hour>=6)&(gromos.time.dt.hour<=18), drop=True)
            somora_dt = somora.where((somora.time.dt.hour>=6)&(somora.time.dt.hour<=18), drop=True)
            gromos_nt = gromos.where(~(gromos.time.dt.hour>=6)&(gromos.time.dt.hour<=18), drop=True)
            somora_nt = somora.where(~(somora.time.dt.hour>=6)&(somora.time.dt.hour<=18), drop=True)
            mls_gromos_dt = mls_gromos.where((mls_gromos.time.dt.hour>=6)&(mls_gromos.time.dt.hour<=18), drop=True)
            mls_somora_dt = mls_somora.where((mls_somora.time.dt.hour>=6)&(mls_somora.time.dt.hour<=18), drop=True)
            mls_gromos_nt = mls_gromos.where(~(mls_gromos.time.dt.hour>=6)&(mls_gromos.time.dt.hour<=18), drop=True)
            mls_somora_nt = mls_somora.where(~(mls_somora.time.dt.hour>=6)&(mls_somora.time.dt.hour<=18), drop=True)
            mls_dt = mls.where((mls.time.dt.hour>=6)&(mls.time.dt.hour<=18), drop=True)
            mls_nt = mls.where(~(mls.time.dt.hour>=6)&(mls.time.dt.hour<=18), drop=True)
            if convolved:
                rel_diff_somora_dt = 100*(somora_dt.o3_x-mls_somora_dt.o3_x)/somora_dt.o3_x.mean(dim='time')
                rel_diff_gromos_dt = 100*(gromos_dt.o3_x-mls_gromos_dt.o3_x)/gromos_dt.o3_x.mean(dim='time')
                rel_diff_somora_nt = 100*(somora_nt.o3_x-mls_somora_nt.o3_x)/somora_nt.o3_x.mean(dim='time')
                rel_diff_gromos_nt = 100*(gromos_nt.o3_x-mls_gromos_nt.o3_x)/gromos_nt.o3_x.mean(dim='time')
            else:
                # If we use collocated measurement, we do not have the same amount of profiles for GROMORA and MLS, TOCHECK ?
                rel_diff_somora_dt = 100*(somora_dt.o3_x-mls_somora_dt.o3_x)/somora_dt.o3_x.mean(dim='time')
                rel_diff_gromos_dt = 100*(gromos_dt.o3_x-mls_gromos_dt.o3_x)/gromos_dt.o3_x.mean(dim='time')
                rel_diff_somora_nt = 100*(somora_nt.o3_x-mls_somora_nt.o3_x)/somora_nt.o3_x.mean(dim='time')
                rel_diff_gromos_nt = 100*(gromos_nt.o3_x-mls_gromos_nt.o3_x)/gromos_nt.o3_x.mean(dim='time')
            mr_somora = somora_dt.o3_mr.data
            mr_gromos = gromos_dt.o3_mr.data
            p_somora_mr = somora_dt.o3_p.data[np.nanmean(mr_somora,0)>=0.8]
            p_gromos_mr = gromos_dt.o3_p.data[np.nanmean(mr_gromos,0)>=0.8]
            mr_somora_nt = somora_nt.o3_mr.data
            mr_gromos_nt = gromos.o3_mr.data
            p_somora_mr_nt = somora_nt.o3_p.data[np.nanmean(mr_somora_nt,0)>=0.8]
            p_gromos_mr_nt = gromos.o3_p.data[np.nanmean(mr_gromos_nt,0)>=0.8]
        else:
            rel_diff_somora = 100*(somora.o3_x-mls_somora.o3_x)/somora.o3_x.mean(dim='time')
            rel_diff_gromos = 100*(gromos.o3_x-mls_gromos.o3_x)/gromos.o3_x.mean(dim='time')
            # rel_diff_somora = rel_diff_somora.reindex_like(rel_diff_gromos.time, method='nearest', tolerance='2H').dropna(dim='time', how='all')
            # rel_diff_gromos = rel_diff_gromos.reindex_like(rel_diff_somora.time, method='nearest', tolerance='2H').dropna(dim='time', how='all')

            for a,p in enumerate(p_min):
                #rel_diff_gromos_prange.extend(rel_diff_gromos.where(rel_diff_gromos.o3_p>p_min[a] , drop=True).where(rel_diff_gromos.o3_p<p_max[a], drop=True).mean(dim='o3_p').data.tolist())
                #rel_diff_somora_prange.extend(rel_diff_somora.where(rel_diff_somora.o3_p>p_min[a] , drop=True).where(rel_diff_somora.o3_p<p_max[a], drop=True).mean(dim='o3_p').data.tolist())
                print('Seasonal rel diff GROMOS - MLS between ',p_max[a],' and ',p_min[a],': ', rel_diff_gromos.where(rel_diff_gromos.o3_p>p_min[a] , drop=True).where(rel_diff_gromos.o3_p<p_max[a], drop=True).mean(dim='o3_p').mean(dim='time').data)
                print('Seasonal rel diff SOMORA - MLS between ',p_max[a],' and ',p_min[a],': ', rel_diff_somora.where(rel_diff_somora.o3_p>p_min[a] , drop=True).where(rel_diff_somora.o3_p<p_max[a], drop=True).mean(dim='o3_p').mean(dim='time').data)
            
            rel_diff_sbuv_gromos = 100*(gromos.sel(time=slice('2009-07-01','2020-12-31')).o3_x.interp(o3_p=sbuv_gromos.p)-sbuv_gromos.ozone)/gromos.o3_x.mean(dim='time').interp(o3_p=sbuv_gromos.p)
            rel_diff_sbuv_somora = 100*(somora.sel(time=slice('2009-07-01','2020-12-31')).o3_x.interp(o3_p=sbuv_gromos.p)-sbuv_somora.ozone)/somora.o3_x.mean(dim='time').interp(o3_p=sbuv_gromos.p)
            rel_diff_sbuv_mls = 100*(mls.sel(time=slice('2009-07-01','2020-12-31')).o3.interp(p=sbuv_mls.p)-sbuv_mls.ozone)/mls.sel(time=slice('2009-07-01','2020-12-31')).o3.mean(dim='time').interp(p=sbuv_mls.p)

            std_somora_colloc = 100*(somora.o3_x-mls_somora.o3_x).std(dim='time')/somora.o3_x.mean(dim='time')
            std_gromos_colloc = 100*(gromos.o3_x-mls_gromos.o3_x).std(dim='time')/gromos.o3_x.mean(dim='time')
            sum_std_gromos = 100* np.sqrt(gromos.o3_x.var(dim='time') + mls_gromos.o3_x.var(dim='time'))/gromos.o3_x.mean(dim='time')
            mr_somora = somora.o3_mr.data
            mr_gromos = gromos.o3_mr.data
            p_somora_mr = somora.o3_p.data[np.nanmean(mr_somora,0)>=0.8]
            p_gromos_mr = gromos.o3_p.data[np.nanmean(mr_gromos,0)>=0.8]
            
            gromos_groups_conv = gromos_conv.groupby('time.season').groups
            somora_groups_conv = somora_conv.groupby('time.season').groups
            gromos_convolved = gromos_conv.isel(time=gromos_groups_conv[s])
            somora_convolved = somora_conv.isel(time=somora_groups_conv[s])
            mls_gromos_convolved = mls_gromos_conv.isel(time=gromos_groups_conv[s])
            mls_somora_convolved = mls_somora_conv.isel(time=somora_groups_conv[s])
            rel_diff_somora_convolved = 100*(somora_convolved.o3_x-mls_somora_convolved.o3_x)/somora_convolved.o3_x.mean(dim='time')
            rel_diff_gromos_convolved = 100*(gromos_convolved.o3_x-mls_gromos_convolved.o3_x)/gromos_convolved.o3_x.mean(dim='time')
            # rel_diff_somora_convolved = rel_diff_somora_convolved.reindex_like(rel_diff_gromos_convolved.time, method='nearest', tolerance='2H').dropna(dim='time', how='all')
            # rel_diff_gromos_convolved = rel_diff_gromos_convolved.reindex_like(rel_diff_somora_convolved.time, method='nearest', tolerance='2H').dropna(dim='time', how='all')

            std_somora_convolved = 100*(somora_convolved.o3_x-mls_somora_convolved.o3_x).std(dim='time')/somora_convolved.o3_x.mean(dim='time')
            std_gromos_convolved = 100*(gromos_convolved.o3_x-mls_gromos_convolved.o3_x).std(dim='time')/gromos_convolved.o3_x.mean(dim='time')

    # rel_diff_somora = 100*(ozone_somora.mean(dim='time') - convolved_MLS_somora.o3_x.mean(dim='time'))/ozone_somora.mean(dim='time')
    # rel_diff_gromos = 100*(ozone_gromos.mean(dim='time') - convolved_MLS_gromos.o3_x.mean(dim='time'))/ozone_gromos.mean(dim='time')
    # rel_diff_gromora = 100*(ozone_gromos.mean(dim='time') - ozone_somora.mean(dim='time'))/ozone_gromos.mean(dim='time')
    
    # error_gromos = 1e6*np.sqrt(gromos_sel.mean(dim='time').o3_eo**2 + gromos_sel.mean(dim='time').o3_es**2)
    # error_somora = 1e6*np.sqrt(somora_sel.mean(dim='time').o3_eo**2 + somora_sel.mean(dim='time').o3_es**2)
        error_gromos = 100*1e6*gromos.mean(dim='time').o3_eo/gromos.o3_x.mean(dim='time')
        error_somora = 100*1e6*somora.mean(dim='time').o3_eo/somora.o3_x.mean(dim='time')
    # color_shading = 'grey'

    # fig, axs = plt.subplots(nrows=1, ncols=2, sharey=True, figsize=(12, 20))

        if split_night:
            mls_dt.o3.mean(dim='time').plot(y='p', ax=axs[0], linestyle='-', color='k', label='MLS')
            sbuv_season.ozone.mean(dim='time').plot(y='p', ax=axs[0], linestyle='-', color=sbuv_color, label='SBUV')
            mls_nt.o3.mean(dim='time').plot(y='p', ax=axs[2], linestyle='-', color='k', label='MLS')
            gromos_dt.o3_x.mean(dim='time').plot(
                y='o3_p', ax=axs[0], color=color_gromos, ls='-', label='GROMOS')
            somora_dt.o3_x.mean(dim='time').plot(
                y='o3_p', ax=axs[0], color=color_somora, ls='-', label='SOMORA')
            mls_gromos_dt.o3_x.mean(dim='time').plot(y='o3_p', ax=axs[0], linestyle='--', color=color_gromos, label='MLS convolved GROMOS')
            mls_somora_dt.o3_x.mean(dim='time').plot(y='o3_p', ax=axs[0], linestyle='--', color=color_somora, label='MLS convolved SOMORA')
            gromos_nt.o3_x.mean(dim='time').plot(
                y='o3_p', ax=axs[2], color=color_gromos, ls='-', label='GROMOS')
            somora_nt.o3_x.mean(dim='time').plot(
                y='o3_p', ax=axs[2], color=color_somora, ls='-', label='SOMORA')
            mls_gromos_nt.o3_x.mean(dim='time').plot(y='o3_p', ax=axs[2], linestyle='--', color=color_gromos, label='MLS convolved GROMOS')
            mls_somora_nt.o3_x.mean(dim='time').plot(y='o3_p', ax=axs[2], linestyle='--', color=color_somora, label='MLS convolved SOMORA')


        else:
            gromos.o3_x.mean(dim='time').plot(
                y='o3_p', ax=axs[0], color=color_gromos, ls='-', label='GROMOS')
            somora.o3_x.mean(dim='time').plot(
                y='o3_p', ax=axs[0], color=color_somora, ls='-', label='SOMORA')

            mls_gromos_convolved.o3_x.mean(dim='time').plot(y='o3_p', ax=axs[0], linestyle=':', color=color_gromos, label='MLS convolved GROMOS')
            mls_somora_convolved.o3_x.mean(dim='time').plot(y='o3_p', ax=axs[0], linestyle=':', color=color_somora, label='MLS convolved SOMORA')
            mls.o3.mean(dim='time').plot(y='p', ax=axs[0], color='k', label='MLS')
            sbuv_season.ozone.mean(dim='time').plot(y='p', ax=axs[0], linestyle='-', color=sbuv_color, label='SBUV')

    # for ax in axs:
    #     ax.axhline(y=p_somora_mr[0],ls='--' ,color=color_somora, lw=1)
    #     ax.axhline(y=p_somora_mr[-1],ls='--', color=color_somora, lw=1)
    #     ax.axhline(y=p_gromos_mr[0],ls=':', color=color_gromos, lw=1)
    #     ax.axhline(y=p_gromos_mr[-1],ls=':', color=color_gromos, lw=1)

        axs[0].invert_yaxis()
        axs[0].set_yscale('log')
        axs[0].legend(loc=1, fontsize=fs-3)
        axs[0].set_ylim(100,0.01)
        axs[0].set_ylabel('Pressure [hPa]', fontsize=fs)

        if split_night:
            axs[2].set_ylabel('') 
            for p in [0,2]:
                axs[p].set_xlim(-0.2, 9)
                axs[p].set_xlabel(r'O$_3$ VMR [ppmv]', fontsize=fs)
                axs[p].grid(axis='x', linewidth=0.5) 
        else: 
            axs[0].set_xlim(-0.2, 9)
            axs[0].set_xlabel(r'O$_3$ VMR [ppmv]', fontsize=fs)
            axs[0].grid(axis='x', linewidth=0.5)
        

        if split_night:
            if convolved:
                rel_diff_gromos_dt.mean(dim='time').plot(y='o3_p', ax=axs[1], color=color_gromos,
                    ls='-', alpha=1, label='GROMOS vs MLS')
                rel_diff_somora_dt.mean(dim='time').plot(y='o3_p', ax=axs[1], color=color_somora,
                    ls='-', alpha=1, label='SOMORA vs MLS')
                rel_diff_gromos_nt.mean(dim='time').plot(y='o3_p', ax=axs[3], color=color_gromos,
                    ls='-', alpha=1, label='GROMOS vs MLS')
                rel_diff_somora_nt.mean(dim='time').plot(y='o3_p', ax=axs[3], color=color_somora,
                    ls='-', alpha=1, label='SOMORA vs MLS')

                axs[1].fill_betweenx(gromos_dt.o3_p, (rel_diff_gromos_dt.mean(dim='time')-error_gromos),(rel_diff_gromos_dt.mean(dim='time')+error_gromos), color=color_gromos, alpha=0.3)
                #  axs[1].fill_betweenx(gromos_dt.o3_p, (rel_diff_gromos_dt.mean(dim='time')-0.5*rel_diff_gromos_dt.std(dim='time')),(rel_diff_gromos_dt.mean(dim='time')+0.5*rel_diff_gromos_dt.std(dim='time')), color='gray', alpha=0.3)

                axs[3].fill_betweenx(gromos_nt.o3_p, (rel_diff_gromos_nt.mean(dim='time')-error_gromos),(rel_diff_gromos_nt.mean(dim='time')+error_gromos), color=color_gromos, alpha=0.3)
                axs[1].fill_betweenx(somora_dt.o3_p, (rel_diff_somora_dt.mean(dim='time')-error_somora),(rel_diff_somora_dt.mean(dim='time')+error_somora), color=color_somora, alpha=0.3)
                axs[3].fill_betweenx(somora_nt.o3_p, (rel_diff_somora_nt.mean(dim='time')-error_somora),(rel_diff_somora_nt.mean(dim='time')+error_somora), color=color_somora, alpha=0.3)
            else:
                rel_diff_gromos_dt.mean(dim='time').plot(y='o3_p', ax=axs[1], color=color_gromos,
                    ls='-', alpha=1, label='GROMOS vs MLS')
                rel_diff_somora_dt.mean(dim='time').plot(y='o3_p', ax=axs[1], color=color_somora,
                    ls='-', alpha=1, label='SOMORA vs MLS')
                rel_diff_gromos_nt.mean(dim='time').plot(y='o3_p', ax=axs[3], color=color_gromos,
                    ls='-', alpha=1, label='GROMOS vs MLS')
                rel_diff_somora_nt.mean(dim='time').plot(y='o3_p', ax=axs[3], color=color_somora,
                    ls='-', alpha=1, label='SOMORA vs MLS')
                axs[1].fill_betweenx(gromos_dt.o3_p, (rel_diff_gromos_dt.mean(dim='time')-error_gromos),(rel_diff_gromos_dt.mean(dim='time')+error_gromos), color=color_gromos, alpha=0.3)
                #  axs[1].fill_betweenx(gromos_dt.o3_p, (rel_diff_gromos_dt.mean(dim='time')-0.5*rel_diff_gromos_dt.std(dim='time')),(rel_diff_gromos_dt.mean(dim='time')+0.5*rel_diff_gromos_dt.std(dim='time')), color='gray', alpha=0.3)

                axs[3].fill_betweenx(gromos_nt.o3_p, (rel_diff_gromos_nt.mean(dim='time')-error_gromos),(rel_diff_gromos_nt.mean(dim='time')+error_gromos), color=color_gromos, alpha=0.3)
                axs[1].fill_betweenx(somora_dt.o3_p, (rel_diff_somora_dt.mean(dim='time')-error_somora),(rel_diff_somora_dt.mean(dim='time')+error_somora), color=color_somora, alpha=0.3)
                axs[3].fill_betweenx(somora_nt.o3_p, (rel_diff_somora_nt.mean(dim='time')-error_somora),(rel_diff_somora_nt.mean(dim='time')+error_somora), color=color_somora, alpha=0.3)

        else:
            rel_diff_gromos.mean(dim='time').plot(y='o3_p', ax=axs[1], color=color_gromos,
                          ls='-', alpha=1, label='GROMOS vs MLS')
            rel_diff_somora.mean(dim='time').plot(y='o3_p', ax=axs[1], color=color_somora,
                          ls='-', alpha=1, label='SOMORA vs MLS')
            # rel_diff_sbuv_gromos.mean(dim='time').plot(y='o3_p', ax=axs[1], color=color_gromos,
            #               ls='--', alpha=1, label='GROMOS vs SBUV')
            # rel_diff_sbuv_somora.mean(dim='time').plot(y='o3_p', ax=axs[1], color=color_somora,
            #               ls='--', alpha=1, label='SOMORA vs SBUV')
            # rel_diff_sbuv_mls.mean(dim='time').plot(y='p', ax=axs[1], color='k',
            #               ls='--', alpha=1, label='MLS vs SBUV')
            rel_diff_gromos_convolved.mean(dim='time').plot(y='o3_p', ax=axs[2], color=color_gromos,
                          ls='-', alpha=1, label='GROMOS vs MLS convolved')
            rel_diff_somora_convolved.mean(dim='time').plot(y='o3_p', ax=axs[2], color=color_somora,
                          ls='-', alpha=1, label='SOMORA vs MLS convolved')
        # rel_diff_gromora.plot(y='o3_p', ax=axs[1], color='k',
        #               ls='-', alpha=1, label='GROMOS vs SOMORA')
            axs[1].fill_betweenx(gromos_conv.o3_p, (rel_diff_gromos.mean(dim='time')-0.5*std_gromos_colloc),(rel_diff_gromos.mean(dim='time')+0.5*std_gromos_colloc), color=color_gromos, alpha=0.2)
            #axs[1].fill_betweenx(gromos_conv.o3_p, (rel_diff_gromos.mean(dim='time')-0.5*sum_std_gromos),(rel_diff_gromos.mean(dim='time')+0.5*sum_std_gromos), color='green', alpha=0.2)
            axs[1].fill_betweenx(somora_conv.o3_p, (rel_diff_somora.mean(dim='time')-0.5*std_somora_colloc),(rel_diff_somora.mean(dim='time')+0.5*std_somora_colloc), color=color_somora, alpha=0.2)
            axs[1].text(
                0.72,
                0.13,
                '{:.0f} profiles'.format(len(rel_diff_gromos.time)),
                transform=axs[1].transAxes,
                verticalalignment="bottom",
                horizontalalignment="left",
                fontsize=fs,
                color=color_gromos
            )
            axs[1].text(
                0.72,
                0.1,
                '{:.0f} profiles'.format(len(rel_diff_somora.time)),
                transform=axs[1].transAxes,
                verticalalignment="bottom",
                horizontalalignment="left",
                fontsize=fs,
                color=color_somora
            )
            axs[2].fill_betweenx(gromos_conv.o3_p, (rel_diff_gromos_convolved.mean(dim='time')-0.5*std_gromos_convolved),(rel_diff_gromos_convolved.mean(dim='time')+0.5*std_gromos_convolved), color=color_gromos, alpha=0.2)
            axs[2].fill_betweenx(somora_conv.o3_p, (rel_diff_somora_convolved.mean(dim='time')-0.5*std_somora_convolved),(rel_diff_somora_convolved.mean(dim='time')+0.5*std_somora_convolved), color=color_somora, alpha=0.2)
            axs[2].text(
                0.72,
                0.13,
                '{:.0f} profiles'.format(len(rel_diff_gromos_convolved.time)),
                transform=axs[2].transAxes,
                verticalalignment="bottom",
                horizontalalignment="left",
                fontsize=fs,
                color=color_gromos
            )
            axs[2].text(
                0.72,
                0.1,
                '{:.0f} profiles'.format(len(rel_diff_somora_convolved.time)),
                transform=axs[2].transAxes,
                verticalalignment="bottom",
                horizontalalignment="left",
                fontsize=fs,
                color=color_somora
            )
    # if convolved:
    #     axs[1].fill_betweenx(gromos_sel.o3_p, (rel_diff_gromos-std_diff_gromos),(ozone_gromos.mean(dim='time')+error_gromos), color=color_gromos, alpha=0.3)
    #     axs[1].fill_betweenx(somora_sel.o3_p, (ozone_somora.mean(dim='time')-error_somora),(ozone_somora.mean(dim='time')+error_somora), color=color_somora, alpha=0.3)

        if split_night:
            for p in [1,3]:
                axs[p].set_xlim(-60, 60)
                axs[p].set_ylabel('')
                axs[p].set_xlabel(r'$\Delta$O$_3$ [\%]', fontsize=fs)
                axs[p].grid(axis='x', linewidth=0.5)
                axs[p].axvline(x=0,ls='--' ,color='k', lw=0.5)
            axs[3].legend(loc=1, fontsize=fs-3)
        else:
            axs[1].set_xlim(-60, 60)
            axs[1].set_ylabel('')
            axs[1].grid(axis='x', linewidth=0.5)
            axs[1].axvline(x=0,ls='-' ,color='k', lw=0.9)
            axs[1].axvline(x=-10,ls='--' ,color='k', lw=0.9)
            axs[1].axvline(x=10,ls='--' ,color='k', lw=0.9)
            axs[1].set_xlabel(r'$\Delta$O$_3$ [\%]', fontsize=fs)

            axs[1].legend(loc=1, fontsize=fs-3)
            axs[2].legend(loc=1, fontsize=fs-3)
            axs[2].axvline(x=0,ls='-' ,color='k', lw=0.9)
            axs[2].axvline(x=-10,ls='--' ,color='k', lw=0.9)
            axs[2].axvline(x=10,ls='--' ,color='k', lw=0.9)
            axs[2].set_xlim(-60, 60)
            axs[2].set_xlabel(r'$\Delta$O$_3$ [\%]', fontsize=fs)
            axs[2].set_ylabel('')

            y1z=1e-3*gromos_colloc.o3_z.mean(dim='time').sel(o3_p=100 ,tolerance=20,method='nearest')
            y2z=1e-3*gromos_colloc.o3_z.mean(dim='time').sel(o3_p=1e-2 ,tolerance=1,method='nearest')
            ax2 = axs[2].twinx()
            ax2.set_yticks(1e-3*gromos.o3_z.mean(dim='time')) #ax2.set_yticks(altitude)
            ax2.set_ylim(y1z,y2z)
            fmt = FormatStrFormatter("%.0f")
            loc=MultipleLocator(base=10)
            ax2.yaxis.set_major_formatter(fmt)
            ax2.yaxis.set_major_locator(loc)
            ax2.tick_params(axis='both', which='major', labelsize=fs)
            ax2.set_ylabel('Altitude [km] ', fontsize=fs)

        # axs[1].set_xlabel(r'$\Delta$O$_3$ [\%]', fontsize=fs)
        

        if split_night:
            for p in [0,1]:
                axs[p].fill_between(axs[p].get_xlim(), p_somora_mr[0],1e4, color=color_shading, alpha=0.2)
                axs[p].fill_between(axs[p].get_xlim(),p_gromos_mr[-1],1e-4, color=color_shading, alpha=0.2)
                axs[p].fill_between(axs[p].get_xlim(),p_gromos_mr[0],1e4, color=color_shading, alpha=0.2)
                axs[p].fill_between(axs[p].get_xlim(),p_gromos_mr[-1],1e-4, color=color_shading, alpha=0.2)
            for p in [2,3]: 
                axs[p].fill_between(axs[p].get_xlim(),p_somora_mr_nt[0],1e4, color=color_shading, alpha=0.2)
                axs[p].fill_between(axs[p].get_xlim(),p_somora_mr_nt[-1],1e-4, color=color_shading, alpha=0.2)
                axs[p].fill_between(axs[p].get_xlim(),p_gromos_mr_nt[0],1e4, color=color_shading, alpha=0.2)
                axs[p].fill_between(axs[p].get_xlim(),p_gromos_mr_nt[-1],1e-4, color=color_shading, alpha=0.2)
            axs[0].set_title('Day') 
            axs[2].set_title('Night')
            axs[2].xaxis.set_minor_locator(MultipleLocator(1))
            axs[2].xaxis.set_major_locator(MultipleLocator(4))
            axs[3].xaxis.set_minor_locator(MultipleLocator(10))
            axs[3].xaxis.set_major_locator(MultipleLocator(30)) 
        else:
            axs[2].xaxis.set_minor_locator(MultipleLocator(10))
            axs[2].xaxis.set_major_locator(MultipleLocator(30))
            for p in [0,1,2]:
                axs[p].fill_between(axs[p].get_xlim(), p_somora_mr[0],1e4, color=color_shading, alpha=0.2)
                axs[p].fill_between(axs[p].get_xlim(),p_somora_mr[-1],1e-4, color=color_shading, alpha=0.2)
                axs[p].fill_between(axs[p].get_xlim(),p_gromos_mr[0],1e4, color=color_shading, alpha=0.2)
                axs[p].fill_between(axs[p].get_xlim(),p_gromos_mr[-1],1e-4, color=color_shading, alpha=0.2)
        axs[0].xaxis.set_minor_locator(MultipleLocator(1))
        axs[0].xaxis.set_major_locator(MultipleLocator(4))
        axs[1].xaxis.set_minor_locator(MultipleLocator(10))
        axs[1].xaxis.set_major_locator(MultipleLocator(30))

        for a in axs:
            a.grid(which='both', axis='y', linewidth=0.5)
            a.grid(which='both', axis='x', linewidth=0.5)
            a.tick_params(axis='both', which='major', labelsize=fs)
            a.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: '{:g}'.format(y)))
    # plt.suptitle('Ozone comparison with ' + str(len(date)) + ' days ' +
    #              pd.to_datetime(ozone_somora.time.mean().data).strftime('%Y-%m-%d %H:%M'))
        fig.suptitle('MLS comparison: '+ s, fontsize=fs+8)
        fig.tight_layout(rect=[0, 0.03, 0.99, 1])
        figures.append(fig)
    if convolved:
        out_str = 'convolved_'
    else:
        out_str = 'collocated_'
    if split_night:
        save_single_pdf(basefolder+'ozone_profile_MLS_'+out_str+str(year)+'_daynight.pdf',figures)
    else:
        save_single_pdf(basefolder+'ozone_profile_MLS_'+out_str+str(year)+'.pdf',figures)
    
    print('#################################################################################################')
    print('#################################################################################################')

def compare_GROMORA_MLS_profiles_egu(gromos_colloc, gromos_conv, mls_gromos_colloc, mls_gromos_conv, somora_colloc, somora_conv, mls_somora_colloc, mls_somora_conv, ds_mls, sbuv, p_min=[0.1, 1, 10] , p_max=[0.9, 5, 50], basefolder='', convolved=True, split_night=True):
    fs = 28
    color_shading = 'gray'

    year=pd.to_datetime(gromos_colloc.time.data[0]).year

    fig, axs = plt.subplots(1, 3, sharey=True, figsize=(24,14))

    gromos_colloc.o3_x.mean(dim='time').plot(
        y='o3_p', ax=axs[0], color=color_gromos, ls='-', label='GROMOS')
    somora_colloc.o3_x.mean(dim='time').plot(
        y='o3_p', ax=axs[0], color=color_somora, ls='-', label='SOMORA')
    mls_gromos_conv.o3_x.mean(dim='time').plot(y='o3_p', ax=axs[0], linestyle=':', color=color_gromos, label='MLS convolved GROMOS')
    mls_somora_conv.o3_x.mean(dim='time').plot(y='o3_p', ax=axs[0], linestyle=':', color=color_somora, label='MLS convolved SOMORA')
    ds_mls.o3.mean(dim='time').plot(y='p', ax=axs[0], color='k', label='MLS')
    sbuv.ozone.mean(dim='time').plot(y='p', ax=axs[0], linestyle='-', color=sbuv_color, label='SBUV')

    axs[0].invert_yaxis()
    axs[0].set_yscale('log')
    axs[0].legend(loc=1, fontsize=fs-3)
    axs[0].set_ylim(100,0.01)
    axs[0].set_ylabel('Pressure [hPa]', fontsize=fs)

    mls_gromos = mls_gromos_colloc.interp_like(gromos_colloc.o3_p)
    mls_somora = mls_somora_colloc.interp_like(somora_colloc.o3_p)
    mls_gromos['time'] = gromos_colloc.time
    mls_somora['time'] = somora_colloc.time
    #mls_gromos_all = mls_gromos.sel(time=gromos_colloc.time, tolerance='2H', method='nearest')
    #mls_somora_all = mls_somora.sel(time=gromos_colloc.time, tolerance='2H', method='nearest')
    rel_diff_somora = 100*(somora_colloc.o3_x-mls_somora.o3_x)/somora_colloc.o3_x.mean(dim='time')
    rel_diff_gromos = 100*(gromos_colloc.o3_x-mls_gromos.o3_x)/gromos_colloc.o3_x.mean(dim='time')
    rel_diff_somora_convolved = 100*(somora_conv.o3_x-mls_somora_conv.o3_x)/somora_conv.o3_x.mean(dim='time')
    rel_diff_gromos_convolved = 100*(gromos_conv.o3_x-mls_gromos_conv.o3_x)/gromos_conv.o3_x.mean(dim='time')
    print('Total collocated profiles GROMOS-MLS: ',len(rel_diff_gromos))
    print('Total collocated profiles SOMORA-MLS: ',len(rel_diff_somora))
    std_somora_colloc = 100*(somora_colloc.o3_x-mls_somora.o3_x).std(dim='time')/somora_colloc.o3_x.mean(dim='time')
    std_gromos_colloc = 100*(gromos_colloc.o3_x-mls_gromos.o3_x).std(dim='time')/gromos_colloc.o3_x.mean(dim='time')
    std_somora_convolved = 100*(somora_conv.o3_x-mls_somora_conv.o3_x).std(dim='time')/somora_conv.o3_x.mean(dim='time')
    std_gromos_convolved = 100*(gromos_conv.o3_x-mls_gromos_conv.o3_x).std(dim='time')/gromos_conv.o3_x.mean(dim='time')
    for a,p in enumerate(p_min):
        rel_diff_gromos_prange = rel_diff_gromos.where(rel_diff_gromos.o3_p>p_min[a] , drop=True).where(rel_diff_gromos.o3_p<p_max[a], drop=True).mean(dim='o3_p').mean(dim='time')
        rel_diff_somora_prange = rel_diff_somora.where(rel_diff_somora.o3_p>p_min[a] , drop=True).where(rel_diff_somora.o3_p<p_max[a], drop=True).mean(dim='o3_p').mean(dim='time')
        rel_diff_gromos_prange_conv = rel_diff_gromos_convolved.where(rel_diff_gromos_convolved.o3_p>p_min[a] , drop=True).where(rel_diff_gromos_convolved.o3_p<p_max[a], drop=True).mean(dim='o3_p').mean(dim='time')
        rel_diff_somora_prange_conv = rel_diff_somora_convolved.where(rel_diff_somora_convolved.o3_p>p_min[a] , drop=True).where(rel_diff_somora_convolved.o3_p<p_max[a], drop=True).mean(dim='o3_p').mean(dim='time')
        std_vertical_diff_gromos_prange = rel_diff_gromos.mean(dim='time').where(rel_diff_gromos.o3_p>p_min[a] , drop=True).where(rel_diff_gromos.o3_p<p_max[a], drop=True).std(dim='o3_p')
        std_vertical_diff_somora_prange = rel_diff_somora.mean(dim='time').where(rel_diff_somora.o3_p>p_min[a] , drop=True).where(rel_diff_somora.o3_p<p_max[a], drop=True).std(dim='o3_p')
        std_vertical_diff_gromos_prange_conv = rel_diff_gromos_convolved.mean(dim='time').where(rel_diff_gromos_convolved.o3_p>p_min[a] , drop=True).where(rel_diff_gromos_convolved.o3_p<p_max[a], drop=True).std(dim='o3_p')
        std_vertical_diff_somora_prange_conv = rel_diff_somora_convolved.mean(dim='time').where(rel_diff_somora_convolved.o3_p>p_min[a] , drop=True).where(rel_diff_somora_convolved.o3_p<p_max[a], drop=True).std(dim='o3_p')

        print('Global rel diff GROMOS - MLS between ',p_max[a],' and ',p_min[a],': ', rel_diff_gromos_prange.values, ', std =', std_vertical_diff_gromos_prange.values)
        print('Global rel diff SOMORA - MLS between ',p_max[a],' and ',p_min[a],': ', rel_diff_somora_prange.values, ', std =', std_vertical_diff_somora_prange.values)
        print('######################')
        print('Convolved:')
        print('Global rel diff GROMOS - MLS between ',p_max[a],' and ',p_min[a],': ', rel_diff_gromos_prange_conv.values, ', std =', std_vertical_diff_gromos_prange_conv.values)
        print('Global rel diff SOMORA - MLS between ',p_max[a],' and ',p_min[a],': ', rel_diff_somora_prange_conv.values, ', std =', std_vertical_diff_somora_prange_conv.values)
        print('############################################')
            
    rel_diff_gromos.mean(dim='time').plot(y='o3_p', ax=axs[1], color=color_gromos,
        ls='-', alpha=1, label='GROMOS vs MLS')
    rel_diff_somora.mean(dim='time').plot(y='o3_p', ax=axs[1], color=color_somora,
        ls='-', alpha=1, label='SOMORA vs MLS')

    rel_diff_gromos_convolved.mean(dim='time').plot(y='o3_p', ax=axs[2], color=color_gromos, ls='-', alpha=1, label='GROMOS vs MLS convolved')
    rel_diff_somora_convolved.mean(dim='time').plot(y='o3_p', ax=axs[2], color=color_somora, ls='-', alpha=1, label='SOMORA vs MLS convolved')

    axs[1].fill_betweenx(gromos_conv.o3_p, (rel_diff_gromos.mean(dim='time')-0.5*std_gromos_colloc),(rel_diff_gromos.mean(dim='time')+0.5*std_gromos_colloc), color=color_gromos, alpha=0.2)
    #axs[1].fill_betweenx(gromos_conv.o3_p, (rel_diff_gromos.mean(dim='time')-0.5*sum_std_gromos),(rel_diff_gromos.mean(dim='time')+0.5*sum_std_gromos), color='green', alpha=0.2)
    axs[1].fill_betweenx(somora_conv.o3_p, (rel_diff_somora.mean(dim='time')-0.5*std_somora_colloc),(rel_diff_somora.mean(dim='time')+0.5*std_somora_colloc), color=color_somora, alpha=0.2)
    axs[1].text(
        0.7,
        0.13,
        '{:.0f} profiles'.format(len(rel_diff_gromos.time)),
        transform=axs[1].transAxes,
        verticalalignment="bottom",
        horizontalalignment="left",
        fontsize=fs,
        color=color_gromos
    )
    axs[1].text(
        0.7,
        0.1,
        '{:.0f} profiles'.format(len(rel_diff_somora.time)),
        transform=axs[1].transAxes,
        verticalalignment="bottom",
        horizontalalignment="left",
        fontsize=fs,
        color=color_somora
    )
    axs[2].fill_betweenx(gromos_conv.o3_p, (rel_diff_gromos_convolved.mean(dim='time')-0.5*std_gromos_convolved),(rel_diff_gromos_convolved.mean(dim='time')+0.5*std_gromos_convolved), color=color_gromos, alpha=0.2)
    axs[2].fill_betweenx(somora_conv.o3_p, (rel_diff_somora_convolved.mean(dim='time')-0.5*std_somora_convolved),(rel_diff_somora_convolved.mean(dim='time')+0.5*std_somora_convolved), color=color_somora, alpha=0.2)
    axs[2].text(
        0.7,
        0.13,
        '{:.0f} profiles'.format(len(rel_diff_gromos_convolved.time)),
        transform=axs[2].transAxes,
        verticalalignment="bottom",
        horizontalalignment="left",
        fontsize=fs,
        color=color_gromos
    )
    axs[2].text(
        0.7,
        0.1,
        '{:.0f} profiles'.format(len(rel_diff_somora_convolved.time)),
        transform=axs[2].transAxes,
        verticalalignment="bottom",
        horizontalalignment="left",
        fontsize=fs,
        color=color_somora
    )

    y1z=1e-3*gromos_colloc.o3_z.mean(dim='time').sel(o3_p=100 ,tolerance=20,method='nearest')
    y2z=1e-3*gromos_colloc.o3_z.mean(dim='time').sel(o3_p=1e-2 ,tolerance=1,method='nearest')
    ax2 = axs[2].twinx()
    ax2.set_yticks(1e-3*gromos_colloc.o3_z.mean(dim='time')) #ax2.set_yticks(altitude)
    ax2.set_ylim(y1z,y2z)
    fmt = FormatStrFormatter("%.0f")
    loc=MultipleLocator(base=10)
    ax2.yaxis.set_major_formatter(fmt)
    ax2.yaxis.set_major_locator(loc)
    ax2.tick_params(axis='both', which='major', labelsize=fs)
    ax2.set_ylabel('Altitude [km] ', fontsize=fs)

    axs[0].invert_yaxis()
    axs[0].set_yscale('log')
    axs[0].legend(loc=1, fontsize=fs-3)
    axs[0].set_ylim(100,0.01)
    axs[0].set_xlim(-0.1,9)
    axs[0].set_ylabel('Pressure [hPa]', fontsize=fs)
    axs[0].set_xlabel(r'O$_3$ [ppmv]', fontsize=fs)
    axs[1].set_xlim(-60, 60)
    axs[1].set_ylabel('')
    axs[1].grid(axis='x', linewidth=0.5)
    axs[1].axvline(x=0,ls='-' ,color='k', lw=0.9)
    axs[1].axvline(x=-10,ls='--' ,color='k', lw=0.9)
    axs[1].axvline(x=10,ls='--' ,color='k', lw=0.9)
    axs[1].set_xlabel(r'$\Delta$O$_3$ [\%]', fontsize=fs)

    axs[1].legend(loc=1, fontsize=fs-3)
    axs[1].set_title('MWR-MLS', fontsize=fs+4)
    axs[2].legend(loc=1, fontsize=fs-3)
    axs[2].axvline(x=0,ls='-' ,color='k', lw=0.9)
    axs[2].axvline(x=-10,ls='--' ,color='k', lw=0.9)
    axs[2].axvline(x=10,ls='--' ,color='k', lw=0.9)
    axs[2].set_xlim(-60, 60)
    axs[2].set_xlabel(r'$\Delta$O$_3$ [\%]', fontsize=fs)
    axs[2].set_ylabel('')
    axs[2].set_title('MWR-MLS (convolved)', fontsize=fs+4)

    axs[2].xaxis.set_minor_locator(MultipleLocator(10))
    axs[2].xaxis.set_major_locator(MultipleLocator(30)) 
    axs[0].xaxis.set_minor_locator(MultipleLocator(1))
    axs[0].xaxis.set_major_locator(MultipleLocator(4))
    axs[1].xaxis.set_minor_locator(MultipleLocator(10))
    axs[1].xaxis.set_major_locator(MultipleLocator(30))
    for a in axs:
        a.grid(which='both', axis='y', linewidth=0.5)
        a.grid(which='both', axis='x', linewidth=0.5)
        a.tick_params(axis='both', which='major', labelsize=fs)
        a.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: '{:g}'.format(y)))
    fig.tight_layout(rect=[0, 0.01, 0.99, 1])
    fig.savefig(basefolder+'ozone_profile_MLS_comparison_'+str(year)+'.pdf', dpi=500)   


def compare_GROMORA_MLS_profiles(gromos_colloc, gromos_conv, mls_gromos_colloc, mls_gromos_conv, somora_colloc, somora_conv, mls_somora_colloc, mls_somora_conv, ds_mls,freq='2D', basefolder='', convolved=True):
    fs = 22
    year=pd.to_datetime(gromos_colloc.time.data[0]).year
    mls_somora_colloc['time']= somora_colloc.time.data
    mls_gromos_colloc['time']= gromos_colloc.time.data
    rel_diff_somora = 100*(somora_colloc.o3_x - mls_somora_colloc.o3_x.interp_like(somora_colloc.o3_p))/somora_colloc.o3_x
    rel_diff_somora_conv = 100*(somora_conv.o3_x - mls_somora_conv.o3_x)/somora_conv.o3_x
    rel_diff_gromos = 100*(gromos_colloc.o3_x - mls_gromos_colloc.o3_x.interp_like(gromos_colloc.o3_p))/gromos_colloc.o3_x
    rel_diff_gromos_conv = 100*(gromos_conv.o3_x - mls_gromos_conv.o3_x)/gromos_conv.o3_x
    figures =list()
    # rel_diff_gromos = 100*(ozone_gromos.mean(dim='time') - convolved_MLS_gromos.o3_x.mean(dim='time'))/ozone_gromos.mean(dim='time')
    # rel_diff_gromora = 100*(ozone_gromos.mean(dim='time') - ozone_somora.mean(dim='time'))/ozone_gromos.mean(dim='time')
    lim=60
    fig1, axs1 = plt.subplots(nrows=2, ncols=1, sharey=True, figsize=(20, 12))
    rel_diff_somora.resample(time=freq).mean().plot(
        x='time',
        y='o3_p',
        ax=axs1[0],
        vmin=-lim,
        vmax=lim,
        yscale='log',
        linewidth=0,
        rasterized=True,
        cmap='coolwarm'
    )
    rel_diff_somora_conv.resample(time=freq).mean().plot(
        x='time',
        y='o3_p',
        ax=axs1[1],
        vmin=-lim,
        vmax=lim,
        yscale='log',
        linewidth=0,
        rasterized=True,
        cmap='coolwarm'
    )
    axs1[0].invert_yaxis()
    axs1[0].set_ylim(100, 1e-2)
    axs1[0].set_xlabel('')
    axs1[1].set_ylabel('Pressure [hPa]', fontsize=fs)
    axs1[0].set_ylabel('Pressure [hPa]', fontsize=fs)
    axs1[0].set_title('SOMORA vs MLS, collocated')
    axs1[1].set_title('SOMORA vs MLS, convolved')
    fig1.tight_layout(rect=[0, 0.01, 0.99, 1])
    figures.append(fig1)

    fig1, axs1 = plt.subplots(nrows=2, ncols=1, sharey=True, figsize=(20, 12))
    rel_diff_gromos.resample(time=freq).mean().plot(
        x='time',
        y='o3_p',
        ax=axs1[0],
        vmin=-lim,
        vmax=lim,
        yscale='log',
        linewidth=0,
        rasterized=True,
        cmap='coolwarm'
    )
    rel_diff_gromos_conv.resample(time=freq).mean().plot(
        x='time',
        y='o3_p',
        ax=axs1[1],
        vmin=-lim,
        vmax=lim,
        yscale='log',
        linewidth=0,
        rasterized=True,
        cmap='coolwarm'
    )
    axs1[0].invert_yaxis()
    axs1[0].set_xlabel('')
    axs1[0].set_ylim(100, 1e-2)
    axs1[1].set_ylabel('Pressure [hPa]', fontsize=fs)
    axs1[0].set_ylabel('Pressure [hPa]', fontsize=fs)
    axs1[0].set_title('GROMOS vs MLS, collocated')
    axs1[1].set_title('GROMOS vs MLS, convolved')
    fig1.tight_layout(rect=[0, 0.01, 0.99, 1])

    figures.append(fig1)

    if convolved:
        ozone_gromos = gromos_conv.o3_x
        ozone_somora = somora_conv.o3_x
        convolved_MLS_gromos = mls_gromos_conv
        convolved_MLS_somora=mls_somora_conv
        error_gromos = 1e6*gromos_conv.mean(dim='time').o3_eo
        error_somora = 1e6*somora_conv.mean(dim='time').o3_eo
    else:
        ozone_gromos = gromos_colloc.o3_x
        ozone_somora = somora_colloc.o3_x
        convolved_MLS_gromos = mls_gromos_colloc.interp_like(gromos_colloc.o3_p)
        convolved_MLS_somora = mls_somora_colloc.interp_like(somora_colloc.o3_p)
        error_gromos = 1e6*gromos_colloc.mean(dim='time').o3_eo
        error_somora = 1e6*somora_colloc.mean(dim='time').o3_eo
    # if convolved:
    #     std_diff_somora = 100*((ozone_somora-convolved_MLS_somora.o3_x)/ozone_somora.mean(dim='time')).std(dim='time') # .plot.hist(bins=np.arange(-5,5,0.1) )
    #     std_diff_gromos = 100*(ozone_somora-convolved_MLS_somora).o3_x.std(dim='time')/ozone_gromos.mean(dim='time') # .plot.hist(bins=np.arange(-5,5,0.1) )

    rel_diff_somora = 100*(ozone_somora.mean(dim='time') - convolved_MLS_somora.o3_x.mean(dim='time'))/ozone_somora.mean(dim='time')
    rel_diff_gromos = 100*(ozone_gromos.mean(dim='time') - convolved_MLS_gromos.o3_x.mean(dim='time'))/ozone_gromos.mean(dim='time')
    rel_diff_gromora = 100*(ozone_gromos.mean(dim='time') - ozone_somora.mean(dim='time'))/ozone_gromos.mean(dim='time')
    # # error_gromos = 1e6*np.sqrt(gromos_sel.mean(dim='time').o3_eo**2 + gromos_sel.mean(dim='time').o3_es**2)
    # # error_somora = 1e6*np.sqrt(somora_sel.mean(dim='time').o3_eo**2 + somora_sel.mean(dim='time').o3_es**2)

    # color_shading = 'grey'

    # mr_somora = somora_sel.o3_mr.data
    # mr_gromos = gromos_sel.o3_mr.data
    # p_somora_mr = somora_sel.o3_p.data[np.mean(mr_somora,0)>=0.8]
    # p_gromos_mr = gromos_sel.o3_p.data[np.mean(mr_gromos,0)>=0.8]
    fig, axs = plt.subplots(nrows=1, ncols=2, sharey=True, figsize=(12, 20))

    ozone_somora.mean(dim='time').plot(
        y='o3_p', ax=axs[0], color=color_somora, ls='-', label='SOMORA')
    ozone_gromos.mean(dim='time').plot(
        y='o3_p', ax=axs[0], color=color_gromos, ls='-', label='GROMOS')


    axs[0].fill_betweenx(ozone_gromos.o3_p, (ozone_gromos.mean(dim='time')-error_gromos),(ozone_gromos.mean(dim='time')+error_gromos), color=color_gromos, alpha=0.3)
    axs[0].fill_betweenx(ozone_somora.o3_p, (ozone_somora.mean(dim='time')-error_somora),(ozone_somora.mean(dim='time')+error_somora), color=color_somora, alpha=0.3)

    convolved_MLS_gromos.o3_x.mean(dim='time').plot(y='o3_p', ax=axs[0], linestyle=':', color=color_gromos, label='MLS convolved GROMOS')
    convolved_MLS_somora.o3_x.mean(dim='time').plot(y='o3_p', ax=axs[0], linestyle=':', color=color_somora, label='MLS convolved SOMORA')

    ds_mls.o3.mean(dim='time').plot(y='p', ax=axs[0], color='k', label='MLS')
    
    # for ax in axs:
    #     ax.axhline(y=p_somora_mr[0],ls='--' ,color=color_somora, lw=1)
    #     ax.axhline(y=p_somora_mr[-1],ls='--', color=color_somora, lw=1)
    #     ax.axhline(y=p_gromos_mr[0],ls=':', color=color_gromos, lw=1)
    #     ax.axhline(y=p_gromos_mr[-1],ls=':', color=color_gromos, lw=1)

    axs[0].invert_yaxis()
    axs[0].set_xlim(-0.2, 9)
    axs[0].set_yscale('log')
    axs[0].legend()
    axs[0].set_ylim(200, 5e-3)
    axs[0].set_ylabel('Pressure [hPa]', fontsize=fs)
    axs[0].set_xlabel(r'O$_3$ VMR [ppmv]', fontsize=fs)
    axs[0].grid(axis='x', linewidth=0.5)
    # axs[0].fill_between(axs[0].get_xlim(),p_somora_mr[0],1e4, color=color_shading, alpha=0.2)
    # axs[0].fill_between(axs[0].get_xlim(),p_somora_mr[-1],1e-4, color=color_shading, alpha=0.2)
    # axs[0].fill_between(axs[0].get_xlim(),p_gromos_mr[0],1e4, color=color_shading, alpha=0.2)
    # axs[0].fill_between(axs[0].get_xlim(),p_gromos_mr[-1],1e-4, color=color_shading, alpha=0.2)

    rel_diff_gromos.plot(y='o3_p', ax=axs[1], color=color_gromos,
                  ls='-', alpha=1, label='GROMOS vs MLS')
    rel_diff_somora.plot(y='o3_p', ax=axs[1], color=color_somora,
                  ls='-', alpha=1, label='SOMORA vs MLS')
    rel_diff_gromora.plot(y='o3_p', ax=axs[1], color='k',
                  ls='-', alpha=1, label='GROMOS vs SOMORA')
    # axs[1].fill_betweenx(gromos_sel.o3_p, (rel_diff_gromos-100*error_gromos/ozone_gromos.mean(dim='time')),(rel_diff_gromos+100*error_gromos/ozone_gromos.mean(dim='time')), color=color_gromos, alpha=0.3)
    # axs[1].fill_betweenx(somora_sel.o3_p, (rel_diff_somora-100*error_somora/ozone_somora.mean(dim='time')),(rel_diff_somora+100*error_somora/ozone_somora.mean(dim='time')), color=color_somora, alpha=0.3)

    # if convolved:
    #     axs[1].fill_betweenx(gromos_sel.o3_p, (rel_diff_gromos-std_diff_gromos),(ozone_gromos.mean(dim='time')+error_gromos), color=color_gromos, alpha=0.3)
    #     axs[1].fill_betweenx(somora_sel.o3_p, (ozone_somora.mean(dim='time')-error_somora),(ozone_somora.mean(dim='time')+error_somora), color=color_somora, alpha=0.3)


    axs[1].set_xlim(-40, 40)
    axs[1].set_xlabel(r'$\Delta$O$_3$ [\%]', fontsize=fs)
    axs[1].legend()
    axs[1].set_ylabel('')
    axs[1].grid(axis='x', linewidth=0.5)
    # ax.axvline(x=0,ls='--' ,color='k', lw=0.5)
    # axs[1].fill_between(axs[1].get_xlim(), p_somora_mr[0],1e4, color=color_shading, alpha=0.2)
    # axs[1].fill_between(axs[1].get_xlim(),p_somora_mr[-1],1e-4, color=color_shading, alpha=0.2)
    # axs[1].fill_between(axs[1].get_xlim(),p_gromos_mr[0],1e4, color=color_shading, alpha=0.2)
    # axs[1].fill_between(axs[1].get_xlim(),p_gromos_mr[-1],1e-4, color=color_shading, alpha=0.2)

    axs[0].xaxis.set_minor_locator(MultipleLocator(1))
    axs[0].xaxis.set_major_locator(MultipleLocator(4))
    axs[1].xaxis.set_minor_locator(MultipleLocator(10))
    axs[1].xaxis.set_major_locator(MultipleLocator(20))
    
    for a in axs:
        a.grid(which='both', axis='y', linewidth=0.5)
        a.grid(which='both', axis='x', linewidth=0.5)
        a.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: '{:g}'.format(y)))
    # plt.suptitle('Ozone comparison with ' + str(len(date)) + ' days ' +
    #              pd.to_datetime(ozone_somora.time.mean().data).strftime('%Y-%m-%d %H:%M'))
    
    fig.tight_layout(rect=[0, 0.01, 0.99, 1])
    figures.append(fig)
    if convolved:
        save_single_pdf(basefolder+'ozone_ts_MLS_convolved_'+str(year)+'.pdf', figures)
       #  fig.savefig(basefolder+'ozone_profile_MLS_convolved_'+str(year)+'.pdf', dpi=500)
    else:
        save_single_pdf(basefolder+'ozone_ts_MLS_collocated_'+str(year)+'.pdf', figures)

       # fig.savefig(basefolder+'ozone_profile_MLS_collocated_'+str(year)+'.pdf', dpi=500)

def read_all_MLS(yrs):
   # ds_mls_v5 = read_MLS(timerange = time_period, vers=5)

    gromos_sel_colloc, gromos_sel_convolved, mls_gromos_coloc, convolved_MLS_GROMOS = read_MLS_convolved(
        instrument_name='GROMOS', 
        folder='/scratch/GROSOM/Level2/MLS/', 
        years=yrs
        )
    somora_sel_colloc, somora_sel_convolved, mls_somora_coloc, convolved_MLS_SOMORA = read_MLS_convolved(
        instrument_name='SOMORA', 
        folder='/scratch/GROSOM/Level2/MLS/', 
        years=yrs
        )

    # gromos_sel, mls_gromos_colloc=select_gromora_corresponding_mls(gromos, convolved_MLS_GROMOS, time_period)
    # somora_sel, mls_somora_colloc=select_gromora_corresponding_mls(somora, convolved_MLS_SOMORA, time_period)

    # gromos_sel['o3'] = gromos_sel.o3_x
    # somora_sel['o3'] = somora_sel.o3_x
    # convolved_MLS_GROMOS['o3'] = convolved_MLS_GROMOS.o3_x
    # convolved_MLS_SOMORA['o3'] = convolved_MLS_SOMORA.o3_x
    # compare_ts_MLS(gromos_sel, mls_gromos_colloc, date_slice=plot_period, freq='1H', basefolder='/scratch/GROSOM/Level2/MLS/', ds_mls=ds_mls_v5)

    return gromos_sel_colloc, gromos_sel_convolved, mls_gromos_coloc, convolved_MLS_GROMOS, somora_sel_colloc, somora_sel_convolved, mls_somora_coloc, convolved_MLS_SOMORA

def compute_correlation_MLS(gromos, mls, freq='1M', pressure_level = [15,20,25], basefolder=''):
    year=pd.to_datetime(gromos.time.data[0]).year
    fs=24
    
    fig, axs = plt.subplots(len(pressure_level),1,figsize=(6, 6*len(pressure_level)))
    error_gromos = np.sqrt( np.square(gromos.o3_eo) + np.square(gromos.o3_es))
    error_gromos = gromos.o3_x.std(dim='time')
    
    error_mls = 1e-6*0.05*mls.o3_x
    error_mls = mls.o3_x.std(dim='time')

    if freq == 'OG':   
        mls['time'] = gromos.time.data

        ds_o3_gromora=xr.merge((
            {'o3_gromos':gromos.o3_x},
            {'o3_mls':mls.o3_x},
            {'error_gromos':error_gromos},
            {'error_mls':error_mls},
            ))
    else:
        ds_o3_gromora=xr.merge((
            {'o3_gromos':gromos.o3_x.resample(time=freq).mean()},
            {'o3_mls':mls.o3_x.resample(time=freq).mean()},
            {'error_gromos':error_gromos},
            {'error_mls':error_mls},
            ))
    maxit=10
    
    for i, p in enumerate(pressure_level): 
        reduced_chi2 =10
        counter=0
        ds_o3_p = ds_o3_gromora.isel(o3_p=p).where(ds_o3_gromora.isel(o3_p=p).o3_gromos.notnull() & ds_o3_gromora.isel(o3_p=p).o3_mls.notnull(), drop=True)
        x = ds_o3_p.o3_gromos#  - ds_o3_p.o3_gromos.mean(dim='time')
        y = ds_o3_p.o3_mls# - ds_o3_p.o3_gromos.mean(dim='time')
        pearson_corr = xr.corr(x,y, dim='time')
        print('Pearson corr coef: ', pearson_corr.values)
        xerr =ds_o3_p.error_gromos
        yerr = ds_o3_p.error_mls
        while (reduced_chi2 > 1.5) and (counter < maxit):
            result, chi2 = regression_xy(#stats.linregress(
                x.values, y.values, x_err = xerr.values, y_err=yerr.values
            )
            reduced_chi2 = chi2/(len(x.values)-2)
            xerr = 1.5*xerr
            yerr = 1.5*yerr
            print('Reduced chi2: ',chi2/(len(x.values)-2))
            print('Sum of square ', result.sum_square)

            counter=counter+1

            print('############################')
        # print('From linregress')
        print('Slope ', result.beta[0], ' +- ', result.sd_beta[0] )
        print('Intercept ', result.beta[1], ' +- ', result.sd_beta[1] )
        # linregress_result = stats.linregress(x, y)
        # print(f"R-squared: {linregress_result.rvalue**2:.6f}")
        # print('pvalue: ', linregress_result.pvalue)
        # print('Slope ', linregress_result.slope, ' +- ', linregress_result.stderr)
       # print('Intercept ', linregress_result.intercept, ' +- ',linregress_result.intercept_stderr )

       # coeff_determination = calcR2_wikipedia(y.values, result.beta[1] + result.beta[0] * x.values)
       # coeff_determination = coefficient_determination(y.values, result.beta[1] + result.beta[0] * x.values)

       # print('r2: ',result.rvalue**2)
        print('R2: ',np.corrcoef(x, y)[0, 1]**2)
        ds_o3_p.plot.scatter(
            ax=axs[i],
            x='o3_gromos', 
            y='o3_mls',
            color='k',
            marker='.'
        )
        axs[i].plot([np.nanmin(x.values),np.nanmax(x.values)],[np.nanmin(y.values), np.nanmax(y)],'k--')
        
       # axs[i].errorbar(x, y, xerr=xerr, yerr=yerr, color='k', linestyle='None', marker='.') 
        axs[i].plot(x,y, '.', color='k') 
        axs[i].plot(x, result.beta[1]  + result.beta[0] * x, color='red') 
        axs[i].set_xlabel(r'GROMOS O$_3$ [ppmv]', fontsize=fs-2)
        
        axs[i].set_ylabel(r'MLS O$_3$ [ppmv]', fontsize=fs-2)

        axs[i].set_title(r'O$_3$ VMR '+f'at p = {gromos.o3_p.data[p]:.3f} hPa')
        axs[i].set_xlim(np.nanmin(x.values),np.nanmax(x.values))
        axs[i].set_ylim(np.nanmin(x.values),np.nanmax(x.values))
        axs[i].xaxis.set_major_locator(MultipleLocator(1))
        axs[i].xaxis.set_minor_locator(MultipleLocator(0.5))
        axs[i].yaxis.set_major_locator(MultipleLocator(1))
        axs[i].yaxis.set_minor_locator(MultipleLocator(0.5))
        print('#########################################################')
        print('#########################################################')
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
    fig.savefig(basefolder+'ozone_scatter_MLS'+freq+'_'+str(year)+'.pdf', dpi=500)

def test_plevel(p, time_period, gromos ,gromos_sel, ds_mls_v5, mls_gromos_colloc, mls_gromos_colloc_convolved):
    fig, axs = plt.subplots(1,1,figsize=(12, 12))
    gromos.sel(time= time_period).sel(o3_p=p, method='nearest').o3_x.plot(ax=axs, marker='x', label='gromora')
    gromos_sel.sel(time= time_period).sel(o3_p=p, method='nearest').o3_x.plot(marker='^', lw=0, ax=axs, label='gromora colloc')
    ds_mls_v5.sel(time=time_period).sel(p=p, method='nearest').o3.plot(ax=axs,marker='x', label='mls')
    mls_gromos_colloc.sel(time=time_period).sel(o3_p=p, method='nearest').o3_x.plot(marker='s', linewidth=0, ax=axs, label='mls colloc')
    mls_gromos_colloc_convolved.sel(time=time_period).sel(o3_p=p, method='nearest').o3_x.plot(marker='d', linewidth=0, ax=axs, label='mls conv')
    axs.legend()

def process_FFT(yrs, time_period):
    gromos = read_GROMORA_all(basefolder='/storage/tub/instruments/gromos/level2/GROMORA/v2/', 
    instrument_name='GROMOS',
    date_slice=time_period, 
    years=yrs,
    prefix='_v2.nc',
    flagged=False
    )
    somora = read_GROMORA_all(basefolder='/storage/tub/instruments/somora/level2/v2/', 
    instrument_name='SOMORA',
    date_slice=time_period, 
    years=yrs,
    prefix='_v2.nc',
    flagged=False
    )

    print('Corrupted retrievals GROMOS : ',len(gromos['o3_x'].where((gromos['o3_x']<0), drop = True))+ len(gromos['o3_x'].where((gromos['o3_x']>1e-5), drop = True))) 
    print('Corrupted retrievals SOMORA : ',len(somora['o3_x'].where((somora['o3_x']<0), drop = True))+ len(somora['o3_x'].where((somora['o3_x']>1e-5), drop = True))) 
    # gromos = gromos.drop(['y', 'yf', 'bad_channels','y_baseline', 'f'] )
    # somora = somora.drop(['y', 'yf', 'bad_channels','y_baseline', 'f'] )
    #gromos['o3_x'] = 1e6*gromos['o3_x'].where((gromos['o3_x']>0)&(gromos['o3_x']<1e-5), drop = True)
    #somora['o3_x'] = 1e6*somora['o3_x'].where((somora['o3_x']>0)&(somora['o3_x']<1e-5), drop = True)
    gromos['o3_x'] = 1e6*gromos['o3_x'].where((gromos['o3_x']>gromos['o3_x'].valid_min)&(gromos['o3_x']<gromos['o3_x'].valid_max), drop = True)
    somora['o3_x'] = 1e6*somora['o3_x'].where((somora['o3_x']>somora['o3_x'].valid_min)&(somora['o3_x']<somora['o3_x'].valid_max), drop = True)
    
    gromos = add_flags_level2_gromora(gromos, 'GROMOS')
    somora = add_flags_level2_gromora(somora, 'SOMORA')

    outfolder = '/scratch/GROSOM/Level2/GROMORA_retrievals_v2/'
    # gromos = gromos.drop(['y', 'yf', 'bad_channels','y_baseline', 'f'] )
    # somora = somora.drop(['y', 'yf', 'bad_channels','y_baseline', 'f'] )

    gromos_clean = gromos.where(gromos.retrieval_quality==1, drop=True).where(gromos.level2_flag==0, drop=True)#.where(~gromos.o3_x.mean('o3_p').isnull(), drop=True)#.where(gromos.o3_mr>0.8)
    somora_clean = somora.where(somora.retrieval_quality==1, drop=True).where(somora.level2_flag==0, drop=True)#.where(~somora.o3_x.mean('o3_p').isnull(), drop=True)#.where(somora.level2_flag==0, drop=True)#.where(gromos.o3_mr>0.8)
    gromos_clean=gromos_clean.where(gromos_clean.o3_avkm.mean(dim='o3_p_avk')<1, drop=True).where(gromos_clean.o3_avkm.mean(dim='o3_p_avk')>-1, drop=True).where(gromos_clean.o3_xa.mean(dim='o3_p')<5e-5, drop=True).where(gromos_clean.o3_xa.mean(dim='o3_p')>-1e-5, drop=True)
    somora_clean=somora_clean.where(somora_clean.o3_avkm.mean(dim='o3_p_avk')<1, drop=True).where(somora_clean.o3_avkm.mean(dim='o3_p_avk')>-1, drop=True).where(somora_clean.o3_xa.mean(dim='o3_p')<5e-5, drop=True).where(somora_clean.o3_xa.mean(dim='o3_p')>-1e-5, drop=True)

    ds_mls_v5 = read_MLS(timerange = time_period, vers=5, filename_MLS='AuraMLS_L2GP-O3_v5_400-800_BERN.nc')

    for yr in yrs:
        plot_period = slice(str(yr)+"-01-01", str(yr)+"-12-31")
        print('Processing year '+str(yr))
       # plot_period = time_period
      #  ds_mls = read_MLS(timerange = plot_period, vers=4.2)
        
    
     #   monthly_mls = ds_mls.o3.resample(time='1D', skipna=True).mean()
    #    compare_MLS(ds_mls.o3, ds_mls_v5.o3)
        
       # gromora_sel, mls_sel = select_gromora_corresponding_mls(gromos, ds_mls_v5)

        #plot_gromora_and_corresponding_MLS(gromora_sel, mls_sel)
    
      #  gromos_sel, convolved_MLS_GROMOS = avk_smooth_mls(gromos_clean, ds_mls_v5, folder='/scratch/GROSOM/Level2/MLS/'+'GROMOS_')
      #   somora_sel, convolved_MLS_SOMORA = avk_smooth_mls(somora_clean, ds_mls_v5, folder='/scratch/GROSOM/Level2/MLS/'+'SOMORA_')

        # gromos_sel, mls_gromos_colloc, mls_gromos_colloc_convolved_new = read_MLS_convolved(
        #     instrument_name='GROMOS', 
        #     folder='/scratch/GROSOM/Level2/MLS/', 
        #     years=[yr]
        #     )
        # somora_sel, mls_somora_colloc, mls_somora_colloc_convolved_new = read_MLS_convolved(
        #     instrument_name='SOMORA', 
        #     folder='/scratch/GROSOM/Level2/MLS/', 
        #     years=[yr]
        #     )

    #     plot_gromora_and_corresponding_MLS(gromos_sel, ds_mls_v5, mls_gromos_colloc_convolved_new,freq='1D', basename='GROMOS_read_daily')
    #     plot_gromora_and_corresponding_MLS(somora_sel, ds_mls_v5, mls_somora_colloc_convolved_new,freq='1D', basename='SOMORA_read_daily')
    #    # test_plevel(1, time_period, gromos ,gromos_sel, ds_mls_v5, mls_gromos_colloc, mls_somora_colloc_convolved_new)
        
        gromos_sel, mls_gromos_colloc=select_gromora_corresponding_mls(
            gromos_clean, 
            'GROMOS',
            ds_mls_v5, 
            plot_period, 
            save_ds=True,
            convolved=False,
            basename='GROMOS_collocation_'
            )

        gromos_sel, mls_gromos_colloc_convolved_new  = avk_smooth_mls_new(
            gromos_sel, mls_gromos_colloc, 'GROMOS',basefolder='/scratch/GROSOM/Level2/MLS/', sel=True, save_ds=True
        )

        somora_sel, mls_somora_colloc=select_gromora_corresponding_mls(
            somora, 
            'SOMORA',
            ds_mls_v5, 
            plot_period, 
            save_ds=True,
            convolved=False,
            basename='SOMORA_collocation_'
            )

        somora_sel, mls_somora_colloc_convolved_new  = avk_smooth_mls_new(
            somora_sel, mls_somora_colloc, 'SOMORA',basefolder='/scratch/GROSOM/Level2/MLS/', sel=True, save_ds=True
        )


    #     somora_sel, mls_somora_colloc=select_gromora_corresponding_mls(somora, convolved_MLS_SOMORA, time_period)
        plot_gromora_and_corresponding_MLS(somora_sel, ds_mls_v5.sel(time=plot_period), mls_somora_colloc_convolved_new, freq='2D', basename='SOMORA_2daily')
        plot_gromora_and_corresponding_MLS(gromos_sel, ds_mls_v5.sel(time=plot_period), mls_gromos_colloc_convolved_new, freq='2D', basename='GROMOS_2daily')
    #     # gromos_sel['o3'] = gromos_sel.o3_x
    #     # somora_sel['o3'] = somora_sel.o3_x
    #     # convolved_MLS_GROMOS['o3'] = convolved_MLS_GROMOS.o3_x
    #     # convolved_MLS_SOMORA['o3'] = convolved_MLS_SOMORA.o3_x

    # #     compare_ts_MLS(gromos_sel, mls_gromos_colloc, date_slice=plot_period, freq='1H', basefolder='/scratch/GROSOM/Level2/MLS/', ds_mls=ds_mls_v5)
    #     compare_GROMORA_MLS_profiles(gromos_sel, somora_sel, mls_gromos_colloc_convolved_new, mls_somora_colloc_convolved_new, ds_mls_v5,basefolder='/scratch/GROSOM/Level2/GROMORA_waccm/')
    #    # compute_corr_profile(somora_sel,mls_somora_colloc_convolved_new,freq='1D',basefolder='/scratch/GROSOM/Level2/MLS/')
        mls_gromos_colloc_convolved_new['time'] = gromos_sel.time.data
        #compute_correlation_MLS(gromos_sel, mls_gromos_colloc_convolved_new, freq='OG', pressure_level=[15,20,25], basefolder='/scratch/GROSOM/Level2/MLS/GROMOS_')
        mls_somora_colloc_convolved_new['time'] = somora_sel.time.data
        #compute_correlation_MLS(somora_sel, mls_somora_colloc_convolved_new, freq='OG', pressure_level=[15,20,25], basefolder='/scratch/GROSOM/Level2/MLS/SOMORA_')
    #     compute_correlation(somora_sel, mls_somora_colloc_convolved_new, freq='1D', pressure_level=[15,20,25], basefolder='/scratch/GROSOM/Level2/MLS/SOMORA_', MLS=True)
    # #    # plot_gromora_and_corresponding_MLS(new_gromos, mls_gromos_colloc)

   #  compare_pressure(gromos_sel, mls_somora_colloc, pressure_level=[31, 25, 21, 15, 12], add_sun=False, freq='1D', basefolder='/scratch/GROSOM/Level2/GROMORA_waccm/')


def process_FB_GROMOS(yrs, time_period):
    gromos = read_GROMORA_all(basefolder='/storage/tub/instruments/gromos/level2/GROMORA/v2/', 
    instrument_name='GROMOS',
    date_slice=time_period, 
    years=yrs,
    prefix='_v2.nc',
    flagged=False,
    decode_time = False
    )

    print('Corrupted retrievals GROMOS : ',len(gromos['o3_x'].where((gromos['o3_x']<0), drop = True))+ len(gromos['o3_x'].where((gromos['o3_x']>1e-5), drop = True))) 
    # gromos = gromos.drop(['y', 'yf', 'bad_channels','y_baseline', 'f'] )
    # somora = somora.drop(['y', 'yf', 'bad_channels','y_baseline', 'f'] )
    #gromos['o3_x'] = 1e6*gromos['o3_x'].where((gromos['o3_x']>0)&(gromos['o3_x']<1e-5), drop = True)
    #somora['o3_x'] = 1e6*somora['o3_x'].where((somora['o3_x']>0)&(somora['o3_x']<1e-5), drop = True)
    gromos['o3_x'] = 1e6*gromos['o3_x'].where((gromos['o3_x']>gromos['o3_x'].valid_min)&(gromos['o3_x']<gromos['o3_x'].valid_max), drop = True)
    
    gromos = add_flags_level2_gromora(gromos, 'GROMOS')

    gromos_clean = gromos.where(gromos.retrieval_quality==1, drop=True).where(gromos.level2_flag==0, drop=True)#.where(~gromos.o3_x.mean('o3_p').isnull(), drop=True)#.where(gromos.o3_mr>0.8)
    gromos_clean=gromos_clean.where(gromos_clean.o3_avkm.mean(dim='o3_p_avk')<1, drop=True).where(gromos_clean.o3_avkm.mean(dim='o3_p_avk')>-1, drop=True).where(gromos_clean.o3_xa.mean(dim='o3_p')<5e-5, drop=True).where(gromos_clean.o3_xa.mean(dim='o3_p')>-1e-5, drop=True)

    ds_mls_v5 = read_MLS(timerange = time_period, vers=5, filename_MLS='AuraMLS_L2GP-O3_v5_400-800_BERN_2004-2022.nc')

    for yr in yrs:
        plot_period = slice(str(yr)+"-01-01", str(yr)+"-12-31")
        print('Processing year '+str(yr))

        gromos_sel, mls_gromos_colloc=select_gromora_corresponding_mls(
            gromos_clean, 
            'GROMOS',
            ds_mls_v5, 
            plot_period, 
            save_ds=True,
            convolved=False,
            basename='GROMOS_collocation_'
            )

        gromos_sel, mls_gromos_colloc_convolved_new  = avk_smooth_mls_new(
            gromos_sel, mls_gromos_colloc, 'GROMOS',basefolder='/scratch/GROSOM/Level2/MLS/', sel=True, save_ds=True
        )

        plot_gromora_and_corresponding_MLS(gromos_sel, ds_mls_v5.sel(time=plot_period), mls_gromos_colloc_convolved_new, freq='2D', basename='GROMOS_2daily')


if __name__ == "__main__":
    time_period = slice("2006-01-01", "2009-12-31")
    yrs = [2006, 2007,2008]#,2019[2009,2010,2011,2012,2013,2014,2015,2016,2017,2018,2019,2020,2021,]
    process_FB_GROMOS(yrs, time_period)