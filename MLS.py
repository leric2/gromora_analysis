#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import datetime
import os

import matplotlib.pyplot as plt
import netCDF4
import numpy as np
import pandas as pd
import scipy.io

import xarray as xr
import matplotlib.ticker as ticker

from matplotlib.ticker import (MultipleLocator, FormatStrFormatter, AutoMinorLocator)

from typhon.collocations import Collocator
from level2_gromora import *

colormap = 'cividis'

# color_gromos = '#d95f02'
# color_somora = '#1b9e77'

def read_MLS_convolved(instrument_name='GROMOS', folder='/scratch/GROSOM/Level2/MLS/', years=[2018]):
    ds_colloc=xr.Dataset()
    ds_mls_conv=xr.Dataset()
    gromora_sel=xr.Dataset()
    counter=0
    for yr in years:
        filename_gromora = instrument_name + '_convolved_'+str(yr)+'.nc'
        filename_colloc = instrument_name + '_collocation_MLS_'+str(yr)+'.nc'
        filename_convolved_mls = instrument_name + '_convolved_MLS_'+str(yr)+'.nc'

        gromora = xr.open_dataset(os.path.join(folder, filename_gromora))
        ds_col = xr.open_dataset(os.path.join(folder, filename_colloc))
        ds_conv = xr.open_dataset(os.path.join(folder, filename_convolved_mls))
        if counter==0:
            ds_colloc=xr.merge([ds_colloc, ds_col])
            ds_mls_conv=xr.merge([ds_mls_conv, ds_conv] )
            gromora_sel=xr.merge([gromora_sel, gromora]    )
            counter=counter+1
        else:
            ds_colloc=xr.concat([ds_colloc, ds_col], dim='time')
            ds_mls_conv=xr.concat([ds_mls_conv, ds_conv] , dim='time')
            gromora_sel=xr.concat([gromora_sel, gromora] , dim='time'   )
        
    return gromora_sel, ds_colloc, ds_mls_conv


def read_MLS(timerange, vers=5):
    if vers == 5:
        MLS_basename = '/storage/tub/atmosphere/AuraMLS/Level2_v5/locations/'
        filename_MLS = 'AuraMLS_L2GP-O3_v5.nc'
    
        ds_mls = xr.open_dataset(os.path.join(MLS_basename, filename_MLS))
        ds_mls.attrs['history']=''
        ds_mls = ds_mls.rename({'value':'o3', 'pressure':'p'})
        ds_mls['o3'] =  ds_mls['o3']*1e6
        ds_mls['p'] =  ds_mls['p']/100
    else:
        MLS_basename = '/home/esauvageat/Documents/AuraMLS/'
        filename_MLS = os.path.join(MLS_basename, 'aura-ozone-at-Bern.mat')
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

def select_gromora_corresponding_mls(gromora, ds_mls, time_period, save_ds=False, basename='/scratch/', convolved=True):
    time_mls = pd.to_datetime(ds_mls.sel(time=time_period).time.data)
    time_gromora = pd.to_datetime(gromora.sel(time=time_period).time.data)
    new_ds = xr.Dataset()
    new_gromora = gromora.sel(time=time_period)
    new_time_list = list()
    mls_colloc_list = list()

    if convolved:
        p_name = 'o3_p'
        o3_name = 'o3_x'
    else:
        p_name = 'p'
        o3_name = 'o3'

    counter = 0
    for i, t in enumerate(time_gromora):
        range = slice(t-datetime.timedelta(minutes=30), t+datetime.timedelta(minutes=30))
        #try
        #mls_colloc = ds_mls.sel(time=t, method='nearest', tolerance='30M')
        mls_colloc = ds_mls.sel(time=range)
        
        if len(mls_colloc.time)>1:
            #print(i)
            #mls_colloc = mls_colloc.mean(dim='time')
            mls_colloc_list.append(mls_colloc[o3_name].mean(dim='time').values)
            new_time_list.append(mls_colloc.time.mean(dim='time').values)
            # gromora_sel = gromora.sel(time=t, method='nearest', tolerance='2H')
            #if counter == 0:
                #new_ds = mls_colloc
                #counter = counter+1
            #else:
                #new_ds=xr.concat([new_ds, mls_colloc], dim='time')
        elif len(mls_colloc.time)==1:
            #print(i)
            mls_colloc_list.append(mls_colloc[o3_name].mean(dim='time').values)
            new_time_list.append(mls_colloc.time.values[0])
        else:
            new_gromora = new_gromora.drop_sel(time=t)
        #except:
         #   new_gromora = new_gromora.drop_sel(time=t)

    new_ds = xr.Dataset(
        data_vars=dict(
            o3_x=(['time','o3_p'], mls_colloc_list)
        ),
        coords=dict(
            time=new_time_list,
            o3_p=ds_mls[p_name].data
        ),
        attrs=dict(description='Collocated ozone time series at bern')
    )
    # new_ds.o3_x.isel(o3_p=12).resample(time='1D').mean().plot()
    # new_mls.o3.isel(p=12).resample(time='1D').mean().plot()
    if save_ds:
        if convolved:
            new_ds.to_netcdf('/scratch/GROSOM/Level2/MLS/'+basename+'MLS_'+t.strftime('%Y')+'.nc')
           # new_gromora.to_netcdf('/scratch/GROSOM/Level2/MLS/'+basename+t.strftime('%Y')+'.nc')
        else:
            new_ds.to_netcdf('/scratch/GROSOM/Level2/MLS/'+basename+'MLS_'+t.strftime('%Y')+'.nc')
           # new_gromora.to_netcdf('/scratch/GROSOM/Level2/MLS/'+basename+t.strftime('%Y')+'.nc')

    return new_gromora, new_ds

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


def avk_smooth_mls_new(gromos_sel, mls_gromos_colloc, basename='/scratch/GROSOM/Level2/', sel=True, save_ds=False):
    time_mls = pd.to_datetime(mls_gromos_colloc.time.data)
    time_gromora = pd.to_datetime(gromos_sel.time.data)
    new_ds = xr.Dataset()
    convolved_MLS_list = []
    time_list = []
    counter = 0
    if sel:
        pname = 'o3_p'
        o3_name = 'o3_x'

  #  ds_mls = ds_mls.resample(time='12H').mean()
    for t in time_mls:
        try:
            gromora_sel = gromos_sel.sel(time=t, method='nearest', tolerance='1H')
            ds_mls_sel= mls_gromos_colloc.sel(time=t)#) , method='nearest', tolerance='2H')
            avkm = gromora_sel.o3_avkm.data

            o3_p_mls = ds_mls_sel[pname].data
            idx = np.argsort(o3_p_mls, kind='heapsort')
            o3_p_mls_sorted = o3_p_mls[idx]  
            o3_mls_sorted = ds_mls_sel[o3_name].data[idx]
    
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
    if save_ds:
        convolved_MLS.to_netcdf(basename+'MLS_'+t.strftime('%Y')+'.nc')
        new_ds.to_netcdf(basename+t.strftime('%Y')+'.nc')
    # new_ds.o3_x.isel(o3_p=12).resample(time='1D').mean().plot()
    # new_mls.o3.isel(p=12).resample(time='1D').mean().plot()
    return new_ds, convolved_MLS


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


def compare_GROMORA_MLS_profiles(gromos_sel, somora_sel, convolved_MLS_gromos, convolved_MLS_somora, ds_mls, basefolder='', convolved=True):
    fs = 22
    ozone_gromos = gromos_sel.o3_x
    ozone_somora = somora_sel.o3_x

    year=pd.to_datetime(gromos_sel.time.data[0]).year

    if not convolved:
        convolved_MLS_gromos = convolved_MLS_gromos.interp_like(somora_sel.o3_p)
        convolved_MLS_somora = convolved_MLS_somora.interp_like(somora_sel.o3_p)

    # if convolved:
    #     std_diff_somora = 100*((ozone_somora-convolved_MLS_somora.o3_x)/ozone_somora.mean(dim='time')).std(dim='time') # .plot.hist(bins=np.arange(-5,5,0.1) )
    #     std_diff_gromos = 100*(ozone_somora-convolved_MLS_somora).o3_x.std(dim='time')/ozone_gromos.mean(dim='time') # .plot.hist(bins=np.arange(-5,5,0.1) )

    rel_diff_somora = 100*(ozone_somora.mean(dim='time') - convolved_MLS_somora.o3_x.mean(dim='time'))/ozone_somora.mean(dim='time')
    rel_diff_gromos = 100*(ozone_gromos.mean(dim='time') - convolved_MLS_gromos.o3_x.mean(dim='time'))/ozone_gromos.mean(dim='time')
    rel_diff_gromora = 100*(ozone_gromos.mean(dim='time') - ozone_somora.mean(dim='time'))/ozone_gromos.mean(dim='time')
    
    # error_gromos = 1e6*np.sqrt(gromos_sel.mean(dim='time').o3_eo**2 + gromos_sel.mean(dim='time').o3_es**2)
    # error_somora = 1e6*np.sqrt(somora_sel.mean(dim='time').o3_eo**2 + somora_sel.mean(dim='time').o3_es**2)
    error_gromos = 1e6*gromos_sel.mean(dim='time').o3_eo
    error_somora = 1e6*somora_sel.mean(dim='time').o3_eo
    color_shading = 'grey'

    mr_somora = somora_sel.o3_mr.data
    mr_gromos = gromos_sel.o3_mr.data
    p_somora_mr = somora_sel.o3_p.data[np.mean(mr_somora,0)>=0.8]
    p_gromos_mr = gromos_sel.o3_p.data[np.mean(mr_gromos,0)>=0.8]

    fig, axs = plt.subplots(nrows=1, ncols=2, sharey=True, figsize=(12, 20))

    ozone_somora.mean(dim='time').plot(
        y='o3_p', ax=axs[0], color=color_somora, ls='-', label='SOMORA')
    ozone_gromos.mean(dim='time').plot(
        y='o3_p', ax=axs[0], color=color_gromos, ls='-', label='GROMOS')


    axs[0].fill_betweenx(gromos_sel.o3_p, (ozone_gromos.mean(dim='time')-error_gromos),(ozone_gromos.mean(dim='time')+error_gromos), color=color_gromos, alpha=0.3)
    axs[0].fill_betweenx(somora_sel.o3_p, (ozone_somora.mean(dim='time')-error_somora),(ozone_somora.mean(dim='time')+error_somora), color=color_somora, alpha=0.3)

    convolved_MLS_gromos.o3_x.mean(dim='time').plot(y='o3_p', ax=axs[0], linestyle=':', color=color_gromos, label='MLS convolved GROMOS')
    convolved_MLS_somora.o3_x.mean(dim='time').plot(y='o3_p', ax=axs[0], linestyle=':', color=color_somora, label='MLS convolved SOMORA')

    ds_mls.o3.mean(dim='time').plot(y='p', ax=axs[0], color='k', label='MLS')
    
    for ax in axs:
        ax.axhline(y=p_somora_mr[0],ls='--' ,color=color_somora, lw=1)
        ax.axhline(y=p_somora_mr[-1],ls='--', color=color_somora, lw=1)
        ax.axhline(y=p_gromos_mr[0],ls=':', color=color_gromos, lw=1)
        ax.axhline(y=p_gromos_mr[-1],ls=':', color=color_gromos, lw=1)

    axs[0].invert_yaxis()
    axs[0].set_xlim(-0.2, 9)
    axs[0].set_yscale('log')
    axs[0].legend()
    axs[0].set_ylim(200, 5e-3)
    axs[0].set_ylabel('Pressure [hPa]', fontsize=fs)
    axs[0].set_xlabel(r'O$_3$ VMR [ppmv]', fontsize=fs)
    axs[0].grid(axis='x', linewidth=0.5)
    axs[0].fill_between(axs[0].get_xlim(),p_somora_mr[0],1e4, color=color_shading, alpha=0.2)
    axs[0].fill_between(axs[0].get_xlim(),p_somora_mr[-1],1e-4, color=color_shading, alpha=0.2)
    axs[0].fill_between(axs[0].get_xlim(),p_gromos_mr[0],1e4, color=color_shading, alpha=0.2)
    axs[0].fill_between(axs[0].get_xlim(),p_gromos_mr[-1],1e-4, color=color_shading, alpha=0.2)

    rel_diff_gromos.plot(y='o3_p', ax=axs[1], color=color_gromos,
                  ls='-', alpha=1, label='GROMOS vs MLS')
    rel_diff_somora.plot(y='o3_p', ax=axs[1], color=color_somora,
                  ls='-', alpha=1, label='SOMORA vs MLS')
    rel_diff_gromora.plot(y='o3_p', ax=axs[1], color='k',
                  ls='-', alpha=1, label='GROMOS vs SOMORA')
    axs[1].fill_betweenx(gromos_sel.o3_p, (rel_diff_gromos-100*error_gromos/ozone_gromos.mean(dim='time')),(rel_diff_gromos+100*error_gromos/ozone_gromos.mean(dim='time')), color=color_gromos, alpha=0.3)
    axs[1].fill_betweenx(somora_sel.o3_p, (rel_diff_somora-100*error_somora/ozone_somora.mean(dim='time')),(rel_diff_somora+100*error_somora/ozone_somora.mean(dim='time')), color=color_somora, alpha=0.3)

    # if convolved:
    #     axs[1].fill_betweenx(gromos_sel.o3_p, (rel_diff_gromos-std_diff_gromos),(ozone_gromos.mean(dim='time')+error_gromos), color=color_gromos, alpha=0.3)
    #     axs[1].fill_betweenx(somora_sel.o3_p, (ozone_somora.mean(dim='time')-error_somora),(ozone_somora.mean(dim='time')+error_somora), color=color_somora, alpha=0.3)


    axs[1].set_xlim(-40, 40)
    axs[1].set_xlabel(r'$\Delta$O$_3$ [\%]', fontsize=fs)
    axs[1].legend()
    axs[1].set_ylabel('')
    axs[1].grid(axis='x', linewidth=0.5)
    ax.axvline(x=0,ls='--' ,color='k', lw=0.5)
    axs[1].fill_between(axs[1].get_xlim(), p_somora_mr[0],1e4, color=color_shading, alpha=0.2)
    axs[1].fill_between(axs[1].get_xlim(),p_somora_mr[-1],1e-4, color=color_shading, alpha=0.2)
    axs[1].fill_between(axs[1].get_xlim(),p_gromos_mr[0],1e4, color=color_shading, alpha=0.2)
    axs[1].fill_between(axs[1].get_xlim(),p_gromos_mr[-1],1e-4, color=color_shading, alpha=0.2)

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
    if convolved:
        fig.savefig(basefolder+'ozone_profile_MLS_convolved_'+str(year)+'.pdf', dpi=500)
    else:
        fig.savefig(basefolder+'ozone_profile_MLS_collocated_'+str(year)+'.pdf', dpi=500)

def MLS_comparison(gromos, somora, yrs, time_period):
   # ds_mls_v5 = read_MLS(timerange = time_period, vers=5)

    gromos_sel, mls_gromos_coloc, convolved_MLS_GROMOS = read_MLS_convolved(
        instrument_name='GROMOS', 
        folder='/scratch/GROSOM/Level2/MLS/', 
        years=yrs
        )
    somora_sel, mls_somora_coloc, convolved_MLS_SOMORA = read_MLS_convolved(
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

    return gromos_sel, mls_gromos_coloc, convolved_MLS_GROMOS, somora_sel, mls_somora_coloc, convolved_MLS_SOMORA

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
if __name__ == "__main__":
    time_period = slice("2010-01-01", "2020-12-31")
    yrs = [2017]#,2019[2012,2013,2014,2015,2016,2017,2018,2019,]
    gromos = read_GROMORA_all(basefolder='/storage/tub/instruments/gromos/level2/GROMORA/v1/', 
    instrument_name='GROMOS',
    date_slice=time_period, 
    years=yrs,
    prefix='_waccm_low_alt_ozone.nc'
    )
    somora = read_GROMORA_all(basefolder='/storage/tub/instruments/somora/level2/v1/', 
    instrument_name='SOMORA',
    date_slice=time_period, 
    years=yrs,
    prefix='_waccm_low_alt_ozone.nc'
    )

    gromos_clean = gromos.where(abs(1-gromos.oem_diagnostics[:,2])<0.2)# .where(gromos.o3_mr>0.8)
    somora_clean = somora.where(abs(1-somora.oem_diagnostics[:,2])<0.1)# .where(somora.o3_mr>0.8)

    for yr in yrs:
        plot_period = slice(str(yr)+"-01-01", str(yr)+"-12-31")
       # plot_period = time_period
      #  ds_mls = read_MLS(timerange = plot_period, vers=4.2)
        ds_mls_v5 = read_MLS(timerange = plot_period, vers=5)
    
     #   monthly_mls = ds_mls.o3.resample(time='1D', skipna=True).mean()
    #    compare_MLS(ds_mls.o3, ds_mls_v5.o3)
        
       # gromora_sel, mls_sel = select_gromora_corresponding_mls(gromos, ds_mls_v5)

        #plot_gromora_and_corresponding_MLS(gromora_sel, mls_sel)
    
      #  gromos_sel, convolved_MLS_GROMOS = avk_smooth_mls(gromos_clean, ds_mls_v5, folder='/scratch/GROSOM/Level2/MLS/'+'GROMOS_')
      #   somora_sel, convolved_MLS_SOMORA = avk_smooth_mls(somora_clean, ds_mls_v5, folder='/scratch/GROSOM/Level2/MLS/'+'SOMORA_')

        gromos_sel, mls_gromos_colloc, mls_gromos_colloc_convolved_new = read_MLS_convolved(
            instrument_name='GROMOS', 
            folder='/scratch/GROSOM/Level2/MLS/', 
            years=[yr]
            )
        somora_sel, mls_somora_colloc, mls_somora_colloc_convolved_new = read_MLS_convolved(
            instrument_name='SOMORA', 
            folder='/scratch/GROSOM/Level2/MLS/', 
            years=[yr]
            )

        plot_gromora_and_corresponding_MLS(gromos_sel, ds_mls_v5, mls_gromos_colloc_convolved_new,freq='1D', basename='GROMOS_read_daily')
        plot_gromora_and_corresponding_MLS(somora_sel, ds_mls_v5, mls_somora_colloc_convolved_new,freq='1D', basename='SOMORA_read_daily')
       # test_plevel(1, time_period, gromos ,gromos_sel, ds_mls_v5, mls_gromos_colloc, mls_somora_colloc_convolved_new)
        # gromos_sel, mls_gromos_colloc=select_gromora_corresponding_mls(
        #     gromos_clean, 
        #     ds_mls_v5, 
        #     time_period, 
        #     save_ds=True,
        #     convolved=False,
        #     basename='GROMOS_collocation_'
        #     )

        # gromos_sel, mls_gromos_colloc_convolved_new  = avk_smooth_mls_new(
        #     gromos_sel, mls_gromos_colloc,basename='/scratch/GROSOM/Level2/MLS/GROMOS_convolved_', sel=True, save_ds=True
        # )

        # somora_sel, mls_somora_colloc=select_gromora_corresponding_mls(
        #     somora, 
        #     ds_mls_v5, 
        #     time_period, 
        #     save_ds=True,
        #     convolved=False,
        #     basename='SOMORA_collocation_'
        #     )

        # somora_sel, mls_somora_colloc_convolved_new  = avk_smooth_mls_new(
        #     somora_sel, mls_somora_colloc,basename='/scratch/GROSOM/Level2/MLS/SOMORA_convolved_', sel=True, save_ds=True
        # )

    #     somora_sel, mls_somora_colloc=select_gromora_corresponding_mls(somora, convolved_MLS_SOMORA, time_period)
        # plot_gromora_and_corresponding_MLS(somora_sel, ds_mls_v5, mls_somora_colloc_convolved_new,freq='2D', basename='SOMORA_2daily')
        # plot_gromora_and_corresponding_MLS(gromos_sel, ds_mls_v5, mls_gromos_colloc_convolved_new,freq='2D', basename='GROMOS_2daily')
    #     # gromos_sel['o3'] = gromos_sel.o3_x
    #     # somora_sel['o3'] = somora_sel.o3_x
    #     # convolved_MLS_GROMOS['o3'] = convolved_MLS_GROMOS.o3_x
    #     # convolved_MLS_SOMORA['o3'] = convolved_MLS_SOMORA.o3_x

    # #     compare_ts_MLS(gromos_sel, mls_gromos_colloc, date_slice=plot_period, freq='1H', basefolder='/scratch/GROSOM/Level2/MLS/', ds_mls=ds_mls_v5)
    #     compare_GROMORA_MLS_profiles(gromos_sel, somora_sel, mls_gromos_colloc_convolved_new, mls_somora_colloc_convolved_new, ds_mls_v5,basefolder='/scratch/GROSOM/Level2/GROMORA_waccm/')
    #    # compute_corr_profile(somora_sel,mls_somora_colloc_convolved_new,freq='1D',basefolder='/scratch/GROSOM/Level2/MLS/')
        mls_gromos_colloc_convolved_new['time'] = gromos_sel.time.data
        compute_correlation_MLS(gromos_sel, mls_gromos_colloc_convolved_new, freq='OG', pressure_level=[15,20,25], basefolder='/scratch/GROSOM/Level2/MLS/GROMOS_')
        mls_somora_colloc_convolved_new['time'] = somora_sel.time.data
        compute_correlation_MLS(somora_sel, mls_somora_colloc_convolved_new, freq='OG', pressure_level=[15,20,25], basefolder='/scratch/GROSOM/Level2/MLS/SOMORA_')
    #     compute_correlation(somora_sel, mls_somora_colloc_convolved_new, freq='1D', pressure_level=[15,20,25], basefolder='/scratch/GROSOM/Level2/MLS/SOMORA_', MLS=True)
    # #    # plot_gromora_and_corresponding_MLS(new_gromos, mls_gromos_colloc)

   #  compare_pressure(gromos_sel, mls_somora_colloc, pressure_level=[31, 25, 21, 15, 12], add_sun=False, freq='1D', basefolder='/scratch/GROSOM/Level2/GROMORA_waccm/')
