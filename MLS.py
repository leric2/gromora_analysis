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
from level2_gromora import *

colormap = 'cividis'

# color_gromos = '#d95f02'
# color_somora = '#1b9e77'


def read_MLS(timerange):
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

    #ds_mls.to_netcdf('/home/esauvageat/Documents/AuraMLS/ozone_bern_ts.nc', format='NETCDF4')
    ds_mls= ds_mls.sel(time=timerange)
    return ds_mls

def select_gromora_corresponding_mls(gromora, ds_mls):
    time_mls = pd.to_datetime(ds_mls.time.data)
    time_gromora = pd.to_datetime(gromora.time.data)
    new_ds = xr.Dataset()
    new_mls = ds_mls
    counter = 0
    for t in time_mls:
        try:
            gromora_sel = gromora.sel(time=t, method='nearest', tolerance='2H')
            if counter == 0:
                new_ds=gromora_sel
                counter = counter+1
            else:
                new_ds=xr.concat([new_ds, gromora_sel], dim='time')
        except:
            new_mls = new_mls.drop_sel(time=t)

    # new_ds.o3_x.isel(o3_p=12).resample(time='1D').mean().plot()
    # new_mls.o3.isel(p=12).resample(time='1D').mean().plot()
    return new_ds, new_mls

def avk_smooth_mls(gromora, ds_mls):
    time_mls = pd.to_datetime(ds_mls.time.data)
    time_gromora = pd.to_datetime(gromora.time.data)
    new_ds = xr.Dataset()
    convolved_MLS_list = []
    time_list = []
    counter = 0
    for t in time_mls:
        try:
            gromora_sel = gromora.sel(time=t, method='nearest', tolerance='2H')
            ds_mls_sel= ds_mls.sel(time=t)
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

    # new_ds.o3_x.isel(o3_p=12).resample(time='1D').mean().plot()
    # new_mls.o3.isel(p=12).resample(time='1D').mean().plot()
    return new_ds, convolved_MLS

def plot_MLS(o3_mls):  
    year=pd.to_datetime(o3_mls.time.data[0]).year  
    fig, ax = plt.subplots(1, 1)
    #monthly_mls.plot(x='time', y='p', ax=ax ,vmin=0, vmax=9).resample(time='24H', skipna=True).mean()
    pl = o3_mls.plot(
        x='time',
        y='p',
        ax=ax,
        vmin=0,
        vmax=10,
        yscale='log',
        linewidth=0,
        rasterized=True,
        cmap=colormap
    )
   # pl.set_edgecolor('face')
    # ax.set_yscale('log')
    #ax.set_ylim(0.01, 500)
    ax.invert_yaxis()
    ax.set_ylabel('P [hPa]')
    plt.tight_layout()
    fig.savefig('/scratch/GROSOM/Level2/GROMORA_waccm/' +
                '/ozone_mls_'+str(year)+'.pdf', dpi=500)

def plot_gromora_and_corresponding_MLS(gromora_sel, mls_sel, freq='1D'):
    fig, axs= plt.subplots(2, 1, figsize=(12,8))
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
    pl_mls = mls_sel.o3.resample(time=freq).mean().plot(
        x='time',
        y='p',
        ax=axs[1],
        vmin=0,
        vmax=10,
        yscale='log',
        linewidth=0,
        rasterized=True,
        cmap=colormap
    )
    axs[1].set_title('MLS OG')
    # pl.set_edgecolor('face')
    # ax.set_yscale('log')
    for ax in axs:
        ax.set_ylim(0.01, 200)
        ax.invert_yaxis()
        ax.set_ylabel('P [hPa]')
    plt.tight_layout()


def compare_GROMORA_MLS_profiles(gromos_sel, somora_sel, convolved_MLS_gromos, convolved_MLS_somora, ds_mls, basefolder=''):
    fs = 22
    ozone_gromos = gromos_sel.o3
    ozone_somora = somora_sel.o3

    year=pd.to_datetime(gromos_sel.time.data[0]).year

    rel_diff_somora = 100*(ozone_somora.mean(dim='time') - convolved_MLS_somora.o3.mean(dim='time'))/ozone_somora.mean(dim='time')
    rel_diff_gromos = 100*(ozone_gromos.mean(dim='time') - convolved_MLS_gromos.o3.mean(dim='time'))/ozone_gromos.mean(dim='time')
    rel_diff_gromora = 100*(ozone_gromos.mean(dim='time') - ozone_somora.mean(dim='time'))/ozone_gromos.mean(dim='time')
    
    error_gromos = 1e6*np.sqrt(gromos_sel.mean(dim='time').o3_eo**2 + gromos_sel.mean(dim='time').o3_es**2)
    error_somora = 1e6*np.sqrt(somora_sel.mean(dim='time').o3_eo**2 + somora_sel.mean(dim='time').o3_es**2)

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

    convolved_MLS_gromos.o3.mean(dim='time').plot(y='o3_p', ax=axs[0], linestyle=':', color=color_gromos, label='MLS convolved GROMOS')
    convolved_MLS_somora.o3.mean(dim='time').plot(y='o3_p', ax=axs[0], linestyle=':', color=color_somora, label='MLS convolved SOMORA')

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
    fig.savefig(basefolder+'ozone_profile_MLS_convolved_new_'+str(year)+'.pdf', dpi=500)

if __name__ == "__main__":
    time_period = slice("2016-01-01", "2016-03-31")
    yrs = [2016]#[2012,2013,2014,2015,2016,2017,2018,2019,]
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
    prefix='_waccm_low_alt_dx10_sinefit_ozone.nc'
    )

    for yr in yrs:
        plot_period = slice(str(yr)+"-01-01", str(yr)+"-03-31")
        ds_mls = read_MLS(timerange = plot_period)
    
        monthly_mls = ds_mls.o3.resample(time='1D', skipna=True).mean()
        plot_MLS(ds_mls.o3)

        gromora_sel, mls_sel = select_gromora_corresponding_mls(gromos, ds_mls)

        #plot_gromora_and_corresponding_MLS(gromora_sel, mls_sel)
    
        gromos_sel, convolved_MLS_GROMOS = avk_smooth_mls(gromos, ds_mls)
        somora_sel, convolved_MLS_SOMORA = avk_smooth_mls(somora, ds_mls)

        gromos_sel['o3'] = gromos_sel.o3_x
        somora_sel['o3'] = somora_sel.o3_x
        convolved_MLS_GROMOS['o3'] = convolved_MLS_GROMOS.o3_x
        convolved_MLS_SOMORA['o3'] = convolved_MLS_SOMORA.o3_x

        compare_GROMORA_MLS_profiles(gromos_sel, somora_sel, convolved_MLS_GROMOS, convolved_MLS_SOMORA, ds_mls,basefolder='/scratch/GROSOM/Level2/GROMORA_waccm/')

    #plot_gromora_and_corresponding_MLS(gromos_sel, convolved_MLS)

    #compare_pressure(gromos_sel, convolved_MLS, pressure_level=[31, 25, 21, 15, 12], add_sun=False, freq='1D', basefolder='/scratch/GROSOM/Level2/GROMORA_waccm/')
