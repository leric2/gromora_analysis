#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 06.01.22

@author: Eric Sauvageat

This is the main script for the MERRA-2 data treatment in the frame of the GROMORA project

This is an old module which needs to be updated to the latest GROMORA v2

"""

import datetime
import os

import matplotlib.pyplot as plt
import netCDF4
import numpy as np
import pandas as pd

import xarray as xr
from level2_gromora_diagnostics import read_GROMORA_all
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter, AutoMinorLocator)
import matplotlib.ticker as ticker

from base_tool import *

colormap = 'cividis'
Mair = 28.9644
Mozone= 47.9982

color_gromos= get_color('GROMOS')
color_somora= get_color('SOMORA')
color_shading = 'grey'

def read_merra2_BRN(years = [2017], months = [10]):

    merra2_basename = '/storage/atmosphere/atmosphere/MERRA2/BRN/'
    filename_merra2 = []
    for y in years:
        #print(y)
        for m in months:
            if m<10:
                str_month = '0'+str(m)
            else:
                str_month = str(m)
            filename = 'MERRA2_BRN_'+str(y)+'_'+str_month+'_diagnostic.h5'
            filename_merra2.append(
                os.path.join(merra2_basename, filename)
            )
    
    merra2_tot = xr.Dataset()
    counter = 0
    for f in filename_merra2:
        merra2_info = xr.open_dataset(
            f,
            group='info',
            decode_times=False,
            decode_coords=True,
            # use_cftime=True,
        )
        #merra2_info.time.attrs = attrs
        merra2_info['time'] = merra2_info.time - merra2_info.time[0]
    #   # construct time vector
        time_merra2 = []
        for i in range(len(merra2_info.time)):
            time_merra2.append(
                datetime.datetime(
                    int(merra2_info.isel(phony_dim_1=0).year.data[i]),
                    int(merra2_info.isel(phony_dim_1=0).month.data[i]),
                    int(merra2_info.isel(phony_dim_1=0).day.data[i]),
                    int(merra2_info.isel(phony_dim_1=0).hour.data[i]),
                    int(merra2_info.isel(phony_dim_1=0)['min'].data[i]),
                    int(merra2_info.isel(phony_dim_1=0).sec.data[i])
                )
            )
        merra2_info['datetime'] = time_merra2
        merra2_decoded = xr.decode_cf(merra2_info)
        merra2 = xr.open_dataset(
            f,
            group='trace_gas',
            decode_times=True,
            decode_coords=True,
            # use_cftime=True,
        )
        merra2_wind = xr.open_dataset(
            f,
            group='wind',
            decode_times=True,
            decode_coords=True,
            # use_cftime=True,
        )

        #o3_merra2 = merra2.O3
        merra2 = merra2.swap_dims({'phony_dim_6': 'altitude'})
        merra2['altitude'] = merra2_decoded.alt.isel(phony_dim_1=0).data
        merra2 = merra2.swap_dims({'phony_dim_5': 'time'})
        merra2['time'] = merra2_decoded.datetime.data

        merra2_wind = merra2_wind.swap_dims({'phony_dim_8': 'altitude'})
        merra2_wind['altitude'] = merra2_decoded.alt.isel(phony_dim_1=0).data
        merra2_wind = merra2_wind.swap_dims({'phony_dim_7': 'time'})
        merra2_wind['time'] = merra2_decoded.datetime.data


        # convert to VMR (in ppm)
        merra2['O3'].data = 1e6*merra2['O3'].data * Mair/Mozone

        #merra2['O3_err'].data = merra2['O3_err'].data*1e6

        merra2 = xr.merge([merra2, merra2_wind])

        if counter == 0:
            merra2_tot = merra2
        else:
            merra2_tot = xr.concat(
                [merra2_tot, merra2], dim='time')
        counter = counter + 1

    #merra2_tot.swap_dims({'datetime','time'})
    return merra2_tot
    
def plot_merra2(merra2_ds):    
    o3_merra2_tot =merra2_ds.O3
    # merra2_ds.sel(datetime=slice("2017-08-15", "2017-12-31"))
    fig, ax = plt.subplots(1, 1)
    o3_merra2_tot.plot(
        x='time', 
        y='altitude', 
        ax=ax, vmin=0, 
        vmax=10, 
        cmap=colormap
    )
    #ax.set_ylim(5, 75)
    # o3_merra2.assign_coords({'altitude':merra2_info.alt.isel(phony_dim_1=0)})
    plt.tight_layout()
    fig.savefig('/scratch/GROSOM/Level2/GROMORA_retrievals_polyfit2/'+'ozone_ts_merra2.pdf')

def plot_merra2_convolved(merra2_convolved):
    fig, axs = plt.subplots(1, 1, sharex=True)
    pl = merra2_convolved.o3.plot(
        x='time',
        y='o3_p',
        ax=axs, 
        vmin=0,
        vmax=10,
        yscale='log',
        linewidth=0,
        rasterized=True,
        cmap='cividis'
    )
    pl.set_edgecolor('face')
    # ax.set_yscale('log')
    axs.invert_yaxis()
    axs.set_ylabel('P [hPa]')

    #fig.savefig('/scratch/GROSOM/Level2/GROMORA_retrievals_polyfit2/'+'ozone_ts_merra2_convolved.pdf')

def avk_smooth_merra2(gromora, ds_merra2):
    time_merra = pd.to_datetime(ds_merra2.time.data)
    time_gromora = pd.to_datetime(gromora.time.data)
    new_ds = xr.Dataset()
    convolved_merra2_list = []
    time_list = []
    counter = 0
    for t in time_merra:
        try:
            gromora_sel = gromora.sel(time=t, method='nearest', tolerance='2H')
            ds_merra2_sel= ds_merra2.sel(time=t)
            avkm = gromora_sel.o3_avkm.data

            o3_p_mls = ds_merra2_sel.p.data/100
            idx = np.argsort(o3_p_mls, kind='heapsort')
            o3_p_mls_sorted = o3_p_mls[idx]  
            o3_mls_sorted = ds_merra2_sel.O3.data[idx]
    
            interpolated_mls = np.interp(np.log(gromora_sel.o3_p.data), np.log(o3_p_mls_sorted),o3_mls_sorted)
            conv = gromora_sel.o3_xa.values*1e6 + np.matmul(avkm,(interpolated_mls - gromora_sel.o3_x.values))

            convolved_merra2_list.append(conv)
            time_list.append(t)

            
            if counter == 0:
                new_ds = gromora_sel
                counter = counter+1
            else:
                new_ds=xr.concat([new_ds, gromora_sel], dim='time')
        except:
            pass
    
    convolved_merra2 = xr.Dataset(
        data_vars=dict(
            o3_x=(['time', 'o3_p'], convolved_merra2_list)
        ),
        coords=dict(
            time=time_list,
            o3_p=gromora_sel.o3_p.data
        ),
        attrs=dict(description='ozone time series at bern')
    )

    # new_ds.o3_x.isel(o3_p=12).resample(time='1D').mean().plot()
    # new_mls.o3.isel(p=12).resample(time='1D').mean().plot()
    return new_ds, convolved_merra2

def convolve_merra2(gromora, dataserie):
    mean_avkm = gromora.o3_avkm.mean(dim='time').values
    z_merra= dataserie.altitude.values*1000

    o3_convolved = np.ones((len(gromora.o3_p), len(dataserie.time)))*np.nan
    for i,t in enumerate(dataserie.time):
        z_gromora = gromora.o3_z.isel(time=i).values
        interpolated_o3 = np.interp(z_gromora, z_merra, dataserie.O3.isel(time=i).values*1e-6)
        # interpolated_o3[np.argwhere(np.isnan(interpolated_o3))] = 0
        xa = gromora.o3_xa.isel(time=i).values
        o3_convolved[:,i] = xa + np.matmul(mean_avkm,(interpolated_o3 - xa))
    
    ozone_ds_convolved = xr.Dataset(
        data_vars=dict(
            o3_x=(['o3_p', 'time'], o3_convolved*1e6),
        ),
        coords=dict(
            time=dataserie.time,
            o3_p=gromora.o3_p.values
        ),
        attrs=dict(description='ozone from MERRA2 convolved with GROMORA AVKM')
    )
    return ozone_ds_convolved

def compare(gromos,merra2_convolved):

    fig1, axs1 = plt.subplots(2, 1, sharey=True)
    pl = gromos.o3_x.plot(
        x='time',
        y='o3_p',
        ax=axs1[0], 
        vmin=0,
        vmax=10,
        yscale='log',
        linewidth=0,
        rasterized=True,
        cmap='cividis'
    )
    pl.set_edgecolor('face')
    # ax.set_yscale('log')
    axs1[0].invert_yaxis()
    axs1[0].set_ylabel('P [hPa]')
    axs1[0].set_xticks([])
    
    pl = merra2_convolved.o3_x.plot(
        x='time',
        y='o3_p',
        ax=axs1[1], 
        vmin=0,
        vmax=10,
        yscale='log',
        linewidth=0,
        rasterized=True,
        cmap='cividis'
    )
    pl.set_edgecolor('face')
    # ax.set_yscale('log')
    axs1[1].invert_yaxis()
    axs1[1].set_ylabel('P [hPa]')
    axs1[1].set_title('MERRA2, convolved')
    for ax in axs1:
        ax.set_ylim((500,0.1))
    plt.tight_layout()
    fig1.savefig('/scratch/GROSOM/Level2/GROMORA_retrievals_polyfit2/'+'comparison_merra2_ts.pdf')

    fig, axs = plt.subplots(1, 2, sharey=True)
    merra2_convolved.o3_x.mean(dim='time').plot(
        y='o3_p',
        ax=axs[0], 
        yscale='log',
        label='MERRA2, conv'
    )
    gromos.o3_x.mean(dim='time').plot(
        y='o3_p',
        ax=axs[0], 
        yscale='log',
        label='MWR'
    )
    rel_diff = 100*(gromos.o3_x.mean(dim='time')-merra2_convolved.o3_x.mean(dim='time'))/merra2_convolved.o3_x.mean(dim='time')
    rel_diff.plot(
        y='o3_p',
        ax=axs[1], 
        yscale='log',
        label='MWR'
    )
    axs[1].set_ylim((1e-1,2e2))
    axs[0].legend()
    axs[0].invert_yaxis()
    axs[0].set_ylabel('P [hPa]')
    axs[1].set_xlim((-20,20))
    axs[1].set_xlabel('rel diff [%]')

    for ax in axs:
        ax.grid()

    fig.savefig('/scratch/GROSOM/Level2/GROMORA_retrievals_polyfit2/'+'SOMORA_comparison_merra2.pdf')

def compare_GROMORA_merra2_profiles(gromos_sel, somora_sel, convolved_merra2_gromos, convolved_merra2_somora, merra2, basefolder=''):
    fs = 22
    ozone_gromos = gromos_sel.o3
    ozone_somora = somora_sel.o3

    year=pd.to_datetime(gromos_sel.time.data[0]).year

    rel_diff_somora = 100*(ozone_somora.mean(dim='time') - convolved_merra2_somora.o3.mean(dim='time'))/ozone_somora.mean(dim='time')
    rel_diff_gromos = 100*(ozone_gromos.mean(dim='time') - convolved_merra2_gromos.o3.mean(dim='time'))/ozone_gromos.mean(dim='time')
    rel_diff_gromora = 100*(ozone_gromos.mean(dim='time') - ozone_somora.mean(dim='time'))/ozone_gromos.mean(dim='time')
    
    error_gromos = 1e6*np.sqrt(gromos_sel.mean(dim='time').o3_eo**2 + gromos_sel.mean(dim='time').o3_es**2)
    error_somora = 1e6*np.sqrt(somora_sel.mean(dim='time').o3_eo**2 + somora_sel.mean(dim='time').o3_es**2)

    mr_somora = somora_sel.o3_mr.data
    mr_gromos = gromos_sel.o3_mr.data
    p_somora_mr = somora_sel.o3_p.data[np.mean(mr_somora,0)>=0.8]
    p_gromos_mr = gromos_sel.o3_p.data[np.mean(mr_gromos,0)>=0.8]

    fig, axs = plt.subplots(nrows=1, ncols=2, sharey=True, figsize=(8, 10))
    axs[0].plot(merra2.O3.mean(dim='time').data, 1e-2*merra2.p.mean(dim='time').data,'-k', label='MERRA2')

    ozone_gromos.mean(dim='time').plot(
        y='o3_p', ax=axs[0], color=color_gromos, ls='-', label='GROMOS')
    ozone_somora.mean(dim='time').plot(
        y='o3_p', ax=axs[0], color=color_somora, ls='-', label='SOMORA')



    axs[0].fill_betweenx(gromos_sel.o3_p, (ozone_gromos.mean(dim='time')-error_gromos),(ozone_gromos.mean(dim='time')+error_gromos), color=color_gromos, alpha=0.3)
    axs[0].fill_betweenx(somora_sel.o3_p, (ozone_somora.mean(dim='time')-error_somora),(ozone_somora.mean(dim='time')+error_somora), color=color_somora, alpha=0.3)

    convolved_merra2_gromos.o3.mean(dim='time').plot(y='o3_p', ax=axs[0], linestyle=':', color=color_gromos, label='MERRA2 convolved GROMOS')
    convolved_merra2_somora.o3.mean(dim='time').plot(y='o3_p', ax=axs[0], linestyle=':', color=color_somora, label='MERRA2 convolved SOMORA')

    

    #ds_mls.O3.mean(dim='time').plot(y='p', ax=axs[0], color='k', label='MERRA2')
    
    for ax in axs:
        ax.axhline(y=p_somora_mr[0],ls='--' ,color=color_somora, lw=1)
        ax.axhline(y=p_somora_mr[-1],ls='--', color=color_somora, lw=1)
        ax.axhline(y=p_gromos_mr[0],ls=':', color=color_gromos, lw=1)
        ax.axhline(y=p_gromos_mr[-1],ls=':', color=color_gromos, lw=1)

    axs[0].invert_yaxis()
    axs[0].set_xlim(-0.2, 9)
    axs[0].set_yscale('log')
    axs[0].legend()
    axs[0].set_ylim(100, 1e-1)
    axs[0].set_ylabel('Pressure [hPa]', fontsize=fs)
    axs[0].set_xlabel(r'O$_3$ VMR [ppmv]', fontsize=fs)
    axs[0].grid(axis='x', linewidth=0.5)
    axs[0].fill_between(axs[0].get_xlim(),p_somora_mr[0],1e4, color=color_shading, alpha=0.2)
    axs[0].fill_between(axs[0].get_xlim(),p_somora_mr[-1],1e-4, color=color_shading, alpha=0.2)
    axs[0].fill_between(axs[0].get_xlim(),p_gromos_mr[0],1e4, color=color_shading, alpha=0.2)
    axs[0].fill_between(axs[0].get_xlim(),p_gromos_mr[-1],1e-4, color=color_shading, alpha=0.2)

    rel_diff_gromos.plot(y='o3_p', ax=axs[1], color=color_gromos,
                  ls='-', alpha=1, label='GROMOS vs MERRA2')
    rel_diff_somora.plot(y='o3_p', ax=axs[1], color=color_somora,
                  ls='-', alpha=1, label='SOMORA vs MERRA2')
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
    fig.savefig(basefolder+'ozone_profile_merra2_convolved_'+str(year)+'.pdf', dpi=500)

if __name__ == "__main__":

    yrs = [2017]

    merra2 = read_merra2_BRN(
        years = yrs, 
        months = [1]
    )

    #o3_merra2 = merra2.O3.where(merra2.altitude<60).data
    #merra2['O3'].data = o3_merra2
    #plot_merra2(merra2)

    time_period = slice("2017-01-01", "2017-01-31")

    fold_somora = '/scratch/GROSOM/Level2/SOMORA/v2/'
    fold_gromos = '/scratch/GROSOM/Level2/GROMOS/v2/'

    gromos = read_GROMORA_all(
        basefolder=fold_gromos, 
        instrument_name='GROMOS',
        date_slice=time_period, 
        years=yrs,
        prefix='_v2.nc',
        flagged=True
    )
    somora = read_GROMORA_all(
        basefolder=fold_somora, 
        instrument_name='SOMORA',
        date_slice=time_period, 
        years=yrs,
        prefix='_v2.nc',
        flagged=True
    )

    #merra2_convolved = convolve_merra2(gromos, merra2)
    #plot_merra2_convolved(merra2_convolved)
    #plot_ozone_ts(gromos)
    gromos_sel, convolved_merra2_gromos = avk_smooth_merra2(gromos, merra2)
    somora_sel, convolved_merra2_somora = avk_smooth_merra2(somora, merra2)

    gromos_sel['o3'] = gromos_sel.o3_x
    somora_sel['o3'] = somora_sel.o3_x
    convolved_merra2_gromos['o3'] = convolved_merra2_gromos.o3_x
    convolved_merra2_somora['o3'] = convolved_merra2_somora.o3_x

    compare_GROMORA_merra2_profiles(gromos_sel, somora_sel, convolved_merra2_gromos, convolved_merra2_somora, merra2 ,basefolder='/scratch/GROSOM/Level2/GROMORA_waccm/')

