#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import datetime
import os

import matplotlib.pyplot as plt
import netCDF4
import numpy as np
import pandas as pd

import xarray as xr
from level2_gromora import *

colormap = 'cividis'
Mair = 28.9644
Mozone= 47.9982

def read_merra2_BRN(years = [2017], months = [10]):

    merra2_basename = '/storage/tub/atmosphere/MERRA2/BRN/'
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

        #o3_merra2 = merra2.O3
        merra2 = merra2.swap_dims({'phony_dim_6': 'altitude'})
        merra2['altitude'] = merra2_decoded.alt.isel(phony_dim_1=0).data
        merra2 = merra2.swap_dims({'phony_dim_5': 'time'})
        merra2['time'] = merra2_decoded.datetime.data


        # convert to VMR (in ppm)
        merra2['O3'].data = 1e6*merra2['O3'].data * Mair/Mozone

        #merra2['O3_err'].data = merra2['O3_err'].data*1e6

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
    o3_merra2_tot.plot(x='time', y='altitude', ax=ax, vmin=0, vmax=15, cmap=colormap)
    #ax.set_ylim(5, 75)
    # o3_merra2.assign_coords({'altitude':merra2_info.alt.isel(phony_dim_1=0)})
    plt.tight_layout()
    #fig.savefig(instrument.level2_folder+'/ozone_ts_17_merra2.pdf')

def plot_merra2_convolved(merra2_convolved):
    fig, axs = plt.subplots(1, 1, sharex=True)
    pl = merra2_convolved.o3_x.plot(
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

def convolve_merra2(gromora, dataserie):
    mean_avkm = gromora.o3_avkm.mean(dim='time').values
    z_merra= dataserie.altitude.values*1000

    o3_convolved = np.ones((len(gromora.o3_p), len(dataserie.time)))*np.nan
    for i,t in enumerate(dataserie.time):
        z_gromora = gromora.o3_z.isel(time=i).values
        interpolated_o3 = np.interp(z_gromora, z_merra, dataserie.O3.isel(time=i).values*1e-6)
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

if __name__ == "__main__":
    merra2 = read_merra2_BRN(
        years = [2015,], 
        months = [1,2,3,4,5,6,7,8,9,10]
    )
    #plot_merra2(merra2)

    filename_gromos = '/storage/tub/instruments/gromos/level2/GROMORA/v1/GROMOS_2015_waccm_cov_yearly_ozone.nc'

    date_slice=slice("2015-01-01", "2015-10-31")
    gromos = read_GROMORA(filename_gromos, date_slice)

    merra2_convolved = convolve_merra2(gromos, merra2)
    plot_merra2_convolved(merra2_convolved)
    plot_ozone_ts(gromos)

