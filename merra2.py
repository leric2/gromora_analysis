#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import datetime
import os

import matplotlib.pyplot as plt
import netCDF4
import numpy as np
import pandas as pd

import xarray as xr
colormap = 'cividis'

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
        merra2 = merra2.swap_dims({'phony_dim_5': 'datetime'})
        merra2['datetime'] = merra2_decoded.datetime.data
        merra2['O3'].data = merra2['O3'].data*1e6
        #merra2['O3_err'].data = merra2['O3_err'].data*1e6
    if counter == 0:
        merra2_tot = merra2
    else:
        merra2_tot = xr.concat(
            [merra2_tot, merra2], dim='datetime')
    counter = counter + 1
    return merra2_tot
    
def plot_merra2(merra2_ds):    

    
    o3_merra2_tot =merra2_ds.O3
    # merra2_ds.sel(datetime=slice("2017-08-15", "2017-12-31"))
    fig, ax = plt.subplots(1, 1)
    o3_merra2_tot.plot(x='datetime', y='altitude', ax=ax, vmin=0, vmax=15, cmap=colormap)
    #ax.set_ylim(5, 75)
    # o3_merra2.assign_coords({'altitude':merra2_info.alt.isel(phony_dim_1=0)})
    plt.tight_layout()
    #fig.savefig(instrument.level2_folder+'/ozone_ts_17_merra2.pdf')


if __name__ == "__main__":
    merra2 = read_merra2_BRN(
        years = [2017,], 
        months = [1,2,3,4,5,6,7]
    )
    plot_merra2(merra2)

