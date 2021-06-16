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

def read_ECMWF(date, location='BERN'):
    ECMWF_folder = '/storage/tub/instruments/gromos/ECMWF_Bern/'
    counter = 0
    for d in date:
        ECMWF_file = os.path.join(
            ECMWF_folder, 'ecmwf_oper_v2_'+location+'_'+d.strftime('%Y%m%d')+'.nc')

        ecmwf_og = xr.open_dataset(
            ECMWF_file,
            decode_times=True,
            decode_coords=True,
            use_cftime=False,
        )
       # ecmwf_og.swap_dims({'level':'pressure'} )
        # for i in range(len(ecmwf_og.time.data)):
        #     ecmwf = ecmwf_og.isel(loc=0, time=i)
        #     ecmwf = read_add_geopotential_altitude(ecmwf)
        if counter == 0:
            ecmwf_ts = ecmwf_og
        else:
            ecmwf_ts = xr.concat([ecmwf_ts, ecmwf_og], dim='time')

        counter = counter + 1

    ecmwf_ts = ecmwf_ts.isel(loc=0)
    ecmwf_ts['pressure'] = ecmwf_ts['pressure']/100
    #o3_ecmwf = ecmwf_ts.isel(loc=0).ozone_mass_mixing_ratio
    return ecmwf_ts
    
    
    # o3_ecmwf.data = o3_ecmwf.data * 1e6



    # from retrievals.data.ecmwf import ECMWFLocationFileStore, levels

    # ecmwf_prefix = 'ecmwf_oper_v2_BERN_%Y%m%d.nc'
    # t1 = date[0]
    # t2 = date[2]
    # ecmwf_store = ECMWFLocationFileStore(ECMWF_folder, ecmwf_prefix)
    # ds_ecmwf = (
    #     ecmwf_store.select_time(t1, t2, combine='by_coords')
    #     .mean(dim='time')
    #     .swap_dims({"level": "pressure"})
    # )

    # ds_ecmwf = read_add_geopotential_altitude(ds_ecmwf)


    # return merra2_tot
    
# def plot_ecmwf(ecmwf_ds):    
#     fig2 = plt.figure(num=1)
#     ax = fig2.subplots(1)
    
#     # ds_ecmwf = (
#     #     ecmwf_store.select_time(t1, t2, combine='by_coords')
#     #     .mean(dim='time')
#     #     .swap_dims({"level": "pressure"})
#     # )

#     o3_ecmwf = ecmwf_ds.ozone_mass_mixing_ratio
#     o3_ecmwf.swap_dims({"level": "pressure"})

#     o3_ecmwf.plot(
#         x='time',
#         y='pressure',
#         vmin=0,
#         vmax=15,
#         cmap='viridis',
#         cbar_kwargs={"label": "ozone [PPM]"}
#     )
#     ax.invert_yaxis()
#     ax.set_yscale('log')
#     ax.set_ylabel('P [hPa]')
#     plt.tight_layout()
    # o3.plot.imshow(x='time')
    #fig2.savefig(instrument.level2_folder+'/'+'ozone_ts_16_ecmwf_payerne.pdf')

if __name__ == "__main__":
    date = pd.date_range(start='2018-01-01', end='2018-01-30')
    ecmwf_ds = read_ECMWF( date, 'Bern')

