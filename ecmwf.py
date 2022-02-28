#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import datetime
import os

import matplotlib.pyplot as plt
import netCDF4
import numpy as np
import pandas as pd

import levels as ecmwf_levels

import cartopy.crs as ccrs
import typhon

import xarray as xr
colormap = 'cividis'

Mair = 28.9644
Mozone= 47.9982

ozone = {
    'varname':'o3',
    'M':47.9982,
    'factor':1e6,
    'colormap':plt.get_cmap('density'),
    'title':'Ozone',
    'colorbar_title': 'PPM',
    'polar_vmin': 0.5,
    'polar_vmax': 4,
    'global_vmin': 0.5,
    'global_vmax': 3,
}
temperature = {
    'varname':'t',
    'M':47.9982,
    'factor':1,
    'colormap':plt.get_cmap('temperature'),
    'title':'Temperature',
    'colorbar_title': 'K',
    'polar_vmin': 185,
    'polar_vmax': 240,
    'global_vmin': 185,
    'global_vmax': 240,
}
humidity = {
    'varname':'q',
    'M':47.9982,
    'factor':1e6,
    'colormap':plt.get_cmap('density'),
    'title':'Specific Humidity',
    'colorbar_title': 'PPM',
    'polar_vmin': 3,
    'polar_vmax': 6,
    'global_vmin': 2,
    'global_vmax': 4,
}
zonal_velocity = {
    'varname':'u',
    'M':47.9982,
    'factor':1,
    'colormap':plt.get_cmap('difference'),
    'title':'Zonal Velocity',
    'colorbar_title': 'm/s',
    'polar_vmin': -40,
    'polar_vmax': 40,
    'global_vmin': -40,
    'global_vmax': 40,
}
meridional_velocity = {
    'varname':'v',
    'M':47.9982,
    'factor':1,
    'colormap':plt.get_cmap('difference'),
    'title':'Meridional Velocity',
    'colorbar_title': 'm/s',
    'polar_vmin': -30,
    'polar_vmax': 30,
    'global_vmin': -30,
    'global_vmax': 30,
}

vertical_velocity = {
    'varname':'w',
    'M':47.9982,
    'factor':1,
    'colormap':plt.get_cmap('vorticity'),
    'title':'Vertical Velocity',
    'colorbar_title': 'Pa/s'
}

relative_vorticity = {
    'varname':'vo',
    'M':47.9982,
    'factor':1e4,
    'colormap':plt.get_cmap('vorticity'),
    'title':'Relative Vorticity',
    'colorbar_title': 's-1',
    'polar_vmin': -1,
    'polar_vmax': 1,
    'global_vmin': -1,
    'global_vmax': 1,
}
geopotential = {
    'varname':'z',
    'M':47.9982,
    'factor':1,
    'colormap':'coolwarm',
    'title':'Geopotential',
    'colorbar_title': 's-1'
}

    


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
    
def pressure_levels_global(lnsp):
    """
    If vector `lnsp` has dims (time,) then return an array with dims (level, time) containing pressure.
    """
    a = ecmwf_levels.hybrid_level_a[np.newaxis, :, np.newaxis, np.newaxis]
    b = ecmwf_levels.hybrid_level_b[np.newaxis, :, np.newaxis, np.newaxis]
    sp = np.exp(lnsp[:, np.newaxis])  # surface pressure
    return a + b * sp

def read_ECMWF_global(date):
    ECMWF_folder = '/storage/tub/atmosphere/ecmwf/oper/'
    year = date.year
    ECMWF_file = os.path.join(ECMWF_folder+str(year), 'ECMWF_OPER_v2_'+date.strftime('%Y%m%d')+'.nc')

    ecmwf_og = xr.open_dataset(
        ECMWF_file,
        decode_times=True,
        decode_coords=True,
        use_cftime=False,
    )
    #
    # for i in range(len(ecmwf_og.time.data)):
    #     ecmwf = ecmwf_og.isel(loc=0, time=i)
    #     ecmwf = read_add_geopotential_altitude(ecmwf)
    #ecmwf_og = ecmwf_og.mean(dim='time')


    #

    pressure = pressure_levels_global(lnsp=ecmwf_og['lnsp'].values[:,0,:,:])
    da_pressure = xr.DataArray(
        data=pressure,
        dims=['time','level','latitude','longitude'],
        coords=[
            ecmwf_og.time,
            ecmwf_og.level,
            ecmwf_og.latitude,
            ecmwf_og.longitude,
        ]
    )

    ecmwf_og['pressure'] = da_pressure

    return ecmwf_og

def extract_var_plevel(ecmwf_global, var=ozone, p_level=100, NH=True, coarsen=True, polar = True):
    if polar:
        mean_p = ecmwf_global.pressure.where(ecmwf_global.latitude>60).mean(dim=['time','latitude','longitude'])/100
    else: 
        mean_p = ecmwf_global.pressure.mean(dim=['time','latitude','longitude'])/100
    lvl = np.abs(mean_p.values-p_level).argmin()

    ecmwf_o3 = ecmwf_global[var['varname']].isel(level=lvl).mean(dim='time')
    #o3_ecmwf = ecmwf_ts.isel(loc=0).ozone_mass_mixing_ratio

    if NH:
        ecmwf_o3 = ecmwf_o3.where(ecmwf_o3.latitude>0, drop=True)
    if coarsen:
        ecmwf_o3 = ecmwf_o3.coarsen(longitude=2).mean().coarsen(latitude=1, boundary='trim').mean()
    
    if (var['varname'] == 'o3'):
        ecmwf_o3 = ecmwf_o3*Mair/var['M']
    
    ecmwf_o3=ecmwf_o3*var['factor']

    return ecmwf_o3


def plot_ozone(o3, ax):
    #fig = plt.figure(figsize=(8, 6))
    #axs = plt.axes(projection=ccrs.Orthographic(central_longitude=7, central_latitude=50, globe=None))
    p = o3.plot(
        ax=ax,
        transform=ccrs.PlateCarree(),
        cbar_kwargs={"label": "ozone ppmv"},
        cmap = colormap,
        #vmin=0,
        #vmax=10,
        linewidth=0,
        rasterized=True,
    ) 


def plot_ts(ts, ax, var, polar):
    #fig = plt.figure(figsize=(8, 6))
    #axs = plt.axes(projection=ccrs.Orthographic(central_longitude=7, central_latitude=50, globe=None))
    if polar:
        minval = var['polar_vmin']
        maxval = var['polar_vmax']
    else:
        minval = var['global_vmin']
        maxval = var['global_vmax']
    
    transform=ccrs.PlateCarree()
    ts.plot(
        ax=ax,
        #levels=8,
        transform=transform,
        cbar_kwargs={"label": var['colorbar_title']},
        cmap = var['colormap'],
        vmin=minval  ,
        vmax=maxval,
        linewidth=0,
        rasterized=True,
    ) 


    # o3_ecmwf.data = o3_ecmwf.data * 1e6

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
    #date = pd.date_range(start='2019-03-03', end='2019-03-03')

    basefolder = '/storage/tub/atmosphere/ecmwf/daily_plots/'
    #ecmwf_ds = read_ECMWF( date, 'Bern')

    #variable = ['t', 'u', 'v', 'o3', 'q','vo'] # Options are t, q, w, vo, o3, z, u, v
    variable = [temperature, zonal_velocity, meridional_velocity, ozone,humidity,relative_vorticity]

    #date=date[0]
    date = datetime.datetime.now()-datetime.timedelta(2)
    datestr = date.strftime('%Y-%m-%d')
    ecmwf_global = read_ECMWF_global(date)

    p_levels = [100, 10, 1]


    for p in p_levels:
        fig, axs = plt.subplots(figsize=(20,10), nrows=2, ncols=3, subplot_kw={'projection': ccrs.Orthographic(central_longitude=7, central_latitude=90, globe=None)})
        fig2, axs2 = plt.subplots(figsize=(18,6), nrows=2, ncols=3, subplot_kw={'projection': ccrs.PlateCarree()})

        #fig, axs = plt.subplots(figsize=(20,10), nrows=2, ncols=3, subplot_kw={'projection': ccrs.PlateCarree(central_longitude=7, globe=None)})
        counter = 0
        for (a,b), ax in np.ndenumerate(axs):
            print('Extracting '+variable[counter]['varname'])
            ts = extract_var_plevel(ecmwf_global, var=variable[counter], p_level=p, NH=False, coarsen=False, polar = True)
        
            plot_ts(ts,  ax, var=variable[counter], polar=True)
            plot_ts(ts,  axs2[a,b], var=variable[counter], polar=False)

            ax.set_title(variable[counter]['title'], fontsize=18)
            axs2[a,b].set_title(variable[counter]['title'], fontsize=18)
            ax.coastlines()
            axs2[a,b].coastlines()
            ax.gridlines()
            axs2[a,b].gridlines()
            #ax.set_extent([-160,-170,30,90]), crs = ccrs.Orthographic(central_longitude=7, central_latitude=90, globe=None))
            counter = counter+1

        fig.suptitle('ECMWF, '+datestr+' at p = ' + str(p)+ ' hPa', fontsize=22)
        fig.tight_layout(rect=[0, 0.01, 0.99, 1])
        
        fig.savefig(basefolder+str(date.year)+'/'+'ECMWF_overview_polar_'+datestr+'_'+str(p)+'hPa'+'.pdf', dpi=500)


        #fig, axs = plt.subplots(figsize=(20,10), nrows=2, ncols=3, subplot_kw={'projection': ccrs.PlateCarree(central_longitude=7, globe=None)})
        fig2.suptitle('ECMWF, '+datestr+' at p = ' + str(p)+ ' hPa', fontsize=22)
        fig2.tight_layout(rect=[0, 0.01, 0.99, 1])
        
        fig2.savefig(basefolder+str(date.year)+'/'+'ECMWF_overview_'+datestr+'_'+str(p)+'hPa'+'.pdf', dpi=500)
