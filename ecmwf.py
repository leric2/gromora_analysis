#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 06.01.22

@author: Eric Sauvageat

This is the main script for the ECMWF data treatment in the frame of the GROMORA project

This module contains the code to read the ECMWF daily global and local files and to do the daily plots.

"""

import datetime
import os

import matplotlib.pyplot as plt
import netCDF4
import numpy as np
import pandas as pd
from base_tool import save_single_pdf

import levels as ecmwf_levels

import cartopy.crs as ccrs
import typhon

import xarray as xr
colormap = 'cividis'

Mair = 28.9644
Mozone= 47.9982

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.sans-serif": ["Free sans"]})

ozone = {
    'varname':'o3',
    'M':47.9982,
    'factor':1e6,
    'colormap':plt.get_cmap('density'),
    'title':'Ozone',
    'colorbar_title': 'VMR [ppmv] ',
    'polar_vmin': 3,
    'polar_vmax': 10,
    'global_vmin': 3,
    'global_vmax': 8,
}
temperature = {
    'varname':'t',
    'M':47.9982,
    'factor':1,
    'colormap':plt.get_cmap('temperature'),
    'title':'Temperature',
    'colorbar_title': 'T [K]',
    'polar_vmin': 185,
    'polar_vmax': 230,
    'global_vmin': 220,
    'global_vmax': 280,
}
humidity = {
    'varname':'q',
    'M':47.9982,
    'factor':1e6/0.62199, # MMR = 0.62199*VMR (see Vaisala conversion)
    'colormap':plt.get_cmap('density'),
    'title':'Water Vapour',
    'colorbar_title': 'VMR [ppmv]',
    'polar_vmin': 3,
    'polar_vmax': 6,
    'global_vmin': 3,
    'global_vmax': 7,
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
    'colorbar_title': r'vo [s$^{-1}$]',
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

def mmr_2_vmr(mmr):
    vmr = ozone['factor']*mmr*Mair/ozone['M']
    return vmr
    

def read_ECMWF(date, location='BERN'):
    ECMWF_folder = '/storage/tub/atmosphere/ecmwf/locations/Bern/'
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

    print('ECMWF data read from:', ECMWF_folder)
    return ecmwf_ts

def read_ERA5(date, years=[2017], location='SwissPlateau', daybyday=False, save=False):
   
    if daybyday:
        ECMWF_folder = '/storage/tub/atmosphere/ecmwf/locations/SwissPlateau/'
        counter = 0
        for d in date:
            ECMWF_file = os.path.join(
                ECMWF_folder, 'ecmwf_era5_'+location+'_'+d.strftime('%Y%m%d')+'.nc')

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
        if save:
            ecmwf_ts.to_netcdf('/scratch/GROSOM/DiurnalCycles/climatology/era5_switzerland_'+d.strftime('%Y')+'.nc')
    else:
        ECMWF_folder = '/scratch/GROSOM/DiurnalCycles/climatology/'
        counter = 0
        for year in years:
            ECMWF_file = os.path.join(
                ECMWF_folder, 'era5_switzerland_'+str(year)+'.nc')

            ecmwf_og = xr.open_dataset(
                ECMWF_file,
                decode_times=True,
                decode_coords=True,
                use_cftime=False,
            )
            if counter == 0:
                ecmwf_ts = ecmwf_og
            else:
                ecmwf_ts = xr.concat([ecmwf_ts, ecmwf_og], dim='time')

            counter = counter + 1

    print('ECMWF data read from:', ECMWF_folder)
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

def read_ERA5_global(date):
    ECMWF_folder = '/storage/tub/atmosphere/ecmwf/era5/europe/'
    year = date.year
    ECMWF_file = os.path.join(ECMWF_folder+str(year), 'ECMWF_ERA5_'+date.strftime('%Y%m%d')+'.nc')

    ecmwf_og = xr.open_dataset(
        ECMWF_file,
        decode_times=True,
        decode_coords=True,
        use_cftime=False,
    )

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

def extract_var_plevel(ecmwf_global, var=ozone, p_level=100, NH=True, coarsen=True, polar = True, zoom=None):
    if polar:
        mean_p = ecmwf_global.pressure.where(ecmwf_global.latitude>60).mean(dim=['time','latitude','longitude'])/100
    else: 
        mean_p = ecmwf_global.pressure.mean(dim=['time','latitude','longitude'])/100
    lvl = np.abs(mean_p.values-p_level).argmin()

    ecmwf_o3 = ecmwf_global[var['varname']].isel(level=lvl)#.mean(dim='time')
    #o3_ecmwf = ecmwf_ts.isel(loc=0).ozone_mass_mixing_ratio

    if NH:
        ecmwf_o3 = ecmwf_o3.where(ecmwf_o3.latitude>0, drop=True)

    if zoom is not None:
        # zoom if a list with [lat0, lon0, lat1, lon1]
        ecmwf_o3 = ecmwf_o3.where(ecmwf_o3.latitude>zoom[0], drop=True).where(ecmwf_o3.longitude>zoom[1], drop=True).where(ecmwf_o3.latitude<zoom[2], drop=True).where(ecmwf_o3.longitude<zoom[3], drop=True)
    
    if coarsen:
        ecmwf_o3 = ecmwf_o3.coarsen(longitude=2).mean().coarsen(latitude=1, boundary='trim').mean()
    
    # if (var['varname'] == 'o3'):
    #     ecmwf_o3 = ecmwf_o3*Mair/var['M']
    
    # ecmwf_o3=ecmwf_o3*var['factor']

    return ecmwf_o3

def extract_plevel_zoom(ecmwf_global, p_level=100, zoom=None):

    #o3_ecmwf = ecmwf_ts.isel(loc=0).ozone_mass_mixing_ratio
    if zoom is not None:
        ecmwf_o3 = ecmwf_global.where(ecmwf_global.latitude>zoom[0], drop=True).where(ecmwf_global.longitude>zoom[1], drop=True).where(ecmwf_global.latitude<zoom[2], drop=True).where(ecmwf_global.longitude<zoom[3], drop=True)
    else:
        ecmwf_o3 = ecmwf_global
   
    mean_p = ecmwf_o3.pressure.mean(dim=['time','latitude','longitude'])/100
    lvl = np.abs(mean_p.values-p_level).argmin()

    ecmwf_o3 = ecmwf_o3.isel(level=lvl)#.mean(dim='time')

    # ecmwf_o3['o3'].data  = ecmwf_o3['o3'].data*ozone['factor']*Mair/ozone['M']
 
    return ecmwf_o3

def plot_ozone(o3, ax):
    #fig = plt.figure(figsize=(8, 6))
    #axs = plt.axes(projection=ccrs.Orthographic(central_longitude=7, central_latitude=50, globe=None))
    p = o3.plot(
        ax=ax,
        transform=ccrs.PlateCarree(),
        cbar_kwargs={"label": "ozone ppmv"},
        cmap = ozone['colormap'],
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
    
    if (var['varname'] == 'o3'):
        ts = ts*Mair/var['M']
    
    ts=ts*var['factor']
    
    transform=ccrs.PlateCarree()
    im = ts.plot(
        ax=ax,
        #levels=8,
        transform=transform,
        add_colorbar=False,
        #cbar_kwargs={"label": var['colorbar_title'], 'shrink':0.85},
        cmap = var['colormap'],
        vmin=minval  ,
        vmax=maxval,
        linewidth=0,
        rasterized=True,
    ) 
    # cbar.ax.tick_params(labelsize=20)
    cbar = ax.figure.colorbar(im, ax=ax, **{"label": var['colorbar_title'],'shrink':0.6})
    cbar.set_label(label=var['colorbar_title'], size=22)
    #cbar.ax.tick_label(labelsize=22) 
    cbar.ax.tick_params(labelsize=20) 
    # im.cbar.set_label(label='Temperature (°C)', size=30, weight='bold')
    ax.set_title(var['title'] +': '+ pd.to_datetime(ts.time.data).strftime('%Y-%m-%d:%H:%M'), fontsize=24)
    # cax = fig.add_axes([ax.get_position().x1+0.01,ax.get_position().y0,0.02,ax.get_position().height])
    # plt.colorbar(im, cax=cax) # Similar to fig.colorbar(im, cax = cax)
    return ax


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

def extract_date(date, p_level = 10, zoom=None, ERA5=True, extra = ''):
    #date=date[0]
    if ERA5:
        ecmwf_global = read_ERA5_global(date) 
    else:
        ecmwf_global = read_ECMWF_global(date)

    ts = extract_plevel_zoom(ecmwf_global, p_level, zoom)
    return ts

def plot_date(date, p_levels = [10], square =True, polar = False, zoom=None, extra = ''):
    basefolder = '/home/esauvageat/Documents/ECMWF/plots/'
    basefolder = '/storage/tub/atmosphere/ecmwf/daily_plots/'+str(date.year)+'/'
    #ecmwf_ds = read_ECMWF( date, 'Bern')

    #variable = ['t', 'u', 'v', 'o3', 'q','vo'] # Options are t, q, w, vo, o3, z, u, v
    variable = [temperature, zonal_velocity, meridional_velocity, ozone,humidity,relative_vorticity]

    #date=date[0]
    ecmwf_global = read_ECMWF_global(date)

    for p in p_levels:
        fig, axs = plt.subplots(figsize=(20,10), nrows=2, ncols=3, subplot_kw={'projection': ccrs.Orthographic(central_longitude=7, central_latitude=90, globe=None)})
        fig2, axs2 = plt.subplots(figsize=(14,6), nrows=2, ncols=3, subplot_kw={'projection': ccrs.PlateCarree()})

        #fig, axs = plt.subplots(figsize=(20,10), nrows=2, ncols=3, subplot_kw={'projection': ccrs.PlateCarree(central_longitude=7, globe=None)})
        counter = 0
        for (a,b), ax in np.ndenumerate(axs):
            print('Extracting '+variable[counter]['varname'])
            ts = extract_var_plevel(ecmwf_global, var=variable[counter], p_level=p, NH=False, coarsen=False, polar = True, zoom = zoom)

            if polar:
                plot_ts(ts.isel(time=0),  ax, var=variable[counter], polar=True)
                ax.set_title(variable[counter]['title'], fontsize=18)
                ax.coastlines()
                ax.gridlines()
            
            if square:
                plot_ts(ts.isel(time=0),  axs2[a,b], var=variable[counter], polar=False)
                axs2[a,b].set_title(variable[counter]['title'], fontsize=18)
                axs2[a,b].coastlines()
                axs2[a,b].gridlines()

            # if variable[counter]['varname'] in save_var:
            #     ts['level'] = p
            #     ts.rename({'level':'pressure'})
            #     to_return = ts

            #ax.set_extent([-160,-170,30,90]), crs = ccrs.Orthographic(central_longitude=7, central_latitude=90, globe=None))
            counter = counter+1


        fig.suptitle('Pressure = ' + str(p)+ ' hPa, '+ str(date), fontsize=18)
        fig.tight_layout(rect=[0, 0.01, 0.99, 1])
        
        if polar:
            fig.savefig(basefolder+'ECMWF_overview_polar_'+extra+str(date)+'_'+str(p)+'hPa'+'.pdf', dpi=500)

        #fig, axs = plt.subplots(figsize=(20,10), nrows=2, ncols=3, subplot_kw={'projection': ccrs.PlateCarree(central_longitude=7, globe=None)})
        fig2.suptitle('Pressure = ' + str(p)+ ' hPa, ' + str(date), fontsize=22)
        fig2.tight_layout(rect=[0, 0.01, 0.99, 1])
        
        if square:
            fig2.savefig(basefolder+'ECMWF_overview_'+extra+str(date)+'_'+str(p)+'hPa'+'.pdf', dpi=500)

def plot_ozone_map(filename, freq='12H'):
    ecmwf = xr.open_dataset(filename)
    ecmwf = ecmwf.resample(time=freq).mean()
    fig_list=list()
    for i,t in enumerate(ecmwf.time.data):
        fig, axs = plt.subplots(figsize=(16,12), nrows=2, ncols=2, subplot_kw={'projection': ccrs.Orthographic(central_longitude=7, central_latitude=47, globe=None)})
        
        h2o_vmr = ecmwf.isel(time=i).q/(1-ecmwf.isel(time=i).q)
        axs[0,0] = plot_ts(ecmwf.isel(time=i).t, axs[0,0], var=temperature, polar=False)
        axs[0,1]= plot_ts(ecmwf.isel(time=i).o3,  axs[0,1], var=ozone, polar=False)
        axs[1,0] = plot_ts(h2o_vmr, axs[1,0], var=humidity, polar=False)
        axs[1,1] = plot_ts(ecmwf.isel(time=i).vo, axs[1,1], var=relative_vorticity, polar=False)
        
        for ax in axs[0,:]:
            ax.coastlines()
            ax.gridlines()

        for ax in axs[1,:]:
            ax.coastlines()
            ax.gridlines()
        # fig.suptitle('Ozone: '+ str(t), fontsize=22)
        fig.tight_layout(rect=[0, 0.01, 0.99, 1])

        fig_list.append(fig)    

    save_single_pdf('/home/esauvageat/Downloads/europe_maps_era_'+str(t)+'.pdf', fig_list)

def degree_formatter(x, pos):
    """Create degree ticklabels for radian data., from https://www.radiativetransfer.org/misc/typhon/doc/typhon.plots.cm.html"""
    return '{:.0f}\N{DEGREE SIGN}'.format(np.rad2deg(x))


def plot_situation_map(filename, freq='12H'):
    ecmwf = xr.open_dataset(filename)
    ecmwf = ecmwf.resample(time=freq).mean()
    fig_list=list()
    for i,t in enumerate(ecmwf.time.data):
        fig, axs = plt.subplots(figsize=(16,12), nrows=2, ncols=2, subplot_kw={'projection': ccrs.Orthographic(central_longitude=7, central_latitude=47, globe=None)})
        
        h2o_vmr = ecmwf.isel(time=i).q/(1-ecmwf.isel(time=i).q)

        # downscaling for wind:
        ecmwf_wind = ecmwf.coarsen(latitude=4, longitude=4, boundary='pad').mean()
        u, v = ecmwf_wind.isel(time=i).u.data, ecmwf_wind.isel(time=i).v.data
        wdir = np.arctan2(u, v) + np.pi

        axs[0,0] = plot_ts(ecmwf.isel(time=i).t, axs[0,0], var=temperature, polar=False)
        axs[0,1]= plot_ts(ecmwf.isel(time=i).o3,  axs[0,1], var=ozone, polar=False)
        #axs[1,0] = plot_ts(h2o_vmr, axs[1,0], var=humidity, polar=False)
        sm = axs[1,0].quiver(ecmwf_wind.isel(time=i).longitude.data ,ecmwf_wind.isel(time=i).latitude.data, u, v, wdir, cmap=plt.get_cmap('phase',8), transform=ccrs.PlateCarree())
        axs[1,1] = plot_ts(ecmwf.isel(time=i).vo, axs[1,1], var=relative_vorticity, polar=False)
        
        # Nice colorbar for wind: https://www.radiativetransfer.org/misc/typhon/doc/typhon.plots.cm.html
        cb = fig.colorbar(sm, ax=axs[1,0], label='Wind direction', format=degree_formatter, pad=0.02, shrink=0.6)
        cb.set_ticks(np.linspace(0, 2 * np.pi, 9))
        cb.set_label(label='Wind direction', size=22)
        cb.ax.tick_params(labelsize=22) 
        axs[1,0].set_title('Winds: ' + pd.to_datetime(t).strftime('%Y-%m-%d:%H:%M'), fontsize=24)
        
        for ax in axs[0,:]:
            ax.coastlines()
            ax.gridlines()

        for ax in axs[1,:]:
            ax.coastlines()
            ax.gridlines()
        # fig.suptitle('Ozone: '+ str(t), fontsize=22)
        fig.tight_layout(rect=[0, 0.01, 0.99, 1])

        fig_list.append(fig)    

    save_single_pdf('/home/esauvageat/Downloads/europe_maps_era_'+str(t)+'.pdf', fig_list)


if __name__ == "__main__":
    ozone_maps_only = True

    # for yr in [2015]:
    dlat = 28
    dlon = 30
    #     range_ecmwf = slice(str(yr)+'-01-01',str(yr)+'-01-10')
    #     date = pd.date_range(start=range_ecmwf.start, end=range_ecmwf.stop)

    #     ecmwf_ts = read_ERA5(date, years=[yr], location='SwissPlateau', daybyday=True, save=False)

    if ozone_maps_only:
        # plot_ozone_map('/scratch/GROSOM/DiurnalCycles/event_2015_april/ozone_ecmwf_all_2015-04-15.nc')

        #plot_ozone_map('/home/esauvageat/Downloads/ecmwf_all_europe_2015_01_15.nc', freq='6H')
        plot_situation_map('/home/esauvageat/Downloads/ecmwf_all_NH_5hPa_2014_03_28.nc', freq='6H')
    else:
        date = pd.date_range(start='2014-03-16', end='2014-03-28', freq='1D')
        ozone_ts = list()
        for d in date:
            print(d.strftime('%Y_%m_%d')+' extracted')
            ts = extract_date(d, p_level=5, zoom=[30, -180, 90, 180], ERA5=False, extra='')
            #o3 = plot_date(d, p_levels = [10], square =False, polar = True, zoom= None, extra='') zoom=[24, -28, 70, 37],
            ozone_ts.append(ts)

        ozone_ds = xr.merge(ozone_ts)
        ozone_ds.to_netcdf('/home/esauvageat/Downloads/ecmwf_all_NH_5hPa_'+d.strftime('%Y_%m_%d')+'.nc')
