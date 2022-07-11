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

color_og= 'k'# '#008837'# '#d95f02'
color_angle= '#2c7bb6' #7b3294' # '#1b9e77'
color_Tcold= '#abd9e9' #d7191c
color_tWindow= '#fdae61'
color_Tprofile='#fc8d59'
symbols =['o','s','<','D','^', 'x','>'] 

ERROR_NAMES = {
    'pointing': 'pointing',
    'Tcold': r'$T_{cold}$' , 
    'window transmittance': 'window transmittance',
    'Tprofile':r'$T_{profile}$',
    'spectroscopy':'spectroscopy',
    'continuum':'continuum',
    'sideband':'sideband ratio',
} 

def read_sensitivity(folder ='/scratch/GROSOM/Level2/GROMORA_sensitivity/', basename='', specific_fnames =[''], v2=True):
    c=0
    for fname in specific_fnames:
        filename = os.path.join(folder, basename + fname)
        ds = xr.open_dataset(
                filename,
                group='',
                decode_times=False,
                decode_coords=True,
                # use_cftime=True,
            )
        if v2:
            ds = ds.drop_vars(['h2o_continuum_x','h2o_continuum_xa','h2o_continuum_mr','h2o_continuum_eo','h2o_continuum_es','h2o_continuum_avkm'] )
        else:
            if fname=='sensitivity_test_continuum.nc':
                ds = ds.drop_vars(['h2o_mpm93_x','h2o_mpm93_xa','h2o_mpm93_mr','h2o_mpm93_eo','h2o_mpm93_es','h2o_mpm93_avkm'] )
            else:
                ds = ds.drop_vars(['h2o_pwr98_x','h2o_pwr98_xa','h2o_pwr98_mr','h2o_pwr98_eo','h2o_pwr98_es','h2o_pwr98_avkm'] )

        ozone =ds# ds['o3_x'].isel(o3_lat=0, o3_lon=0)
        if c == 0:
            sensi_ds = ozone
            c = c+1
        else:
            sensi_ds =  xr.concat([ sensi_ds, ozone] , dim='param', coords='minimal')

    sensi_ds['o3_p'] = 1e-2*sensi_ds['o3_p'] 
    # ozone = xr.DataArray(
    #     data=stacked_vmr,
    #     dims=['p','time'],
    #     coords=dict(
    #         time=df.time.data,
    #         p=np.array([0.5, 0.7, 1.0 ,1.5  ,2.0 ,3.0,4.0 ,5.0 ,7.0 ,10.0,15.0,20.0,30.0,40.0,50.0])
    #     )
    # )
    return sensi_ds

def compare_sensi(sensi_ds, param, outname):
    color_somora='b'
    fs = 22
   #  symbols =['o','x','s','D','*','^','>'] 
    ozone = 1e6*sensi_ds.o3_x #.isel(o3_lat=0, o3_lon=0)
   #error_retrieval = 1e6*np.sqrt(sensi_ds.o3_eo**2 + sensi_ds.o3_es**2).isel(param=0,o3_lat=0, o3_lon=0)
    error_retrieval = 1e6*sensi_ds.o3_eo.isel(param=0) #,o3_lat=0, o3_lon=0)
    error_smoothing = 1e6*sensi_ds.o3_es.isel(param=0) #,o3_lat=0, o3_lon=0)

    color_shading = 'grey'

    mr = sensi_ds.o3_mr.isel(param=0).data[0]

    p_mr = sensi_ds.o3_p.data[mr>=0.8]

    fig, axs = plt.subplots(nrows=1, ncols=3, sharey=True, figsize=(16, 12))
    ozone.isel(param=0).plot(
        y='o3_p', ax=axs[0], color='k', ls='-', label='OG')
    error_retrieval.plot(
        y='o3_p', ax=axs[1], color=color_somora, lw=0, marker='X', label='obs error')
    error_smoothing.plot(
        y='o3_p', ax=axs[1], color=color_somora, lw=0, marker='o', label='smoothinh error')

    ref_diff_eo = 100*error_retrieval//ozone.isel(param=0)
    ref_diff_es = 100*error_smoothing//ozone.isel(param=0)

    ref_diff_eo.plot(
        y='o3_p', ax=axs[2], color=color_somora, lw=0, marker='X', label='obs error')
    ref_diff_es.plot(
        y='o3_p', ax=axs[2], color=color_somora, lw=0, marker='o', label='smoothing error')
    total_error = np.fabs((ozone - ozone.isel(param=0))).sum(dim='param')
    for i, pa in enumerate(param):
        diff = np.fabs(ozone.isel(param=0)-ozone.isel(param=i+1))
        rel_diff = 100*diff/ozone.isel(param=0)

        ozone.isel(param=i).plot(
            y='o3_p', ax=axs[0],ls='-', label='angle')


   ######################### 
        diff.plot(
            y='o3_p', ax=axs[1], color='k',lw=0, marker=symbols[i] , label=pa)

  
    ##################################### 
        rel_diff.plot(
            y='o3_p', ax=axs[2], color='k',lw=0,marker=symbols[i] , label=pa)


    total_error.plot(
        y='o3_p', ax=axs[1], color='red', lw=1, label='total')
    rel_diff_tot = 100*total_error/ ozone.isel(param=0)
    rel_diff_tot.plot(
        y='o3_p', ax=axs[2], color='red', lw=1, label='total')
    axs[1].xaxis.set_minor_locator(MultipleLocator(0.1))
    axs[1].xaxis.set_major_locator(MultipleLocator(0.2))
    axs[2].xaxis.set_minor_locator(MultipleLocator(1))
    axs[2].xaxis.set_major_locator(MultipleLocator(5))

   # axs[0].fill_betweenx(ozone.o3_p, (ozone.isel(param=0)-error_retrieval),(ozone.isel(param=0)+error_retrieval), color=color_gromos, alpha=0.3)
    # axs[0].fill_between(axs[0].get_xlim(),p_mr[0],1e4, color=color_shading, alpha=0.5)
    # axs[0].fill_between(axs[0].get_xlim(),p_mr[-1],1e-4, color=color_shading, alpha=0.5)
        
    axs[0].invert_yaxis()
    axs[0].set_xlim(-0.2, 9)
    axs[0].set_yscale('log')
    
    axs[0].set_ylim(200, 1e-2)
    axs[0].set_ylabel('Pressure [hPa]', fontsize=fs)
    axs[0].set_xlabel(r'O$_3$ VMR [ppmv]', fontsize=fs)
    axs[0].grid(axis='x', linewidth=0.5)

    axs[1].legend(fontsize='small') 
    axs[1].set_xlabel(r'$\Delta$O$_3$ VMR [ppmv]', fontsize=fs)
    axs[1].set_ylabel('', fontsize=fs)
    axs[1].set_xlim(-0.01, 0.4)
    axs[2].set_xlabel(r'$\Delta$O$_3$ [\%]', fontsize=fs)
    axs[2].set_xlim(-1,20)
    axs[2].set_ylabel('', fontsize=fs)

    for ax in axs:
        ax.grid(which='both', axis='y', linewidth=0.2)
        ax.grid(which='both', axis='x', linewidth=0.2)
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: '{:g}'.format(y)))

    fig.savefig(outname)


def plot_uncertainty_budget(instrument_name, sensi_ds, param, outname):
    fs = 22
    if instrument_name == 'GROMOS':
        color_plot = color_gromos
    else:
        color_plot = color_somora
    ozone = 1e6*sensi_ds.o3_x#.isel(o3_lat=0, o3_lon=0)
    eo = 1e6*sensi_ds.o3_eo.isel(param=0)
    color_shading = 'grey'
    error_retrieval = 1e6*np.sqrt(sensi_ds.o3_eo**2 + sensi_ds.o3_es**2).isel(param=0)

    mr = sensi_ds.o3_mr.isel(param=0).data

    p_mr = sensi_ds.o3_p.data[mr>=0.8]

    fig, axs = plt.subplots(nrows=1, ncols=2, sharey=True, figsize=(22, 16))
    ozone.isel(param=0).plot(
        y='o3_p', ax=axs[0], color=color_plot, ls='-', label='OG')

    total_error = np.fabs((ozone - ozone.isel(param=0))).sum(dim='param')

    # ax1 = axs[1].twiny()

    # rel_diff_tot = 100*total_error/ ozone.isel(param=0)
    # rel_diff_tot.plot(
    #     y='o3_p', ax=ax1, color='red', lw=1, label='total')
    # ax1.set_xlim(0, 50)
    # ax1.set_xlabel(r'Relative uncertainties [$\%$]', fontsize =fs, color='r')
    # ax1.xaxis.label.set_color('red')
    # ax1.tick_params(axis='x', colors='red')

    # adding altitude axis, thanks Leonie :)
    y1z=1e-3*sensi_ds.o3_z.mean(dim='param').sel(o3_p=100 ,tolerance=20,method='nearest')
    y2z=1e-3*sensi_ds.o3_z.mean(dim='param').sel(o3_p=1e-2 ,tolerance=1,method='nearest')
    ax2 = axs[1].twinx()
    ax2.set_yticks(1e-3*sensi_ds.o3_z.mean(dim='param')) #ax2.set_yticks(altitude)
    ax2.set_ylim(y1z,y2z)
    fmt = FormatStrFormatter("%.0f")
    loc=MultipleLocator(base=10)
    ax2.yaxis.set_major_formatter(fmt)
    ax2.yaxis.set_major_locator(loc)
    ax2.set_ylabel('Altitude [km] ', fontsize=fs)

    for i, pa in enumerate(param):
        if (pa == 'continuum') or (pa == 'spectroscopy'):    
            if (pa == 'spectroscopy'):
                spectro_cont_error= np.fabs(ozone.isel(param=5)- ozone.isel(param=0)) + np.fabs(ozone.isel(param=6)  - ozone.isel(param=0))
                spectro_cont_error.plot(
                    y='o3_p', ax=axs[1], color='k',lw=0, marker=symbols[i] , label='spectroscopy')
        else:
            diff = np.fabs(ozone.isel(param=0)-ozone.isel(param=param[pa]))
            rel_diff = 100*diff/ozone.isel(param=0)

            ######################### 
            diff.plot(
                y='o3_p', ax=axs[1], color='k',lw=0, marker=symbols[i] , label=ERROR_NAMES[pa])

  
    # ##################################### 
    #     rel_diff.plot(
    #         y='o3_p', ax=axs[2], color='k',lw=0,marker=symbols[i] , label=pa)

    # total_error.plot(
    #     y='o3_p', ax=axs[1], color='red', lw=1, label='total')
    eo.plot(
        y='o3_p', ax=axs[1], color=color_plot, lw=0, marker='X', label='measurement error')
    axs[1].xaxis.set_minor_locator(MultipleLocator(0.05))
    axs[1].xaxis.set_major_locator(MultipleLocator(0.1))
    # axs[2].xaxis.set_minor_locator(MultipleLocator(1))
    # axs[2].xaxis.set_major_locator(MultipleLocator(5))

    axs[0].fill_betweenx(ozone.o3_p, (ozone.isel(param=0)-eo),(ozone.isel(param=0)+eo), color=color_plot, alpha=0.3)

    axs[0].invert_yaxis()
    axs[0].set_xlim(-0.2, 9)
    axs[0].set_yscale('log')
    
    axs[0].set_ylim(100, 1e-2)
    axs[0].set_ylabel('Pressure [hPa]', fontsize=fs)

    axs[0].set_xlabel(r'O$_3$ VMR [ppmv]', fontsize=fs)

    axs[1].legend(fontsize='small', loc='best', bbox_to_anchor=(0.45, 0.45, 0.5, 0.5)) 
    axs[1].set_xlabel(r'$\Delta $O$_3$ VMR [ppmv]', fontsize=fs)
    axs[1].set_ylabel('', fontsize=fs)
    axs[1].set_xlim(-0.01, 0.5)

    # for ax in axs:

        
    # axs[2].set_xlabel(r'$\Delta $O$_3$ [%]', fontsize=fs)
    # axs[2].set_xlim(-1,20)
    # axs[2].set_ylabel('', fontsize=fs)

    for ax in axs:
        # ax.fill_between(ax.get_xlim(),p_mr[0],1e4, color=color_shading, alpha=0.5)
        # ax.fill_between(ax.get_xlim(),p_mr[-1],1e-4, color=color_shading, alpha=0.5)
        ax.grid(which='both', axis='y', linewidth=0.75)
        ax.grid(which='both', axis='x', linewidth=0.75)
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: '{:g}'.format(y)))
    
    fig.savefig(outname)

def plot_sensi_fig_gromora_paper(instrument_name = 'GROMOS'):
    # if instrument_name == 'GROMOS':
    #     outN1 = 'fig5.pdf'
    #     outN2 = 'figA1.pdf'
    # else:
    #     outN1 = 'fig6.pdf'
    #     outN2 = 'figA2.pdf'
    folder =  '/scratch/GROSOM/Level2/GROMORA_sensitivity/v3/'    
    outfolder = '/scratch/GROSOM/Level2/GROMORA_paper_plots/'
    specific_fnames = [
        'sensitivity_test_og.nc',
        'sensitivity_test_angle.nc',
        'Tcold_.nc',
        'tWindow_.nc',
        'sensitivity_test_Tprofile.nc',
        'sensitivity_test_spectroscopy_new.nc',
        'sensitivity_test_continuum.nc',
        'sensitivity_test_SB.nc',
    ] 
    param_names ={
        'pointing':1,
        'Tcold':2,
        'window transmittance':3,
        'Tprofile':4,
        'spectroscopy':5,
        'continuum':6,
        'sideband':7,
    } 
    #######################################################################################
    # Low opacity case:
    date = datetime.date(2018, 2, 26)
    basename = instrument_name + '_level2_AC240_' + date.strftime('%Y_%m_%d') + '_'
    sensi_ds = read_sensitivity( folder, basename, specific_fnames = specific_fnames)
    sensi_ds=sensi_ds.isel(time=0, o3_lon=0, o3_lat=0).drop_vars(['time','o3_lon', 'o3_lat'])
    plot_uncertainty_budget(instrument_name, sensi_ds,param=param_names, outname=outfolder+instrument_name+'_sensitivity_test_v2_'+ date.strftime('%Y_%m_%d')+'_plots.pdf')

    #######################################################################################
    # High opacity case:
    date = datetime.date(2018, 6, 9)
    
    #######################################
    basename = instrument_name + '_level2_AC240_' + date.strftime('%Y_%m_%d') + '_'
    sensi_ds = read_sensitivity( folder, basename, specific_fnames = specific_fnames)
    sensi_ds=sensi_ds.isel(time=0, o3_lon=0, o3_lat=0).drop_vars(['time','o3_lon', 'o3_lat'])
    plot_uncertainty_budget(instrument_name, sensi_ds,param=param_names, outname=outfolder+instrument_name+'_sensitivity_test_v2_'+ date.strftime('%Y_%m_%d')+'_plots.pdf')

if __name__ == "__main__":
    #date = datetime.date(2018, 2, 26)
    date = datetime.date(2018, 6, 9)
    folder =  '/scratch/GROSOM/Level2/GROMORA_sensitivity/v3/'
    instrument_name = 'SOMORA'
    basename = instrument_name + '_level2_AC240_' + date.strftime('%Y_%m_%d') + '_'
   # GROMOS_level2_AC240_2018_02_26_sensitivity_test_og.nc
    specific_fnames = [
        'sensitivity_test_og.nc',
        'sensitivity_test_angle.nc',
        'Tcold_.nc',
        'tWindow_.nc',
        'sensitivity_test_Tprofile.nc',
        'sensitivity_test_spectroscopy_new.nc',
        'sensitivity_test_continuum.nc',
        'sensitivity_test_SB.nc',
    ] 
    sensi_ds = read_sensitivity( folder, basename, specific_fnames = specific_fnames)
    sensi_ds=sensi_ds.isel(time=0, o3_lon=0, o3_lat=0).drop_vars(['time','o3_lon', 'o3_lat'])
    
    param_names ={
        'pointing':1,
        'Tcold':2,
        'window transmittance':3,
        'Tprofile':4,
        'spectroscopy':5,
        'continuum':6,
        'sideband':7,
    } 
    compare_sensi(sensi_ds, param=param_names, outname=folder+instrument_name+'_sensitivity_test_v2_'+ date.strftime('%Y_%m_%d')+'.pdf')
    plot_uncertainty_budget(instrument_name, sensi_ds,param=param_names, outname=folder+instrument_name+'_sensitivity_test_v2_'+ date.strftime('%Y_%m_%d')+'_plots.pdf')