#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Fri Apr 10 11:37:52 2020

@author: eric

Function to make the diagnostics plots of the GROMORA level 2

"""

from multiprocessing.sharedctypes import Value
import sys

# sys.path.insert(0, '/home/esauvageat/Documents/GROMORA/Analysis/GROMORA-harmo/scripts/retrieval/')
# sys.path.insert(0, '/home/esauvageat/Documents/GROMORA/Analysis/GROMORA-harmo/scripts/pyretrievals/')

import datetime
import os
from abc import ABC

import matplotlib.pyplot as plt
import netCDF4
import numpy as np
import pandas as pd
import scipy.io
import xarray as xr
from dotenv import load_dotenv
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.ticker import (AutoMinorLocator, FormatStrFormatter,
                               MultipleLocator, FuncFormatter)
import matplotlib
from level2_gromora import read_GROMORA_all, read_GROMORA_concatenated, read_gromos_v2021, read_old_SOMORA, plot_ozone_ts, read_gromos_old_FB

from base_tool import get_color, save_single_pdf
cmap = matplotlib.cm.get_cmap('plasma')

#from cmcrameri import cm
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.sans-serif": ["Free sans"]})

def plot_L2_v3(instrument_name = "GROMOS", date = [datetime.date(2017,1,9)], cycles=[14], ex = '_waccm_low_alt_dx10_v2', spectro='AC240'):
    sys.path.insert(0, './GROMORA_harmo/scripts/retrieval/')

    if instrument_name=='GROMOS':
        folder = '/storage/tub/instruments/gromos/level2/GROMORA/v2/'+date.strftime("%Y")+'/'
    else:
        folder = '/storage/tub/instruments/somora/level2/v2/'+date.strftime("%Y")+'/'

    filename = folder+instrument_name+'_level2_'+spectro+'_'+date.strftime("%Y_%m_%d")+ex+'.nc'
    
    ds = xr.open_dataset(filename)

    for c in cycles:
        plot_gromora_ds(ds.isel(time=c), spectro=spectro, outname='/scratch/GROSOM/Level2/Diagnostics_v3/'+instrument_name+'_'+spectro+'_'+date.strftime("%Y_%m_%d")+'_'+str(c)+'.pdf', instrument_name=instrument_name)

    return ds


def plot_L2(instrument_name = "GROMOS", date = [datetime.date(2017,1,9)], cycles=[14], ex = '_waccm_low_alt_dx10_v2'):
    sys.path.insert(0, './GROMORA_harmo/scripts/retrieval/')
    int_time = date[0].strftime("%Y_%m_%D")

    integration_strategy = 'classic'
    spectros = ['AC240'] 


    new_L2 = False

    plotfolder = '/scratch/GROSOM/Level2/GROMORA_paper_plots/'
    #plotfolder = '/storage/tub/instruments/somora/level2/v2/'

    cont_name = 'h2o_continuum_x' 

    colormap = 'cividis'  # 'viridis' #, batlow_map cmap_crameri cividis

    if instrument_name == "GROMOS":
        import gromos_classes as gc
        basename_lvl1 = "/storage/tub/instruments/gromos/level1/GROMORA/"+str(date[0].year)
        #basename_lvl2 = "/scratch/GROSOM/Level2/GROMORA_retrievals_polyfit2/"
        if new_L2:
            basename_lvl2 = "/storage/tub/instruments/gromos/level2/GROMORA/v2/"+str(date[0].year)
        else:
            basename_lvl2 = "/storage/tub/instruments/gromos/level2/GROMORA/v2/"+str(date[0].year)
        instrument = gc.GROMOS_LvL2(
            date=date[0],
            basename_lvl1=basename_lvl1,
            basename_lvl2=basename_lvl2,
            integration_strategy=integration_strategy,
            integration_time=int_time
        )
    elif instrument_name == "SOMORA":
        import somora_classes as sm
        basename_lvl1 = "/scratch/GROSOM/Level1/"
        if new_L2:
            #basename_lvl2 = "/scratch/GROSOM/Level2/GROMORA_retrievals_polyfit2/"
            basename_lvl2 = "/storage/tub/instruments/somora/level2/v2/"+str(date[0].year)
        else:
            basename_lvl2 = "/storage/tub/instruments/somora/level2/v2/"+str(date[0].year)
        instrument = sm.SOMORA_LvL2(
            date=date[0],
            basename_lvl1=basename_lvl1,
            basename_lvl2=basename_lvl2,
            integration_strategy=integration_strategy,
            integration_time=int_time
        )

    level2_dataset = instrument.read_level2(
        spectrometers=spectros,
        extra_base=ex
    )
    F0 = instrument.observation_frequency

    outname = plotfolder+'/'+instrument.basename_plot_level2 + instrument.datestr + ex + '_plot_sel_polyfit2'
    # if instrument_name == 'GROMOS':
    #     outname=plotfolder+'/'+'fig1'
    # elif instrument_name == 'SOMORA':
    #     outname=plotfolder+'/'+'fig2'

    instrument.plot_ozone_sel(
        level2_dataset,
        outname,
        spectro='AC240',
        cycles=cycles,
        altitude = False,
        add_baselines = False, 
        to_ppm = 1e6  
    )

def plot_L2_gromora(date = [datetime.date(2017,1,9)], cycles=[14]):
    sys.path.insert(0, './GROMORA_harmo/scripts/retrieval/')
    int_time = 1

    integration_strategy = 'classic'
    spectros = ['AC240'] 

    ex = '_v2'

    new_L2 = True

    plotfolder = '/scratch/GROSOM/Level2/GROMORA_paper_plots/'
    #plotfolder = '/storage/tub/instruments/somora/level2/v2/'

    cont_name = 'h2o_continuum_x' 

    colormap = 'cividis'  # 'viridis' #, batlow_map cmap_crameri cividis

    import gromos_classes as gc
    basename_lvl1 = "/storage/tub/instruments/gromos/level1/GROMORA/"+str(date[0].year)
    #basename_lvl2 = "/scratch/GROSOM/Level2/GROMORA_retrievals_polyfit2/"
    if new_L2:
        basename_lvl2 = "/storage/tub/instruments/gromos/level2/GROMORA/v2/"+str(date[0].year)
    else:
        basename_lvl2 = "/storage/tub/instruments/gromos/level2/GROMORA/v2/"+str(date[0].year)
    gromos_class = gc.GROMOS_LvL2(
        date=date[0],
        basename_lvl1=basename_lvl1,
        basename_lvl2=basename_lvl2,
        integration_strategy=integration_strategy,
        integration_time=int_time
    )
    import somora_classes as sm
    basename_lvl1 = "/scratch/GROSOM/Level1/"
    if new_L2:
    #basename_lvl2 = "/scratch/GROSOM/Level2/GROMORA_retrievals_polyfit2/"
        basename_lvl2 = "/storage/tub/instruments/somora/level2/v2/"+str(date[0].year)
    else:
        basename_lvl2 = "/storage/tub/instruments/somora/level2/v2/"+str(date[0].year)
    somora_class = sm.SOMORA_LvL2(
        date=date[0],
        basename_lvl1=basename_lvl1,
        basename_lvl2=basename_lvl2,
        integration_strategy=integration_strategy,
        integration_time=int_time
    )

    gromos = gromos_class.read_level2(
        spectrometers=spectros,
        extra_base=ex
    )
    somora = somora_class.read_level2(
        spectrometers=spectros,
        extra_base=ex
    )
    
    outname = plotfolder+'gromora_L2_defense_'+ gromos_class.datestr + ex + ''
    # if instrument_name == 'GROMOS':
    #     outname=plotfolder+'/'+'fig1'
    # elif instrument_name == 'SOMORA':
    #     outname=plotfolder+'/'+'fig2'

    # instrument.plot_ozone_sel(
    #     level2_dataset,
    #     outname,
    #     spectro='AC240',
    #     cycles=cycles,
    #     altitude = False,
    #     add_baselines = False, 
    #     to_ppm = 1e6  
    # )

    assert(len(cycles)==1) 
    to_ppm = 1e6 
    i = cycles[0]
    spectro = spectros[0]

    grom = gromos[spectro].isel(time=i)
    som = somora[spectro].isel(time=i)
    width = 2

    col_gromos = get_color('GROMOS')
    col_somora = get_color('SOMORA')

    fig, axs = plt.subplots(nrows=1, ncols=5, sharey=True, figsize=(28,16))
    fs = 28
    o3_grom = grom.o3_x
    o3_som = som.o3_x
    o3_apriori = grom.o3_xa
    o3_z = grom.o3_z
    fwhm=grom.o3_fwhm 
    offset=grom.o3_offset
    o3_p = grom.o3_p
    #mr = grom.o3_mr
    mr = 1e-2*grom.o3_p.data[grom.o3_mr.data>=0.75]

    #error = lvl2[spectro].isel(time=i).o3_eo +  lvl2[spectro].isel(time=i).o3_es
    error_grom = np.sqrt(grom.o3_eo**2 +  grom.o3_es**2)
    error_som = np.sqrt(som.o3_eo**2 +  som.o3_es**2)

    y_axis = o3_p/100
    y_lab = 'Pressure [hPa] '
    #axs[0].plot( error_grom*to_ppm, y_axis, '--', color=col_gromos)
    #axs[0].plot( error_som*to_ppm,y_axis, '--', color=col_somora)
    axs[0].fill_betweenx(y_axis, (o3_grom-error_grom)*to_ppm,(o3_grom+error_grom)*to_ppm, color=col_gromos, alpha=0.5)
    axs[0].fill_betweenx(y_axis, (o3_som-error_som)*to_ppm,(o3_som+error_som)*to_ppm, color=col_somora, alpha=0.5)
    axs[0].plot(o3_grom*to_ppm, y_axis,'-', linewidth=width, label='GROMOS',color=col_gromos)
    axs[0].plot(o3_som*to_ppm, y_axis,'-', linewidth=width, label='SOMORA',color=col_somora)

    axs[0].plot(o3_apriori*to_ppm, y_axis, '--', linewidth=1.6, label='apriori',color='k')
    #axs[0].set_title('O$_3$ VMR')
    axs[0].set_xlim(-0.5,9)
    axs[0].set_title("Ozone retrievals", fontsize=fs+2)

    axs[0].set_yscale('log')
    axs[0].invert_yaxis()
    axs[0].set_ylim(500,0.005)
               # axs[0].yaxis.set_major_locator(MultipleLocator(10))
              #  axs[0].yaxis.set_minor_locator(MultipleLocator(5))
    axs[0].yaxis.set_major_formatter(FuncFormatter(lambda y, _: '{:g}'.format(y)))
    axs[0].set_xlabel('O$_3$ VMR [ppmv]', fontsize=fs)

    axs[0].xaxis.set_major_locator(MultipleLocator(4))
    axs[0].xaxis.set_minor_locator(MultipleLocator(1))
    axs[0].grid(which='both',  axis='x', linewidth=0.5)
    axs[0].set_ylabel(y_lab, fontsize=fs)
    axs[0].legend(fontsize=fs)
    axs[1].plot(grom.o3_mr/4, y_axis,color='k', label='MR/4')
    axs[2].plot(som.o3_mr/4, y_axis,color='k', label='MR/4')
    axs[0].text(
    0.9,
    0.01,
    'a)',
    transform=axs[0].transAxes,
    verticalalignment="bottom",
    horizontalalignment="left",
    fontsize=fs+8
    )
    counter=0
    color_count = 0
    for j, avk in enumerate(grom.o3_avkm):
        if 0.6 <= np.sum(avk) <= 1.4:
            counter=counter+1
            if np.mod(counter,8)==0:
                axs[1].plot(avk, y_axis, color=cmap(color_count*0.25+0.01))#label='z = '+f'{o3_z.sel(o3_p=avk.o3_p).values/1e3:.0f}'+' km'
                color_count = color_count +1
            else:
                if counter==1:
                    axs[1].plot(avk,y_axis, color='silver', label='AVKs')
                else:
                    axs[1].plot(avk, y_axis, color='silver')
    color_count=0
    counter = 0
    for j, avk in enumerate(som.o3_avkm):
        if 0.6 <= np.sum(avk) <= 1.4:
            counter=counter+1
            if np.mod(counter,8)==0:
                axs[2].plot(avk, y_axis, color=cmap(color_count*0.25+0.01))#label='z = '+f'{o3_z.sel(o3_p=avk.o3_p).values/1e3:.0f}'+' km'
                color_count = color_count +1
            else:
                if counter==1:
                    axs[2].plot(avk, y_axis, color='silver', label='AVKs')
                else:
                    axs[2].plot(avk, y_axis, color='silver')

            # counter=0
            # for avk in level2_data[spectro].isel(time=i).o3_avkm:
            #     if 0.8 <= np.sum(avk) <= 1.2:
            #         counter=counter+1
            #         if np.mod(counter,5)==0:
            #             axs[1].plot(avk, o3_z / 1e3, label='z='+f'{o3_z.sel(o3_p=avk.o3_p).values/1e3:.0f}'+'km', color='r')
            #         else:
            #             axs[1].plot(avk, o3_z / 1e3, color='k')
    axs[1].set_xlabel("Averaging Kernels", fontsize=fs)
    axs[1].set_ylabel("", fontsize=fs)
    axs[1].set_xlim(-0.08,0.35)
    axs[1].set_title("GROMOS", fontsize=fs+2)
    axs[2].set_xlim(-0.08,0.35)
    axs[1].xaxis.set_major_locator(MultipleLocator(0.1))
    axs[1].xaxis.set_minor_locator(MultipleLocator(0.05))
    axs[1].legend(loc=1, fontsize=fs-2)
    axs[1].grid(which='both',  axis='x', linewidth=0.5)
    axs[1].text(
    0.9,
    0.01,
    'b)',
    transform=axs[1].transAxes,
    verticalalignment="bottom",
    horizontalalignment="left",
    fontsize=fs+8
    )
    axs[2].set_xlabel("Averaging Kernels", fontsize=fs)
    axs[2].xaxis.set_major_locator(MultipleLocator(0.1))
    axs[2].xaxis.set_minor_locator(MultipleLocator(0.05))
    axs[2].legend(loc=1, fontsize=fs-2)
    axs[2].set_title("SOMORA", fontsize=fs+2)
    axs[2].grid(which='both',  axis='x', linewidth=0.5)
    axs[2].text(
    0.9,
    0.01,
    'c)',
    transform=axs[2].transAxes,
    verticalalignment="bottom",
    horizontalalignment="left",
    fontsize=fs+8
    )
    # adding altitude axis, thanks Leonie :)
    y1z=1e-3*o3_z.sel(o3_p=48696, tolerance=100,method='nearest')
    y2z=1e-3*o3_z.sel(o3_p=0.5, tolerance=1,method='nearest')
    ax2 = axs[4].twinx()
    ax2.set_yticks(grom.o3_z) #ax2.set_yticks(altitude)
    ax2.set_ylim(y1z,y2z)
    fmt = FormatStrFormatter("%.0f")
    loc=MultipleLocator(base=10)
    ax2.yaxis.set_major_formatter(fmt)
    ax2.yaxis.set_major_locator(loc)
    ax2.set_ylabel('Altitude [km] ', fontsize=fs)
    ax2.tick_params(axis='both', which='major', labelsize=fs)
    for a in axs:
        #a.set_ylim(10,80)
        a.grid(which='both', axis='y', linewidth=0.5)
        a.grid(which='both', axis='x', linewidth=0.5)
        a.tick_params(axis='both', which='major', labelsize=fs)
    
    axs[3].plot(grom.o3_es * 1e6, y_axis, '-',lw=width, color=col_gromos, label="smoothing error")
    axs[3].plot(grom.o3_eo * 1e6, y_axis,lw=width, dashes=[6, 4] ,color=col_gromos, label="measurement error")
    axs[3].plot(som.o3_es * 1e6, y_axis,'-', lw=width, color=col_somora, label="smoothing error")
    axs[3].plot(som.o3_eo * 1e6, y_axis,lw=width, dashes=[6, 4] ,color=col_somora, label="measurement error")
    axs[3].set_xlabel(r"$e$ [ppmv]", fontsize=fs)
    axs[3].set_title("Errors", fontsize=fs+2)

    axs[3].set_ylabel("", fontsize=fs)
    axs[3].set_xlim(-0.08,1)
    axs[3].xaxis.set_major_locator(MultipleLocator(0.5))
    axs[3].xaxis.set_minor_locator(MultipleLocator(0.1))
    
    #axs2[0].legend(loc=1, fontsize=fs-2)
    legend_elements = [
        Line2D([0], [0], color='k', label='smoothing'),
        Line2D([0], [0], dashes=[6, 4], color='k', label='measurement')
        ]
    axs[3].legend(loc=1,handles=legend_elements, fontsize=fs-3)
    axs[3].grid(axis='x', linewidth=0.5)
    axs[3].text(
    0.9,
    0.01,
    'd)',
    transform=axs[3].transAxes,
    verticalalignment="bottom",
    horizontalalignment="left",
    fontsize=fs+8
    )
    axs[3].set_yscale('log')
    axs[3].invert_yaxis()
    axs[3].set_ylim(500,0.005)
    #axs[3].set_ylabel(y_lab, fontsize=fs)
    axs[3].yaxis.set_major_formatter(FuncFormatter(lambda y, _: '{:g}'.format(y)))

    axs[4].plot(grom.o3_fwhm/1e3, y_axis, lw=width,  color=col_gromos, label='FWHM')
    axs[4].plot(grom.o3_offset /1e3, y_axis,lw=width, dashes=[6, 4], color=col_gromos, label='AVKs offset')
    axs[4].plot(som.o3_fwhm/1e3, y_axis,lw=width, color=col_somora, label='FWHM')
    axs[4].plot(som.o3_offset /1e3, y_axis,lw=width, dashes=[6, 4], color=col_somora, label='AVKs offset')
    axs[4].set_xlim(-15,20)
    axs[4].set_xlabel(r"$\Delta z$ [km]", fontsize=fs)
    axs[4].set_title("Resolution and offset", fontsize=fs+2)

    axs[4].set_ylabel("", fontsize=fs)
    axs[4].xaxis.set_minor_locator(MultipleLocator(5))
    axs[4].xaxis.set_major_locator(MultipleLocator(10))
    axs[4].grid(which='both', axis='x', linewidth=0.5)
    #axs2[1].legend(loc=2, fontsize=fs-2)
    legend_elements = [
        Line2D([0], [0], color='k', label='FWHM'),
        Line2D([0], [0],  dashes=[6, 4], color='k', label='AVKs offset')
        ]
    axs[4].legend(loc=2,handles=legend_elements, fontsize=fs-2)
    axs[4].text(
    0.9,
    0.01,
    'e)',
    transform=axs[4].transAxes,
    verticalalignment="bottom",
    horizontalalignment="left",
    fontsize=fs+8
    )

    for a in axs:
        a.fill_between(a.get_xlim(),mr[0],1e4, color='gray', alpha=0.2)
        a.fill_between(a.get_xlim(),mr[-1],1e-4, color='gray', alpha=0.2)
    #fig.suptitle('O$_3$ retrievals: '+pd.to_datetime(grom.time.data).strftime('%Y-%m-%d %H:%M'), fontsize=fs+4)
    fig.tight_layout(rect=[0, 0.01, 1, 0.99])
    fig.savefig(outname+'.pdf')    

    ###################################################################################
    fig2, axs2 = plt.subplots(nrows=1, ncols=2, sharey=True, figsize=(16,16))
    width = 2
    axs2[0].plot(grom.o3_es * 1e6, y_axis, '-',lw=width, color=col_gromos, label="smoothing error")
    axs2[0].plot(grom.o3_eo * 1e6, y_axis,lw=width, dashes=[6, 4] ,color=col_gromos, label="measurement error")
    axs2[0].plot(som.o3_es * 1e6, y_axis,'-', lw=width, color=col_somora, label="smoothing error")
    axs2[0].plot(som.o3_eo * 1e6, y_axis,lw=width, dashes=[6, 4] ,color=col_somora, label="measurement error")
    axs2[0].set_xlabel("Errors [ppmv]", fontsize=fs)
    axs2[0].set_ylabel("", fontsize=fs)
    axs2[0].set_xlim(-0.08,0.801)
    axs2[0].xaxis.set_major_locator(MultipleLocator(0.4))
    axs2[0].xaxis.set_minor_locator(MultipleLocator(0.1))
    
    #axs2[0].legend(loc=1, fontsize=fs-2)
    legend_elements = [
        Line2D([0], [0], color='k', label='smoothing error'),
        Line2D([0], [0], dashes=[6, 4], color='k', label='measurement error')
        ]
    axs2[0].legend(loc=4,handles=legend_elements, fontsize=fs-2)
    axs2[0].grid(axis='x', linewidth=0.5)
    # axs2[0].text(
    # 0.9,
    # 0.01,
    # 'a)',
    # transform=axs2[0].transAxes,
    # verticalalignment="bottom",
    # horizontalalignment="left",
    # fontsize=fs+8
    # )
    axs2[0].set_yscale('log')
    axs2[0].invert_yaxis()
    axs2[0].set_ylim(500,0.005)
    axs2[0].set_ylabel(y_lab, fontsize=fs)
    axs2[0].yaxis.set_major_formatter(FuncFormatter(lambda y, _: '{:g}'.format(y)))

    axs2[1].plot(grom.o3_fwhm/1e3, y_axis, lw=width,  color=col_gromos, label='FWHM')
    axs2[1].plot(grom.o3_offset /1e3, y_axis,lw=width, dashes=[6, 4], color=col_gromos, label='AVKs offset')
    axs2[1].plot(som.o3_fwhm/1e3, y_axis,lw=width, color=col_somora, label='FWHM')
    axs2[1].plot(som.o3_offset /1e3, y_axis,lw=width, dashes=[6, 4], color=col_somora, label='AVKs offset')
    axs2[1].set_xlim(-15,20)
    axs2[1].set_xlabel("Resolution and offset [km]", fontsize=fs)
    axs2[1].set_ylabel("", fontsize=fs)
    axs2[1].xaxis.set_minor_locator(MultipleLocator(5))
    axs2[1].xaxis.set_major_locator(MultipleLocator(10))
    axs2[1].grid(which='both', axis='x', linewidth=0.5)
    #axs2[1].legend(loc=2, fontsize=fs-2)
    legend_elements = [
        Line2D([0], [0], color='k', label='FWHM'),
        Line2D([0], [0],  dashes=[6, 4], color='k', label='AVKs offset')
        ]
    axs2[1].legend(loc=2,handles=legend_elements, fontsize=fs-2)
    # axs2[1].text(
    #     0.9,
    #     0.01,
    #     'b)',
    #     transform=axs2[1].transAxes,
    #     verticalalignment="bottom",
    #     horizontalalignment="left",
    #     fontsize=fs+8
    #     )
        #axs[3].plot(level2_data[spectro].isel(time=i).h2o_x * 1e6, o3_z / 1e3, label="retrieved")
        # axs[3].set_xlabel("$VMR$ [ppm]")
        # axs[3].set_ylabel("Altitude [km]")
        # axs[3].legend()
        #axs[3].grid(axis='x', linewidth=0.5)

        # adding altitude axis, thanks Leonie :)
    y1z=1e-3*o3_z.sel(o3_p=48696, tolerance=100,method='nearest')
    y2z=1e-3*o3_z.sel(o3_p=0.5, tolerance=1,method='nearest')
    ax2 = axs2[1].twinx()
    ax2.set_yticks(grom.o3_z) #ax2.set_yticks(altitude)
    ax2.set_ylim(y1z,y2z)
    fmt = FormatStrFormatter("%.0f")
    loc=MultipleLocator(base=10)
    ax2.yaxis.set_major_formatter(fmt)
    ax2.yaxis.set_major_locator(loc)
    ax2.set_ylabel('Altitude [km] ', fontsize=fs)
    ax2.tick_params(axis='both', which='major', labelsize=fs)

    for a in axs2:
        #a.set_ylim(10,80)
        a.grid(which='both', axis='y', linewidth=0.5)
        a.grid(which='both', axis='x', linewidth=0.5)
        a.tick_params(axis='both', which='major', labelsize=fs)
    #fig2.suptitle('O$_3$ retrievals: '+pd.to_datetime(grom.time.data).strftime('%Y-%m-%d %H:%M'), fontsize=fs+4)
    fig2.tight_layout(rect=[0, 0.01, 1, 0.99])
    fig2.savefig(outname+'_error.pdf')    

def plot_gromora_ds(gromora, spectro, outname, instrument_name):
    width = 2
    add_baselines = False
    fs=26
    figure_list = list()
    to_ppm  = 1e6
    col_gromos = get_color(instrument_name)
    F0 = 142.175e9

    f_backend = np.where(gromora.bad_channels==0,gromora.f.data,  np.nan)
    y = np.where(gromora.bad_channels==0, gromora.y.data, np.nan)# gromora.y.data
    yf = np.where(gromora.bad_channels==0, gromora.yf.data, np.nan)# gromora.yf.data
    lim1 = np.nanmedian(yf)-5
    lim2 = np.nanmedian(yf)+20
    bl = np.where(gromora.bad_channels==0, gromora.y_baseline.data, np.nan)# gromora.y_baseline.data 
    # yf = 0.65*yf + 0.37*np.nanmean(yf)
    r = y - yf
    r = np.where(np.isnan(r), 0, r)
    fig, axs = plt.subplots(nrows=2, ncols=1, sharex=True, figsize=(18,12))
   # 
    axs[0].plot((f_backend - F0) / 1e6, y, color='silver', label="fitted", alpha=1)
    r_smooth = np.convolve(r, np.ones(128) / 128, mode="same")
    axs[0].plot((f_backend - F0) / 1e6, yf, color='k', label="modelled")
    axs[0].set_ylabel("$T_B$ [K]", fontsize=fs)
    axs[0].set_ylim(lim1, lim2)
    axs[0].set_xlim(-0.5,15)
    axs[0].legend(fontsize=fs)
    axs[1].plot((f_backend - F0) / 1e6, r, color='silver', label="residuals", alpha=1)
    if spectro != 'FB':
        axs[1].plot((f_backend - F0) / 1e6, r_smooth, color='k', label="residuals smooth")
    if add_baselines:
        axs[1].plot((f_backend - F0) / 1e6, bl, label="baseline")
    axs[1].set_ylim(-4, 4)
    axs[1].legend(fontsize=fs)
    axs[1].set_xlabel("RF - {:.3f} GHz [MHz]".format(F0 / 1e9), fontsize=fs)
    axs[1].set_ylabel(r"$\Delta T_B$ [K]", fontsize=fs)
    for ax in axs:
        ax.grid()
        ax.set_xlim([np.nanmin((f_backend - F0) / 1e6), np.nanmax((f_backend - F0) / 1e6)])
        ax.tick_params(axis='both', which='major', labelsize=fs)
    axs[0].set_title(instrument_name +' O$_3$ spectrum: '+pd.to_datetime(gromora.time.data).strftime('%Y-%m-%d %H:%M'), fontsize=fs+6)
    fig.tight_layout(rect=[0, 0.03, 1, 0.99])
    figure_list.append(fig)

    ###################################################################################
    fig1, axs = plt.subplots(nrows=1, ncols=4, sharey=True, figsize=(28,16))
    fig_o3, ax_o3 = plt.subplots(nrows=1, ncols=1, sharey=True, figsize=(12,12))
    fs = 28
    o3_grom = gromora.o3_x
    o3_apriori = gromora.o3_xa
    o3_z = gromora.o3_z
    fwhm=gromora.o3_fwhm 
    offset=gromora.o3_offset
    o3_p = gromora.o3_p
    mr = gromora.o3_mr
    #error = lvl2[spectro].isel(time=i).o3_eo +  lvl2[spectro].isel(time=i).o3_es
    error_grom = np.sqrt(gromora.o3_eo**2 +  gromora.o3_es**2)
    y_axis = o3_p/100
    y_lab = 'Pressure [hPa] '
    #axs[0].plot( error_grom*to_ppm, y_axis, '--', color=col_gromos)
    #axs[0].plot( error_som*to_ppm,y_axis, '--', color=col_somora)
    #axs[0].fill_betweenx(y_axis, (o3_som-error_som)*to_ppm,(o3_som+error_som)*to_ppm, color=col_somora, alpha=0.5)
    axs[0].plot(o3_grom*to_ppm, y_axis,'-', linewidth=width, label='GROMOS',color=col_gromos)

    axs[0].plot(o3_apriori*to_ppm, y_axis, '--', linewidth=1.6, label='apriori',color='k')
    #axs[0].set_title('O$_3$ VMR')
    axs[0].set_xlim(-0.5,9)
    axs[0].set_title("Ozone retrievals", fontsize=fs+2)

    axs[0].set_yscale('log')
    axs[0].invert_yaxis()
    axs[0].set_ylim(500,0.005)
               # axs[0].yaxis.set_major_locator(MultipleLocator(10))
              #  axs[0].yaxis.set_minor_locator(MultipleLocator(5))
    axs[0].yaxis.set_major_formatter(FuncFormatter(lambda y, _: '{:g}'.format(y)))
    axs[0].set_xlabel('O$_3$ VMR [ppmv]', fontsize=fs)

    axs[0].xaxis.set_major_locator(MultipleLocator(4))
    axs[0].xaxis.set_minor_locator(MultipleLocator(1))
    axs[0].grid(which='both',  axis='x', linewidth=0.5)
    axs[0].set_ylabel(y_lab, fontsize=fs)
    axs[0].legend(fontsize=fs)
    axs[1].plot(gromora.o3_mr/4, y_axis,color='k', label='MR/4')
    axs[0].text(
    0.9,
    0.01,
    'a)',
    transform=axs[0].transAxes,
    verticalalignment="bottom",
    horizontalalignment="left",
    fontsize=fs+8
    )
    counter=0
    color_count = 0
    for j, avk in enumerate(gromora.o3_avkm):
        if 0.6 <= np.sum(avk) <= 1.4:
            counter=counter+1
            if np.mod(counter,8)==0:
                axs[1].plot(avk, y_axis, color=cmap(color_count*0.25+0.01))#label='z = '+f'{o3_z.sel(o3_p=avk.o3_p).values/1e3:.0f}'+' km'
                color_count = color_count +1
            else:
                if counter==1:
                    axs[1].plot(avk,y_axis, color='silver', label='AVKs')
                else:
                    axs[1].plot(avk, y_axis, color='silver')
    color_count=0
    counter = 0

            # counter=0
            # for avk in level2_data[spectro].isel(time=i).o3_avkm:
            #     if 0.8 <= np.sum(avk) <= 1.2:
            #         counter=counter+1
            #         if np.mod(counter,5)==0:
            #             axs[1].plot(avk, o3_z / 1e3, label='z='+f'{o3_z.sel(o3_p=avk.o3_p).values/1e3:.0f}'+'km', color='r')
            #         else:
            #             axs[1].plot(avk, o3_z / 1e3, color='k')
    axs[1].set_xlabel("Averaging Kernels", fontsize=fs)
    axs[1].set_ylabel("", fontsize=fs)
    axs[1].set_xlim(-0.08,0.35)
    axs[1].set_title("GROMOS", fontsize=fs+2)
    axs[2].set_xlim(-0.08,0.35)
    axs[1].xaxis.set_major_locator(MultipleLocator(0.1))
    axs[1].xaxis.set_minor_locator(MultipleLocator(0.05))
    axs[1].legend(loc=1, fontsize=fs-2)
    axs[1].grid(which='both',  axis='x', linewidth=0.5)
    axs[1].text(
    0.9,
    0.01,
    'b)',
    transform=axs[1].transAxes,
    verticalalignment="bottom",
    horizontalalignment="left",
    fontsize=fs+8
    )

    # adding altitude axis, thanks Leonie :)
    y1z=1e-3*o3_z.sel(o3_p=48696, tolerance=100,method='nearest')
    y2z=1e-3*o3_z.sel(o3_p=0.5, tolerance=1,method='nearest')
    ax2 = axs[3].twinx()
    ax2.set_yticks(gromora.o3_z) #ax2.set_yticks(altitude)
    ax2.set_ylim(y1z,y2z)
    fmt = FormatStrFormatter("%.0f")
    loc=MultipleLocator(base=10)
    ax2.yaxis.set_major_formatter(fmt)
    ax2.yaxis.set_major_locator(loc)
    ax2.set_ylabel('Altitude [km] ', fontsize=fs)
    ax2.tick_params(axis='both', which='major', labelsize=fs)
    for a in axs:
        #a.set_ylim(10,80)
        a.grid(which='both', axis='y', linewidth=0.5)
        a.grid(which='both', axis='x', linewidth=0.5)
        a.tick_params(axis='both', which='major', labelsize=fs)
    
    axs[2].plot(gromora.o3_es * 1e6, y_axis, '-',lw=width, color=col_gromos, label="smoothing error")
    axs[2].plot(gromora.o3_eo * 1e6, y_axis,lw=width, dashes=[6, 4] ,color=col_gromos, label="measurement error")
    axs[2].set_xlabel(r"$e$ [ppmv]", fontsize=fs)
    axs[2].set_title("Errors", fontsize=fs+2)

    axs[2].set_ylabel("", fontsize=fs)
    axs[2].set_xlim(-0.08,1)
    axs[2].xaxis.set_major_locator(MultipleLocator(0.5))
    axs[2].xaxis.set_minor_locator(MultipleLocator(0.1))
    
    #axs2[0].legend(loc=1, fontsize=fs-2)
    legend_elements = [
        Line2D([0], [0], color='k', label='smoothing'),
        Line2D([0], [0], dashes=[6, 4], color='k', label='measurement')
        ]
    axs[2].legend(loc=1,handles=legend_elements, fontsize=fs-3)
    axs[2].grid(axis='x', linewidth=0.5)
    axs[2].text(
    0.9,
    0.01,
    'd)',
    transform=axs[2].transAxes,
    verticalalignment="bottom",
    horizontalalignment="left",
    fontsize=fs+8
    )
    axs[3].set_yscale('log')
    axs[3].invert_yaxis()
    axs[3].set_ylim(500,0.005)
    #axs[3].set_ylabel(y_lab, fontsize=fs)
    axs[3].yaxis.set_major_formatter(FuncFormatter(lambda y, _: '{:g}'.format(y)))

    axs[3].plot(gromora.o3_fwhm/1e3, y_axis, lw=width,  color=col_gromos, label='FWHM')
    axs[3].plot(gromora.o3_offset /1e3, y_axis,lw=width, dashes=[6, 4], color=col_gromos, label='AVKs offset')
    axs[3].set_xlim(-15,20)
    axs[3].set_xlabel(r"$\Delta z$ [km]", fontsize=fs)
    axs[3].set_title("Resolution and offset", fontsize=fs+2)

    axs[3].set_ylabel("", fontsize=fs)
    axs[3].xaxis.set_minor_locator(MultipleLocator(5))
    axs[3].xaxis.set_major_locator(MultipleLocator(10))
    axs[3].grid(which='both', axis='x', linewidth=0.5)
    #axs2[1].legend(loc=2, fontsize=fs-2)
    legend_elements = [
        Line2D([0], [0], color='k', label='FWHM'),
        Line2D([0], [0],  dashes=[6, 4], color='k', label='AVKs offset')
        ]
    axs[3].legend(loc=2,handles=legend_elements, fontsize=fs-2)
    axs[3].text(
    0.9,
    0.01,
    'e)',
    transform=axs[3].transAxes,
    verticalalignment="bottom",
    horizontalalignment="left",
    fontsize=fs+8
    )
    #fig.suptitle('O$_3$ retrievals: '+pd.to_datetime(gromora.time.data).strftime('%Y-%m-%d %H:%M'), fontsize=fs+4)
    fig1.tight_layout(rect=[0, 0.01, 1, 0.99])
    figure_list.append(fig1)
    #fig.savefig(outname+'.pdf')   
    # 
    # 
    ax_o3.plot(o3_grom*to_ppm, y_axis,'-', linewidth=width, label='retrieved',color='k')
    ax_o3.fill_betweenx(y_axis, (o3_grom-error_grom)*to_ppm,(o3_grom+error_grom)*to_ppm, color='silver', alpha=1)

    ax_o3.plot(o3_apriori*to_ppm, y_axis, '--', linewidth=1.6, label='apriori', color='k')
    ax_o3.set_xlim(-0.5,9)
    ax_o3.set_title("Ozone retrievals", fontsize=fs+2)
    ax_o3.set_yscale('log')
    ax_o3.invert_yaxis()
    ax_o3.set_ylim(500,0.005)
    ax_o3.yaxis.set_major_formatter(FuncFormatter(lambda y, _: '{:g}'.format(y)))
    ax_o3.set_xlabel('O$_3$ VMR [ppmv]', fontsize=fs)
    ax_o3.xaxis.set_major_locator(MultipleLocator(4))
    ax_o3.xaxis.set_minor_locator(MultipleLocator(1))
    ax_o3.grid(which='both',  axis='x', linewidth=0.5)
    ax_o3.set_ylabel(y_lab, fontsize=fs)
    ax_o3.legend(fontsize=fs)
    # adding altitude axis, thanks Leonie :)

    ax2_o3 = ax_o3.twinx()
    ax2_o3.set_yticks(gromora.o3_z) #ax2.set_yticks(altitude)
    ax2_o3.set_ylim(y1z,y2z)
    ax2_o3.yaxis.set_major_formatter(fmt)
    ax2_o3.yaxis.set_major_locator(loc)
    ax2_o3.set_ylabel('Altitude [km] ', fontsize=fs)
    ax2_o3.tick_params(axis='both', which='major', labelsize=fs)

    ax_o3.grid(which='both', axis='y', linewidth=0.5)
    ax_o3.grid(which='both', axis='x', linewidth=0.5)
    ax_o3.tick_params(axis='both', which='major', labelsize=fs)

    fig_o3.tight_layout(rect=[0, 0.01, 1, 0.99])
    fig_o3.savefig(outname+'o3.pdf')  

    ###################################################################################
    fig2, axs2 = plt.subplots(nrows=1, ncols=2, sharey=True, figsize=(16,16))
    width = 2
    axs2[0].plot(gromora.o3_es * 1e6, y_axis, '-',lw=width, color=col_gromos, label="smoothing error")
    axs2[0].plot(gromora.o3_eo * 1e6, y_axis,lw=width, dashes=[6, 4] ,color=col_gromos, label="measurement error")
    axs2[0].set_xlabel("Errors [ppmv]", fontsize=fs)
    axs2[0].set_ylabel("", fontsize=fs)
    axs2[0].set_xlim(-0.08,0.801)
    axs2[0].xaxis.set_major_locator(MultipleLocator(0.4))
    axs2[0].xaxis.set_minor_locator(MultipleLocator(0.1))
    
    #axs2[0].legend(loc=1, fontsize=fs-2)
    legend_elements = [
        Line2D([0], [0], color='k', label='smoothing error'),
        Line2D([0], [0], dashes=[6, 4], color='k', label='measurement error')
        ]
    axs2[0].legend(loc=4,handles=legend_elements, fontsize=fs-2)
    axs2[0].grid(axis='x', linewidth=0.5)
    # axs2[0].text(
    # 0.9,
    # 0.01,
    # 'a)',
    # transform=axs2[0].transAxes,
    # verticalalignment="bottom",
    # horizontalalignment="left",
    # fontsize=fs+8
    # )
    axs2[0].set_yscale('log')
    axs2[0].invert_yaxis()
    axs2[0].set_ylim(500,0.005)
    axs2[0].set_ylabel(y_lab, fontsize=fs)
    axs2[0].yaxis.set_major_formatter(FuncFormatter(lambda y, _: '{:g}'.format(y)))

    axs2[1].plot(gromora.o3_fwhm/1e3, y_axis, lw=width,  color=col_gromos, label='FWHM')
    axs2[1].plot(gromora.o3_offset /1e3, y_axis,lw=width, dashes=[6, 4], color=col_gromos, label='AVKs offset')

    axs2[1].set_xlim(-15,20)
    axs2[1].set_xlabel("Resolution and offset [km]", fontsize=fs)
    axs2[1].set_ylabel("", fontsize=fs)
    axs2[1].xaxis.set_minor_locator(MultipleLocator(5))
    axs2[1].xaxis.set_major_locator(MultipleLocator(10))
    axs2[1].grid(which='both', axis='x', linewidth=0.5)
    #axs2[1].legend(loc=2, fontsize=fs-2)
    legend_elements = [
        Line2D([0], [0], color='k', label='FWHM'),
        Line2D([0], [0],  dashes=[6, 4], color='k', label='AVKs offset')
        ]
    axs2[1].legend(loc=2,handles=legend_elements, fontsize=fs-2)
    # axs2[1].text(
    #     0.9,
    #     0.01,
    #     'b)',
    #     transform=axs2[1].transAxes,
    #     verticalalignment="bottom",
    #     horizontalalignment="left",
    #     fontsize=fs+8
    #     )
        #axs[3].plot(level2_data[spectro].isel(time=i).h2o_x * 1e6, o3_z / 1e3, label="retrieved")
        # axs[3].set_xlabel("$VMR$ [ppm]")
        # axs[3].set_ylabel("Altitude [km]")
        # axs[3].legend()
        #axs[3].grid(axis='x', linewidth=0.5)

        # adding altitude axis, thanks Leonie :)
    y1z=1e-3*o3_z.sel(o3_p=48696, tolerance=100,method='nearest')
    y2z=1e-3*o3_z.sel(o3_p=0.5, tolerance=1,method='nearest')
    ax2 = axs2[1].twinx()
    ax2.set_yticks(gromora.o3_z) #ax2.set_yticks(altitude)
    ax2.set_ylim(y1z,y2z)
    fmt = FormatStrFormatter("%.0f")
    loc=MultipleLocator(base=10)
    ax2.yaxis.set_major_formatter(fmt)
    ax2.yaxis.set_major_locator(loc)
    ax2.set_ylabel('Altitude [km] ', fontsize=fs)
    ax2.tick_params(axis='both', which='major', labelsize=fs)

    for a in axs2:
        #a.set_ylim(10,80)
        a.grid(which='both', axis='y', linewidth=0.5)
        a.grid(which='both', axis='x', linewidth=0.5)
        a.tick_params(axis='both', which='major', labelsize=fs)
    #fig2.suptitle('O$_3$ retrievals: '+pd.to_datetime(gromora.time.data).strftime('%Y-%m-%d %H:%M'), fontsize=fs+4)
    fig2.tight_layout(rect=[0, 0.01, 1, 0.99])
    figure_list.append(fig2)
    #fig2.savefig(outname+'_error.pdf')    
    save_single_pdf(outname, figure_list)

if __name__ == "__main__":
    spectro = 'AC240'
    instrument_name = "SOMORA"
    #plot_figures_gromora_paper(do_sensitivity = False, do_L2=True, do_comp=False, do_old=False)
    plot_L2_gromora(date = [datetime.date(2017,1,9)], cycles=[14])
    #plot_L2(date = [datetime.date(2017,1,9)], cycles=[14])
    #gromos = plot_L2_v3(instrument_name, date =datetime.date(2017,1,9), cycles=[1], ex='_v2', spectro=spectro)

    #plot_gromora_ds(gromos.isel(time=21), spectro=spectro, outname='/scratch/GROSOM/Level2/Diagnostics_v3/'+instrument_name+'_test_'+spectro+'.pdf')