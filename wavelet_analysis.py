#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 01.22

@author: Eric Sauvageat

First draft code for wavelet analysis of the GROMORA v2 -> NOT WORKING at the moment.

"""
import datetime
import os

import matplotlib.pyplot as plt
import netCDF4
import numpy as np
from numpy.lib.shape_base import dsplit
import pandas as pd

import xarray as xr

from matplotlib.ticker import (MultipleLocator, FormatStrFormatter, AutoMinorLocator)
from matplotlib.lines import Line2D
import matplotlib.ticker as ticker
import matplotlib
from matplotlib.gridspec import GridSpec
cmap = matplotlib.cm.get_cmap('plasma')

import pywt
from wavelets.wave_python.waveletFunctions import wavelet, wave_signif
from level2_gromora_diagnostics import read_GROMORA_all

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.sans-serif": ["Free sans"]})

plt.rcParams['xtick.labelsize'] = 22
plt.rcParams['ytick.labelsize'] = 22
plt.rcParams['font.size'] = 22
plt.rcParams['axes.titlesize'] = 22

color_gromos= '#d7191c'# '#008837'# '#d95f02'
color_somora= '#2c7bb6' #7b3294' # '#1b9e77'

def wavelet_analysis(o3):
    o3 = o3 - np.mean(o3)
    variance = np.std(o3, ddof=1) ** 2
    print("variance = ", variance)

    # ----------C-O-M-P-U-T-A-T-I-O-N------S-T-A-R-T-S------H-E-R-E---------------

    # normalize by standard deviation (not necessary, but makes it easier
    # to compare with plot on Interactive Wavelet page, at
    # "http://paos.colorado.edu/research/wavelets/plot/"
    if 0:
        variance = 1.0
        o3 = o3 / np.std(o3, ddof=1)
    n = len(o3)
    dt = 1/(365)
    time = np.arange(len(o3)) * dt  # construct time array
    #xlim = ([2016, 2016])  # plotting range
    pad = 1  # pad the time series with zeroes (recommended)
    dj = 0.1  # this will do 4 sub-octaves per octave
    s0 = 10 * dt  # this says start at a scale of 6 months
    j1 = 7 / dj  # this says do 7 powers-of-two with dj sub-octaves each
    lag1 = 0.72  # lag-1 autocorrelation for red noise background
    print("lag1 = ", lag1)
    mother = 'MORLET'

    # Wavelet transform:
    wave, period, scale, coi = wavelet(o3, dt, pad, dj, s0, j1, mother)
    power = (np.abs(wave)) ** 2  # compute wavelet power spectrum
    global_ws = (np.sum(power, axis=1) / n)  # time-average over all times

    # Significance levels:
    signif = wave_signif(([variance]), dt=dt, sigtest=0, scale=scale,
        lag1=lag1, mother=mother)
    # expand signif --> (J+1)x(N) array
    sig95 = signif[:, np.newaxis].dot(np.ones(n)[np.newaxis, :])
    sig95 = power / sig95  # where ratio > 1, power is significant

    # Global wavelet spectrum & significance levels:
    dof = n - scale  # the -scale corrects for padding at edges
    global_signif = wave_signif(variance, dt=dt, scale=scale, sigtest=1,
        lag1=lag1, dof=dof, mother=mother)

    # Scale-average between El Nino periods of 2--8 years
    avg = np.logical_and(scale >= 2, scale < 8)
    Cdelta = 0.776  # this is for the MORLET wavelet
    # expand scale --> (J+1)x(N) array
    scale_avg = scale[:, np.newaxis].dot(np.ones(n)[np.newaxis, :])
    scale_avg = power / scale_avg  # [Eqn(24)]
    scale_avg = dj * dt / Cdelta * sum(scale_avg[avg, :])  # [Eqn(24)]
    scaleavg_signif = wave_signif(variance, dt=dt, scale=scale, sigtest=2,
        lag1=lag1, dof=([2, 7.9]), mother=mother)

    # ------------------------------------------------------ Plotting

    # --- Plot time series
    fig = plt.figure(figsize=(9, 10))
    gs = GridSpec(3, 4, hspace=0.4, wspace=0.75)
    plt.subplots_adjust(left=0.1, bottom=0.05, right=0.9, top=0.95,
                        wspace=0, hspace=0)
    plt.subplot(gs[0, 0:3])
    plt.plot(time, o3, 'k')
    #plt.xlim(xlim[:])
    plt.xlabel('Time (year)')
    #plt.ylabel('NINO3 SST (\u00B0C)')
    plt.title('a) NINO3 Sea Surface Temperature (seasonal)')

    # --- Contour plot wavelet power spectrum
    # plt3 = plt.subplot(3, 1, 2)
    plt3 = plt.subplot(gs[1, 0:3])
    levels = [0, 0.5, 1, 2, 4, 999]
    # *** or use 'contour'
    CS = plt.contourf(time, period, power, len(levels))
    im = plt.contourf(CS, levels=levels,
        colors=['white', 'bisque', 'orange', 'orangered', 'darkred'])
    plt.xlabel('Time (year)')
    plt.ylabel('Period (years)')
    plt.title('b) Wavelet Power Spectrum (contours at 0.5,1,2,4\u00B0C$^2$)')
    #plt.xlim(xlim[:])
    # 95# significance contour, levels at -99 (fake) and 1 (95# signif)
    plt.contour(time, period, sig95, [-99, 1], colors='k')
    # cone-of-influence, anything "below" is dubious
    plt.fill_between(time, coi * 0 + period[-1], coi, facecolor="none",
        edgecolor="#00000040", hatch='x')
    plt.plot(time, coi, 'k')
    # format y-scale
    plt3.set_yscale('log', base=2, subs=None)
    plt.ylim([np.min(period), np.max(period)])
    ax = plt.gca().yaxis
    ax.set_major_formatter(ticker.ScalarFormatter())
    plt3.ticklabel_format(axis='y', style='plain')
    plt3.invert_yaxis()
    # set up the size and location of the colorbar
    # position=fig.add_axes([0.5,0.36,0.2,0.01])
    # plt.colorbar(im, cax=position, orientation='horizontal')
    #   , fraction=0.05, pad=0.5)

    # plt.subplots_adjust(right=0.7, top=0.9)

    # --- Plot global wavelet spectrum
    plt4 = plt.subplot(gs[1, -1])
    plt.plot(global_ws, period)
    plt.plot(global_signif, period, '--')
    plt.xlabel('Power (\u00B0C$^2$)')
    plt.title('c) Global Wavelet Spectrum')
    plt.xlim([0, 1.25 * np.max(global_ws)])
    # format y-scale
    plt4.set_yscale('log', base=2, subs=None)
    plt.ylim([np.min(period), np.max(period)])
    ax = plt.gca().yaxis
    ax.set_major_formatter(ticker.ScalarFormatter())
    plt4.ticklabel_format(axis='y', style='plain')
    plt4.invert_yaxis()

    # --- Plot 2--8 yr scale-average time series
    plt.subplot(gs[2, 0:3])
    plt.plot(time, scale_avg, 'k')
    plt.xlim(xlim[:])
    plt.xlabel('Time (year)')
    plt.ylabel('Avg variance (\u00B0C$^2$)')
    plt.title('d) 2-8 yr Scale-average Time Series')
    #plt.plot(xlim, scaleavg_signif + [0, 0], '--')

    plt.show()

if __name__ == "__main__":
    yr = 2016
    date_slice=slice(str(yr)+'-01-01',str(yr)+'-12-31')

    instrument = 'comp'
    if instrument == 'GROMOS':
        instNameGROMOS = 'GROMOS'
        instNameSOMORA = 'GROMOS'
        fold_somora = '/storage/tub/instruments/gromos/level2/GROMORA/v1/'
        fold_gromos =  '/storage/tub/instruments/gromos/level2/GROMORA/v1/'
    elif instrument == 'SOMORA':
        instNameGROMOS = 'SOMORA'
        instNameSOMORA = 'SOMORA'
        fold_somora ='/storage/tub/instruments/somora/level2/v1/'
        fold_gromos ='/storage/tub/instruments/somora/level2/v1/'
    else:
        instNameGROMOS = 'GROMOS'
        instNameSOMORA = 'SOMORA'
        fold_somora = '/scratch/GROSOM/Level2/SOMORA/v2/'
        fold_gromos = '/scratch/GROSOM/Level2/GROMOS/v2/'
        prefix_all='.nc'
    #basefolder=_waccm_low_alt_dx10_v2_SB_ozone
    gromos = read_GROMORA_all(basefolder=fold_gromos, 
    instrument_name=instNameGROMOS,
    date_slice=date_slice, 
    years=[yr],#[2011, 2012,2013,2014,2015,2016,2017,2018,2019,2020],#[2011,2012,2013,2014,2015,2016,2017,2018,2019,],#[yr],#[],#
    prefix=prefix_all,
    flagged=True,
    )
    somora = read_GROMORA_all(basefolder=fold_somora, 
    instrument_name=instNameSOMORA,
    date_slice=date_slice, 
    years=[yr],#[2011, 2012,2013,2014,2015,2016,2017,2018,2019,2020],#years=[2010, 2011, 2012,2013,2014,2015,2016,2017,2018,2019,2020],#[yr],#[2011, 2012,2013,2014,2015,2016,2017,2018,2019,2020],##[yr],#[2011,2012,2013,2014,2015,2016,2017,2018,2019,]
    prefix=prefix_all,
    flagged=True,
    )

    o3_strat = 1e6*gromos.o3_x.interpolate_na(dim='time', method='nearest').sel(o3_p=10 ,method='nearest').resample(time='1H').mean(dim='time').data


    o3_strat[np.isnan(o3_strat)] = 0
    wavelet_analysis(o3_strat)
    
    # dt=1/(265*24)
    # time=gromos.o3_x.time.resample(time='1H').mean(dim='time').data
    # N = len(time)
    # o3_norm = waipy.normalize(o3_strat)
    # alpha = np.corrcoef(o3_norm[0:-1], o3_norm[1:])[0,1]; 
    # pad = 1         # pad the time series with zeroes (recommended)
    # dj = 0.25       # this will do 4 sub-octaves per octave
    # s0 = 2*dt       # this says start at a scale of 6 months if dt =annual
    # j1 = 7/dj       # this says do 7 powers-of-two with dj sub-octaves each
    # lag1 = 0.72     # lag-1 autocorrelation for red noise background
    # param = 6
    # mother = 'Morlet'
    # result = waipy.cwt(o3_norm, dt, pad, dj, s0, j1, lag1, param, mother='Morlet',name='Stratospheric Ozone')
    
    # waipy.wavelet_plot('title', np.arange(1,len(time)+1), o3_norm, 1, result); 