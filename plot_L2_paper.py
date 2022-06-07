#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Fri Apr 10 11:37:52 2020

@author: eric

Integration script for IAP instruments

Example:
    E...

        $ python example_google.py

Attributes:
    module_level_variable1 (int): Module level variables may be documented in
        either the ``Attributes`` section of the module docstring, or in an
        inline docstring immediately following the variable.

        Either form is acceptable, but the two should not be mixed. Choose
        one convention to document module level variables and be consistent
        with it.

Todo: all

"""

import sys

#sys.path.insert(0, '/home/esauvageat/Documents/GROMORA/Analysis/GROMORA-harmo/scripts/retrieval/')
#sys.path.insert(0, '/home/esauvageat/Documents/GROMORA/Analysis/GROMORA-harmo/scripts/pyretrievals/')

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
                               MultipleLocator)
from xarray.backends import file_manager

from compare_gromora_v2 import plot_figures_gromora_paper

#from cmcrameri import cm
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.sans-serif": ["Free sans"]})

load_dotenv('/home/esauvageat/Documents/ARTS/.env.moench-arts2.4')

# date = pd.date_range(start='2019-01-30', end='219-06-18')
#date = pd.date_range(start='2017-01-09', end='2017-01-09')
#date = pd.date_range(start=sys.argv[1], end=sys.argv[2])

def plot_L2(instrument_name = "GROMOS", date = [datetime.date(2017,1,9)], cycles=[14]):
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

    if instrument_name == "GROMOS":
        import gromos_classes as gc
        basename_lvl1 = "/storage/tub/instruments/gromos/level1/GROMORA/"+str(date[0].year)
        #basename_lvl2 = "/scratch/GROSOM/Level2/GROMORA_retrievals_polyfit2/"
        if new_L2:
            basename_lvl2 = "/storage/tub/instruments/gromos/level2/GROMORA/v2/"+str(date[0].year)
        else:
            basename_lvl2 = "/storage/tub/instruments/gromos/level2/GROMORA/v1/"+str(date[0].year)
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
            basename_lvl2 = "/storage/tub/instruments/somora/level2/v1/"+str(date[0].year)
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

#plot_figures_gromora_paper(do_sensitivity = False, do_L2=True, do_comp=False, do_old=False)