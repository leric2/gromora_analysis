#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on 17.03.22

@author: Eric Sauvageat

Library containing some basic piece of code to deal with GROMORA L2 data

Attributes:
    module_level_variable1 (int): Module level variables may be documented in
        either the ``Attributes`` section of the module docstring, or in an
        inline docstring immediately following the variable.

        Either form is acceptable, but the two should not be mixed. Choose
        one convention to document module level variables and be consistent
        with it.


"""

#%%
import datetime
from multiprocessing.sharedctypes import Value
import os
from re import A
from typing import ValuesView
from matplotlib import units

import matplotlib.pyplot as plt
import netCDF4
import numpy as np
from numpy.lib.shape_base import dsplit
import pandas as pd
import scipy
from scipy.odr.odrpack import RealData
from scipy.stats.stats import RepeatedResults
from secretstorage import search_items

import xarray as xr
from scipy import stats
from scipy.odr import *

from GROMORA_harmo.scripts.retrieval import GROMORA_time

from matplotlib.ticker import (MultipleLocator, FormatStrFormatter, AutoMinorLocator)
from matplotlib.lines import Line2D
import matplotlib.ticker as ticker
import matplotlib

from matplotlib.backends.backend_pdf import PdfPages


MONTH_STR = ['Jan', 'Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']

def get_color(instrument_name):
    if instrument_name=='GROMOS':
        return '#d7191c'# '#008837'# '#d95f02'
    elif instrument_name=='SOMORA':
        return '#2c7bb6' #7b3294' # '#1b9e77'
    elif instrument_name=='sbuv':
        return'#fdae61'
    elif instrument_name=='MLS':
        return 'k'
    else:
        raise ValueError

def utc_to_lst(gromora):
    lsts = list()
    sunrise = list()
    sunset = list()
    for i, t in enumerate(gromora.time.data):
        #print('from : ',t)
        lst, ha, sza, night, tc= GROMORA_time.get_LST_from_GROMORA(t, gromora.obs_lat.data[i], gromora.obs_lon.data[i])
        #print('to :',lst)
        lsts.append(lst)

        sunr, suns = GROMORA_time.get_sunset_lst_from_lst(lst, gromora.obs_lat.data[i])
        sunrise.append(sunr)
        sunset.append(suns)

    gromora['time'] = lsts
    gromora['time'].attrs = {'description':'Local solar time'}

    sunrise_da = xr.DataArray(
        data = sunrise,
        dims=['time'],
        coords=dict(time=gromora['time']),
        attrs=dict(description='sunrise')
    )
    sunset_da = xr.DataArray(
        data = sunset,
        dims=['time'],
        coords=dict(time=gromora['time']),
        attrs=dict(description='sunset')
    )
    gromora['sunrise'] = sunrise_da
    gromora['sunset'] = sunset_da
    return gromora

def linear(p, x):
    m, c = p
    return m*x+c

def regression_xy(x, y, x_err, y_err, lin=True):
    if lin:
        lin_model = Model(linear)
    else:
        print('Linear regression only')
    
    data = RealData(x, y, sx=x_err, sy=y_err)
    
   # odr = ODR(data, lin_model, beta0=[1.,0.])
    odr_obj = ODR(data, scipy.odr.unilinear)

    result = odr_obj.run()
   # result.pprint()
    print('Reduced Chi-Squared: ', result.res_var)
    #  chi_squared = np.sum(np.divide(np.square(y - result.beta[0] - result.beta[1]*x),(x_err**2 + np.square(result.beta[1]*y_err))))

    # plt.errorbar(x, y, xerr=x_err, yerr=y_err, linestyle='None', marker='.')
    # plt.plot(x_fit, y_fit)

    # plt.show()

    return result

# def coefficient_determination(y, y_pred):
#     SST = np.sum((y - np.mean(y))**2)
#     SSReg = np.sum((y_pred - np.mean(y))**2)
#     R2 = SSReg/SST
#     return R2

def calcR2_wikipedia(y, y_pred):
    # Mean value of the observed data y.
    y_mean = np.mean(y)
    # Total sum of squares.
    SS_tot = np.sum((y - y_mean)**2)
    # Residual sum of squares.
    SS_res = np.sum((y - y_pred)**2)
    # Coefficient of determination.
    R2 = 1.0 - (SS_res / SS_tot)
    return R2

def save_single_pdf(filename, figures):
    """
    Save all `figures` to a single PDF. taken from Jonas
    """
    with PdfPages(filename) as pdf:
        for fig in figures:
            pdf.savefig(fig)

if __name__ == "__main__":
    pass