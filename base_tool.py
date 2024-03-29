#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on 17.03.22

@author: Eric Sauvageat

Library containing some basic piece of code to deal with GROMORA L2 data

"""
#%%
import datetime
import os

import matplotlib.pyplot as plt
import netCDF4
import numpy as np
import pandas as pd
import scipy

import xarray as xr
from scipy import stats
from scipy.odr import *

from GROMORA_harmo.scripts.retrieval import gromora_time
from matplotlib.backends.backend_pdf import PdfPages

from datetime import datetime, timedelta

from typhon.plots import (figsize, cmap2rgba, cmap2txt)



# ax.set_prop_cycle(color=)

MONTH_STR = ['Jan', 'Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']

def get_color(instrument_name):
    colors = cmap2rgba('qualitative1', 7)

    if instrument_name=='GROMOS':
        return colors[0]# return '#d7191c'# '#008837'# '#d95f02'
    elif instrument_name=='SOMORA':
        return colors[1]#return '#2c7bb6' #7b3294' # '#1b9e77'
    elif instrument_name=='SBUV':
        return colors[2]#return '#fdae61'
    elif instrument_name=='MLS':
        return 'k' #return '#abd9e9'  colors[3]
    elif instrument_name=='GDOC':
        return colors[4] #'#fdae61'#colors[3]#colors[4]#return '#fdae61'
    elif instrument_name=='WACCM':
        return colors[6]##
    elif instrument_name=='ECMWF':
        return 'k' ##
    elif instrument_name=='BASCOE':
        return colors[2] ##
    else:
        raise ValueError

def utc_to_lst(gromora):
    lsts = list()
    sunrise = list()
    sunset = list()
    for i, t in enumerate(gromora.time.data):
        #print('from : ',t)
        lst, ha, sza, night, tc= gromora_time.get_LST_from_GROMORA(t, gromora.obs_lat.data[i], gromora.obs_lon.data[i])
        #print('to :',lst)
        lsts.append(lst)

        sunr, suns = gromora_time.get_sunset_lst_from_lst(lst, gromora.obs_lat.data[i])
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

def get_LST_from_UTC(date, lat, lon, print_option=False, check_format=True):
    # local time zone for GROMOS and SOMORA:
    local_timezone = gromora_time.timezone('Europe/Zurich')

    #dt = utc.localize(datetime64_2_datetime(date))
    if check_format:
        if np.issubdtype(date.dtype, np.datetime64):
            date = gromora_time.datetime64_2_datetime(date).replace(tzinfo=gromora_time.timezone('UTC'))

    if print_option:
        print('UTC time: ',date) 
    local_time =  date.astimezone(local_timezone)
    if print_option:
        print('Local time: ',local_time)

    doy = pd.to_datetime(date).dayofyear

    eot = gromora_time.equation_of_time(doy-1)
    if print_option:
        print('Equation of time : ', str(eot))

    lstm = 15*local_time.utcoffset().seconds/3600
    tc = gromora_time.time_correction_factor(lon, lstm, eot)

    lst = local_time + timedelta(minutes=tc)
    
    lst = lst.replace(tzinfo=None)

    ha = gromora_time.hour_angle(lst)

    #ha_sunset, ha_NOAA= hour_angle_sunset(doy, lat)
    if print_option:
        print('Hour angle: ', str(ha))
    #  print('Hour angle sunset: ', str(ha_sunset))
    # print('Hour angle NOAA: ', str(ha_NOAA))
    
    # if np.abs(ha) > np.abs(ha_NOAA):
    #     night = True
    # else:
    #     night = False

    sza, night = gromora_time.solar_zenith_angle(ha, doy, lat)
    return lst, ha, sza, night, tc

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