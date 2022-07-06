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


def read_SBUV_dailyMean(timerange, SBUV_basename = '/home/esauvageat/Documents/GROMORA/Data/SBUV/', specific_fname='sbuv_v87.mod_v2r1.vmr.payerne_156.txt'):
    filename = os.path.join(SBUV_basename, specific_fname)
    col_names = ['year', 'mon' ,'day' ,'DOY' , 'L1', 'L2', 'L3' ,'L4', 'L5' ,'L6' ,'L7' ,'L8' ,'L9' ,'L10' ,'L11' ,'L12' ,'L13' ,'L14' ,'L15' ,'ptot']
    SBUV_overpass = pd.read_fwf(
        filename,
        sep='',
        skiprows=30,
        names=col_names,
        na_values=999.000,
        colspecs='infer'
    )

    dt = [datetime.date(SBUV_overpass.year[i],SBUV_overpass.mon[i], SBUV_overpass.day[i]) for i in SBUV_overpass.index]
    #dateInd = datetime.date(SBUV_overpass.year.values,SBUV_overpass.mon.values, SBUV_overpass.day.values)
    SBUV_overpass['time'] = pd.to_datetime(dt)
    SBUV_overpass=SBUV_overpass.set_index('time')
    
    stacked_vmr = SBUV_overpass['L1'].values
    for plevel in col_names[5:19]:
        stacked_vmr = np.vstack((stacked_vmr, SBUV_overpass[plevel].values))

    df = SBUV_overpass.to_xarray()

    ozone = xr.DataArray(
        data=stacked_vmr,
        dims=['p','time'],
        coords=dict(
            time=df.time.data,
            p=np.array([0.5, 0.7, 1.0 ,1.5  ,2.0 ,3.0,4.0 ,5.0 ,7.0 ,10.0,15.0,20.0,30.0,40.0,50.0])
        )
    )

    df['ozone'] = ozone

    #df.drop(('L1', 'L2', 'L3' ,'L4', 'L5' ,'L6' ,'L7' ,'L8' ,'L9' ,'L10' ,'L11' ,'L12' ,'L13' ,'L14' ,'L15' ))

    return df.drop(('L1', 'L2', 'L3' ,'L4', 'L5' ,'L6' ,'L7' ,'L8' ,'L9' ,'L10' ,'L11' ,'L12' ,'L13' ,'L14' ,'L15' )).sel(time=timerange)


if __name__ == "__main__":
    time_period = slice("2018-01-01", "2018-12-31")
    bn = '/home/esauvageat/Documents/GROMORA/Data/SBUV/'
    sbuv = read_SBUV_dailyMean(time_period, SBUV_basename = bn, specific_fname='sbuv_v87.mod_v2r1.vmr.payerne_156.txt')
    sbuv_arosa = read_SBUV_dailyMean(time_period, SBUV_basename = bn, specific_fname='sbuv_v87.mod_v2r1.vmr.arosa_035.txt')