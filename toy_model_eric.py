#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import datetime
import os

import matplotlib.pyplot as plt
import netCDF4
import numpy as np
import pandas as pd

import xarray as xr

filename = '/storage/tub/instruments/gromos/level2/GROMORA/v1/2018/full_2018_waccm_cov_yearly_ozone.nc'

gromos = xr.open_dataset(
    filename,
    decode_times=True,
    decode_coords=True,
    # use_cftime=True,
)

pandas_time = pd.to_datetime(gromos.time.data)


hours = pandas_time.hour
gromos = gromos.sel(time=slice("2018-01-01", "2018-01-04"))

fig, ax = plt.subplots(1, 1)
pl = gromos.o3_x.plot(
    x='time',
    y='o3_p',
    ax=ax,
    vmin=0,
    vmax=10,
    yscale='log',
    linewidth=0,
    rasterized=True,
    cmap='viridis'
)
pl.set_edgecolor('face')
# ax.set_yscale('log')
ax.invert_yaxis()
ax.set_ylabel('P [hPa]')
plt.tight_layout()