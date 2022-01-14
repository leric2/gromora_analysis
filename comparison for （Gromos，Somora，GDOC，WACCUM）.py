#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created

@author: eric

Collection of functions for dealing with time


Including :
    * a-priori data
"""
import os
import numpy as np
# import retrievals
import xarray as xr
import pandas as pd
import math
import netCDF4
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from pytz import timezone, utc
from matplotlib.pylab import date2num, num2date
from pysolar import solar

#load 17-18 GROMOS ozone profies
filename = 'F:/PyCharm Community Edition 2019.3/practice2021 april/GROMOS_2017_12_31_waccm_low_alt_ozone.nc' #load ozone profile

gromos = xr.open_dataset(
    filename,
    decode_times=True,
    decode_coords=True,
    # use_cftime=True,
)

filename = 'F:/PyCharm Community Edition 2019.3/practice2021 april/GROMOS_2018_12_31_waccm_low_alt_ozone.nc'

gromos_1 = xr.open_dataset(
    filename,
    decode_times=True,
    decode_coords=True,
    # use_cftime=True,
)

#load 17-18 SOMORA ozone profies
filename = 'F:/PyCharm Community Edition 2019.3/practice2021 april/SOMORA_2017_12_31_waccm_low_alt_ozone.nc' #load ozone profile

somora = xr.open_dataset(
    filename,
    decode_times=True,
    decode_coords=True,
    # use_cftime=True,
)

filename = 'F:/PyCharm Community Edition 2019.3/practice2021 april/SOMORA_2018_12_31_waccm_low_alt_ozone.nc'

somora_1 = xr.open_dataset(
    filename,
    decode_times=True,
    decode_coords=True,
    # use_cftime=True,
)

#Load GDOC ozone profile
filename = 'F:/PyCharm Community Edition 2019.3/practice2021 april/GDOC_ver1.nc'

GDOC = xr.open_dataset(
    filename,
    decode_times=True,
    decode_coords=True,
    # use_cftime=True,
)

#load WACCM ozone profile
WACCM_file = 'F:/WACCUM data/WACCUM_full_year_data.nc'

WACCM = xr.open_dataset(
    WACCM_file,
    decode_times=True,
    decode_coords=True,
    # use_cftime=True,
)


########################################################UTC convert to LST
gromora_tz = timezone('Europe/Zurich')


def solar_zenith_angle(ha, lst, lat):
    # sun declination
    declination = -23.44 * np.cos(np.deg2rad((pd.to_datetime(lst).dayofyear + 10) * 360 / 365))
    cos_declination = np.cos(np.deg2rad(declination))
    sin_declination = np.sin(np.deg2rad(declination))

    # Hour angle
    cos_solar_hour_angle = np.cos(np.deg2rad(ha))

    # Sunrise/Sunset:
    cos_hour_angle_night = -np.tan(np.deg2rad(lat)) * np.tan(np.deg2rad(declination))
    if cos_solar_hour_angle < cos_hour_angle_night:
        night = True
    else:
        night = False
    cos_sza = sin_declination * np.sin(np.deg2rad(lat)) + np.cos(
        np.deg2rad(lat)) * cos_declination * cos_solar_hour_angle
    sza = np.rad2deg(np.arccos(cos_sza))
    return sza, night
#####################################################
def hour_angle_sunset(lst, lat):
    declination = -23.44*np.cos(np.deg2rad((pd.to_datetime(lst).dayofyear+10)*360/365))
    cos_declination = np.cos(np.deg2rad(declination))
    sin_declination = np.sin(np.deg2rad(declination))

    # Sunrise/Sunset:
    cos_hour_angle_sunset = -np.tan(np.deg2rad(lat))*np.tan(np.deg2rad(declination))
    return np.rad2deg(np.arccos(cos_hour_angle_sunset))
#########################################################
def pysolar_sza(date, lat, lon):
    # Using pysolar package:
    time = utc.localize(datetime64_2_datetime(date))
    sza_pysolar = solar.get_altitude(lat, lon, time)
    print('solar elevation angle = ', sza_pysolar)
    return sza_pysolar


def datetime64_2_datetime(dt64):
    unix_epoch = np.datetime64(0, 's')
    one_second = np.timedelta64(1, 's')
    seconds_since_epoch = (dt64 - unix_epoch) / one_second
    dt = datetime.utcfromtimestamp(seconds_since_epoch)
    return dt


def equation_of_time(doy):
    B = (360 / 365) * (doy - 81)
    eot = 9.87 * np.sin(np.deg2rad(2 * B)) - 7.53 * np.cos(np.deg2rad(B)) - 1.5 * np.sin(np.deg2rad(B))
    return eot


def time_correction_factor(lon, lstm, eot):
    return 4 * (lon - lstm) + eot


def local_solar_time(local_time, tc):
    return local_time + tc / 60


def hour_angle(lst):
    hours_from_midnight = (lst - lst.replace(hour=0, minute=0, second=0, microsecond=0)).total_seconds() / 3600
    return 15 * (hours_from_midnight - 12)
###################################################
def lst_sunset_from_hour_angle(sunset_ha, midnight_lst):
    sunrise = midnight_lst + timedelta(hours=12) + timedelta(hours=-np.abs(sunset_ha)/15)
    sunset = midnight_lst + timedelta(hours=12) + timedelta(hours=np.abs(sunset_ha)/15)
    return sunrise, sunset

def get_sunset_lst_from_lst(lst, lat):
    sunset_ha = hour_angle_sunset(lst,lat)

    sunrise_lst, sunset_lst = lst_sunset_from_hour_angle(
        sunset_ha,
        midnight_lst = lst.replace(hour=0, minute=0,second=0, microsecond=0)
    )
    return sunrise_lst, sunset_lst
####################################################
def get_LST_from_UTC(date, lat, lon):
    dt = utc.localize(datetime64_2_datetime(date))
    # print('UTC time: ',dt)
    local_time = dt.astimezone(gromora_tz)
    # print('Local time: ',local_time)

    doy = pd.to_datetime(dt).dayofyear

    eot = equation_of_time(doy)

    lstm = 15 * local_time.utcoffset().seconds / 3600
    tc = time_correction_factor(lon, lstm, eot)

    seconds_from_midnight = (local_time - local_time.replace(hour=0, minute=0, second=0, microsecond=0)).total_seconds()
    minutes_from_midnight = seconds_from_midnight / 60

    lst = local_time + timedelta(hours=tc / 60)
    ha = hour_angle(lst)

    # print('Local solar time: ',lst)
    # print('Hour angle: ', str(ha))

    sza, night = solar_zenith_angle(ha, lst, lat)

    # print('solar elevation angle : ', 90-sza)

    return lst, ha, sza, night



################################################################

gromos_month_1 = gromos.o3_x.sel(time=slice("2017-01-01", "2017-01-31"))
gromos_month_2 = gromos.o3_x.sel(time=slice("2017-02-01", "2017-02-28"))  # 2017 slice month o3 data
gromos_month_3 = gromos.o3_x.sel(time=slice("2017-03-01", "2017-03-31"))
gromos_month_4 = gromos.o3_x.sel(time=slice("2017-04-01", "2017-04-30"))
gromos_month_5 = gromos.o3_x.sel(time=slice("2017-05-01", "2017-05-31"))
gromos_month_6 = gromos.o3_x.sel(time=slice("2017-06-01", "2017-06-30"))
gromos_month_7 = gromos.o3_x.sel(time=slice("2017-07-01", "2017-07-31"))
gromos_month_8 = gromos.o3_x.sel(time=slice("2017-08-01", "2017-08-31"))
gromos_month_9 = gromos.o3_x.sel(time=slice("2017-09-01", "2017-09-30"))
gromos_month_10 = gromos.o3_x.sel(time=slice("2017-10-01", "2017-10-31"))
gromos_month_11 = gromos.o3_x.sel(time=slice("2017-11-01", "2017-11-30"))
gromos_month_12 = gromos.o3_x.sel(time=slice("2017-12-01", "2017-12-31"))

gromos_month_lst_1 = gromos.sel(time=slice("2017-01-01", "2017-01-31"))
gromos_month_lst_2 = gromos.sel(time=slice("2017-02-01", "2017-02-28"))  # 2017 slice month data
gromos_month_lst_3 = gromos.sel(time=slice("2017-03-01", "2017-03-31"))
gromos_month_lst_4 = gromos.sel(time=slice("2017-04-01", "2017-04-30"))
gromos_month_lst_5 = gromos.sel(time=slice("2017-05-01", "2017-05-31"))
gromos_month_lst_6 = gromos.sel(time=slice("2017-06-01", "2017-06-30"))
gromos_month_lst_7 = gromos.sel(time=slice("2017-07-01", "2017-07-31"))
gromos_month_lst_8 = gromos.sel(time=slice("2017-08-01", "2017-08-31"))
gromos_month_lst_9 = gromos.sel(time=slice("2017-09-01", "2017-09-30"))
gromos_month_lst_10 = gromos.sel(time=slice("2017-10-01", "2017-10-31"))
gromos_month_lst_11 = gromos.sel(time=slice("2017-11-01", "2017-11-30"))
gromos_month_lst_12 = gromos.sel(time=slice("2017-12-01", "2017-12-31"))

gromos_1_month_1 = gromos_1.o3_x.sel(time=slice("2018-01-01", "2018-01-31"))
gromos_1_month_2 = gromos_1.o3_x.sel(time=slice("2018-02-01", "2018-02-28"))  # 2018 slice month o3 data
gromos_1_month_3 = gromos_1.o3_x.sel(time=slice("2018-03-01", "2018-03-31"))
gromos_1_month_4 = gromos_1.o3_x.sel(time=slice("2018-04-01", "2018-04-30"))
gromos_1_month_5 = gromos_1.o3_x.sel(time=slice("2018-05-01", "2018-05-31"))
gromos_1_month_6 = gromos_1.o3_x.sel(time=slice("2018-06-01", "2018-06-30"))
gromos_1_month_7 = gromos_1.o3_x.sel(time=slice("2018-07-01", "2018-07-31"))
gromos_1_month_8 = gromos_1.o3_x.sel(time=slice("2018-08-01", "2018-08-31"))
gromos_1_month_9 = gromos_1.o3_x.sel(time=slice("2018-09-01", "2018-09-30"))
gromos_1_month_10 = gromos_1.o3_x.sel(time=slice("2018-10-01", "2018-10-31"))
gromos_1_month_11 = gromos_1.o3_x.sel(time=slice("2018-11-01", "2018-11-30"))
gromos_1_month_12 = gromos_1.o3_x.sel(time=slice("2018-12-01", "2018-12-31"))

gromos_1_month_lst_1 = gromos_1.sel(time=slice("2018-01-01", "2018-01-31"))
gromos_1_month_lst_2 = gromos_1.sel(time=slice("2018-02-01", "2018-02-28"))  # 2018 slice month data
gromos_1_month_lst_3 = gromos_1.sel(time=slice("2018-03-01", "2018-03-31"))
gromos_1_month_lst_4 = gromos_1.sel(time=slice("2018-04-01", "2018-04-30"))
gromos_1_month_lst_5 = gromos_1.sel(time=slice("2018-05-01", "2018-05-31"))
gromos_1_month_lst_6 = gromos_1.sel(time=slice("2018-06-01", "2018-06-30"))
gromos_1_month_lst_7 = gromos_1.sel(time=slice("2018-07-01", "2018-07-31"))
gromos_1_month_lst_8 = gromos_1.sel(time=slice("2018-08-01", "2018-08-31"))
gromos_1_month_lst_9 = gromos_1.sel(time=slice("2018-09-01", "2018-09-30"))
gromos_1_month_lst_10 = gromos_1.sel(time=slice("2018-10-01", "2018-10-31"))
gromos_1_month_lst_11 = gromos_1.sel(time=slice("2018-11-01", "2018-11-30"))
gromos_1_month_lst_12 = gromos_1.sel(time=slice("2018-12-01", "2018-12-31"))

somora_month_1 = somora.o3_x.sel(time=slice("2017-01-01", "2017-01-31"))
somora_month_2 = somora.o3_x.sel(time=slice("2017-02-01", "2017-02-28"))  # 2017 slice month o3 data
somora_month_3 = somora.o3_x.sel(time=slice("2017-03-01", "2017-03-31"))
somora_month_4 = somora.o3_x.sel(time=slice("2017-04-01", "2017-04-30"))
somora_month_5 = somora.o3_x.sel(time=slice("2017-05-01", "2017-05-31"))
somora_month_6 = somora.o3_x.sel(time=slice("2017-06-01", "2017-06-30"))
somora_month_7 = somora.o3_x.sel(time=slice("2017-07-01", "2017-07-31"))
somora_month_8 = somora.o3_x.sel(time=slice("2017-08-01", "2017-08-31"))
somora_month_9 = somora.o3_x.sel(time=slice("2017-09-01", "2017-09-30"))
somora_month_10 = somora.o3_x.sel(time=slice("2017-10-01", "2017-10-31"))
somora_month_11 = somora.o3_x.sel(time=slice("2017-11-01", "2017-11-30"))
somora_month_12 = somora.o3_x.sel(time=slice("2017-12-01", "2017-12-31"))

somora_month_lst_1 = somora.sel(time=slice("2017-01-01", "2017-01-31"))
somora_month_lst_2 = somora.sel(time=slice("2017-02-01", "2017-02-28"))  # 2017 slice month data
somora_month_lst_3 = somora.sel(time=slice("2017-03-01", "2017-03-31"))
somora_month_lst_4 = somora.sel(time=slice("2017-04-01", "2017-04-30"))
somora_month_lst_5 = somora.sel(time=slice("2017-05-01", "2017-05-31"))
somora_month_lst_6 = somora.sel(time=slice("2017-06-01", "2017-06-30"))
somora_month_lst_7 = somora.sel(time=slice("2017-07-01", "2017-07-31"))
somora_month_lst_8 = somora.sel(time=slice("2017-08-01", "2017-08-31"))
somora_month_lst_9 = somora.sel(time=slice("2017-09-01", "2017-09-30"))
somora_month_lst_10 = somora.sel(time=slice("2017-10-01", "2017-10-31"))
somora_month_lst_11 = somora.sel(time=slice("2017-11-01", "2017-11-30"))
somora_month_lst_12 = somora.sel(time=slice("2017-12-01", "2017-12-31"))

somora_1_month_1 = somora_1.o3_x.sel(time=slice("2018-01-01", "2018-01-31"))
somora_1_month_2 = somora_1.o3_x.sel(time=slice("2018-02-01", "2018-02-28"))  # 2018 slice month o3 data
somora_1_month_3 = somora_1.o3_x.sel(time=slice("2018-03-01", "2018-03-31"))
somora_1_month_4 = somora_1.o3_x.sel(time=slice("2018-04-01", "2018-04-30"))
somora_1_month_5 = somora_1.o3_x.sel(time=slice("2018-05-01", "2018-05-31"))
somora_1_month_6 = somora_1.o3_x.sel(time=slice("2018-06-01", "2018-06-30"))
somora_1_month_7 = somora_1.o3_x.sel(time=slice("2018-07-01", "2018-07-31"))
somora_1_month_8 = somora_1.o3_x.sel(time=slice("2018-08-01", "2018-08-31"))
somora_1_month_9 = somora_1.o3_x.sel(time=slice("2018-09-01", "2018-09-30"))
somora_1_month_10 = somora_1.o3_x.sel(time=slice("2018-10-01", "2018-10-31"))
somora_1_month_11 = somora_1.o3_x.sel(time=slice("2018-11-01", "2018-11-30"))
somora_1_month_12 = somora_1.o3_x.sel(time=slice("2018-12-01", "2018-12-31"))

somora_1_month_lst_1 = somora_1.sel(time=slice("2018-01-01", "2018-01-31"))
somora_1_month_lst_2 = somora_1.sel(time=slice("2018-02-01", "2018-02-28"))  # 2018 slice month data
somora_1_month_lst_3 = somora_1.sel(time=slice("2018-03-01", "2018-03-31"))
somora_1_month_lst_4 = somora_1.sel(time=slice("2018-04-01", "2018-04-30"))
somora_1_month_lst_5 = somora_1.sel(time=slice("2018-05-01", "2018-05-31"))
somora_1_month_lst_6 = somora_1.sel(time=slice("2018-06-01", "2018-06-30"))
somora_1_month_lst_7 = somora_1.sel(time=slice("2018-07-01", "2018-07-31"))
somora_1_month_lst_8 = somora_1.sel(time=slice("2018-08-01", "2018-08-31"))
somora_1_month_lst_9 = somora_1.sel(time=slice("2018-09-01", "2018-09-30"))
somora_1_month_lst_10 = somora_1.sel(time=slice("2018-10-01", "2018-10-31"))
somora_1_month_lst_11 = somora_1.sel(time=slice("2018-11-01", "2018-11-30"))
somora_1_month_lst_12 = somora_1.sel(time=slice("2018-12-01", "2018-12-31"))

#extract GDOC four dimensional data
GDOC_month = GDOC.month.data
GDOC_hour = GDOC.hour.data
GDOC_lat = GDOC.lat.data
GDOC_pressure = GDOC.zstar_pr.data

for lat_num, lat_GDOC in enumerate(GDOC_lat): #locate latitude close to Bern
    if lat_GDOC == 47.5:
        lat_certain = lat_num

GDOC_month_1 = GDOC.GDOC.sel(number_of_months=0, latitude=lat_certain) #slice 12 months
GDOC_month_2 = GDOC.GDOC.sel(number_of_months=1, latitude=lat_certain)
GDOC_month_3 = GDOC.GDOC.sel(number_of_months=2, latitude=lat_certain)
GDOC_month_4 = GDOC.GDOC.sel(number_of_months=3, latitude=lat_certain)
GDOC_month_5 = GDOC.GDOC.sel(number_of_months=4, latitude=lat_certain)
GDOC_month_6 = GDOC.GDOC.sel(number_of_months=5, latitude=lat_certain)
GDOC_month_7 = GDOC.GDOC.sel(number_of_months=6, latitude=lat_certain)
GDOC_month_8 = GDOC.GDOC.sel(number_of_months=7, latitude=lat_certain)
GDOC_month_9 = GDOC.GDOC.sel(number_of_months=8, latitude=lat_certain)
GDOC_month_10 = GDOC.GDOC.sel(number_of_months=9, latitude=lat_certain)
GDOC_month_11 = GDOC.GDOC.sel(number_of_months=10, latitude=lat_certain)
GDOC_month_12 = GDOC.GDOC.sel(number_of_months=11, latitude=lat_certain)
###################################################WACCM UTC to LST preparison
CFT_time = xr.CFTimeIndex(WACCM.time.values)    #build up time dimensional information
waccm_pdtime = pd.DataFrame({'year': 2018,
                             'month': CFT_time.month,
                             'day': CFT_time.day,
                             # 'dayofyear': CFT_time.dayofyear,
                             # 'days_in_month': CFT_time.days_in_month,
                             'hour': CFT_time.hour,
                             'minute': CFT_time.minute
                             })
waccm_datetime = pd.to_datetime(waccm_pdtime)         #covert time to panda mode
WACCM['datetime']=waccm_datetime
WACCM_new = xr.Dataset(                              #set up new waccm ozone profile (time x pressure)
    {'O3': (('datetime', 'lev'), WACCM.O3)},
    coords={
        'datetime': WACCM.datetime.values,
        'lev': WACCM.lev
    },
)

WACCM_month_1 = WACCM_new.sel(datetime=slice("2018-01-01", "2018-01-31"))
WACCM_month_2 = WACCM_new.sel(datetime=slice("2018-02-01", "2018-02-28"))  # 2018 slice month O3 data
WACCM_month_3 = WACCM_new.sel(datetime=slice("2018-03-01", "2018-03-31"))
WACCM_month_4 = WACCM_new.sel(datetime=slice("2018-04-01", "2018-04-30"))
WACCM_month_5 = WACCM_new.sel(datetime=slice("2018-05-01", "2018-05-31"))
WACCM_month_6 = WACCM_new.sel(datetime=slice("2018-06-01", "2018-06-30"))
WACCM_month_7 = WACCM_new.sel(datetime=slice("2018-07-01", "2018-07-31"))
WACCM_month_8 = WACCM_new.sel(datetime=slice("2018-08-01", "2018-08-31"))
WACCM_month_9 = WACCM_new.sel(datetime=slice("2018-09-01", "2018-09-30"))
WACCM_month_10 = WACCM_new.sel(datetime=slice("2018-10-01", "2018-10-31"))
WACCM_month_11 = WACCM_new.sel(datetime=slice("2018-11-01", "2018-11-30"))
WACCM_month_12 = WACCM_new.sel(datetime=slice("2018-12-01", "2018-12-31"))

# build 12 month ozone profile tuples for four datasets
GDOC_month_12_data = (GDOC_month_1, GDOC_month_2, GDOC_month_3, GDOC_month_4, GDOC_month_5, GDOC_month_6, GDOC_month_7,
                      GDOC_month_8, GDOC_month_9, GDOC_month_10, GDOC_month_11, GDOC_month_12)
month_time_lst_12 = (
gromos_month_lst_1, gromos_month_lst_2, gromos_month_lst_3, gromos_month_lst_4, gromos_month_lst_5, gromos_month_lst_6,
gromos_month_lst_7, gromos_month_lst_8, gromos_month_lst_9, gromos_month_lst_10, gromos_month_lst_11,
gromos_month_lst_12)
month_12_o3_data = (
gromos_month_1, gromos_month_2, gromos_month_3, gromos_month_4, gromos_month_5, gromos_month_6, gromos_month_7,
gromos_month_8, gromos_month_9, gromos_month_10, gromos_month_11, gromos_month_12)

month_time_lst_somora_12 = (
somora_month_lst_1, somora_month_lst_2, somora_month_lst_3, somora_month_lst_4, somora_month_lst_5, somora_month_lst_6,
somora_month_lst_7, somora_month_lst_8, somora_month_lst_9, somora_month_lst_10, somora_month_lst_11,
somora_month_lst_12)
month_12_o3_somora_data = (
somora_month_1, somora_month_2, somora_month_3, somora_month_4, somora_month_5, somora_month_6, somora_month_7,
somora_month_8, somora_month_9, somora_month_10, somora_month_11, somora_month_12)

#build up title tuples
month_title_12 = ('1718av_gromos-January-stratospheric-diurnal-ozone-variation', '1718av_gromos-February-stratospheric-diurnal-ozone-variation',
                  '1718av_gromos-March-stratospheric-diurnal-ozone-variation'
                  , '1718av_gromos-April-stratospheric-diurnal-ozone-variation', '1718av_gromos-May-stratospheric-diurnal-ozone-variation',
                  '1718av_gromos-June-stratospheric-diurnal-ozone-variation'
                  , '1718av_gromos-July-stratospheric-diurnal-ozone-variation', '1718av_gromos-August-stratospheric-diurnal-ozone-variation',
                  '1718av_gromos-September-stratospheric-diurnal-ozone-variation'
                  , '1718av_gromos-October-stratospheric-diurnal-ozone-variation', '1718av_gromos-November-stratospheric-diurnal-ozone-variation',
                  '1718av_gromos-December-stratospheric-diurnal-ozone-variation')

month_title_somora_12 = ('1718av_somora-January-stratospheric-diurnal-ozone-variation', '1718av_somora-February-stratospheric-diurnal-ozone-variation',
                  '1718av_somora-March-stratospheric-diurnal-ozone-variation'
                  , '1718av_somora-April-stratospheric-diurnal-ozone-variation', '1718av_somora-May-stratospheric-diurnal-ozone-variation',
                  '1718av_somora-June-stratospheric-diurnal-ozone-variation'
                  , '1718av_somora-July-stratospheric-diurnal-ozone-variation', '1718av_somora-August-stratospheric-diurnal-ozone-variation',
                  '1718av_somora-September-stratospheric-diurnal-ozone-variation'
                  , '1718av_somora-October-stratospheric-diurnal-ozone-variation', '1718av_somora-November-stratospheric-diurnal-ozone-variation',
                  '1718av_somora-December-stratospheric-diurnal-ozone-variation')

'''
month_title_12 = ('1618av-January-stratospheric-diurnal-ozone-variation', '1618av-February-stratospheric-diurnal-ozone-variation',
                  '1618av-March-stratospheric-diurnal-ozone-variation'
                  , '1618av-April-stratospheric-diurnal-ozone-variation', '1618av-May-stratospheric-diurnal-ozone-variation',
                  '1618av-June-stratospheric-diurnal-ozone-variation'
                  , '1618av-July-stratospheric-diurnal-ozone-variation', '1618av-August-stratospheric-diurnal-ozone-variation',
                  '1618av-September-stratospheric-diurnal-ozone-variation'
                  , '1618av-October-stratospheric-diurnal-ozone-variation', '1618av-November-stratospheric-diurnal-ozone-variation',
                  '1618av-December-stratospheric-diurnal-ozone-variation')
'''
month_title_GDOC_12 = ('1718av-GDOC-stratospheric-January-diurnal-ozone-variation','1718av-GDOC-stratospheric-February-diurnal-ozone-variation',
                       '1718av-GDOC-stratospheric-March-diurnal-ozone-variation', '1718av-GDOC-stratospheric-April-diurnal-ozone-variation',
                       '1718av-GDOC-stratospheric-May-diurnal-ozone-variation', '1718av-GDOC-stratospheric-June-diurnal-ozone-variation',
                       '1718av-GDOC-stratospheric-July-diurnal-ozone-variation', '1718av-GDOC-stratospheric-August-diurnal-ozone-variation',
                       '1718av-GDOC-stratospheric-September-diurnal-ozone-variation', '1718av-GDOC-stratospheric-October-diurnal-ozone-variation',
                       '1718av-GDOC-stratospheric-November-diurnal-ozone-variation', '1718av-GDOC-stratospheric-December-diurnal-ozone-variation')

month_title_WACCM_12 = ('WACCM-January-stratospheric-diurnal-ozone-variation', 'WACCM-February-stratospheric-diurnal-ozone-variation',
                  'WACCM-March-stratospheric-diurnal-ozone-variation'
                  , 'WACCM-April-stratospheric-diurnal-ozone-variation', 'WACCM-May-stratospheric-diurnal-ozone-variation',
                  'WACCM-June-stratospheric-diurnal-ozone-variation'
                  , 'WACCM-July-stratospheric-diurnal-ozone-variation', 'WACCM-August-stratospheric-diurnal-ozone-variation',
                  'WACCM-September-stratospheric-diurnal-ozone-variation'
                  , 'WACCM-October-stratospheric-diurnal-ozone-variation', 'WACCM-November-stratospheric-diurnal-ozone-variation',
                  'WACCM-December-stratospheric-diurnal-ozone-variation')

Comparison_month_title_12 = ('Comparison-January-stratospheric-diurnal-ozone-variation', 'Comparison-February-stratospheric-diurnal-ozone-variation',
                  'Comparison-March-stratospheric-diurnal-ozone-variation'
                  , 'Comparison-April-stratospheric-diurnal-ozone-variation', 'Comparison-May-stratospheric-diurnal-ozone-variation',
                  'Comparison-June-stratospheric-diurnal-ozone-variation'
                  , 'Comparison-July-stratospheric-diurnal-ozone-variation', 'Comparison-August-stratospheric-diurnal-ozone-variation',
                  'Comparison-September-stratospheric-diurnal-ozone-variation'
                  , 'Comparison-October-stratospheric-diurnal-ozone-variation', 'Comparison-November-stratospheric-diurnal-ozone-variation',
                  'Comparison-December-stratospheric-diurnal-ozone-variation')

# build 12 month ozone profile tuples for four datasets
WACCM_pressure = WACCM.lev.data
WACCM_month_12_data = (
WACCM_month_1, WACCM_month_2, WACCM_month_3, WACCM_month_4, WACCM_month_5,
WACCM_month_6, WACCM_month_7, WACCM_month_8, WACCM_month_9,
WACCM_month_10, WACCM_month_11, WACCM_month_12)

month_time_lst_12_1 = (
gromos_1_month_lst_1, gromos_1_month_lst_2, gromos_1_month_lst_3, gromos_1_month_lst_4, gromos_1_month_lst_5, gromos_1_month_lst_6,
gromos_1_month_lst_7, gromos_1_month_lst_8, gromos_1_month_lst_9, gromos_1_month_lst_10, gromos_1_month_lst_11,
gromos_1_month_lst_12)
month_12_o3_data_1 = (
gromos_1_month_1, gromos_1_month_2, gromos_1_month_3, gromos_1_month_4, gromos_1_month_5, gromos_1_month_6, gromos_1_month_7,
gromos_1_month_8, gromos_1_month_9, gromos_1_month_10, gromos_1_month_11, gromos_1_month_12)

month_time_lst_somora_12_1 = (
somora_1_month_lst_1, somora_1_month_lst_2, somora_1_month_lst_3, somora_1_month_lst_4, somora_1_month_lst_5, somora_1_month_lst_6,
somora_1_month_lst_7, somora_1_month_lst_8, somora_1_month_lst_9, somora_1_month_lst_10, somora_1_month_lst_11,
somora_1_month_lst_12)
month_12_o3_somora_data_1 = (
somora_1_month_1, somora_1_month_2, somora_1_month_3, somora_1_month_4, somora_1_month_5, somora_1_month_6, somora_1_month_7,
somora_1_month_8, somora_1_month_9, somora_1_month_10, somora_1_month_11, somora_1_month_12)


for month_data, month_title in enumerate(np.arange(12)):

    gromos_month = month_12_o3_data[month_data]
    gromos_month_time_lst = month_time_lst_12[month_data]
    lst_gromos = np.array([])
    for z in range(gromos_month_time_lst.time.shape[0]):    # utc convert to lst for each month of GROMOS 2017
        lst_save = get_LST_from_UTC(gromos_month_time_lst.time.values[z],
                                    gromos_month_time_lst.obs_lat.values[z],
                                    gromos_month_time_lst.obs_lon.values[z])
        lst_gromos = np.append(lst_gromos, lst_save[0])
    pandas_time_test = pd.to_datetime(lst_gromos)
    hours = pandas_time_test.hour #pick up hour points

    gromos_month_1 = month_12_o3_data_1[month_data]
    gromos_month_time_lst_1 = month_time_lst_12_1[month_data]
    lst_gromos_1 = np.array([])
    for z in range(gromos_month_time_lst_1.time.shape[0]):   # utc convert to lst for each month of GROMOS 2018
        lst_save = get_LST_from_UTC(gromos_month_time_lst_1.time.values[z],
                                    gromos_month_time_lst_1.obs_lat.values[z],
                                    gromos_month_time_lst_1.obs_lon.values[z])
        lst_gromos_1 = np.append(lst_gromos_1, lst_save[0]) # lst_save[0] is time term
    pandas_time_test_1 = pd.to_datetime(lst_gromos_1)
    hours_1 = pandas_time_test_1.hour #pick up hour points

    somora_month = month_12_o3_somora_data[month_data]
    somora_month_time_lst = month_time_lst_somora_12[month_data]
    lst_somora = np.array([])
    for z in range(somora_month_time_lst.time.shape[0]):   # utc convert to lst for each month of SOMORA 2017
        lst_save = get_LST_from_UTC(somora_month_time_lst.time.values[z],
                                    somora_month_time_lst.obs_lat.values[z],
                                    somora_month_time_lst.obs_lon.values[z])
        lst_somora = np.append(lst_somora, lst_save[0])
    pandas_time_test_2 = pd.to_datetime(lst_somora)
    hours_2 = pandas_time_test_2.hour #pick up hour points

    somora_month_1 = month_12_o3_somora_data_1[month_data]
    somora_month_time_lst_1 = month_time_lst_somora_12_1[month_data]
    lst_somora_1 = np.array([])
    for z in range(somora_month_time_lst_1.time.shape[0]):   # utc convert to lst for each month of SOMORA 2018
        lst_save = get_LST_from_UTC(somora_month_time_lst_1.time.values[z],
                                    somora_month_time_lst_1.obs_lat.values[z],
                                    somora_month_time_lst_1.obs_lon.values[z])
        lst_somora_1 = np.append(lst_somora_1, lst_save[0]) # lst_save[0] is time term
    pandas_time_test_3 = pd.to_datetime(lst_somora_1)
    hours_3 = pandas_time_test_3.hour #pick up hour points

    #sunrise and sunset calculation
    sunrise_set_day_order = np.array([])
    sunrise_ft = np.array([])
    sunset_ft = np.array([])
    for x, y in enumerate(hours):
        if y == 8:  # pick 8 am as the referece to pick the order of day
            sunrise_set_day_order = np.append(np.array(sunrise_set_day_order, dtype=int), x)
    for sunrise_dayofmonth in sunrise_set_day_order:
        sunrise_set_lst_covert = lst_gromos[sunrise_dayofmonth]
        sunrise_set_lst = get_sunset_lst_from_lst(sunrise_set_lst_covert, gromos_month_time_lst.obs_lat.values[
            0])  # calculate sunrise and set lst ; surise_set_lst_covert[0] is that pick time variable
        sunrise_datetime = pd.to_datetime(sunrise_set_lst)
        #date(str) to num to compute mean and transfer back to date
        sunrise_str = (str(sunrise_datetime[0].hour) + ':' + str(sunrise_datetime[0].minute) + ':' + str(
            sunrise_datetime[0].second))
        sunrise_str_midnumber = datetime.strptime(sunrise_str, '%H:%M:%S')
        sunrise_str_to_date = date2num(sunrise_str_midnumber)
        sunrise_ft = np.append(sunrise_ft, sunrise_str_to_date)
        sunset_str = (str(sunrise_datetime[1].hour) + ':' + str(sunrise_datetime[1].minute) + ':' + str(
            sunrise_datetime[1].second))
        sunset_str_midnumber = datetime.strptime(sunset_str, '%H:%M:%S')
        sunset_str_to_date = date2num(sunset_str_midnumber)
        sunset_ft = np.append(sunset_ft, sunset_str_to_date)
    sunrise_ft_av = num2date(np.average(sunrise_ft))
    sunset_ft_av = num2date(np.average(sunset_ft))
    sunrise_transfer_num = sunrise_ft_av.hour + sunrise_ft_av.minute / 60 # convert to mathematical format
    sunset_transfer_num = sunset_ft_av.hour + sunset_ft_av.minute / 60
    print(sunrise_ft_av)
    print(sunset_ft_av)
################################################gromos diurnal ozone cycle
    gromos_month_data_av_full = np.zeros((1, 47))  # build a one row full 0 array
    for x in range(0, 24):
        time_month_data = np.array([])
        for i, j in enumerate(hours):
            if j == x:
                time_month_data = np.append(np.array(time_month_data, dtype=int),i)  # pick up indexs of certain hour for instance j=0 pick up index of 0 oclock
        gromos_month_data = gromos_month[time_month_data, :]  # pick up o3_x data corresponding to index hour
        gromos_month_data_sum = np.sum(gromos_month_data, axis=0)  # sum all the data along row
        gromos_month_data_av = gromos_month_data_sum / time_month_data.size  # average the certain hour of whole month data
        gromos_month_data_av_full = np.vstack((gromos_month_data_av_full, gromos_month_data_av))  # build a diuranl ts array

    gromos_month_data_av_full = np.delete(gromos_month_data_av_full, 0, axis=0)  # delete the first row (full 0 array)
    average_midnight = (gromos_month_data_av_full[0] + gromos_month_data_av_full[23]) / 2  # average 0am and 23pm as midnight reference
    gromos_month_data_av_full_rate_midnight = np.transpose(gromos_month_data_av_full / average_midnight)  # get a rate arrary (every row divide into midnight row)
    gromos_month_data_av_full_rate_midnight_1 = gromos_month_data_av_full_rate_midnight

    gromos_month_1_data_av_full = np.zeros((1, 47))  # build a one row full 0 array
    for x in range(0, 24):
        time_month_data = np.array([])
        for i, j in enumerate(hours_1):
            if j == x:
                time_month_data = np.append(np.array(time_month_data, dtype=int),
                                            i)  # pick up indexs of certain hour for instance j=0 pick up index of 0 oclock
        gromos_month_1_data = gromos_month_1[time_month_data, :]  # pick up o3_x data corresponding to index hour
        gromos_month_1_data_sum = np.sum(gromos_month_1_data, axis=0)  # sum all the data along row
        gromos_month_1_data_av = gromos_month_1_data_sum / time_month_data.size  # average the certain hour of whole month data
        gromos_month_1_data_av_full = np.vstack((gromos_month_1_data_av_full, gromos_month_1_data_av))  # build a diuranl ts array

    gromos_month_1_data_av_full = np.delete(gromos_month_1_data_av_full, 0, axis=0)  # delete the first row (full 0 array)
    average_midnight = (gromos_month_1_data_av_full[0] + gromos_month_1_data_av_full[23]) / 2  # average 0am and 23pm as midnight reference
    gromos_month_1_data_av_full_rate_midnight = np.transpose(gromos_month_1_data_av_full / average_midnight)  # get a rate arrary (every row divide into midnight row)
    gromos_month_data_av_full_rate_midnight_2 = gromos_month_1_data_av_full_rate_midnight
    gromos_month_data_av_full_rate_midnight_12 = (gromos_month_data_av_full_rate_midnight_1 + gromos_month_data_av_full_rate_midnight_2) / 2

####################################################################somora
    somora_month_data_av_full = np.zeros((1, 47))  # build a one row full 0 array
    for x in range(0, 24):
        time_month_data = np.array([])
        for i, j in enumerate(hours_2):
            if j == x:
                time_month_data = np.append(np.array(time_month_data, dtype=int),
                                            i)  # pick up indexs of certain hour for instance j=0 pick up index of 0 oclock
        somora_month_data = somora_month[time_month_data, :]  # pick up o3_x data corresponding to index hour
        somora_month_data_sum = np.sum(somora_month_data, axis=0)  # sum all the data along row
        somora_month_data_av = somora_month_data_sum / time_month_data.size  # average the certain hour of whole month data
        somora_month_data_av_full = np.vstack(
            (somora_month_data_av_full, somora_month_data_av))  # build a diuranl ts array

    somora_month_data_av_full = np.delete(somora_month_data_av_full, 0, axis=0)  # delete the first row (full 0 array)
    average_midnight = (somora_month_data_av_full[0] + somora_month_data_av_full[
        23]) / 2  # average 0am and 23pm as midnight reference
    somora_month_data_av_full_rate_midnight = np.transpose(
        somora_month_data_av_full / average_midnight)  # get a rate arrary (every row divide into midnight row)
    somora_month_data_av_full_rate_midnight_1 = somora_month_data_av_full_rate_midnight

    somora_month_1_data_av_full = np.zeros((1, 47))  # build a one row full 0 array
    for x in range(0, 24):
        time_month_data = np.array([])
        for i, j in enumerate(hours_3):
            if j == x:
                time_month_data = np.append(np.array(time_month_data, dtype=int),
                                            i)  # pick up indexs of certain hour for instance j=0 pick up index of 0 oclock
        somora_month_1_data = somora_month_1[time_month_data, :]  # pick up o3_x data corresponding to index hour
        somora_month_1_data_sum = np.sum(somora_month_1_data, axis=0)  # sum all the data along row
        somora_month_1_data_av = somora_month_1_data_sum / time_month_data.size  # average the certain hour of whole month data
        somora_month_1_data_av_full = np.vstack(
            (somora_month_1_data_av_full, somora_month_1_data_av))  # build a diuranl ts array

    somora_month_1_data_av_full = np.delete(somora_month_1_data_av_full, 0,
                                            axis=0)  # delete the first row (full 0 array)
    average_midnight = (somora_month_1_data_av_full[0] + somora_month_1_data_av_full[23]) / 2  # average 0am and 23pm as midnight reference
    somora_month_1_data_av_full_rate_midnight = np.transpose(
        somora_month_1_data_av_full / average_midnight)  # get a rate arrary (every row divide into midnight row)
    somora_month_data_av_full_rate_midnight_2 = somora_month_1_data_av_full_rate_midnight
    somora_month_data_av_full_rate_midnight_12 = (somora_month_data_av_full_rate_midnight_1 + somora_month_data_av_full_rate_midnight_2) / 2



###################################################GDOC
    GDOC_month = GDOC_month_12_data[month_data]
    GDOC_month = GDOC_month.transpose()
#####################################################WACCUM

    #WACCM_month = WACCM_month_12_data[month_data].O3
    #CFT_time = xr.CFTimeIndex(WACCM_month.time.values)
    #hours_WACCM = CFT_time.hour

    WACCM_month_o3 = WACCM_month_12_data[month_data].O3
    WACCM_month = WACCM_month_12_data[month_data]
    lst_WACCM = np.array([])
    for z in range(WACCM_month.datetime.size):  # utc convert to lst
        lst_save = get_LST_from_UTC(WACCM_month.datetime[z],
                                    46.00000,
                                    7.50000)
        lst_WACCM = np.append(lst_WACCM, lst_save[0])
    pandas_time_test = pd.to_datetime(lst_WACCM)
    hours_WACCM = pandas_time_test.hour

    WACCM_month_data_av_full = np.zeros((1, 66))  # build a one row full 0 array
    for x in range(0, 24):
        time_month_data = np.array([])
        for i, j in enumerate(hours_WACCM):
            if j == x:
                time_month_data = np.append(np.array(time_month_data, dtype=int),
                                            i)  # pick up indexs of certain hour for instance j=0 pick up index of 0 oclock
        WACCM_month_data = WACCM_month_o3[time_month_data, :]  # pick up o3_x data corresponding to index hour
        WACCM_month_data_sum = np.sum(WACCM_month_data, axis=0)  # sum all the data along row
        WACCM_month_data_av = WACCM_month_data_sum / time_month_data.size  # average the certain hour of whole month data
        WACCM_month_data_av_full = np.vstack((WACCM_month_data_av_full, WACCM_month_data_av))  # build WACCM_ diuranl ts array

    WACCM_month_data_av_full = np.delete(WACCM_month_data_av_full, 0, axis=0)  # delete the first row (full 0 array)
    average_midnight = WACCM_month_data_av_full[0] # average 0am as midnight reference
    WACCM_month_data_av_full_rate_midnight = np.transpose(
        WACCM_month_data_av_full / average_midnight)  # get a rate arrary (every row divide into midnight row)
###########################################################################3



    interval = np.arange(0.93, 1.07, 0.01)  # range for levels in contour plot
    #interval = np.arange(0.10, 1.70, 0.1)
    fig = plt.figure(dpi=100,figsize=(18,16))
    ax1 = fig.add_subplot(221)
    ax1.set_yscale('log')
    ax1.set_xlabel('Local Solar Time[Hour]', fontsize=7)
    ax1.set_ylabel('Pressure[hPa]', fontsize=7)
    ax1.set_title(month_title_12[month_title],fontsize=7)
    ax1.set_ylim(1, 100)
    #ax1.set_ylim(0.01, 100)
    cs = plt.contourf(np.arange(0,24), gromos.o3_p / 100, gromos_month_data_av_full_rate_midnight_12, levels=(interval),
                      cmap='coolwarm',fontsize=30, extend="both")  # colors='k' is mono color line
    plt.gca().invert_yaxis()  # change the order of y axis
    #labels = ax.set_xticklabels(['0:00', '1:00', '2:00', '3:00', '4:00', '5:00', '6:00', '7:00', '8:00', '9:00', '10:00', '11:00','12:00','13:00', '14:00', '15:00', '16:00', '17:00', '18:00', '19:00', '20:00', '21:00', '22:00', '23:00'])
    cs.cmap.set_under('MediumBlue')  # set the color over colorbar low boundary
    cs.cmap.set_over('Crimson')  # set the color over colorbar upper boundary
    ax1.axhline(y=0.02, color='black', linestyle='dashed')
    ax1.axhline(y=110, color='black', linestyle='dashed')
    ax1.axvline(x=sunrise_transfer_num, color='white', linestyle='dashed')
    ax1.axvline(x=sunset_transfer_num, color='black', linestyle='dashed')
    ax1.text(0, 0.02, 'MR', rotation=45,color='red')
    #ax1.text(0, 110, 'MR', rotation=45,color='red')
    # plt.clabel(cs, inline=True, colors='black', fontsize=7)
    cb = plt.colorbar()
    cb.set_label('Ratio to ozone at midnight')

    ax2 = fig.add_subplot(222)
    ax2.set_yscale('log')
    ax2.set_xlabel('Local Solar Time[Hour]',fontsize=7)
    ax2.set_ylabel('Pressure[hPa]',fontsize=7)
    ax2.set_title(month_title_somora_12[month_title],fontsize=7)
    ax2.set_ylim(1, 100)
    #ax2.set_ylim(0.01, 100)
    cs = plt.contourf(np.arange(0,24), gromos.o3_p / 100, somora_month_data_av_full_rate_midnight_12, levels=(interval),
                      cmap='coolwarm', extend="both")  # colors='k' is mono color line
    plt.gca().invert_yaxis()  # change the order of y axis
    #ticks = ax.set_xticks([np.arange(0, 24)])
    #labels = ax.set_xticklabels(['0:00', '1:00', '2:00', '3:00', '4:00', '5:00', '6:00', '7:00', '8:00', '9:00', '10:00', '11:00','12:00','13:00', '14:00', '15:00', '16:00', '17:00', '18:00', '19:00', '20:00', '21:00', '22:00', '23:00'])
    cs.cmap.set_under('MediumBlue')  # set the color over colorbar low boundary
    cs.cmap.set_over('Crimson')  # set the color over colorbar upper boundary
    ax2.axhline(y=0.02, color='black', linestyle='dashed')
    ax2.axhline(y=110, color='black', linestyle='dashed')
    ax2.axvline(x=sunrise_transfer_num, color='white', linestyle='dashed')
    ax2.axvline(x=sunset_transfer_num, color='black', linestyle='dashed')
    ax2.text(0, 0.02, 'MR', rotation=45,color='red')
    #ax2.text(0, 110, 'MR', rotation=45,color='red')
    # plt.clabel(cs, inline=True, colors='black', fontsize=7)
    cb = plt.colorbar()
    cb.set_label('Ratio to ozone at midnight')


    ax3 = fig.add_subplot(223)
    ax3.set_yscale('log')
    ax3.set_xlabel('Local Solar Time[Hour]', fontsize=7)
    ax3.set_ylabel('Pressure[hPa]', fontsize=7)
    ax3.set_title(month_title_GDOC_12[month_title],fontsize=7)
    ax3.set_ylim(1, 100)
    #ax3.set_ylim(0.01, 100)
    cs = plt.contourf(GDOC_hour, GDOC_pressure, GDOC_month, levels=(interval), cmap='coolwarm',extend="both")
    cs.cmap.set_under('MediumBlue')  # set the color over colorbar low boundary
    cs.cmap.set_over('Crimson')  # set the color over colorbar upper boundary
    ax3.axvline(x=sunrise_transfer_num, color='white', linestyle='dashed')
    ax3.axvline(x=sunset_transfer_num, color='black', linestyle='dashed')
    plt.gca().invert_yaxis()
    cb = plt.colorbar()
    cb.set_label('Ratio to ozone at midnight')


    ax4 = fig.add_subplot(224)
    ax4.set_yscale('log')
    ax4.set_xlabel('Local Solar Time[Hour]', fontsize=7)
    ax4.set_ylabel('Pressure[hPa]', fontsize=7)
    ax4.set_title(month_title_WACCM_12[month_data],fontsize=7)
    ax4.set_ylim(1, 1e2)
    #ax4.set_ylim(0.01, 100)
    cs = plt.contourf(np.arange(0, 24), WACCM_pressure, WACCM_month_data_av_full_rate_midnight, levels=(interval),
                      cmap='coolwarm', extend="both")  # colors='k' is mono color line
    plt.gca().invert_yaxis()  # change the order of y ax4is
    # ticks = ax4.set_xticks([np.arange(0, 24)])
    # labels = ax4.set_xticklabels(['0:00', '1:00', '2:00', '3:00', '4:00', '5:00', '6:00', '7:00', '8:00', '9:00', '10:00', '11:00','12:00','13:00', '14:00', '15:00', '16:00', '17:00', '18:00', '19:00', '20:00', '21:00', '22:00', '23:00'])
    cs.cmap.set_under('MediumBlue')  # set the color over colorbar low boundary
    cs.cmap.set_over('Crimson')  # set the color over colorbar upper boundary
    # ax4.ax4hline(y=0.02, color='black', linestyle='dashed')
    # ax4.ax4hline(y=110, color='black', linestyle='dashed')
    ax4.axvline(x=sunrise_transfer_num, color='white', linestyle='dashed')
    ax4.axvline(x=sunset_transfer_num, color='black', linestyle='dashed')
    # ax4.text(0, 0.02, 'MR', rotation=45,color='red')
    # ax4.text(0, 110, 'MR', rotation=45,color='red')
    # plt.clabel(cs, inline=True, colors='black', fontsize=7)
    cb = plt.colorbar()
    cb.set_label('Ratio to ozone at midnight')
    plt.suptitle(Comparison_month_title_12[month_data],fontsize=30)
    #plt.savefig('C://Users//Hou//Desktop//master thesis//figures for 2018 data//2021.11//stratospheric comparison//' + Comparison_month_title_12[month_title] + '.png', dpi=100)
    #plt.savefig('C:/Users/Hou/Desktop/master thesis/figures for 2018 data/2021.11/comparison_full_pressure/' + Comparison_month_title_12[month_title] + '.png', dpi=100)
    plt.show()


