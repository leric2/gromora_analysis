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

filename = 'F:\PyCharm Community Edition 2019.3\practice2021 april/GROMOS_2018_waccm_continuum_ozone.nc'

gromos = xr.open_dataset(
    filename,
    decode_times=True,
    decode_coords=True,
    # use_cftime=True,
)

filename = 'F:\PyCharm Community Edition 2019.3\practice2021 april/GROMOS_2016_waccm_monthly_scaled_h2o_ozone.nc'

gromos_1 = xr.open_dataset(
    filename,
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


'''
if __name__ == "__main__":
    #date = spectro_dataset.time.data[0]


    date = np.datetime64('2017-12-30 08:12:20.123456')
    date = gromos.time.values[0]
    lat = gromos.obs_lat.values[0]
    lon = gromos.obs_lon.values[0]

    sza_pysolar = pysolar_sza(date, lat, lon)

    lst, ha, sza, night = get_LST_from_UTC(date, lat, lon)

    lst_simone = datetime64_2_datetime(date).astimezone(gromora_tz) + timedelta(hours=lon*24/360)
    print('LST diff:', (lst - lst_simone).total_seconds() / 60, 'min')

    if night:
        print('Night !')
    else:
        print('Day !')
#####################################################
    sunset_ha = hour_angle_sunset(lst, lat)

    sunrise_lst, sunset_lst = lst_sunset_from_hour_angle(
        sunset_ha,
        midnight_lst=lst.replace(hour=0, minute=0, second=0, microsecond=0)
    )

    print('Sunrise: ', sunrise_lst, ' and sunset: ', sunset_lst)
    print('Daylight hours: ', (sunset_lst - sunrise_lst).total_seconds() / 3600, 'hours')
    if np.abs(ha) > sunset_ha:
        print('Night !')
    else:
        print('Day !')

'''
########################################################################################set up sunrise and sunset codes
'''
gromos_month_1 = gromos.o3_x.sel(time=slice("2018-06-21", "2018-06-21"))
gromos_month_lst_1 = gromos.sel(time=slice("2018-06-21", "2018-06-21"))
gromos_month = gromos_month_1
gromos_month_time_lst = gromos_month_lst_1
lst_gromos = np.array([])
for z in range(gromos_month_time_lst.time.shape[0]):    #2018 utc convert to lst
    lst_save = get_LST_from_UTC(gromos_month_time_lst.time.values[z],
                                gromos_month_time_lst.obs_lat.values[z],
                                gromos_month_time_lst.obs_lon.values[z])
    lst_gromos = np.append(lst_gromos, lst_save[0])
pandas_time_test = pd.to_datetime(lst_gromos)
hours = pandas_time_test.hour

sunrise_set_day_order = np.array([])
sunrise_ft = np.array([])
sunset_ft = np.array([])
for x, y in enumerate(hours):
    if y == 8: #pick 8 am as the referece to pick the order of day
        sunrise_set_day_order = np.append(np.array(sunrise_set_day_order, dtype=int), x)
for sunrise_dayofmonth in sunrise_set_day_order:
    sunrise_set_lst_covert = lst_gromos[sunrise_dayofmonth]
    surnise_set_lst = get_sunset_lst_from_lst(sunrise_set_lst_covert, gromos_month_time_lst.obs_lat.values[0]) #surise_set_lst_covert[0] is that pick time variable
    sunrise_datetime = pd.to_datetime(surnise_set_lst)
    sunrise_str = (str(sunrise_datetime[0].hour)+':'+str(sunrise_datetime[0].minute)+':'+str(sunrise_datetime[0].second))
    sunrise_str_midnumber = datetime.strptime(sunrise_str,'%H:%M:%S')
    sunrise_str_to_date = date2num(sunrise_str_midnumber)
    sunrise_ft = np.append(sunrise_ft,sunrise_str_to_date)
    sunset_str = (str(sunrise_datetime[1].hour) + ':' + str(sunrise_datetime[1].minute) + ':' + str(sunrise_datetime[1].second))
    sunset_str_midnumber = datetime.strptime(sunset_str, '%H:%M:%S')
    sunset_str_to_date = date2num(sunset_str_midnumber)
    sunset_ft = np.append(sunset_ft, sunset_str_to_date)
sunrise_ft_av = num2date(np.average(sunrise_ft))
sunset_ft_av = num2date(np.average(sunset_ft))
sunrise_transfer_num = sunrise_ft_av.hour + sunrise_ft_av.minute/60
sunset_transfer_num = sunset_ft_av.hour + sunset_ft_av.minute/60

print(sunrise_ft_av)
print(sunset_ft_av)
'''
################################################################

gromos_month_1 = gromos.o3_x.sel(time=slice("2018-01-01", "2018-01-31"))
gromos_month_2 = gromos.o3_x.sel(time=slice("2018-02-01", "2018-02-28"))  # 2018 slice month o3 data
gromos_month_3 = gromos.o3_x.sel(time=slice("2018-03-01", "2018-03-31"))
gromos_month_4 = gromos.o3_x.sel(time=slice("2018-04-01", "2018-04-30"))
gromos_month_5 = gromos.o3_x.sel(time=slice("2018-05-01", "2018-05-31"))
gromos_month_6 = gromos.o3_x.sel(time=slice("2018-06-01", "2018-06-30"))
gromos_month_7 = gromos.o3_x.sel(time=slice("2018-07-01", "2018-07-31"))
gromos_month_8 = gromos.o3_x.sel(time=slice("2018-08-01", "2018-08-31"))
gromos_month_9 = gromos.o3_x.sel(time=slice("2018-09-01", "2018-09-30"))
gromos_month_10 = gromos.o3_x.sel(time=slice("2018-10-01", "2018-10-31"))
gromos_month_11 = gromos.o3_x.sel(time=slice("2018-11-01", "2018-11-30"))
gromos_month_12 = gromos.o3_x.sel(time=slice("2018-12-01", "2018-12-31"))

gromos_month_lst_1 = gromos.sel(time=slice("2018-01-01", "2018-01-31"))
gromos_month_lst_2 = gromos.sel(time=slice("2018-02-01", "2018-02-28"))  # 2018 slice month data
gromos_month_lst_3 = gromos.sel(time=slice("2018-03-01", "2018-03-31"))
gromos_month_lst_4 = gromos.sel(time=slice("2018-04-01", "2018-04-30"))
gromos_month_lst_5 = gromos.sel(time=slice("2018-05-01", "2018-05-31"))
gromos_month_lst_6 = gromos.sel(time=slice("2018-06-01", "2018-06-30"))
gromos_month_lst_7 = gromos.sel(time=slice("2018-07-01", "2018-07-31"))
gromos_month_lst_8 = gromos.sel(time=slice("2018-08-01", "2018-08-31"))
gromos_month_lst_9 = gromos.sel(time=slice("2018-09-01", "2018-09-30"))
gromos_month_lst_10 = gromos.sel(time=slice("2018-10-01", "2018-10-31"))
gromos_month_lst_11 = gromos.sel(time=slice("2018-11-01", "2018-11-30"))
gromos_month_lst_12 = gromos.sel(time=slice("2018-12-01", "2018-12-31"))

gromos_1_month_1 = gromos_1.o3_x.sel(time=slice("2016-01-01", "2016-01-31"))
gromos_1_month_2 = gromos_1.o3_x.sel(time=slice("2016-02-01", "2016-02-28"))  # 2016 slice month o3 data
gromos_1_month_3 = gromos_1.o3_x.sel(time=slice("2016-03-01", "2016-03-31"))
gromos_1_month_4 = gromos_1.o3_x.sel(time=slice("2016-04-01", "2016-04-30"))
gromos_1_month_5 = gromos_1.o3_x.sel(time=slice("2016-05-01", "2016-05-31"))
gromos_1_month_6 = gromos_1.o3_x.sel(time=slice("2016-06-01", "2016-06-30"))
gromos_1_month_7 = gromos_1.o3_x.sel(time=slice("2016-07-01", "2016-07-31"))
gromos_1_month_8 = gromos_1.o3_x.sel(time=slice("2016-08-01", "2016-08-31"))
gromos_1_month_9 = gromos_1.o3_x.sel(time=slice("2016-09-01", "2016-09-30"))
gromos_1_month_10 = gromos_1.o3_x.sel(time=slice("2016-10-01", "2016-10-31"))
gromos_1_month_11 = gromos_1.o3_x.sel(time=slice("2016-11-01", "2016-11-30"))
gromos_1_month_12 = gromos_1.o3_x.sel(time=slice("2016-12-01", "2016-12-31"))

gromos_1_month_lst_1 = gromos_1.sel(time=slice("2016-01-01", "2016-01-31"))
gromos_1_month_lst_2 = gromos_1.sel(time=slice("2016-02-01", "2016-02-28"))  # 2016 slice month data
gromos_1_month_lst_3 = gromos_1.sel(time=slice("2016-03-01", "2016-03-31"))
gromos_1_month_lst_4 = gromos_1.sel(time=slice("2016-04-01", "2016-04-30"))
gromos_1_month_lst_5 = gromos_1.sel(time=slice("2016-05-01", "2016-05-31"))
gromos_1_month_lst_6 = gromos_1.sel(time=slice("2016-06-01", "2016-06-30"))
gromos_1_month_lst_7 = gromos_1.sel(time=slice("2016-07-01", "2016-07-31"))
gromos_1_month_lst_8 = gromos_1.sel(time=slice("2016-08-01", "2016-08-31"))
gromos_1_month_lst_9 = gromos_1.sel(time=slice("2016-09-01", "2016-09-30"))
gromos_1_month_lst_10 = gromos_1.sel(time=slice("2016-10-01", "2016-10-31"))
gromos_1_month_lst_11 = gromos_1.sel(time=slice("2016-11-01", "2016-11-30"))
gromos_1_month_lst_12 = gromos_1.sel(time=slice("2016-12-01", "2016-12-31"))

month_time_lst_12 = (
gromos_month_lst_1, gromos_month_lst_2, gromos_month_lst_3, gromos_month_lst_4, gromos_month_lst_5, gromos_month_lst_6,
gromos_month_lst_7, gromos_month_lst_8, gromos_month_lst_9, gromos_month_lst_10, gromos_month_lst_11,
gromos_month_lst_12)
month_12_o3_data = (
gromos_month_1, gromos_month_2, gromos_month_3, gromos_month_4, gromos_month_5, gromos_month_6, gromos_month_7,
gromos_month_8, gromos_month_9, gromos_month_10, gromos_month_11, gromos_month_12)
'''
month_title_12 = ('1618av-January-stratospheric-ozone-diurnal-ts', '1618av-February-stratospheric-ozone-diurnal-ts',
                  '1618av-March-stratospheric-ozone-diurnal-ts'
                  , '1618av-April-stratospheric-ozone-diurnal-ts', '1618av-May-stratospheric-ozone-diurnal-ts',
                  '1618av-June-stratospheric-ozone-diurnal-ts'
                  , '1618av-July-stratospheric-ozone-diurnal-ts', '1618av-August-stratospheric-ozone-diurnal-ts',
                  '1618av-September-stratospheric-ozone-diurnal-ts'
                  , '1618av-October-stratospheric-ozone-diurnal-ts', '1618av-November-stratospheric-ozone-diurnal-ts',
                  '1618av-December-stratospheric-ozone-diurnal-ts')
'''
month_title_12 = ('1618av-January-full-pressure-ozone-diurnal-ts', '1618av-February-full-pressure-ozone-diurnal-ts',
                  '1618av-March-full-pressure-ozone-diurnal-ts'
                  , '1618av-April-full-pressure-ozone-diurnal-ts', '1618av-May-full-pressure-ozone-diurnal-ts',
                  '1618av-June-full-pressure-ozone-diurnal-ts'
                  , '1618av-July-full-pressure-ozone-diurnal-ts', '1618av-August-full-pressure-ozone-diurnal-ts',
                  '1618av-September-full-pressure-ozone-diurnal-ts'
                  , '1618av-October-full-pressure-ozone-diurnal-ts', '1618av-November-full-pressure-ozone-diurnal-ts',
                  '1618av-December-full-pressure-ozone-diurnal-ts')

month_time_lst_12_1 = (
gromos_1_month_lst_1, gromos_1_month_lst_2, gromos_1_month_lst_3, gromos_1_month_lst_4, gromos_1_month_lst_5, gromos_1_month_lst_6,
gromos_1_month_lst_7, gromos_1_month_lst_8, gromos_1_month_lst_9, gromos_1_month_lst_10, gromos_1_month_lst_11,
gromos_1_month_lst_12)
month_12_o3_data_1 = (
gromos_1_month_1, gromos_1_month_2, gromos_1_month_3, gromos_1_month_4, gromos_1_month_5, gromos_1_month_6, gromos_1_month_7,
gromos_1_month_8, gromos_1_month_9, gromos_1_month_10, gromos_1_month_11, gromos_1_month_12)

for month_data, month_title in enumerate(np.arange(12)):
    gromos_month = month_12_o3_data[month_data]
    gromos_month_time_lst = month_time_lst_12[month_data]
    lst_gromos = np.array([])
    for z in range(gromos_month_time_lst.time.shape[0]):    #2018 utc convert to lst
        lst_save = get_LST_from_UTC(gromos_month_time_lst.time.values[z],
                                    gromos_month_time_lst.obs_lat.values[z],
                                    gromos_month_time_lst.obs_lon.values[z])
        lst_gromos = np.append(lst_gromos, lst_save[0])
    pandas_time_test = pd.to_datetime(lst_gromos)
    hours = pandas_time_test.hour

    gromos_month_1 = month_12_o3_data_1[month_data]
    gromos_month_time_lst_1 = month_time_lst_12_1[month_data]
    lst_gromos_1 = np.array([])
    for z in range(gromos_month_time_lst_1.time.shape[0]):   #2016 utc convert to lst
        lst_save = get_LST_from_UTC(gromos_month_time_lst_1.time.values[z],
                                    gromos_month_time_lst_1.obs_lat.values[z],
                                    gromos_month_time_lst_1.obs_lon.values[z])
        lst_gromos_1 = np.append(lst_gromos_1, lst_save[0]) # lst_save[0] is time term
    pandas_time_test_1 = pd.to_datetime(lst_gromos_1)
    hours_1 = pandas_time_test_1.hour

    sunrise_set_day_order = np.array([])
    sunrise_ft = np.array([])
    sunset_ft = np.array([])
    for x, y in enumerate(hours):
        if y == 8:  # pick 8 am as the referece to pick the order of day
            sunrise_set_day_order = np.append(np.array(sunrise_set_day_order, dtype=int), x)
    for sunrise_dayofmonth in sunrise_set_day_order:
        sunrise_set_lst_covert = lst_gromos[sunrise_dayofmonth]
        surnise_set_lst = get_sunset_lst_from_lst(sunrise_set_lst_covert, gromos_month_time_lst.obs_lat.values[
            0])  # surise_set_lst_covert[0] is that pick time variable
        sunrise_datetime = pd.to_datetime(surnise_set_lst)
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
    sunrise_transfer_num = sunrise_ft_av.hour + sunrise_ft_av.minute / 60
    sunset_transfer_num = sunset_ft_av.hour + sunset_ft_av.minute / 60
    print(sunrise_ft_av)
    print(sunset_ft_av)

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

    #interval = np.arange(0.93, 1.07, 0.01)  # range for levels in contour plot
    interval = np.arange(0.20, 1.90, 0.1)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_yscale('log')
    ax.set_xlabel('Time[Hour]')
    ax.set_ylabel('Pressure\n[hPa]')
    ax.set_title(month_title_12[month_title])
    #ax.set_ylim(1, 100)
    cs = plt.contourf(np.arange(0,24), gromos.o3_p / 100, gromos_month_data_av_full_rate_midnight_12, levels=(interval),
                      cmap='coolwarm', extend="both")  # colors='k' is mono color line
    plt.gca().invert_yaxis()  # change the order of y axis
    #ticks = ax.set_xticks([np.arange(0, 24)])
    #labels = ax.set_xticklabels(['0:00', '1:00', '2:00', '3:00', '4:00', '5:00', '6:00', '7:00', '8:00', '9:00', '10:00', '11:00','12:00','13:00', '14:00', '15:00', '16:00', '17:00', '18:00', '19:00', '20:00', '21:00', '22:00', '23:00'])
    cs.cmap.set_under('MediumBlue')  # set the color over colorbar low boundary
    cs.cmap.set_over('Crimson')  # set the color over colorbar upper boundary
    ax.axhline(y=0.02, color='black', linestyle='dashed')
    ax.axhline(y=110, color='black', linestyle='dashed')
    ax.axvline(x=sunrise_transfer_num, color='white', linestyle='dashed')
    ax.axvline(x=sunset_transfer_num, color='black', linestyle='dashed')
    ax.text(0, 0.02, 'MR', rotation=45,color='red')
    ax.text(0, 110, 'MR', rotation=45,color='red')
    # plt.clabel(cs, inline=True, colors='black', fontsize=10)
    cb = plt.colorbar()
    cb.set_label('Ratio to ozone at midnight')
    #plt.savefig('C://Users//Hou//Desktop//master thesis//figures for 2018 data//2021.7.5 lst_av//stratospheric part//' + month_title_12[month_title] + '.png', dpi=100)
    plt.savefig('C://Users//Hou//Desktop//master thesis//figures for 2018 data//2021.7.5 lst_av//full pressure//' +month_title_12[month_title] + '.png', dpi=100)
    plt.show()