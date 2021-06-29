import datetime
import os

import matplotlib.pyplot as plt
import netCDF4
import numpy as np
import pandas as pd

import xarray as xr

filename = 'F:\PyCharm Community Edition 2019.3\practice2021 april/GROMOS_2018_waccm_continuum_ozone.nc'

gromos = xr.open_dataset(
    filename,
    decode_times=True,
    decode_coords=True,
    # use_cftime=True,
)


#####################################################Jan
gromos_jan_av = gromos.o3_x.sel(time=slice("2018-01-01", "2018-01-31"))
pandas_time_av = pd.to_datetime(gromos_jan_av.time.data)
hours_av = pandas_time_av.hour

gromos_jan_av_data8_av_full = np.zeros((1, 47))
for x in range(0, 24):
    time_jan_av_data8 = np.array([])
    for i, j in enumerate(hours_av):
        if j == x:
            time_jan_av_data8 = np.append(np.array(time_jan_av_data8, dtype=int), i)
    gromos_jan_av_data8 = gromos_jan_av[time_jan_av_data8, :]
    gromos_jan_av_data8_sum = np.sum(gromos_jan_av_data8, axis=0)
    gromos_jan_av_data8_av = gromos_jan_av_data8_sum / time_jan_av_data8.size
    gromos_jan_av_data8_av_full = np.vstack((gromos_jan_av_data8_av_full, gromos_jan_av_data8_av))

gromos_jan_av_data8_av_full = np.transpose(np.delete(gromos_jan_av_data8_av_full, 0, axis=0))


interval_av=np.arange(0.,9,0.3)
fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_yscale('log')
ax.set_xlabel('Time[Hour]')
ax.set_ylabel('Pressure\n[hPa]')
ax.set_title('2018-January_av-ozone-diurnal-ts')
ax.axhline(y=0.02, color='black', linestyle='dashed')
ax.axhline(y=110, color='black', linestyle='dashed')
ax.text(0, 0.02, 'MR', rotation=45,color='yellow')
ax.text(0, 110, 'MR', rotation=45,color='yellow')
#ax.set_ylim(1,100)
cs = plt.contourf(range(0, 24), gromos.o3_p / 100, gromos_jan_av_data8_av_full,levels=interval_av,cmap='cividis')  # colors='k' is mono color line
plt.gca().invert_yaxis()  # change the order of y axis
#plt.clabel(cs, inline=True, colors='black', fontsize=10)
cb = plt.colorbar()
cb.set_label('Ozone[ppm]')
plt.show()


#####################################################Feb
gromos_feb_av = gromos.o3_x.sel(time=slice("2018-02-01", "2018-02-28"))
pandas_time_av = pd.to_datetime(gromos_feb_av.time.data)
hours_av = pandas_time_av.hour

gromos_feb_av_data8_av_full = np.zeros((1, 47))
for x in range(0, 24):
    time_feb_av_data8 = np.array([])
    for i, j in enumerate(hours_av):
        if j == x:
            time_feb_av_data8 = np.append(np.array(time_feb_av_data8, dtype=int), i)
    gromos_feb_av_data8 = gromos_feb_av[time_feb_av_data8, :]
    gromos_feb_av_data8_sum = np.sum(gromos_feb_av_data8, axis=0)
    gromos_feb_av_data8_av = gromos_feb_av_data8_sum / time_feb_av_data8.size
    gromos_feb_av_data8_av_full = np.vstack((gromos_feb_av_data8_av_full, gromos_feb_av_data8_av))

gromos_feb_av_data8_av_full = np.transpose(np.delete(gromos_feb_av_data8_av_full, 0, axis=0))


interval_av=np.arange(0.,9,0.3)
fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_yscale('log')
ax.set_xlabel('Time[Hour]')
ax.set_ylabel('Pressure\n[hPa]')
ax.set_title('2018-February_av-ozone-diurnal-ts')
ax.axhline(y=0.02, color='black', linestyle='dashed')
ax.axhline(y=110, color='black', linestyle='dashed')
ax.text(0, 0.02, 'MR', rotation=45,color='yellow')
ax.text(0, 110, 'MR', rotation=45,color='yellow')
#ax.set_ylim(1,100)
cs = plt.contourf(range(0, 24), gromos.o3_p / 100, gromos_feb_av_data8_av_full,levels=interval_av,cmap='cividis')  # colors='k' is mono color line
plt.gca().invert_yaxis()  # change the order of y axis
#plt.clabel(cs, inline=True, colors='black', fontsize=10)
cb = plt.colorbar()
cb.set_label('Ozone[ppm]')
plt.show()

#####################################################March
gromos_mar_av = gromos.o3_x.sel(time=slice("2018-03-01", "2018-03-31"))
pandas_time_av = pd.to_datetime(gromos_mar_av.time.data)
hours_av = pandas_time_av.hour

gromos_mar_av_data8_av_full = np.zeros((1, 47))
for x in range(0, 24):
    time_mar_av_data8 = np.array([])
    for i, j in enumerate(hours_av):
        if j == x:
            time_mar_av_data8 = np.append(np.array(time_mar_av_data8, dtype=int), i)
    gromos_mar_av_data8 = gromos_mar_av[time_mar_av_data8, :]
    gromos_mar_av_data8_sum = np.sum(gromos_mar_av_data8, axis=0)
    gromos_mar_av_data8_av = gromos_mar_av_data8_sum / time_mar_av_data8.size
    gromos_mar_av_data8_av_full = np.vstack((gromos_mar_av_data8_av_full, gromos_mar_av_data8_av))

gromos_mar_av_data8_av_full = np.transpose(np.delete(gromos_mar_av_data8_av_full, 0, axis=0))


interval_av=np.arange(0.,9,0.3)
fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_yscale('log')
ax.set_xlabel('Time[Hour]')
ax.set_ylabel('Pressure\n[hPa]')
ax.set_title('2018-March_av-ozone-diurnal-ts')
ax.axhline(y=0.02, color='black', linestyle='dashed')
ax.axhline(y=110, color='black', linestyle='dashed')
ax.text(0, 0.02, 'MR', rotation=45,color='yellow')
ax.text(0, 110, 'MR', rotation=45,color='yellow')
#ax.set_ylim(1,100)
cs = plt.contourf(range(0, 24), gromos.o3_p / 100, gromos_mar_av_data8_av_full,levels=interval_av,cmap='cividis')  # colors='k' is mono color line
plt.gca().invert_yaxis()  # change the order of y axis
#plt.clabel(cs, inline=True, colors='black', fontsize=10)
cb = plt.colorbar()
cb.set_label('Ozone[ppm]')
plt.show()


#####################################################April
gromos_apr_av = gromos.o3_x.sel(time=slice("2018-04-01", "2018-04-30"))
pandas_time_av = pd.to_datetime(gromos_apr_av.time.data)
hours_av = pandas_time_av.hour

gromos_apr_av_data8_av_full = np.zeros((1, 47))
for x in range(0, 24):
    time_apr_av_data8 = np.array([])
    for i, j in enumerate(hours_av):
        if j == x:
            time_apr_av_data8 = np.append(np.array(time_apr_av_data8, dtype=int), i)
    gromos_apr_av_data8 = gromos_apr_av[time_apr_av_data8, :]
    gromos_apr_av_data8_sum = np.sum(gromos_apr_av_data8, axis=0)
    gromos_apr_av_data8_av = gromos_apr_av_data8_sum / time_apr_av_data8.size
    gromos_apr_av_data8_av_full = np.vstack((gromos_apr_av_data8_av_full, gromos_apr_av_data8_av))

gromos_apr_av_data8_av_full = np.transpose(np.delete(gromos_apr_av_data8_av_full, 0, axis=0))


interval_av=np.arange(0.,9,0.3)
fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_yscale('log')
ax.set_xlabel('Time[Hour]')
ax.set_ylabel('Pressure\n[hPa]')
ax.set_title('2018-April_av-ozone-diurnal-ts')
ax.axhline(y=0.02, color='black', linestyle='dashed')
ax.axhline(y=110, color='black', linestyle='dashed')
ax.text(0, 0.02, 'MR', rotation=45,color='yellow')
ax.text(0, 110, 'MR', rotation=45,color='yellow')
#ax.set_ylim(1,100)
cs = plt.contourf(range(0, 24), gromos.o3_p / 100, gromos_apr_av_data8_av_full,levels=interval_av,cmap='cividis')  # colors='k' is mono color line
plt.gca().invert_yaxis()  # change the order of y axis
#plt.clabel(cs, inline=True, colors='black', fontsize=10)
cb = plt.colorbar()
cb.set_label('Ozone[ppm]')
plt.show()


#####################################################May
gromos_may_av = gromos.o3_x.sel(time=slice("2018-05-01", "2018-05-31"))
pandas_time_av = pd.to_datetime(gromos_may_av.time.data)
hours_av = pandas_time_av.hour

gromos_may_av_data8_av_full = np.zeros((1, 47))
for x in range(0, 24):
    time_may_av_data8 = np.array([])
    for i, j in enumerate(hours_av):
        if j == x:
            time_may_av_data8 = np.append(np.array(time_may_av_data8, dtype=int), i)
    gromos_may_av_data8 = gromos_may_av[time_may_av_data8, :]
    gromos_may_av_data8_sum = np.sum(gromos_may_av_data8, axis=0)
    gromos_may_av_data8_av = gromos_may_av_data8_sum / time_may_av_data8.size
    gromos_may_av_data8_av_full = np.vstack((gromos_may_av_data8_av_full, gromos_may_av_data8_av))

gromos_may_av_data8_av_full = np.transpose(np.delete(gromos_may_av_data8_av_full, 0, axis=0))


interval_av=np.arange(0.,9,0.3)
fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_yscale('log')
ax.set_xlabel('Time[Hour]')
ax.set_ylabel('Pressure\n[hPa]')
ax.set_title('2018-May_av-ozone-diurnal-ts')
ax.axhline(y=0.02, color='black', linestyle='dashed')
ax.axhline(y=110, color='black', linestyle='dashed')
ax.text(0, 0.02, 'MR', rotation=45,color='yellow')
ax.text(0, 110, 'MR', rotation=45,color='yellow')
#ax.set_ylim(1,100)
cs = plt.contourf(range(0, 24), gromos.o3_p / 100, gromos_may_av_data8_av_full,levels=interval_av,cmap='cividis')  # colors='k' is mono color line
plt.gca().invert_yaxis()  # change the order of y axis
#plt.clabel(cs, inline=True, colors='black', fontsize=10)
cb = plt.colorbar()
cb.set_label('Ozone[ppm]')
plt.show()


#####################################################June
gromos_june_av = gromos.o3_x.sel(time=slice("2018-06-01", "2018-06-30"))
pandas_time_av = pd.to_datetime(gromos_june_av.time.data)
hours_av = pandas_time_av.hour

gromos_june_av_data8_av_full = np.zeros((1, 47))
for x in range(0, 24):
    time_june_av_data8 = np.array([])
    for i, j in enumerate(hours_av):
        if j == x:
            time_june_av_data8 = np.append(np.array(time_june_av_data8, dtype=int), i)
    gromos_june_av_data8 = gromos_june_av[time_june_av_data8, :]
    gromos_june_av_data8_sum = np.sum(gromos_june_av_data8, axis=0)
    gromos_june_av_data8_av = gromos_june_av_data8_sum / time_june_av_data8.size
    gromos_june_av_data8_av_full = np.vstack((gromos_june_av_data8_av_full, gromos_june_av_data8_av))

gromos_june_av_data8_av_full = np.transpose(np.delete(gromos_june_av_data8_av_full, 0, axis=0))


interval_av=np.arange(0.,9,0.3)
fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_yscale('log')
ax.set_xlabel('Time[Hour]')
ax.set_ylabel('Pressure\n[hPa]')
ax.set_title('2018-June_av-ozone-diurnal-ts')
ax.axhline(y=0.02, color='black', linestyle='dashed')
ax.axhline(y=110, color='black', linestyle='dashed')
ax.text(0, 0.02, 'MR', rotation=45,color='yellow')
ax.text(0, 110, 'MR', rotation=45,color='yellow')
#ax.set_ylim(1,100)
cs = plt.contourf(range(0, 24), gromos.o3_p / 100, gromos_june_av_data8_av_full,levels=interval_av,cmap='cividis')  # colors='k' is mono color line
plt.gca().invert_yaxis()  # change the order of y axis
#plt.clabel(cs, inline=True, colors='black', fontsize=10)
cb = plt.colorbar()
cb.set_label('Ozone[ppm]')
plt.show()


#####################################################July
gromos_july_av = gromos.o3_x.sel(time=slice("2018-07-01", "2018-07-31"))
pandas_time_av = pd.to_datetime(gromos_july_av.time.data)
hours_av = pandas_time_av.hour

gromos_july_av_data8_av_full = np.zeros((1, 47))
for x in range(0, 24):
    time_july_av_data8 = np.array([])
    for i, j in enumerate(hours_av):
        if j == x:
            time_july_av_data8 = np.append(np.array(time_july_av_data8, dtype=int), i)
    gromos_july_av_data8 = gromos_july_av[time_july_av_data8, :]
    gromos_july_av_data8_sum = np.sum(gromos_july_av_data8, axis=0)
    gromos_july_av_data8_av = gromos_july_av_data8_sum / time_july_av_data8.size
    gromos_july_av_data8_av_full = np.vstack((gromos_july_av_data8_av_full, gromos_july_av_data8_av))

gromos_july_av_data8_av_full = np.transpose(np.delete(gromos_july_av_data8_av_full, 0, axis=0))


interval_av=np.arange(0.,9,0.3)
fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_yscale('log')
ax.set_xlabel('Time[Hour]')
ax.set_ylabel('Pressure\n[hPa]')
ax.set_title('2018-July_av-ozone-diurnal-ts')
ax.axhline(y=0.02, color='black', linestyle='dashed')
ax.axhline(y=110, color='black', linestyle='dashed')
ax.text(0, 0.02, 'MR', rotation=45,color='yellow')
ax.text(0, 110, 'MR', rotation=45,color='yellow')
#ax.set_ylim(1,100)
cs = plt.contourf(range(0, 24), gromos.o3_p / 100, gromos_july_av_data8_av_full,levels=interval_av,cmap='cividis')  # colors='k' is mono color line
plt.gca().invert_yaxis()  # change the order of y axis
#plt.clabel(cs, inline=True, colors='black', fontsize=10)
cb = plt.colorbar()
cb.set_label('Ozone[ppm]')
plt.show()


#####################################################August
gromos_august_av = gromos.o3_x.sel(time=slice("2018-08-01", "2018-08-31"))
pandas_time_av = pd.to_datetime(gromos_august_av.time.data)
hours_av = pandas_time_av.hour

gromos_august_av_data8_av_full = np.zeros((1, 47))
for x in range(0, 24):
    time_august_av_data8 = np.array([])
    for i, j in enumerate(hours_av):
        if j == x:
            time_august_av_data8 = np.append(np.array(time_august_av_data8, dtype=int), i)
    gromos_august_av_data8 = gromos_august_av[time_august_av_data8, :]
    gromos_august_av_data8_sum = np.sum(gromos_august_av_data8, axis=0)
    gromos_august_av_data8_av = gromos_august_av_data8_sum / time_august_av_data8.size
    gromos_august_av_data8_av_full = np.vstack((gromos_august_av_data8_av_full, gromos_august_av_data8_av))

gromos_august_av_data8_av_full = np.transpose(np.delete(gromos_august_av_data8_av_full, 0, axis=0))


interval_av=np.arange(0.,9,0.3)
fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_yscale('log')
ax.set_xlabel('Time[Hour]')
ax.set_ylabel('Pressure\n[hPa]')
ax.set_title('2018-August_av-ozone-diurnal-ts')
ax.axhline(y=0.02, color='black', linestyle='dashed')
ax.axhline(y=110, color='black', linestyle='dashed')
ax.text(0, 0.02, 'MR', rotation=45,color='yellow')
ax.text(0, 110, 'MR', rotation=45,color='yellow')
#ax.set_ylim(1,100)
cs = plt.contourf(range(0, 24), gromos.o3_p / 100, gromos_august_av_data8_av_full,levels=interval_av,cmap='cividis')  # colors='k' is mono color line
plt.gca().invert_yaxis()  # change the order of y axis
#plt.clabel(cs, inline=True, colors='black', fontsize=10)
cb = plt.colorbar()
cb.set_label('Ozone[ppm]')
plt.show()

#####################################################September
gromos_sep_av = gromos.o3_x.sel(time=slice("2018-09-01", "2018-09-30"))
pandas_time_av = pd.to_datetime(gromos_sep_av.time.data)
hours_av = pandas_time_av.hour

gromos_sep_av_data8_av_full = np.zeros((1, 47))
for x in range(0, 24):
    time_sep_av_data8 = np.array([])
    for i, j in enumerate(hours_av):
        if j == x:
            time_sep_av_data8 = np.append(np.array(time_sep_av_data8, dtype=int), i)
    gromos_sep_av_data8 = gromos_sep_av[time_sep_av_data8, :]
    gromos_sep_av_data8_sum = np.sum(gromos_sep_av_data8, axis=0)
    gromos_sep_av_data8_av = gromos_sep_av_data8_sum / time_sep_av_data8.size
    gromos_sep_av_data8_av_full = np.vstack((gromos_sep_av_data8_av_full, gromos_sep_av_data8_av))

gromos_sep_av_data8_av_full = np.transpose(np.delete(gromos_sep_av_data8_av_full, 0, axis=0))


interval_av=np.arange(0.,9,0.3)
fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_yscale('log')
ax.set_xlabel('Time[Hour]')
ax.set_ylabel('Pressure\n[hPa]')
ax.set_title('2018-September_av-ozone-diurnal-ts')
ax.axhline(y=0.02, color='black', linestyle='dashed')
ax.axhline(y=110, color='black', linestyle='dashed')
ax.text(0, 0.02, 'MR', rotation=45,color='yellow')
ax.text(0, 110, 'MR', rotation=45,color='yellow')
#ax.set_ylim(1,100)
cs = plt.contourf(range(0, 24), gromos.o3_p / 100, gromos_sep_av_data8_av_full,levels=interval_av,cmap='cividis')  # colors='k' is mono color line
plt.gca().invert_yaxis()  # change the order of y axis
#plt.clabel(cs, inline=True, colors='black', fontsize=10)
cb = plt.colorbar()
cb.set_label('Ozone[ppm]')
plt.show()


#####################################################October
gromos_oct_av = gromos.o3_x.sel(time=slice("2018-10-01", "2018-10-31"))
pandas_time_av = pd.to_datetime(gromos_oct_av.time.data)
hours_av = pandas_time_av.hour

gromos_oct_av_data8_av_full = np.zeros((1, 47))
for x in range(0, 24):
    time_oct_av_data8 = np.array([])
    for i, j in enumerate(hours_av):
        if j == x:
            time_oct_av_data8 = np.append(np.array(time_oct_av_data8, dtype=int), i)
    gromos_oct_av_data8 = gromos_oct_av[time_oct_av_data8, :]
    gromos_oct_av_data8_sum = np.sum(gromos_oct_av_data8, axis=0)
    gromos_oct_av_data8_av = gromos_oct_av_data8_sum / time_oct_av_data8.size
    gromos_oct_av_data8_av_full = np.vstack((gromos_oct_av_data8_av_full, gromos_oct_av_data8_av))

gromos_oct_av_data8_av_full = np.transpose(np.delete(gromos_oct_av_data8_av_full, 0, axis=0))


interval_av=np.arange(0.,9,0.3)
fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_yscale('log')
ax.set_xlabel('Time[Hour]')
ax.set_ylabel('Pressure\n[hPa]')
ax.set_title('2018-October_av-ozone-diurnal-ts')
ax.axhline(y=0.02, color='black', linestyle='dashed')
ax.axhline(y=110, color='black', linestyle='dashed')
ax.text(0, 0.02, 'MR', rotation=45,color='yellow')
ax.text(0, 110, 'MR', rotation=45,color='yellow')
#ax.set_ylim(1,100)
cs = plt.contourf(range(0, 24), gromos.o3_p / 100, gromos_oct_av_data8_av_full,levels=interval_av,cmap='cividis')  # colors='k' is mono color line
plt.gca().invert_yaxis()  # change the order of y axis
#plt.clabel(cs, inline=True, colors='black', fontsize=10)
cb = plt.colorbar()
cb.set_label('Ozone[ppm]')
plt.show()


#####################################################November
gromos_nov_av = gromos.o3_x.sel(time=slice("2018-11-01", "2018-11-30"))
pandas_time_av = pd.to_datetime(gromos_nov_av.time.data)
hours_av = pandas_time_av.hour

gromos_nov_av_data8_av_full = np.zeros((1, 47))
for x in range(0, 24):
    time_nov_av_data8 = np.array([])
    for i, j in enumerate(hours_av):
        if j == x:
            time_nov_av_data8 = np.append(np.array(time_nov_av_data8, dtype=int), i)
    gromos_nov_av_data8 = gromos_nov_av[time_nov_av_data8, :]
    gromos_nov_av_data8_sum = np.sum(gromos_nov_av_data8, axis=0)
    gromos_nov_av_data8_av = gromos_nov_av_data8_sum / time_nov_av_data8.size
    gromos_nov_av_data8_av_full = np.vstack((gromos_nov_av_data8_av_full, gromos_nov_av_data8_av))

gromos_nov_av_data8_av_full = np.transpose(np.delete(gromos_nov_av_data8_av_full, 0, axis=0))


interval_av=np.arange(0.,9,0.3)
fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_yscale('log')
ax.set_xlabel('Time[Hour]')
ax.set_ylabel('Pressure\n[hPa]')
ax.set_title('2018-November_av-ozone-diurnal-ts')
ax.axhline(y=0.02, color='black', linestyle='dashed')
ax.axhline(y=110, color='black', linestyle='dashed')
ax.text(0, 0.02, 'MR', rotation=45,color='yellow')
ax.text(0, 110, 'MR', rotation=45,color='yellow')
#ax.set_ylim(1,100)
cs = plt.contourf(range(0, 24), gromos.o3_p / 100, gromos_nov_av_data8_av_full,levels=interval_av,cmap='cividis')  # colors='k' is mono color line
plt.gca().invert_yaxis()  # change the order of y axis
#plt.clabel(cs, inline=True, colors='black', fontsize=10)
cb = plt.colorbar()
cb.set_label('Ozone[ppm]')
plt.show()


#####################################################December
gromos_dec_av = gromos.o3_x.sel(time=slice("2018-12-01", "2018-12-31"))
pandas_time_av = pd.to_datetime(gromos_dec_av.time.data)
hours_av = pandas_time_av.hour

gromos_dec_av_data8_av_full = np.zeros((1, 47))
for x in range(0, 24):
    time_dec_av_data8 = np.array([])
    for i, j in enumerate(hours_av):
        if j == x:
            time_dec_av_data8 = np.append(np.array(time_dec_av_data8, dtype=int), i)
    gromos_dec_av_data8 = gromos_dec_av[time_dec_av_data8, :]
    gromos_dec_av_data8_sum = np.sum(gromos_dec_av_data8, axis=0)
    gromos_dec_av_data8_av = gromos_dec_av_data8_sum / time_dec_av_data8.size
    gromos_dec_av_data8_av_full = np.vstack((gromos_dec_av_data8_av_full, gromos_dec_av_data8_av))

gromos_dec_av_data8_av_full = np.transpose(np.delete(gromos_dec_av_data8_av_full, 0, axis=0))


interval_av=np.arange(0.,9,0.3)
fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_yscale('log')
ax.set_xlabel('Time[Hour]')
ax.set_ylabel('Pressure\n[hPa]')
ax.set_title('2018-December_av-ozone-diurnal-ts')
ax.axhline(y=0.02, color='black', linestyle='dashed')
ax.axhline(y=110, color='black', linestyle='dashed')
ax.text(0, 0.02, 'MR', rotation=45,color='yellow')
ax.text(0, 110, 'MR', rotation=45,color='yellow')
#ax.set_ylim(1,100)
cs = plt.contourf(range(0, 24), gromos.o3_p / 100, gromos_dec_av_data8_av_full,levels=interval_av,cmap='cividis')  # colors='k' is mono color line
plt.gca().invert_yaxis()  # change the order of y axis
#plt.clabel(cs, inline=True, colors='black', fontsize=10)
cb = plt.colorbar()
cb.set_label('Ozone[ppm]')
plt.show()


##################################################################
################################################################## ratio to midnight
###########################################################Jan
gromos_jan = gromos.o3_x.sel(time=slice("2018-01-01", "2018-01-31"))#slice month data
pandas_time_1 = pd.to_datetime(gromos_jan.time.data) #pick up time data
hours_1 = pandas_time_1.hour #only pick up hour(0-23)

gromos_jan_data8_av_full = np.zeros((1, 47)) #build a one row full 0 array


for x in range(0, 24):
    time_jan_data8 = np.array([])
    for i, j in enumerate(hours_1):
        if j == x:
            time_jan_data8 = np.append(np.array(time_jan_data8, dtype=int), i) #pick up indexs of certain hour for instance j=0 pick up index of 0 oclock
    gromos_jan_data8 = gromos_jan[time_jan_data8, :]                           #pick up o3_x data corresponding to index hour
    gromos_jan_data8_sum = np.sum(gromos_jan_data8, axis=0)                    #sum all the data along row
    gromos_jan_data8_av = gromos_jan_data8_sum / time_jan_data8.size           #average the certain hour of whole month data
    gromos_jan_data8_av_full = np.vstack((gromos_jan_data8_av_full, gromos_jan_data8_av))  #build a diuranl ts array

gromos_jan_data8_av_full = np.delete(gromos_jan_data8_av_full, 0, axis=0)      #delete the first row (full 0 array)
average_midnight = (gromos_jan_data8_av_full[0] + gromos_jan_data8_av_full[23]) / 2
gromos_jan_data8_av_full_rate_midnight = np.transpose(gromos_jan_data8_av_full / average_midnight)  #get a rate arrary (every row divide into midnight row)

interval=np.arange(0.93,1.07,0.01)   #range for levels in contour plot

fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_yscale('log')
ax.set_xlabel('Time[Hour]')
ax.set_ylabel('Pressure\n[hPa]')
ax.set_title('2018-January-stratospheric-ozone-diurnal-ts')
ax.set_ylim(1,100)
cs = plt.contourf(range(0, 24), gromos.o3_p / 100, gromos_jan_data8_av_full_rate_midnight,levels=(interval),cmap='coolwarm')  # colors='k' is mono color line
plt.gca().invert_yaxis()  # change the order of y axis
ax.axhline(y=0.02, color='black', linestyle='dashed')
ax.axhline(y=110, color='black', linestyle='dashed')
#ax.text(0, 0.02, 'RCIL', rotation=45)
#ax.text(0, 110, 'RCIL', rotation=45)
#plt.clabel(cs, inline=True, colors='black', fontsize=10)
cb = plt.colorbar()
cb.set_label('Ratio to ozone at midnight')
plt.show()

###########################################################Feb
gromos_Feb = gromos.o3_x.sel(time=slice("2018-02-01", "2018-02-28"))#slice month data
pandas_time_2 = pd.to_datetime(gromos_Feb.time.data) #pick up time data
hours_2 = pandas_time_2.hour #only pick up hour(0-23)

gromos_Feb_data8_av_full = np.zeros((1, 47)) #build a one row full 0 array


for x in range(0, 24):
    time_Feb_data8 = np.array([])
    for i, j in enumerate(hours_2):
        if j == x:
            time_Feb_data8 = np.append(np.array(time_Feb_data8, dtype=int), i) #pick up indexs of certain hour for instance j=0 pick up index of 0 oclock
    gromos_Feb_data8 = gromos_Feb[time_Feb_data8, :]                           #pick up o3_x data corresponding to index hour
    gromos_Feb_data8_sum = np.sum(gromos_Feb_data8, axis=0)                    #sum all the data along row
    gromos_Feb_data8_av = gromos_Feb_data8_sum / time_Feb_data8.size           #average the certain hour of whole month data
    gromos_Feb_data8_av_full = np.vstack((gromos_Feb_data8_av_full, gromos_Feb_data8_av))  #build a diuranl ts array

gromos_Feb_data8_av_full = np.delete(gromos_Feb_data8_av_full, 0, axis=0)      #delete the first row (full 0 array)
average_midnight = (gromos_Feb_data8_av_full[0] + gromos_Feb_data8_av_full[23]) / 2
gromos_Feb_data8_av_full_rate_midnight = np.transpose(gromos_Feb_data8_av_full / average_midnight)  #get a rate arrary (every row divide into midnight row)

interval=np.arange(0.93,1.07,0.01)   #range for levels in contour plot

fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_yscale('log')
ax.set_xlabel('Time[Hour]')
ax.set_ylabel('Pressure\n[hPa]')
ax.set_title('2018-February-stratospheric-ozone-diurnal-ts')
ax.set_ylim(1,100)
cs = plt.contourf(range(0, 24), gromos.o3_p / 100, gromos_Feb_data8_av_full_rate_midnight,levels=(interval),cmap='coolwarm')  # colors='k' is mono color line
plt.gca().invert_yaxis()  # change the order of y axis
ax.axhline(y=0.02, color='black', linestyle='dashed')
ax.axhline(y=110, color='black', linestyle='dashed')
#ax.text(0, 0.02, 'RCIL', rotation=45)
#ax.text(0, 110, 'RCIL', rotation=45)
#plt.clabel(cs, inline=True, colors='black', fontsize=10)
cb = plt.colorbar()
cb.set_label('Ratio to ozone at midnight')
plt.show()


###########################################################March
gromos_mar = gromos.o3_x.sel(time=slice("2018-03-01", "2018-03-31"))#slice month data
pandas_time_3 = pd.to_datetime(gromos_mar.time.data) #pick up time data
hours_3 = pandas_time_3.hour #only pick up hour(0-23)

gromos_mar_data8_av_full = np.zeros((1, 47)) #build a one row full 0 array


for x in range(0, 24):
    time_mar_data8 = np.array([])
    for i, j in enumerate(hours_3):
        if j == x:
            time_mar_data8 = np.append(np.array(time_mar_data8, dtype=int), i) #pick up indexs of certain hour for instance j=0 pick up index of 0 oclock
    gromos_mar_data8 = gromos_mar[time_mar_data8, :]                           #pick up o3_x data corresponding to index hour
    gromos_mar_data8_sum = np.sum(gromos_mar_data8, axis=0)                    #sum all the data along row
    gromos_mar_data8_av = gromos_mar_data8_sum / time_mar_data8.size           #average the certain hour of whole month data
    gromos_mar_data8_av_full = np.vstack((gromos_mar_data8_av_full, gromos_mar_data8_av))  #build a diuranl ts array

gromos_mar_data8_av_full = np.delete(gromos_mar_data8_av_full, 0, axis=0)      #delete the first row (full 0 array)
average_midnight = (gromos_mar_data8_av_full[0] + gromos_mar_data8_av_full[23]) / 2
gromos_mar_data8_av_full_rate_midnight = np.transpose(gromos_mar_data8_av_full / average_midnight)  #get a rate arrary (every row divide into midnight row)

interval=np.arange(0.93,1.07,0.01)   #range for levels in contour plot

fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_yscale('log')
ax.set_xlabel('Time[Hour]')
ax.set_ylabel('Pressure\n[hPa]')
ax.set_title('2018-March-stratospheric-ozone-diurnal-ts')
ax.set_ylim(1,100)
cs = plt.contourf(range(0, 24), gromos.o3_p / 100, gromos_mar_data8_av_full_rate_midnight,levels=(interval),cmap='coolwarm')  # colors='k' is mono color line
plt.gca().invert_yaxis()  # change the order of y axis
ax.axhline(y=0.02, color='black', linestyle='dashed')
ax.axhline(y=110, color='black', linestyle='dashed')
#ax.text(0, 0.02, 'RCIL', rotation=45)
#ax.text(0, 110, 'RCIL', rotation=45)
#plt.clabel(cs, inline=True, colors='black', fontsize=10)
cb = plt.colorbar()
cb.set_label('Ratio to ozone at midnight')
plt.show()

###########################################################April
gromos_apr = gromos.o3_x.sel(time=slice("2018-04-01", "2018-04-30"))#slice month data
pandas_time_4 = pd.to_datetime(gromos_apr.time.data) #pick up time data
hours_4 = pandas_time_4.hour #only pick up hour(0-23)

gromos_apr_data8_av_full = np.zeros((1, 47)) #build a one row full 0 array


for x in range(0, 24):
    time_apr_data8 = np.array([])
    for i, j in enumerate(hours_4):
        if j == x:
            time_apr_data8 = np.append(np.array(time_apr_data8, dtype=int), i) #pick up indexs of certain hour for instance j=0 pick up index of 0 oclock
    gromos_apr_data8 = gromos_apr[time_apr_data8, :]                           #pick up o3_x data corresponding to index hour
    gromos_apr_data8_sum = np.sum(gromos_apr_data8, axis=0)                    #sum all the data along row
    gromos_apr_data8_av = gromos_apr_data8_sum / time_apr_data8.size           #average the certain hour of whole month data
    gromos_apr_data8_av_full = np.vstack((gromos_apr_data8_av_full, gromos_apr_data8_av))  #build a diuranl ts array

gromos_apr_data8_av_full = np.delete(gromos_apr_data8_av_full, 0, axis=0)      #delete the first row (full 0 array)
average_midnight = (gromos_apr_data8_av_full[0] + gromos_apr_data8_av_full[23]) / 2
gromos_apr_data8_av_full_rate_midnight = np.transpose(gromos_apr_data8_av_full / average_midnight)  #get a rate arrary (every row divide into midnight row)

interval=np.arange(0.93,1.07,0.01)   #range for levels in contour plot

fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_yscale('log')
ax.set_xlabel('Time[Hour]')
ax.set_ylabel('Pressure\n[hPa]')
ax.set_title('2018-April-stratospheric-ozone-diurnal-ts')
ax.set_ylim(1,100)
cs = plt.contourf(range(0, 24), gromos.o3_p / 100, gromos_apr_data8_av_full_rate_midnight,levels=(interval),cmap='coolwarm')  # colors='k' is mono color line
plt.gca().invert_yaxis()  # change the order of y axis
ax.axhline(y=0.02, color='black', linestyle='dashed')
ax.axhline(y=110, color='black', linestyle='dashed')
#ax.text(0, 0.02, 'RCIL', rotation=45)
#ax.text(0, 110, 'RCIL', rotation=45)
#plt.clabel(cs, inline=True, colors='black', fontsize=10)
cb = plt.colorbar()
cb.set_label('Ratio to ozone at midnight')
plt.show()


###########################################################May
gromos_may = gromos.o3_x.sel(time=slice("2018-05-01", "2018-05-31"))#slice month data
pandas_time_5 = pd.to_datetime(gromos_may.time.data) #pick up time data
hours_5 = pandas_time_5.hour #only pick up hour(0-23)

gromos_may_data8_av_full = np.zeros((1, 47)) #build a one row full 0 array


for x in range(0, 24):
    time_may_data8 = np.array([])
    for i, j in enumerate(hours_5):
        if j == x:
            time_may_data8 = np.append(np.array(time_may_data8, dtype=int), i) #pick up indexs of certain hour for instance j=0 pick up index of 0 oclock
    gromos_may_data8 = gromos_may[time_may_data8, :]                           #pick up o3_x data corresponding to index hour
    gromos_may_data8_sum = np.sum(gromos_may_data8, axis=0)                    #sum all the data along row
    gromos_may_data8_av = gromos_may_data8_sum / time_may_data8.size           #average the certain hour of whole month data
    gromos_may_data8_av_full = np.vstack((gromos_may_data8_av_full, gromos_may_data8_av))  #build a diuranl ts array

gromos_may_data8_av_full = np.delete(gromos_may_data8_av_full, 0, axis=0)      #delete the first row (full 0 array)
average_midnight = (gromos_may_data8_av_full[0] + gromos_may_data8_av_full[23]) / 2
gromos_may_data8_av_full_rate_midnight = np.transpose(gromos_may_data8_av_full / average_midnight)  #get a rate arrary (every row divide into midnight row)

interval=np.arange(0.93,1.07,0.01)   #range for levels in contour plot

fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_yscale('log')
ax.set_xlabel('Time[Hour]')
ax.set_ylabel('Pressure\n[hPa]')
ax.set_title('2018-May-stratospheric-ozone-diurnal-ts')
ax.set_ylim(1,100)
cs = plt.contourf(range(0, 24), gromos.o3_p / 100, gromos_may_data8_av_full_rate_midnight,levels=(interval),cmap='coolwarm')  # colors='k' is mono color line
plt.gca().invert_yaxis()  # change the order of y axis
ax.axhline(y=0.02, color='black', linestyle='dashed')
ax.axhline(y=110, color='black', linestyle='dashed')
#ax.text(0, 0.02, 'RCIL', rotation=45)
#ax.text(0, 110, 'RCIL', rotation=45)
#plt.clabel(cs, inline=True, colors='black', fontsize=10)
cb = plt.colorbar()
cb.set_label('Ratio to ozone at midnight')
plt.show()


###########################################################June
gromos_june = gromos.o3_x.sel(time=slice("2018-06-01", "2018-06-30"))#slice month data
pandas_time_6 = pd.to_datetime(gromos_june.time.data) #pick up time data
hours_6 = pandas_time_6.hour #only pick up hour(0-23)

gromos_june_data8_av_full = np.zeros((1, 47)) #build a one row full 0 array

for x in range(0, 24):
    time_june_data8 = np.array([])
    for i, j in enumerate(hours_6):
        if j == x:
            time_june_data8 = np.append(np.array(time_june_data8, dtype=int), i) #pick up indexs of certain hour for instance j=0 pick up index of 0 oclock
    gromos_june_data8 = gromos_june[time_june_data8, :]                           #pick up o3_x data corresponding to index hour
    gromos_june_data8_sum = np.sum(gromos_june_data8, axis=0)                    #sum all the data along row
    gromos_june_data8_av = gromos_june_data8_sum / time_june_data8.size           #average the certain hour of whole month data
    gromos_june_data8_av_full = np.vstack((gromos_june_data8_av_full, gromos_june_data8_av))  #build a diuranl ts array

gromos_june_data8_av_full = np.delete(gromos_june_data8_av_full, 0, axis=0)      #delete the first row (full 0 array)
average_midnight = (gromos_june_data8_av_full[0] + gromos_june_data8_av_full[23]) / 2
gromos_june_data8_av_full_rate_midnight = np.transpose(gromos_june_data8_av_full / average_midnight)  #get a rate arrary (every row divide into midnight row)

interval=np.arange(0.93,1.07,0.01)   #range for levels in contour plot

fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_yscale('log')
ax.set_xlabel('Time[Hour]')
ax.set_ylabel('Pressure\n[hPa]')
ax.set_title('2018-June-stratospheric-ozone-diurnal-ts')
ax.set_ylim(1,100)
cs = plt.contourf(range(0, 24), gromos.o3_p / 100, gromos_june_data8_av_full_rate_midnight,levels=(interval),cmap='coolwarm')  # colors='k' is mono color line
plt.gca().invert_yaxis()  # change the order of y axis
ax.axhline(y=0.02, color='black', linestyle='dashed')
ax.axhline(y=110, color='black', linestyle='dashed')
#ax.text(0, 0.02, 'RCIL', rotation=45)
#ax.text(0, 110, 'RCIL', rotation=45)
#plt.clabel(cs, inline=True, colors='black', fontsize=10)
cb = plt.colorbar()
cb.set_label('Ratio to ozone at midnight')
plt.show()

###########################################################July
gromos_july = gromos.o3_x.sel(time=slice("2018-07-01", "2018-07-31"))#slice month data
pandas_time_7 = pd.to_datetime(gromos_july.time.data) #pick up time data
hours_7 = pandas_time_7.hour #only pick up hour(0-23)

gromos_july_data8_av_full = np.zeros((1, 47)) #build a one row full 0 array

for x in range(0, 24):
    time_july_data8 = np.array([])
    for i, j in enumerate(hours_7):
        if j == x:
            time_july_data8 = np.append(np.array(time_july_data8, dtype=int), i) #pick up indexs of certain hour for instance j=0 pick up index of 0 oclock
    gromos_july_data8 = gromos_july[time_july_data8, :]                           #pick up o3_x data corresponding to index hour
    gromos_july_data8_sum = np.sum(gromos_july_data8, axis=0)                    #sum all the data along row
    gromos_july_data8_av = gromos_july_data8_sum / time_july_data8.size           #average the certain hour of whole month data
    gromos_july_data8_av_full = np.vstack((gromos_july_data8_av_full, gromos_july_data8_av))  #build a diuranl ts array

gromos_july_data8_av_full = np.delete(gromos_july_data8_av_full, 0, axis=0)      #delete the first row (full 0 array)
average_midnight = (gromos_july_data8_av_full[0] + gromos_july_data8_av_full[23]) / 2
gromos_july_data8_av_full_rate_midnight = np.transpose(gromos_july_data8_av_full / average_midnight)  #get a rate arrary (every row divide into midnight row)

interval=np.arange(0.93,1.07,0.01)   #range for levels in contour plot

fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_yscale('log')
ax.set_xlabel('Time[Hour]')
ax.set_ylabel('Pressure\n[hPa]')
ax.set_title('2018-July-stratospheric-ozone-diurnal-ts')
ax.set_ylim(1,100)
cs = plt.contourf(range(0, 24), gromos.o3_p / 100, gromos_july_data8_av_full_rate_midnight,levels=(interval),cmap='coolwarm')  # colors='k' is mono color line
plt.gca().invert_yaxis()  # change the order of y axis
ax.axhline(y=0.02, color='black', linestyle='dashed')
ax.axhline(y=110, color='black', linestyle='dashed')
#ax.text(0, 0.02, 'RCIL', rotation=45)
#ax.text(0, 110, 'RCIL', rotation=45)
#plt.clabel(cs, inline=True, colors='black', fontsize=10)
cb = plt.colorbar()
cb.set_label('Ratio to ozone at midnight')
plt.show()

###########################################################August
gromos_august = gromos.o3_x.sel(time=slice("2018-08-01", "2018-08-31"))#slice month data
pandas_time_8 = pd.to_datetime(gromos_august.time.data) #pick up time data
hours_8 = pandas_time_8.hour #only pick up hour(0-23)

gromos_august_data8_av_full = np.zeros((1, 47)) #build a one row full 0 array

for x in range(0, 24):
    time_august_data8 = np.array([])
    for i, j in enumerate(hours_8):
        if j == x:
            time_august_data8 = np.append(np.array(time_august_data8, dtype=int), i) #pick up indexs of certain hour for instance j=0 pick up index of 0 oclock
    gromos_august_data8 = gromos_august[time_august_data8, :]                           #pick up o3_x data corresponding to index hour
    gromos_august_data8_sum = np.sum(gromos_august_data8, axis=0)                    #sum all the data along row
    gromos_august_data8_av = gromos_august_data8_sum / time_august_data8.size           #average the certain hour of whole month data
    gromos_august_data8_av_full = np.vstack((gromos_august_data8_av_full, gromos_august_data8_av))  #build a diuranl ts array

gromos_august_data8_av_full = np.delete(gromos_august_data8_av_full, 0, axis=0)      #delete the first row (full 0 array)
average_midnight = (gromos_august_data8_av_full[0] + gromos_august_data8_av_full[23]) / 2
gromos_august_data8_av_full_rate_midnight = np.transpose(gromos_august_data8_av_full / average_midnight)  #get a rate arrary (every row divide into midnight row)

interval=np.arange(0.93,1.07,0.01)   #range for levels in contour plot

fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_yscale('log')
ax.set_xlabel('Time[Hour]')
ax.set_ylabel('Pressure\n[hPa]')
ax.set_title('2018-August-stratospheric-ozone-diurnal-ts')
ax.set_ylim(1,100)
cs = plt.contourf(range(0, 24), gromos.o3_p / 100, gromos_august_data8_av_full_rate_midnight,levels=(interval),cmap='coolwarm')  # colors='k' is mono color line
plt.gca().invert_yaxis()  # change the order of y axis
ax.axhline(y=0.02, color='black', linestyle='dashed')
ax.axhline(y=110, color='black', linestyle='dashed')
#ax.text(0, 0.02, 'RCIL', rotation=45)
#ax.text(0, 110, 'RCIL', rotation=45)
#plt.clabel(cs, inline=True, colors='black', fontsize=10)
cb = plt.colorbar()
cb.set_label('Ratio to ozone at midnight')
plt.show()

###########################################################Sep
gromos__sep = gromos.o3_x.sel(time=slice("2018-09-01", "2018-09-30"))#slice month data
pandas_time_9 = pd.to_datetime(gromos__sep.time.data) #pick up time data
hours_9 = pandas_time_9.hour #only pick up hour(0-23)

gromos__sep_data8_av_full = np.zeros((1, 47)) #build a one row full 0 array

for x in range(0, 24):
    time__sep_data8 = np.array([])
    for i, j in enumerate(hours_9):
        if j == x:
            time__sep_data8 = np.append(np.array(time__sep_data8, dtype=int), i) #pick up indexs of certain hour for instance j=0 pick up index of 0 oclock
    gromos__sep_data8 = gromos__sep[time__sep_data8, :]                           #pick up o3_x data corresponding to index hour
    gromos__sep_data8_sum = np.sum(gromos__sep_data8, axis=0)                    #sum all the data along row
    gromos__sep_data8_av = gromos__sep_data8_sum / time__sep_data8.size           #average the certain hour of whole month data
    gromos__sep_data8_av_full = np.vstack((gromos__sep_data8_av_full, gromos__sep_data8_av))  #build a diuranl ts array

gromos__sep_data8_av_full = np.delete(gromos__sep_data8_av_full, 0, axis=0)      #delete the first row (full 0 array)
average_midnight = (gromos__sep_data8_av_full[0] + gromos__sep_data8_av_full[23]) / 2
gromos__sep_data8_av_full_rate_midnight = np.transpose(gromos__sep_data8_av_full / average_midnight)  #get a rate arrary (every row divide into midnight row)

interval=np.arange(0.93,1.07,0.01)   #range for levels in contour plot

fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_yscale('log')
ax.set_xlabel('Time[Hour]')
ax.set_ylabel('Pressure\n[hPa]')
ax.set_title('2018-September-stratospheric-ozone-diurnal-ts')
ax.set_ylim(1,100)
cs = plt.contourf(range(0, 24), gromos.o3_p / 100, gromos__sep_data8_av_full_rate_midnight,levels=(interval),cmap='coolwarm')  # colors='k' is mono color line
plt.gca().invert_yaxis()  # change the order of y axis
ax.axhline(y=0.02, color='black', linestyle='dashed')
ax.axhline(y=110, color='black', linestyle='dashed')
#ax.text(0, 0.02, 'RCIL', rotation=45)
#ax.text(0, 110, 'RCIL', rotation=45)
#plt.clabel(cs, inline=True, colors='black', fontsize=10)
cb = plt.colorbar()
cb.set_label('Ratio to ozone at midnight')
plt.show()


###########################################################Oct
gromos_oct = gromos.o3_x.sel(time=slice("2018-10-01", "2018-10-31"))#slice month data
pandas_time_10 = pd.to_datetime(gromos_oct.time.data) #pick up time data
hours_10 = pandas_time_10.hour #only pick up hour(0-23)

gromos_oct_data8_av_full = np.zeros((1, 47)) #build a one row full 0 array

for x in range(0, 24):
    time_oct_data8 = np.array([])
    for i, j in enumerate(hours_10):
        if j == x:
            time_oct_data8 = np.append(np.array(time_oct_data8, dtype=int), i) #pick up indexs of certain hour for instance j=0 pick up index of 0 oclock
    gromos_oct_data8 = gromos_oct[time_oct_data8, :]                           #pick up o3_x data corresponding to index hour
    gromos_oct_data8_sum = np.sum(gromos_oct_data8, axis=0)                    #sum all the data along row
    gromos_oct_data8_av = gromos_oct_data8_sum / time_oct_data8.size           #average the certain hour of whole month data
    gromos_oct_data8_av_full = np.vstack((gromos_oct_data8_av_full, gromos_oct_data8_av))  #build a diuranl ts array

gromos_oct_data8_av_full = np.delete(gromos_oct_data8_av_full, 0, axis=0)      #delete the first row (full 0 array)
average_midnight = (gromos_oct_data8_av_full[0] + gromos_oct_data8_av_full[23]) / 2
gromos_oct_data8_av_full_rate_midnight = np.transpose(gromos_oct_data8_av_full / average_midnight)  #get a rate arrary (every row divide into midnight row)

interval=np.arange(0.93,1.07,0.01)   #range for levels in contour plot

fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_yscale('log')
ax.set_xlabel('Time[Hour]')
ax.set_ylabel('Pressure\n[hPa]')
ax.set_title('2018-oct-stratospheric-ozone-diurnal-ts')
ax.set_ylim(1,100)
cs = plt.contourf(range(0, 24), gromos.o3_p / 100, gromos_oct_data8_av_full_rate_midnight,levels=(interval),cmap='coolwarm')  # colors='k' is mono color line
plt.gca().invert_yaxis()  # change the order of y axis
ax.axhline(y=0.02, color='black', linestyle='dashed')
ax.axhline(y=110, color='black', linestyle='dashed')
#ax.text(0, 0.02, 'RCIL', rotation=45)
#ax.text(0, 110, 'RCIL', rotation=45)
#plt.clabel(cs, inline=True, colors='black', fontsize=10)
cb = plt.colorbar()
cb.set_label('Ratio to ozone at midnight')
plt.show()


###########################################################Nov
gromos_nov = gromos.o3_x.sel(time=slice("2018-11-01", "2018-11-30"))#slice month data
pandas_time_11 = pd.to_datetime(gromos_nov.time.data) #pick up time data
hours_11 = pandas_time_11.hour #only pick up hour(0-23)

gromos_nov_data8_av_full = np.zeros((1, 47)) #build a one row full 0 array

for x in range(0, 24):
    time_nov_data8 = np.array([])
    for i, j in enumerate(hours_11):
        if j == x:
            time_nov_data8 = np.append(np.array(time_nov_data8, dtype=int), i) #pick up indexs of certain hour for instance j=0 pick up index of 0 oclock
    gromos_nov_data8 = gromos_nov[time_nov_data8, :]                           #pick up o3_x data corresponding to index hour
    gromos_nov_data8_sum = np.sum(gromos_nov_data8, axis=0)                    #sum all the data along row
    gromos_nov_data8_av = gromos_nov_data8_sum / time_nov_data8.size           #average the certain hour of whole month data
    gromos_nov_data8_av_full = np.vstack((gromos_nov_data8_av_full, gromos_nov_data8_av))  #build a diuranl ts array

gromos_nov_data8_av_full = np.delete(gromos_nov_data8_av_full, 0, axis=0)      #delete the first row (full 0 array)
average_midnight = (gromos_nov_data8_av_full[0] + gromos_nov_data8_av_full[23]) / 2
gromos_nov_data8_av_full_rate_midnight = np.transpose(gromos_nov_data8_av_full / average_midnight)  #get a rate arrary (every row divide into midnight row)

interval=np.arange(0.93,1.07,0.01)   #range for levels in contour plot

fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_yscale('log')
ax.set_xlabel('Time[Hour]')
ax.set_ylabel('Pressure\n[hPa]')
ax.set_title('2018-November-stratospheric-ozone-diurnal-ts')
ax.set_ylim(1,100)
cs = plt.contourf(range(0, 24), gromos.o3_p / 100, gromos_nov_data8_av_full_rate_midnight,levels=(interval),cmap='coolwarm')  # colors='k' is mono color line
plt.gca().invert_yaxis()  # change the order of y axis
ax.axhline(y=0.02, color='black', linestyle='dashed')
ax.axhline(y=110, color='black', linestyle='dashed')
#ax.text(0, 0.02, 'RCIL', rotation=45)
#ax.text(0, 110, 'RCIL', rotation=45)
#plt.clabel(cs, inline=True, colors='black', fontsize=10)
cb = plt.colorbar()
cb.set_label('Ratio to ozone at midnight')
plt.show()


###########################################################Dec
gromos_dec = gromos.o3_x.sel(time=slice("2018-12-01", "2018-12-31"))#slice month data
pandas_time_12 = pd.to_datetime(gromos_dec.time.data) #pick up time data
hours_12 = pandas_time_12.hour #only pick up hour(0-23)

gromos_dec_data8_av_full = np.zeros((1, 47)) #build a one row full 0 array

for x in range(0, 24):
    time_dec_data8 = np.array([])
    for i, j in enumerate(hours_12):
        if j == x:
            time_dec_data8 = np.append(np.array(time_dec_data8, dtype=int), i) #pick up indexs of certain hour for instance j=0 pick up index of 0 oclock
    gromos_dec_data8 = gromos_dec[time_dec_data8, :]                           #pick up o3_x data corresponding to index hour
    gromos_dec_data8_sum = np.sum(gromos_dec_data8, axis=0)                    #sum all the data along row
    gromos_dec_data8_av = gromos_dec_data8_sum / time_dec_data8.size           #average the certain hour of whole month data
    gromos_dec_data8_av_full = np.vstack((gromos_dec_data8_av_full, gromos_dec_data8_av))  #build a diuranl ts array

gromos_dec_data8_av_full = np.delete(gromos_dec_data8_av_full, 0, axis=0)      #delete the first row (full 0 array)
average_midnight = (gromos_dec_data8_av_full[0] + gromos_dec_data8_av_full[23]) / 2
gromos_dec_data8_av_full_rate_midnight = np.transpose(gromos_dec_data8_av_full / average_midnight)  #get a rate arrary (every row divide into midnight row)

interval=np.arange(0.93,1.07,0.01)   #range for levels in contour plot

fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_yscale('log')
ax.set_xlabel('Time[Hour]')
ax.set_ylabel('Pressure\n[hPa]')
ax.set_title('2018-December-stratospheric-ozone-diurnal-ts')
ax.set_ylim(1,100)
cs = plt.contourf(range(0, 24), gromos.o3_p / 100, gromos_dec_data8_av_full_rate_midnight,levels=(interval),cmap='coolwarm')  # colors='k' is mono color line
plt.gca().invert_yaxis()  # change the order of y axis
ax.axhline(y=0.02, color='black', linestyle='dashed')
ax.axhline(y=110, color='black', linestyle='dashed')
#ax.text(0, 0.02, 'RCIL', rotation=45)
#ax.text(0, 110, 'RCIL', rotation=45)
#plt.clabel(cs, inline=True, colors='black', fontsize=10)
cb = plt.colorbar()
cb.set_label('Ratio to ozone at midnight')
plt.show()