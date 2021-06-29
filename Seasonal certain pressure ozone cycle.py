import datetime
import os

import matplotlib.pyplot as plt
import netCDF4
import numpy as np
import pandas as pd
from netCDF4 import Dataset
import xarray as xr

filename = 'F:\PyCharm Community Edition 2019.3\practice2021 april/GROMOS_2018_waccm_continuum_ozone.nc'

gromos = xr.open_dataset(
    filename,
    decode_times=True,
    decode_coords=True,
    # use_cftime=True,
)

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


###########################################################Sep
gromos_sep = gromos.o3_x.sel(time=slice("2018-09-01", "2018-09-30"))#slice month data
pandas_time_9 = pd.to_datetime(gromos_sep.time.data) #pick up time data
hours_9 = pandas_time_9.hour #only pick up hour(0-23)

gromos_sep_data8_av_full = np.zeros((1, 47)) #build a one row full 0 array

for x in range(0, 24):
    time_sep_data8 = np.array([])
    for i, j in enumerate(hours_9):
        if j == x:
            time_sep_data8 = np.append(np.array(time_sep_data8, dtype=int), i) #pick up indexs of certain hour for instance j=0 pick up index of 0 oclock
    gromos_sep_data8 = gromos_sep[time_sep_data8, :]                           #pick up o3_x data corresponding to index hour
    gromos_sep_data8_sum = np.sum(gromos_sep_data8, axis=0)                    #sum all the data along row
    gromos_sep_data8_av = gromos_sep_data8_sum / time_sep_data8.size           #average the certain hour of whole month data
    gromos_sep_data8_av_full = np.vstack((gromos_sep_data8_av_full, gromos_sep_data8_av))  #build a diuranl ts array

gromos_sep_data8_av_full = np.delete(gromos_sep_data8_av_full, 0, axis=0)      #delete the first row (full 0 array)
average_midnight = (gromos_sep_data8_av_full[0] + gromos_sep_data8_av_full[23]) / 2
gromos_sep_data8_av_full_rate_midnight = np.transpose(gromos_sep_data8_av_full / average_midnight)  #get a rate arrary (every row divide into midnight row)


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


######################################################## in 0.5hPa
for x_1,y_1 in enumerate(gromos.o3_p/100):
    if 0.40 < y_1 < 0.60:
        gromos_p1 = x_1 #x_1 represents march and pressure 0,5hpa

gromos_mar_data8_av_full_rate_midnight_p1 = gromos_mar_data8_av_full_rate_midnight[gromos_p1,:] # the data of march and 0.5hpa
gromos_june_data8_av_full_rate_midnight_p1 = gromos_june_data8_av_full_rate_midnight[gromos_p1,:] # the data of march and 0.5hpa
gromos_sep_data8_av_full_rate_midnight_p1 = gromos_sep_data8_av_full_rate_midnight[gromos_p1,:] # the data of march and 0.5hpa
gromos_dec_data8_av_full_rate_midnight_p1 = gromos_dec_data8_av_full_rate_midnight[gromos_p1,:] # the data of march and 0.5hpa
daytime=np.arange(0,24)
fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_ylim(0.75,1.10)
ax.set_xlim(0,23)
ax.set_xlabel('Time[Hour]')
ax.set_ylabel('Normalized ozone')
ax.set_title('Gromos diurnal cycle 0.5[hPa]')
ax.plot(daytime,gromos_mar_data8_av_full_rate_midnight_p1,color='green',label='March')
ax.plot(daytime,gromos_june_data8_av_full_rate_midnight_p1,color='yellow',label='June')
ax.plot(daytime,gromos_sep_data8_av_full_rate_midnight_p1,color='red',label='September')
ax.plot(daytime,gromos_dec_data8_av_full_rate_midnight_p1,color='blue',label='December')
plt.legend()
plt.show()

################################################################### 1hpa
for x_2,y_2 in enumerate(gromos.o3_p/100):
    if 0.80 < y_2 < 1.20:
        gromos_p2 = x_2 #x_2 represents march and pressure 1hpa

#a=gromos.o3_p/100
gromos_mar_data8_av_full_rate_midnight_p2 = gromos_mar_data8_av_full_rate_midnight[gromos_p2,:] # the data of march and 1hpa
gromos_june_data8_av_full_rate_midnight_p2 = gromos_june_data8_av_full_rate_midnight[gromos_p2,:] # the data of march and 1hpa
gromos_sep_data8_av_full_rate_midnight_p2 = gromos_sep_data8_av_full_rate_midnight[gromos_p2,:] # the data of march and 1hpa
gromos_dec_data8_av_full_rate_midnight_p2 = gromos_dec_data8_av_full_rate_midnight[gromos_p2,:] # the data of march and 1hpa
daytime=np.arange(0,24)
fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_ylim(0.90,1.05)
ax.set_xlim(0,23)
ax.set_xlabel('Time[Hour]')
ax.set_ylabel('Normalized ozone')
ax.set_title('Gromos diurnal cycle 1[hPa]')
ax.plot(daytime,gromos_mar_data8_av_full_rate_midnight_p2,color='green',label='March')
ax.plot(daytime,gromos_june_data8_av_full_rate_midnight_p2,color='yellow',label='June')
ax.plot(daytime,gromos_sep_data8_av_full_rate_midnight_p2,color='red',label='September')
ax.plot(daytime,gromos_dec_data8_av_full_rate_midnight_p2,color='blue',label='December')
plt.legend()
plt.show()


######################################################## in 3.0hPa
for x_3,y_3 in enumerate(gromos.o3_p/100):
    if 2.70 < y_3 < 3.30:
        gromos_p3 = x_3 #x_1 represents march and pressure 0,5hpa

gromos_mar_data8_av_full_rate_midnight_p3 = gromos_mar_data8_av_full_rate_midnight[gromos_p3,:] # the data of march and 0.5hpa
gromos_june_data8_av_full_rate_midnight_p3 = gromos_june_data8_av_full_rate_midnight[gromos_p3,:] # the data of march and 0.5hpa
gromos_sep_data8_av_full_rate_midnight_p3 = gromos_sep_data8_av_full_rate_midnight[gromos_p3,:] # the data of march and 0.5hpa
gromos_dec_data8_av_full_rate_midnight_p3 = gromos_dec_data8_av_full_rate_midnight[gromos_p3,:] # the data of march and 0.5hpa
daytime=np.arange(0,24)
fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_ylim(0.96,1.07)
ax.set_xlim(0,23)
ax.set_xlabel('Time[Hour]')
ax.set_ylabel('Normalized ozone')
ax.set_title('Gromos diurnal cycle 3[hPa]')
ax.plot(daytime,gromos_mar_data8_av_full_rate_midnight_p3,color='green',label='March')
ax.plot(daytime,gromos_june_data8_av_full_rate_midnight_p3,color='yellow',label='June')
ax.plot(daytime,gromos_sep_data8_av_full_rate_midnight_p3,color='red',label='September')
ax.plot(daytime,gromos_dec_data8_av_full_rate_midnight_p3,color='blue',label='December')
plt.legend()
plt.show()

######################################################## in 5.0hPa
for x_4,y_4 in enumerate(gromos.o3_p/100):
    if 4.70 < y_4 < 5.30:
        gromos_p4 = x_4 #x_1 represents march and pressure 0,5hpa

gromos_mar_data8_av_full_rate_midnight_p4 = gromos_mar_data8_av_full_rate_midnight[gromos_p4,:] # the data of march and 0.5hpa
gromos_june_data8_av_full_rate_midnight_p4 = gromos_june_data8_av_full_rate_midnight[gromos_p4,:] # the data of march and 0.5hpa
gromos_sep_data8_av_full_rate_midnight_p4 = gromos_sep_data8_av_full_rate_midnight[gromos_p4,:] # the data of march and 0.5hpa
gromos_dec_data8_av_full_rate_midnight_p4 = gromos_dec_data8_av_full_rate_midnight[gromos_p4,:] # the data of march and 0.5hpa
daytime=np.arange(0,24)
fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_ylim(0.95,1.05)
ax.set_xlim(0,23)
ax.set_xlabel('Time[Hour]')
ax.set_ylabel('Normalized ozone')
ax.set_title('Gromos diurnal cycle 5[hPa]')
ax.plot(daytime,gromos_mar_data8_av_full_rate_midnight_p4,color='green',label='March')
ax.plot(daytime,gromos_june_data8_av_full_rate_midnight_p4,color='yellow',label='June')
ax.plot(daytime,gromos_sep_data8_av_full_rate_midnight_p4,color='red',label='September')
ax.plot(daytime,gromos_dec_data8_av_full_rate_midnight_p4,color='blue',label='December')
plt.legend()
plt.show()

########################################
######################################## GDOC part
nc=Dataset('GDOC_ver1.nc')
month=nc.variables['month'][:].data
hour=nc.variables['hour'][:].data
latitude=nc.variables['lat'][:].data
zstar=nc.variables['zstar_pr'][:].data
GDOC=nc.variables['GDOC'][:].data

################################################### 0.5hPa
part_1_1=np.array(GDOC[:,2,36,28])
part_1_2=np.array(GDOC[:,5,36,28])
part_1_3=np.array(GDOC[:,8,36,28])
part_1_4=np.array(GDOC[:,11,36,28])
daytime_GDOC=np.arange(0,24,0.5)

fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_xlabel('Time[Hour]')
ax.set_ylabel('Normalized ozone')
ax.set_title('GDOC diurnal cycle 0.5[hPa]')
ax.set_ylim(0.75,1.10)
ax.plot(daytime_GDOC,part_1_1,color='green',label='March')
ax.plot(daytime_GDOC,part_1_2,color='yellow',label='June')
ax.plot(daytime_GDOC,part_1_3,color='red',label='September')
ax.plot(daytime_GDOC,part_1_4,color='blue',label='December')
plt.legend()
plt.show()

################################################### 1hPa
part_2_1=np.array(GDOC[:,2,31,28])
part_2_2=np.array(GDOC[:,5,31,28])
part_2_3=np.array(GDOC[:,8,31,28])
part_2_4=np.array(GDOC[:,11,31,28])

fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_xlabel('Time[Hour]')
ax.set_ylabel('Normalized ozone')
ax.set_title('GDOC diurnal cycle 1[hPa]')
ax.set_ylim(0.90,1.05)
ax.plot(daytime_GDOC,part_2_1,color='green',label='March')
ax.plot(daytime_GDOC,part_2_2,color='yellow',label='June')
ax.plot(daytime_GDOC,part_2_3,color='red',label='September')
ax.plot(daytime_GDOC,part_2_4,color='blue',label='December')
plt.legend()
plt.show()


################################################### 3hPa
part_3_1=np.array(GDOC[:,2,23,28])
part_3_2=np.array(GDOC[:,5,23,28])
part_3_3=np.array(GDOC[:,8,23,28])
part_3_4=np.array(GDOC[:,11,23,28])

fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_xlabel('Time[Hour]')
ax.set_ylabel('Normalized ozone')
ax.set_title('GDOC diurnal cycle 3[hPa]')
ax.set_ylim(0.96,1.07)
ax.plot(daytime_GDOC,part_3_1,color='green',label='March')
ax.plot(daytime_GDOC,part_3_2,color='yellow',label='June')
ax.plot(daytime_GDOC,part_3_3,color='red',label='September')
ax.plot(daytime_GDOC,part_3_4,color='blue',label='December')
plt.legend()
plt.show()



################################################### 5hPa
part_4_1=np.array(GDOC[:,2,20,28])
part_4_2=np.array(GDOC[:,5,20,28])
part_4_3=np.array(GDOC[:,8,20,28])
part_4_4=np.array(GDOC[:,11,20,28])

fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_xlabel('Time[Hour]')
ax.set_ylabel('Normalized ozone')
ax.set_title('GDOC diurnal cycle 5[hPa]')
ax.set_ylim(0.95,1.05)
ax.plot(daytime_GDOC,part_4_1,color='green',label='March')
ax.plot(daytime_GDOC,part_4_2,color='yellow',label='June')
ax.plot(daytime_GDOC,part_4_3,color='red',label='September')
ax.plot(daytime_GDOC,part_4_4,color='blue',label='December')
plt.legend()
plt.show()



###########################################################
########################################################### comparison between gromos and GDOC
######################################################## in 0.5hPa
fig = plt.figure(figsize=(13,6))
ax_1 = fig.add_subplot(121)
ax_1.set_ylim(0.75,1.10)
ax_1.set_xlim(0,23)
ax_1.set_xlabel('Time[Hour]')
ax_1.set_ylabel('Normalized ozone')
ax_1.set_title('Gromos diurnal cycle 0.5[hPa]')
ax_1.plot(daytime,gromos_mar_data8_av_full_rate_midnight_p1,color='green',label='March')
ax_1.plot(daytime,gromos_june_data8_av_full_rate_midnight_p1,color='yellow',label='June')
ax_1.plot(daytime,gromos_sep_data8_av_full_rate_midnight_p1,color='red',label='September')
ax_1.plot(daytime,gromos_dec_data8_av_full_rate_midnight_p1,color='blue',label='December')
plt.legend(loc='best')
ax_2 = fig.add_subplot(122)
ax_2.set_xlabel('Time[Hour]')
ax_2.set_ylabel('Normalized ozone')
ax_2.set_title('GDOC diurnal cycle 0.5[hPa]')
ax_2.set_ylim(0.75,1.10)
ax_2.plot(daytime_GDOC,part_1_1,color='green',label='March')
ax_2.plot(daytime_GDOC,part_1_2,color='yellow',label='June')
ax_2.plot(daytime_GDOC,part_1_3,color='red',label='September')
ax_2.plot(daytime_GDOC,part_1_4,color='blue',label='December')
plt.legend(loc='best')
plt.show()


######################################################## in 1hPa
fig = plt.figure(figsize=(13,6))
ax_1 = fig.add_subplot(121)
ax_1.set_ylim(0.90,1.05)
ax_1.set_xlim(0,23)
ax_1.set_xlabel('Time[Hour]')
ax_1.set_ylabel('Normalized ozone')
ax_1.set_title('Gromos diurnal cycle 1[hPa]')
ax_1.plot(daytime,gromos_mar_data8_av_full_rate_midnight_p2,color='green',label='March')
ax_1.plot(daytime,gromos_june_data8_av_full_rate_midnight_p2,color='yellow',label='June')
ax_1.plot(daytime,gromos_sep_data8_av_full_rate_midnight_p2,color='red',label='September')
ax_1.plot(daytime,gromos_dec_data8_av_full_rate_midnight_p2,color='blue',label='December')
plt.legend()
ax_2 = fig.add_subplot(122)
ax_2.set_xlabel('Time[Hour]')
ax_2.set_ylabel('Normalized ozone')
ax_2.set_title('GDOC diurnal cycle 1[hPa]')
ax_2.set_ylim(0.90,1.05)
ax_2.plot(daytime_GDOC,part_2_1,color='green',label='March')
ax_2.plot(daytime_GDOC,part_2_2,color='yellow',label='June')
ax_2.plot(daytime_GDOC,part_2_3,color='red',label='September')
ax_2.plot(daytime_GDOC,part_2_4,color='blue',label='December')
plt.legend()
plt.show()


######################################################## in 3hPa
fig = plt.figure(figsize=(13,6))
ax_1 = fig.add_subplot(121)
ax_1.set_ylim(0.96,1.07)
ax_1.set_xlim(0,23)
ax_1.set_xlabel('Time[Hour]')
ax_1.set_ylabel('Normalized ozone')
ax_1.set_title('Gromos diurnal cycle 3[hPa]')
ax_1.plot(daytime,gromos_mar_data8_av_full_rate_midnight_p3,color='green',label='March')
ax_1.plot(daytime,gromos_june_data8_av_full_rate_midnight_p3,color='yellow',label='June')
ax_1.plot(daytime,gromos_sep_data8_av_full_rate_midnight_p3,color='red',label='September')
ax_1.plot(daytime,gromos_dec_data8_av_full_rate_midnight_p3,color='blue',label='December')
plt.legend()
ax_2 = fig.add_subplot(122)
ax_2.set_xlabel('Time[Hour]')
ax_2.set_ylabel('Normalized ozone')
ax_2.set_title('GDOC diurnal cycle 3[hPa]')
ax_2.set_ylim(0.96,1.07)
ax_2.plot(daytime_GDOC,part_3_1,color='green',label='March')
ax_2.plot(daytime_GDOC,part_3_2,color='yellow',label='June')
ax_2.plot(daytime_GDOC,part_3_3,color='red',label='September')
ax_2.plot(daytime_GDOC,part_3_4,color='blue',label='December')
plt.legend()
plt.show()

######################################################## in 5hPa
fig = plt.figure(figsize=(13,6))
ax_1 = fig.add_subplot(121)
ax_1.set_ylim(0.95,1.05)
ax_1.set_xlim(0,23)
ax_1.set_xlabel('Time[Hour]')
ax_1.set_ylabel('Normalized ozone')
ax_1.set_title('Gromos diurnal cycle 5[hPa]')
ax_1.plot(daytime,gromos_mar_data8_av_full_rate_midnight_p4,color='green',label='March')
ax_1.plot(daytime,gromos_june_data8_av_full_rate_midnight_p4,color='yellow',label='June')
ax_1.plot(daytime,gromos_sep_data8_av_full_rate_midnight_p4,color='red',label='September')
ax_1.plot(daytime,gromos_dec_data8_av_full_rate_midnight_p4,color='blue',label='December')
plt.legend()
ax_2 = fig.add_subplot(122)
ax_2.set_xlabel('Time[Hour]')
ax_2.set_ylabel('Normalized ozone')
ax_2.set_title('GDOC diurnal cycle 5[hPa]')
ax_2.set_ylim(0.95,1.05)
ax_2.plot(daytime_GDOC,part_4_1,color='green',label='March')
ax_2.plot(daytime_GDOC,part_4_2,color='yellow',label='June')
ax_2.plot(daytime_GDOC,part_4_3,color='red',label='September')
ax_2.plot(daytime_GDOC,part_4_4,color='blue',label='December')
plt.legend()
plt.show()