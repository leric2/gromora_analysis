import datetime
import os
import matplotlib.pyplot as plt
import netCDF4
from netCDF4 import Dataset
import numpy as np
import pandas as pd

import xarray as xr
#############################input gromos data
filename = 'F:\PyCharm Community Edition 2019.3\practice2021 april/GROMOS_2018_waccm_continuum_ozone.nc'

gromos = xr.open_dataset(
    filename,
    decode_times=True,
    decode_coords=True,
    # use_cftime=True,
)
#############################input GDOC data
nc=Dataset('GDOC_ver1.nc')
month=nc.variables['month'][:].data
hour=nc.variables['hour'][:].data
latitude=nc.variables['lat'][:].data
zstar=nc.variables['zstar_pr'][:].data
GDOC=nc.variables['GDOC'][:].data


###########################################################Gromos_January
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

fig = plt.figure(figsize=(13,6))
ax_1 = fig.add_subplot(121)
ax_1.set_yscale('log')
ax_1.set_xlabel('Time[Hour]')
ax_1.set_ylabel('Pressure\n[hPa]')
ax_1.set_title('2018-January-stratospheric-ozone-diurnal-ts')
ax_1.set_ylim(1,100)
cs = plt.contourf(range(0, 24), gromos.o3_p / 100, gromos_jan_data8_av_full_rate_midnight,levels=(interval),cmap='coolwarm')  # colors='k' is mono color line
plt.gca().invert_yaxis()  # change the order of y ax_1is
ax_1.axhline(y=0.02, color='black', linestyle='dashed')
ax_1.axhline(y=110, color='black', linestyle='dashed')
#ax_1.text(0, 0.02, 'RCIL', rotation=45)
#ax_1.text(0, 110, 'RCIL', rotation=45)
#plt.clabel(cs, inline=True, colors='black', fontsize=10)
cb = plt.colorbar()
cb.set_label('Ratio to ozone at midnight')


#######################################GDOC_janil
GDOC_diurnal_s1 =np.transpose(GDOC[:,0,:,28]) #np.tanspose exchange x,y

interval=np.arange(0.93,1.07,0.01)
ax_2 = fig.add_subplot(122)
ax_2.set_yscale('log')
ax_2.set_xlabel('Time[Hour]')
ax_2.set_ylabel('Pressure\n[hPa]')
ax_2.set_title(r'$2018-January \quad 47.5^\circ N$')
ax_2.set_ylim(1,100)
cs=plt.contourf(hour,zstar,GDOC_diurnal_s1,levels=(interval),cmap='coolwarm')
plt.gca().invert_yaxis()
#plt.clabel(cs,inline=True, colors='black',fontsize=10)
cb = plt.colorbar()
cb.set_label('Ratio to ozone at midnight')
plt.show()



###########################################################Gromos_April
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

fig = plt.figure(figsize=(13,6))
ax_1 = fig.add_subplot(121)
ax_1.set_yscale('log')
ax_1.set_xlabel('Time[Hour]')
ax_1.set_ylabel('Pressure\n[hPa]')
ax_1.set_title('2018-April-stratospheric-ozone-diurnal-ts')
ax_1.set_ylim(1,100)
cs = plt.contourf(range(0, 24), gromos.o3_p / 100, gromos_apr_data8_av_full_rate_midnight,levels=(interval),cmap='coolwarm')  # colors='k' is mono color line
plt.gca().invert_yaxis()  # change the order of y ax_1is
ax_1.axhline(y=0.02, color='black', linestyle='dashed')
ax_1.axhline(y=110, color='black', linestyle='dashed')
#ax_1.text(0, 0.02, 'RCIL', rotation=45)
#ax_1.text(0, 110, 'RCIL', rotation=45)
#plt.clabel(cs, inline=True, colors='black', fontsize=10)
cb = plt.colorbar()
cb.set_label('Ratio to ozone at midnight')


#######################################GDOC_April
GDOC_diurnal_s4 =np.transpose(GDOC[:,3,:,28]) #np.tanspose exchange x,y

interval=np.arange(0.93,1.07,0.01)
ax_2 = fig.add_subplot(122)
ax_2.set_yscale('log')
ax_2.set_xlabel('Time[Hour]')
ax_2.set_ylabel('Pressure\n[hPa]')
ax_2.set_title(r'$2018-April \quad 47.5^\circ N$')
ax_2.set_ylim(1,100)
cs=plt.contourf(hour,zstar,GDOC_diurnal_s4,levels=(interval),cmap='coolwarm')
plt.gca().invert_yaxis()
#plt.clabel(cs,inline=True, colors='black',fontsize=10)
cb = plt.colorbar()
cb.set_label('Ratio to ozone at midnight')
plt.show()



###########################################################Gromos_June
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

fig = plt.figure(figsize=(13,6))
ax_1 = fig.add_subplot(121)
ax_1.set_yscale('log')
ax_1.set_xlabel('Time[Hour]')
ax_1.set_ylabel('Pressure\n[hPa]')
ax_1.set_title('2018-June-stratospheric-ozone-diurnal-ts')
ax_1.set_ylim(1,100)
cs = plt.contourf(range(0, 24), gromos.o3_p / 100, gromos_june_data8_av_full_rate_midnight,levels=(interval),cmap='coolwarm')  # colors='k' is mono color line
plt.gca().invert_yaxis()  # change the order of y ax_1is
ax_1.axhline(y=0.02, color='black', linestyle='dashed')
ax_1.axhline(y=110, color='black', linestyle='dashed')
#ax_1.text(0, 0.02, 'RCIL', rotation=45)
#ax_1.text(0, 110, 'RCIL', rotation=45)
#plt.clabel(cs, inline=True, colors='black', fontsize=10)
cb = plt.colorbar()
cb.set_label('Ratio to ozone at midnight')


#######################################GDOC_june
GDOC_diurnal_s6 =np.transpose(GDOC[:,5,:,28]) #np.tanspose exchange x,y

interval=np.arange(0.93,1.07,0.01)
ax_2 = fig.add_subplot(122)
ax_2.set_yscale('log')
ax_2.set_xlabel('Time[Hour]')
ax_2.set_ylabel('Pressure\n[hPa]')
ax_2.set_title(r'$2018-June \quad 47.5^\circ N$')
ax_2.set_ylim(1,100)
cs=plt.contourf(hour,zstar,GDOC_diurnal_s6,levels=(interval),cmap='coolwarm')
plt.gca().invert_yaxis()
#plt.clabel(cs,inline=True, colors='black',fontsize=10)
cb = plt.colorbar()
cb.set_label('Ratio to ozone at midnight')
plt.show()


###########################################################Gromos_July
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

fig = plt.figure(figsize=(13,6))
ax_1 = fig.add_subplot(121)
ax_1.set_yscale('log')
ax_1.set_xlabel('Time[Hour]')
ax_1.set_ylabel('Pressure\n[hPa]')
ax_1.set_title('2018-July-stratospheric-ozone-diurnal-ts')
ax_1.set_ylim(1,100)
cs = plt.contourf(range(0, 24), gromos.o3_p / 100, gromos_july_data8_av_full_rate_midnight,levels=(interval),cmap='coolwarm')  # colors='k' is mono color line
plt.gca().invert_yaxis()  # change the order of y ax_1is
ax_1.axhline(y=0.02, color='black', linestyle='dashed')
ax_1.axhline(y=110, color='black', linestyle='dashed')
#ax_1.text(0, 0.02, 'RCIL', rotation=45)
#ax_1.text(0, 110, 'RCIL', rotation=45)
#plt.clabel(cs, inline=True, colors='black', fontsize=10)
cb = plt.colorbar()
cb.set_label('Ratio to ozone at midnight')


#######################################GDOC_July
GDOC_diurnal_s7 =np.transpose(GDOC[:,6,:,28]) #np.tanspose exchange x,y

interval=np.arange(0.93,1.07,0.01)
ax_2 = fig.add_subplot(122)
ax_2.set_yscale('log')
ax_2.set_xlabel('Time[Hour]')
ax_2.set_ylabel('Pressure\n[hPa]')
ax_2.set_title(r'$2018-July \quad 47.5^\circ N$')
ax_2.set_ylim(1,100)
cs=plt.contourf(hour,zstar,GDOC_diurnal_s7,levels=(interval),cmap='coolwarm')
plt.gca().invert_yaxis()
#plt.clabel(cs,inline=True, colors='black',fontsize=10)
cb = plt.colorbar()
cb.set_label('Ratio to ozone at midnight')
plt.show()



###########################################################Gromos_August
gromos_aug = gromos.o3_x.sel(time=slice("2018-08-01", "2018-08-31"))#slice month data
pandas_time_7 = pd.to_datetime(gromos_aug.time.data) #pick up time data
hours_7 = pandas_time_7.hour #only pick up hour(0-23)

gromos_aug_data8_av_full = np.zeros((1, 47)) #build a one row full 0 array


for x in range(0, 24):
    time_aug_data8 = np.array([])
    for i, j in enumerate(hours_7):
        if j == x:
            time_aug_data8 = np.append(np.array(time_aug_data8, dtype=int), i) #pick up indexs of certain hour for instance j=0 pick up index of 0 oclock
    gromos_aug_data8 = gromos_aug[time_aug_data8, :]                           #pick up o3_x data corresponding to index hour
    gromos_aug_data8_sum = np.sum(gromos_aug_data8, axis=0)                    #sum all the data along row
    gromos_aug_data8_av = gromos_aug_data8_sum / time_aug_data8.size           #average the certain hour of whole month data
    gromos_aug_data8_av_full = np.vstack((gromos_aug_data8_av_full, gromos_aug_data8_av))  #build a diuranl ts array

gromos_aug_data8_av_full = np.delete(gromos_aug_data8_av_full, 0, axis=0)      #delete the first row (full 0 array)
average_midnight = (gromos_aug_data8_av_full[0] + gromos_aug_data8_av_full[23]) / 2
gromos_aug_data8_av_full_rate_midnight = np.transpose(gromos_aug_data8_av_full / average_midnight)  #get a rate arrary (every row divide into midnight row)

interval=np.arange(0.93,1.07,0.01)   #range for levels in contour plot

fig = plt.figure(figsize=(13,6))
ax_1 = fig.add_subplot(121)
ax_1.set_yscale('log')
ax_1.set_xlabel('Time[Hour]')
ax_1.set_ylabel('Pressure\n[hPa]')
ax_1.set_title('2018-August-stratospheric-ozone-diurnal-ts')
ax_1.set_ylim(1,100)
cs = plt.contourf(range(0, 24), gromos.o3_p / 100, gromos_aug_data8_av_full_rate_midnight,levels=(interval),cmap='coolwarm')  # colors='k' is mono color line
plt.gca().invert_yaxis()  # change the order of y ax_1is
ax_1.axhline(y=0.02, color='black', linestyle='dashed')
ax_1.axhline(y=110, color='black', linestyle='dashed')
#ax_1.text(0, 0.02, 'RCIL', rotation=45)
#ax_1.text(0, 110, 'RCIL', rotation=45)
#plt.clabel(cs, inline=True, colors='black', fontsize=10)
cb = plt.colorbar()
cb.set_label('Ratio to ozone at midnight')


#######################################GDOC_aug
GDOC_diurnal_s8 =np.transpose(GDOC[:,7,:,28]) #np.tanspose exchange x,y

interval=np.arange(0.93,1.07,0.01)
ax_2 = fig.add_subplot(122)
ax_2.set_yscale('log')
ax_2.set_xlabel('Time[Hour]')
ax_2.set_ylabel('Pressure\n[hPa]')
ax_2.set_title(r'$2018-August \quad 47.5^\circ N$')
ax_2.set_ylim(1,100)
cs=plt.contourf(hour,zstar,GDOC_diurnal_s8,levels=(interval),cmap='coolwarm')
plt.gca().invert_yaxis()
#plt.clabel(cs,inline=True, colors='black',fontsize=10)
cb = plt.colorbar()
cb.set_label('Ratio to ozone at midnight')
plt.show()


###########################################################Gromos_October
gromos_oct = gromos.o3_x.sel(time=slice("2018-10-01", "2018-10-31"))#slice month data
pandas_time_7 = pd.to_datetime(gromos_oct.time.data) #pick up time data
hours_7 = pandas_time_7.hour #only pick up hour(0-23)

gromos_oct_data8_av_full = np.zeros((1, 47)) #build a one row full 0 array


for x in range(0, 24):
    time_oct_data8 = np.array([])
    for i, j in enumerate(hours_7):
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

fig = plt.figure(figsize=(13,6))
ax_1 = fig.add_subplot(121)
ax_1.set_yscale('log')
ax_1.set_xlabel('Time[Hour]')
ax_1.set_ylabel('Pressure\n[hPa]')
ax_1.set_title('2018-October-stratospheric-ozone-diurnal-ts')
ax_1.set_ylim(1,100)
cs = plt.contourf(range(0, 24), gromos.o3_p / 100, gromos_oct_data8_av_full_rate_midnight,levels=(interval),cmap='coolwarm')  # colors='k' is mono color line
plt.gca().invert_yaxis()  # change the order of y ax_1is
ax_1.axhline(y=0.02, color='black', linestyle='dashed')
ax_1.axhline(y=110, color='black', linestyle='dashed')
#ax_1.text(0, 0.02, 'RCIL', rotation=45)
#ax_1.text(0, 110, 'RCIL', rotation=45)
#plt.clabel(cs, inline=True, colors='black', fontsize=10)
cb = plt.colorbar()
cb.set_label('Ratio to ozone at midnight')


#######################################GDOC_oct
GDOC_diurnal_s10 =np.transpose(GDOC[:,9,:,28]) #np.tanspose exchange x,y

interval=np.arange(0.93,1.07,0.01)
ax_2 = fig.add_subplot(122)
ax_2.set_yscale('log')
ax_2.set_xlabel('Time[Hour]')
ax_2.set_ylabel('Pressure\n[hPa]')
ax_2.set_title(r'$2018-October \quad 47.5^\circ N$')
ax_2.set_ylim(1,100)
cs=plt.contourf(hour,zstar,GDOC_diurnal_s10,levels=(interval),cmap='coolwarm')
plt.gca().invert_yaxis()
#plt.clabel(cs,inline=True, colors='black',fontsize=10)
cb = plt.colorbar()
cb.set_label('Ratio to ozone at midnight')
plt.show()