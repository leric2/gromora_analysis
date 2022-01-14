import os
import numpy as np
# import retrievals
import xarray as xr
from glob import glob
import pandas as pd
import math
import netCDF4 as nc
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from pytz import timezone, utc
from matplotlib.pylab import date2num, num2date
from pysolar import solar


'''
#based on each day completed profiles to extract ozone and related dimensions to build oneyear ozone profile
WACCUM_O3_bern_av_full = np.zeros((66,24))
path='Z://atmos_legacy//waccm2//f2000-t900-lat4lon5//atm//hist/'  #设置存储路径
files= os.listdir(path) #attain all file name in the fold(得到文件夹下的所有文件名称)
files_final = np.array([])
for file_date in np.arange(len(files)):
    file_comb = os.path.join('Z://atmos_legacy//waccm2//f2000-t900-lat4lon5//atm//hist/'+ files[file_date]) #combine all days file name into array
    files_final = np.append(files_final,file_comb)

# extract ozone and related dimensions for each day and store them
for file_date, file in enumerate(files_final):  #按照顺序在 files 里面进行每一个文件的 数据名称 循环读取
    WACCUM = xr.open_dataset(
        file,
        decode_times=True,
        decode_coords=True,
        # use_cftime=True,
    )    #如是nc文件，将上一行换成：f =nc.Dataset(file,'r') 即可

    WACCUM_time = WACCUM.time.data
    WACCUM_lon = WACCUM.lon.data
    WACCUM_lat = WACCUM.lat.data
    WACCUM_pressure = WACCUM.lev.data
    WACCUM_O3 = WACCUM.O3.sel(lat=46.00000, lon=5.00000).data * 1e6

    WACCUM_new = xr.Dataset(
        {'O3': (('time', 'lev'), WACCUM_O3)},
        coords={
            'time': WACCUM_time,
            'lev': WACCUM_pressure
        },
    )
    WACCUM_new.to_netcdf(os.path.join('F://waccum data/'+files[file_date]))

# combine oneyear ozone profile
def read_netcdfs(files, dim):
    # glob expands paths with * to a list of files, like the unix shell
    paths = sorted(glob(files))
    datasets = [xr.open_dataset(p) for p in paths]
    combined = xr.concat(datasets, dim)
    return combined

#combined = read_netcdfs('F:\PyCharm Community Edition 2019.3\practice2021 april\gromos_data\*.nc', dim='time')
combined = read_netcdfs('F:\PyCharm Community Edition 2019.3\practice2021 april\somora_data\*.nc', dim='time')
#combined.to_netcdf('F:\PyCharm Community Edition 2019.3\practice2021 april\gromos_data\gromos_nine_year_data.nc')
combined.to_netcdf('F:\PyCharm Community Edition 2019.3\practice2021 april\somora_data\somora_nine_year_data.nc')
'''
######################################################################################
#filename = 'F:\PyCharm Community Edition 2019.3\practice2021 april\somora_data\somora_nine_year_data.nc'
filename = 'F:\PyCharm Community Edition 2019.3\practice2021 april\gromos_data\gromos_nine_year_data.nc'
gromos = xr.open_dataset(
    filename,
    decode_times=True,
    decode_coords=True,
    # use_cftime=True,
)

gromos = gromos.sel(time=slice("2012-01-01", "2019-12-31"))

outlier_number = gromos.oem_diagnostics[:,2]

gromos_real = np.array([])       #remove outlier
for order, index in enumerate(outlier_number):
    if 0.8 < index <1.2:
        gromos_real = np.append(gromos_real,order)


#build new datasets after removement
gromos_real = gromos_real.astype(int)
gromos_time = gromos.time[gromos_real]
gromos_newo3 = gromos.o3_x[gromos_real]
gromos_p = gromos.o3_p
gromos_lat = gromos.obs_lat
gromos_lon = gromos.obs_lon
gromos = xr.Dataset(
    {'o3_x':(["time","o3_p"], gromos_newo3)},
    #{'obs_lat':(["time"],gromos_lat)},
    #{'obs_lon':(["time"],gromos_lon)},
    coords=dict({
            'time': gromos_time,
            'o3_p': gromos_p,
        }),
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
#################################################################################################
nine_year = ["2012","2013","2014","2015","2016","2017","2018","2019"]
#nine_year = ["2012","2013","2014","2015","2016","2017","2018","2019"]

#pick up certain pressure lev for stra and mesos
for pressure_num, pressure_gromos in enumerate(gromos.o3_p/100):
    if pressure_gromos > 4 and pressure_gromos < 6:
         pressure_certain_stra = pressure_num
         print(pressure_certain_stra)

for pressure_num, pressure_gromos in enumerate(gromos.o3_p/100):
    if pressure_gromos > 0.4 and pressure_gromos < 0.6:
         pressure_certain_mes = pressure_num
         print(pressure_certain_mes)

sunrise_seasonal = np.array([])
sunset_seasonal = np.array([])
gromos_seasonal = np.zeros((1, 24))
gromos_seasonal_1 = np.zeros((1, 24))



for year_num, each_year in enumerate(nine_year):
    gromos_month_1 = gromos.o3_x.sel(time=slice(nine_year[year_num]+"-"+"01-01", nine_year[year_num]+"-"+"01-31"))
    gromos_month_2 = gromos.o3_x.sel(time=slice(nine_year[year_num]+"-"+"02-01", nine_year[year_num]+"-"+"02-28"))  # slice month o3 data
    gromos_month_3 = gromos.o3_x.sel(time=slice(nine_year[year_num]+"-"+"03-01", nine_year[year_num]+"-"+"03-31"))
    gromos_month_4 = gromos.o3_x.sel(time=slice(nine_year[year_num]+"-"+"04-01", nine_year[year_num]+"-"+"04-30"))
    gromos_month_5 = gromos.o3_x.sel(time=slice(nine_year[year_num]+"-"+"05-01", nine_year[year_num]+"-"+"05-31"))
    gromos_month_6 = gromos.o3_x.sel(time=slice(nine_year[year_num]+"-"+"06-01", nine_year[year_num]+"-"+"06-30"))
    gromos_month_7 = gromos.o3_x.sel(time=slice(nine_year[year_num]+"-"+"07-01", nine_year[year_num]+"-"+"07-31"))
    gromos_month_8 = gromos.o3_x.sel(time=slice(nine_year[year_num]+"-"+"08-01", nine_year[year_num]+"-"+"08-31"))
    gromos_month_9 = gromos.o3_x.sel(time=slice(nine_year[year_num]+"-"+"09-01", nine_year[year_num]+"-"+"09-30"))
    gromos_month_10 = gromos.o3_x.sel(time=slice(nine_year[year_num]+"-"+"10-01", nine_year[year_num]+"-"+"10-31"))
    gromos_month_11 = gromos.o3_x.sel(time=slice(nine_year[year_num]+"-"+"11-01", nine_year[year_num]+"-"+"11-30"))
    gromos_month_12 = gromos.o3_x.sel(time=slice(nine_year[year_num]+"-"+"12-01", nine_year[year_num]+"-"+"12-31"))

    gromos_month_lst_1 = gromos.sel(time=slice(nine_year[year_num]+"-"+"01-01", nine_year[year_num]+"-"+"01-31"))
    gromos_month_lst_2 = gromos.sel(time=slice(nine_year[year_num]+"-"+"02-01", nine_year[year_num]+"-"+"02-28"))  # slice month data
    gromos_month_lst_3 = gromos.sel(time=slice(nine_year[year_num]+"-"+"03-01", nine_year[year_num]+"-"+"03-31"))
    gromos_month_lst_4 = gromos.sel(time=slice(nine_year[year_num]+"-"+"04-01", nine_year[year_num]+"-"+"04-30"))
    gromos_month_lst_5 = gromos.sel(time=slice(nine_year[year_num]+"-"+"05-01", nine_year[year_num]+"-"+"05-31"))
    gromos_month_lst_6 = gromos.sel(time=slice(nine_year[year_num]+"-"+"06-01", nine_year[year_num]+"-"+"06-30"))
    gromos_month_lst_7 = gromos.sel(time=slice(nine_year[year_num]+"-"+"07-01", nine_year[year_num]+"-"+"07-31"))
    gromos_month_lst_8 = gromos.sel(time=slice(nine_year[year_num]+"-"+"08-01", nine_year[year_num]+"-"+"08-31"))
    gromos_month_lst_9 = gromos.sel(time=slice(nine_year[year_num]+"-"+"09-01", nine_year[year_num]+"-"+"09-30"))
    gromos_month_lst_10 = gromos.sel(time=slice(nine_year[year_num]+"-"+"10-01", nine_year[year_num]+"-"+"10-31"))
    gromos_month_lst_11 = gromos.sel(time=slice(nine_year[year_num]+"-"+"11-01", nine_year[year_num]+"-"+"11-30"))
    gromos_month_lst_12 = gromos.sel(time=slice(nine_year[year_num]+"-"+"12-01", nine_year[year_num]+"-"+"12-31"))

    # build 12 month ozone profile tuples for four datasets
    month_time_lst_12 = (
    gromos_month_lst_1, gromos_month_lst_2, gromos_month_lst_3, gromos_month_lst_4, gromos_month_lst_5, gromos_month_lst_6,
    gromos_month_lst_7, gromos_month_lst_8, gromos_month_lst_9, gromos_month_lst_10, gromos_month_lst_11,
    gromos_month_lst_12)
    month_12_o3_data = (
    gromos_month_1, gromos_month_2, gromos_month_3, gromos_month_4, gromos_month_5, gromos_month_6, gromos_month_7,
    gromos_month_8, gromos_month_9, gromos_month_10, gromos_month_11, gromos_month_12)

    for month_data, month_title in enumerate(np.arange(12)):
        gromos_month = month_12_o3_data[month_data]
        gromos_month_time_lst = month_time_lst_12[month_data]
        lst_gromos = np.array([])
        for z in range(gromos_month_time_lst.time.shape[0]):    #utc convert to lst
            lst_save = get_LST_from_UTC(gromos_month_time_lst.time.values[z],
                                        46.82,
                                        6.95)
            lst_gromos = np.append(lst_gromos, lst_save[0])
        pandas_time_test = pd.to_datetime(lst_gromos)
        hours = pandas_time_test.hour

        sunrise_set_day_order = np.array([])
        sunrise_ft = np.array([])
        sunset_ft = np.array([])

        for x, y in enumerate(hours):
            if y == 8:  # pick 8 am as the referece to pick the order of day
                sunrise_set_day_order = np.append(np.array(sunrise_set_day_order, dtype=int), x)
        if sunrise_set_day_order.size > 0:
            for sunrise_dayofmonth in sunrise_set_day_order:
                sunrise_set_lst_covert = lst_gromos[sunrise_dayofmonth]
                sunrise_set_lst = get_sunset_lst_from_lst(sunrise_set_lst_covert, 46.82)  # calculate sunrise and set lst ; surise_set_lst_covert[0] is that pick time variable
                sunrise_datetime = pd.to_datetime(sunrise_set_lst)
                # date(str) to num to compute mean and transfer back to date
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
            sunrise_seasonal = np.append(sunrise_seasonal,sunrise_transfer_num)
            sunset_seasonal = np.append(sunset_seasonal,sunset_transfer_num)
            print(sunrise_ft_av)
            print(sunset_ft_av)
        else:  #if no data in the month minus 1
            sunrise_seasonal = np.append(sunrise_seasonal, sunrise_seasonal[-1])
            sunset_seasonal = np.append(sunset_seasonal, sunset_seasonal[-1])

        gromos_month_data_av_full = np.zeros((1, 47))  # build a one row full 0 array
        for x in range(0, 24):
            time_month_data = np.array([])
            for i, j in enumerate(hours):
                if j == x:
                    time_month_data = np.append(np.array(time_month_data, dtype=int),
                                                i)  # pick up indexs of certain hour for instance j=0 pick up index of 0 oclock
            gromos_month_data = gromos_month[time_month_data, :]  # pick up o3_x data corresponding to index hour
            gromos_month_data_sum = np.sum(gromos_month_data, axis=0)  # sum all the data along row
            gromos_month_data_av = gromos_month_data_sum / time_month_data.size  # average the certain hour of whole month data
            gromos_month_data_av_full = np.vstack(
                (gromos_month_data_av_full, gromos_month_data_av))  # build a diuranl ts array

        gromos_month_data_av_full = np.delete(gromos_month_data_av_full, 0, axis=0)  # delete the first row (full 0 array)
        average_midnight = (gromos_month_data_av_full[0] + gromos_month_data_av_full[23]) / 2  # average 0am and 23pm as midnight reference
        gromos_month_data_av_full_rate_midnight = np.transpose(gromos_month_data_av_full / average_midnight)  # get a rate arrary (every row divide into midnight row)
        gromos_month_data_av_full_rate_midnight = np.array(gromos_month_data_av_full_rate_midnight[pressure_certain_mes, :])
        #gromos_month_data_av_full_rate_midnight = np.array(gromos_month_data_av_full_rate_midnight[pressure_certain_stra,:]) #np.array 修正了ndarray的格式 # pressure change for mesos or stratos
        gromos_seasonal = np.vstack((gromos_seasonal, gromos_month_data_av_full_rate_midnight))


gromos_seasonal_finish = np.delete(gromos_seasonal, 0, axis=0)


#interval = np.arange(0.95, 1.05, 0.01)  # for stra
interval = np.arange(0.80, 1.02, 0.02) # for mes
fig = plt.figure()
ax1 = fig.add_subplot(111)

ax1.set_xlabel('Local Solar Time[Hour]')
ax1.set_ylabel('Year[2012 to 2019]')
#ax1.set_title('1718av seasonal ozone diurnal cycle (somora)[4.86hPa]')
#ax1.set_title('1718av seasonal ozone diurnal cycle (gromos)[4.86hPa]')
ax1.set_title('8 years ozone diurnal cycle (gromos)[4.86hPa]')
cs = plt.contourf(np.arange(0, 24), np.arange(1,97), gromos_seasonal_finish, levels=(interval),
                  cmap='coolwarm', extend="both")  # colors='k' is mono color line
plt.gca().invert_yaxis()  # change the order of y axis
cs.cmap.set_under('MediumBlue')  # set the color over colorbar low boundary
cs.cmap.set_over('Crimson')  # set the color over colorbar upper boundary
plt.yticks(np.arange(1,97,12),nine_year)
ax1.plot(sunrise_seasonal,np.arange(1,97),color='white', linestyle='dashed')
ax1.plot(sunset_seasonal,np.arange(1,97),color='black', linestyle='dashed')
#ax1.axvline(x=sunrise_transfer_num, color='white', linestyle='dashed')
#ax1.axvline(x=sunset_transfer_num, color='black', linestyle='dashed')
# plt.clabel(cs, inline=True, colors='black', fontsize=10)
cb = plt.colorbar()
cb.set_label('Ratio to ozone at midnight')
#plt.savefig('C:/Users/Hou/Desktop/master thesis/figures for 2018 data/2021.11/annual diurnal ozone cycle stra and mes/' + 'annual diurnal ozone cycle (gromos)[0.486hPa]' + '.pdf', dpi=100)

print(gromos.o3_p.values[18])
print(gromos.o3_p.values[26])