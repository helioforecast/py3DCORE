import numpy as np
import pandas as pd
from datetime import datetime, timedelta
#from spacepy import pycdf
import spiceypy
# import os
import glob
import urllib.request
import os.path
import json


def filter_bad_data(df, col, bad_val):
    if bad_val < 0:
        mask = df[col] < bad_val  # boolean mask for all bad values
    else:
        mask = df[col] > bad_val  # boolean mask for all bad values
    cols = [x for x in df.columns if x != 'timestamp']
    df.loc[mask, cols] = np.nan
    return df

import urllib.request
import json
import pandas as pd
import numpy as np
from sunpy.coordinates import frames, get_body_heliographic_stonyhurst
from astropy.coordinates import SkyCoord
import astropy.units as u

def get_noaa_mag_realtime_7days_heeq():
    #mag data request produces file in GSM coords
    request_mag=urllib.request.urlopen('https://services.swpc.noaa.gov/products/solar-wind/mag-7-day.json')
    file_mag = request_mag.read()
    data_mag = json.loads(file_mag)
    noaa_mag_gsm = pd.DataFrame(data_mag[1:], columns=['timestamp', 'b_x', 'b_y', 'b_z', 'lon_gsm', 'lat_gsm', 'b_tot'])

    # Convert timestamp to datetime and magnetic field components to float
    noaa_mag_gsm['timestamp'] = pd.to_datetime(noaa_mag_gsm['timestamp'])
    noaa_mag_gsm['b_x'] = noaa_mag_gsm['b_x'].astype('float')
    noaa_mag_gsm['b_y'] = noaa_mag_gsm['b_y'].astype('float')
    noaa_mag_gsm['b_z'] = noaa_mag_gsm['b_z'].astype('float')
    noaa_mag_gsm['b_tot'] = noaa_mag_gsm['b_tot'].astype('float')

    # Transform magnetic field components from GSM to HEEQ coordinates
    time = noaa_mag_gsm['timestamp'].values
    gsm_coord = SkyCoord(noaa_mag_gsm['lon_gsm'].values * u.deg, noaa_mag_gsm['lat_gsm'].values * u.deg, 1 * u.AU,
                         obstime=time, frame=frames.HeliographicStonyhurst)
    heeq_coord = gsm_coord.transform_to(frames.HeliocentricEarthEcliptic)
    noaa_mag_heeq = noaa_mag_gsm.copy()
    noaa_mag_heeq['b_x'], noaa_mag_heeq['b_y'], noaa_mag_heeq['b_z'] = heeq_coord.cartesian.xyz.value.T

    # Calculate spacecraft position in HEEQ coordinates
    spacecraft_heeq = get_body_heliographic_stonyhurst('earth', time).transform_to(frames.HeliocentricEarthEcliptic)
    noaa_mag_heeq['x'] = spacecraft_heeq.cartesian.x.value
    noaa_mag_heeq['y'] = spacecraft_heeq.cartesian.y.value
    noaa_mag_heeq['z'] = spacecraft_heeq.cartesian.z.value

    return noaa_mag_heeq

def get_noaa_mag_realtime_7days():
    #mag data request produces file in GSM coords
    request_mag=urllib.request.urlopen('https://services.swpc.noaa.gov/products/solar-wind/mag-7-day.json')
    file_mag = request_mag.read()
    data_mag = json.loads(file_mag)
    noaa_mag_gsm = pd.DataFrame(data_mag[1:], columns=['timestamp', 'b_x', 'b_y', 'b_z', 'lon_gsm', 'lat_gsm', 'b_tot'])

    noaa_mag_gsm['timestamp'] = pd.to_datetime(noaa_mag_gsm['timestamp'])
    noaa_mag_gsm['b_x'] = noaa_mag_gsm['b_x'].astype('float')
    noaa_mag_gsm['b_y'] = noaa_mag_gsm['b_y'].astype('float')
    noaa_mag_gsm['b_z'] = noaa_mag_gsm['b_z'].astype('float')
    noaa_mag_gsm['b_tot'] = noaa_mag_gsm['b_tot'].astype('float')
    return noaa_mag_gsm


def get_noaa_plas_realtime_7days():
    #plasma data request returns bulk parameters: density, v_bulk, temperature
    request_plas=urllib.request.urlopen('https://services.swpc.noaa.gov/products/solar-wind/plasma-7-day.json')
    file_plas = request_plas.read()
    data_plas = json.loads(file_plas)
    print(data_plas)
    noaa_plas = pd.DataFrame(data_plas[1:], columns=['timestamp', 'density', 'v_bulk', 'temperature'])

    noaa_plas['timestamp'] = pd.to_datetime(noaa_plas['timestamp'])
    noaa_plas['density'] = noaa_plas['density'].astype('float')
    noaa_plas['v_bulk'] = noaa_plas['v_bulk'].astype('float')
    noaa_plas['temperature'] = noaa_plas['temperature'].astype('float')
    return noaa_plas


def get_noaa_realtime_alt(path = '/Volumes/External/Data/DSCOVR'):

    filename = os.listdir(path)[0]
    noaa_alt = pd.read_table(f'{path}/{filename}', header=9, sep='\s+')
    noaa_alt = noaa_alt.reset_index()
    noaa_alt['timestamp'] = pd.to_datetime(noaa_alt['index'] + ' ' + noaa_alt['Timestamp'], format='%Y-%m-%d %H:%M:%S')
    noaa_alt = noaa_alt.drop(columns=['index', 'Timestamp'])

    noaa_alt.rename(columns={'Bt-med': 'b_tot', 'Bx-med': 'b_x', 'By-med': 'b_y', 'Bz-med': 'b_z'}, inplace=True)
    noaa_alt.rename(columns={'Dens-med': 'density', 'Speed-med': 'v_bulk', 'Temp-med': 'temperature'}, inplace=True)

    noaa_alt.drop(columns = ['Source', 'Bt-min', 'Bt-max', 'Bx-min', 'Bx-max', 'By-min', 'By-max', 'Bz-min', 'Bz-max'], inplace=True)
    noaa_alt.drop(columns = ['Phi-mean', 'Phi-min', 'Phi-max', 'Theta-med', 'Theta-min', 'Theta-max'], inplace=True)
    noaa_alt.drop(columns = ['Dens-min', 'Dens-max', 'Speed-min', 'Speed-max', 'Temp-min', 'Temp-max'], inplace=True)

    return noaa_alt