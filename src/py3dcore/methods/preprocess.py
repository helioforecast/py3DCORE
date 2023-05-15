# -*- coding: utf-8 -*-

import datetime
from typing import Any, List, Optional, Sequence, Tuple, Union

import heliosat
import numba
import numpy as np
import os
from scipy.signal import detrend, welch
######
import matplotlib.dates as mdates
from matplotlib.dates import  DateFormatter
import datetime
from datetime import timedelta

from sunpy.coordinates import frames, get_horizons_coord
from sunpy.time import parse_time

import urllib.request

import cdflib
import pickle

import logging
import json 

import pandas as pds

logger = logging.getLogger(__name__)

    


def cdftopickle(magpath, swapath, sc):
    
    '''
    creating a pickle file from cdf
    
    magpath        path to directory with magnetic field data
    swapath        path to directory with solar wind data (not needed for 3DCORE)
    sc             spacecraft
    
    '''
    
    
    if sc == 'solo':
        fullname = 'solar orbiter'
    if sc == 'psp':
        fullname = 'Parker Solar Probe'
    if sc == 'wind':
        fullname = 'Wind'    
    
    timep = np.zeros(0,dtype=[('time',object)])
    den = np.zeros(0)
    temp = np.zeros(0)
    vr = np.zeros(0)
    vt = np.zeros(0)
    vn = np.zeros(0)
        
    if os.path.exists(swapath):
        ll_path = swapath
    
        files = os.listdir(ll_path)
        files.sort()
        llfiles = [os.path.join(ll_path, f) for f in files if f.endswith('.cdf')]
    

        for i in np.arange(0,len(llfiles)):
            p1 = cdflib.CDF(llfiles[i])

            den1 = p1.varget('N')
            speed1 = p1.varget('V_RTN')
            temp1 = p1.varget('T')

            vr1 = speed1[:, 0]
            vt1 = speed1[:, 1]
            vn1 = speed1[:, 2]

            vr = np.append(vr1, vr)
            vt = np.append(vt1, vt)
            vn = np.append(vn1, vn)


            temp = np.append(temp1, temp)
            den = np.append(den1, den)


            time1 = p1.varget('EPOCH')
            t1 = parse_time(cdflib.cdfastropy.convert_to_astropy(time1, format=None)).datetime
            timep = np.append(timep, t1)

            temp = temp*(1.602176634*1e-19) / (1.38064852*1e-23) # from ev to K 

    ll_path = magpath

    files = os.listdir(ll_path)
    files.sort()
    llfiles = [os.path.join(ll_path, f) for f in files if f.endswith('.cdf')]
    
    br1 = np.zeros(0)
    bt1 = np.zeros(0)
    bn1 = np.zeros(0)
    time1 = np.zeros(0,dtype=[('time',object)])
    
    for i in np.arange(0,len(llfiles)):
        m1 = cdflib.CDF(llfiles[i])
        
        if sc == 'solo':
            print('solo cdf')
            b = m1.varget('B_RTN')
            time = m1.varget('EPOCH')
        
        elif sc == 'psp':
            print('psp cdf')
            b = m1.varget('psp_fld_l2_mag_RTN_1min')
            time = m1.varget('epoch_mag_RTN_1min')
        
        elif sc =='wind':
            print('wind cdf')
            b = m1.varget('BRTN')
            time = m1.varget('Epoch')
                
        else:
            print('No data from cdf file')
            
        br = b[:, 0]
        bt = b[:, 1]
        bn = b[:, 2]

        br1 = np.append(br1, br)
        bt1 = np.append(bt1, bt)
        bn1 = np.append(bn1, bn)

        #try:
        #    time = m1.varget('EPOCH')
        #except:
        #    time = m1.varget('epoch_mag_RTN_1min')
            
        t1 = parse_time(cdflib.cdfastropy.convert_to_astropy(time, format=None)).datetime
        time1 = np.append(time1,t1)
        
    starttime = time1[0].replace(hour = 0, minute = 0, second=0, microsecond=0)
    endtime = time1[-1].replace(hour = 0, minute = 0, second=0, microsecond=0)
    
    time_int = []
    while starttime < endtime:
        time_int.append(starttime)
        starttime += timedelta(minutes=1)


    time_int_mat = mdates.date2num(time_int)
    time1_mat = mdates.date2num(time1)
    timep_mat = mdates.date2num(timep)  
    
    if os.path.exists(swapath):
        ll = np.zeros(np.size(time_int),dtype=[('time',object),('bx', float),('by', float),('bz', float),('bt', float),('r', float),('lat', float),('lon', float),('x', float),('y', float),('z', float),('vx', float),('vy', float),('vz', float),('vt', float),('tp', float),('np', float) ] )
    else:
        ll = np.zeros(np.size(time_int),dtype=[('time',object),('bx', float),('by', float),('bz', float),('bt', float),('r', float),('lat', float),('lon', float),('x', float),('y', float),('z', float)] )
        

    ll = ll.view(np.recarray)  
    
    # replace all unreasonable large and small values with nan
    thresh = 1e6
    br1[br1 > thresh] = np.nan
    br1[br1 < -thresh] = np.nan
    bt1[bt1 > thresh] = np.nan
    bt1[bt1 < -thresh] = np.nan
    bn1[bn1 > thresh] = np.nan
    bn1[bn1 < -thresh] = np.nan

    # interpolate between non nan values
    ll.time = time_int
    ll.bx = np.interp(time_int_mat, time1_mat[~np.isnan(br1)], br1[~np.isnan(br1)])
    ll.by = np.interp(time_int_mat, time1_mat[~np.isnan(bt1)], bt1[~np.isnan(bt1)])
    ll.bz = np.interp(time_int_mat, time1_mat[~np.isnan(bn1)], bn1[~np.isnan(bn1)])
    ll.bt = np.sqrt(ll.bx**2 + ll.by**2 + ll.bz**2)
    
    if os.path.exists(swapath):
        ll.np = np.interp(time_int_mat, timep_mat, den)
        ll.tp = np.interp(time_int_mat, timep_mat, temp) 
        ll.vx = np.interp(time_int_mat, timep_mat, vr)
        ll.vy = np.interp(time_int_mat, timep_mat, vt)
        ll.vz = np.interp(time_int_mat, timep_mat, vn)
        ll.vt = np.sqrt(ll.vx**2 + ll.vy**2 + ll.vz**2)

    if sc == 'solo':
        # Solar Orbiter position with sunpy
        coord = get_horizons_coord(sc, time={'start': time_int[0], 'stop': time_int[-1], 'step': '1m'})  
        heeq = coord.transform_to(frames.HeliographicStonyhurst) #HEEQ
        
        ll.r = heeq.radius.value
        ll.lon = heeq.lon.value
        ll.lat = heeq.lat.value
        
        ll = sphere2cart(ll)
        
        print(ll.x, ll.y, ll.z)

    elif sc == 'psp':    
        # PSP position with sunpy
        coord = get_horizons_coord(sc, time={'start': time_int[0], 'stop': time_int[-1], 'step': '1m'})  
        heeq = coord.transform_to(frames.HeliographicStonyhurst) #HEEQ
        
        ll.r = heeq.radius.value
        ll.lon = heeq.lon.value
        ll.lat = heeq.lat.value
        
        ll = sphere2cart(ll)
        
        print(ll.x, ll.y, ll.z)

    elif sc == 'wind':
        # Wind position with sunpy
        
        # use EM-L1 as proxy for the Wind spacecraft position
        # this is valid from 2004 on, where Wind was placed in orbit around L1
        coord = get_horizons_coord('EM-L1', time={'start': time_int[0], 'stop': time_int[-1], 'step': '1m'})  
        heeq = coord.transform_to(frames.HeliographicStonyhurst) #HEEQ
        
        ll.r = heeq.radius.value
        ll.lon = heeq.lon.value
        ll.lat = heeq.lat.value
        
        ll = sphere2cart(ll)
        
        print(ll.x, ll.y, ll.z)
     
    else:
        print('no spacecraft position')
        
    return ll
    
    
def sphere2cart(ll):
        
    ll.x = ll.r * np.cos(np.deg2rad(ll.lon)) * np.cos(np.deg2rad(ll.lat))
    ll.y = ll.r * np.sin(np.deg2rad(ll.lon)) * np.cos(np.deg2rad(ll.lat))
    ll.z = ll.r * np.sin(np.deg2rad(ll.lat))
    
    return ll
        
def createpicklefiles(self, data_path):
    name = data_path.split('.')[0]
    sc = name.split('_')[0]
    ev = name.split('_')[1]

    magpath = 'py3dcore/custom_data/' + sc +'_mag_'+ ev
    swapath = 'py3dcore/custom_data/' + sc +'_swa_'+ ev

    ll = cdftopickle(magpath, swapath, sc)

    filename= sc +'_'+ ev + '.p'

    pickle.dump(ll, open('py3dcore/custom_data/' + filename, "wb"))
    logger.info("Created pickle file from cdf: %s", filename)
                