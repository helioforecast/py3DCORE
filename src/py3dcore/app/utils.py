import datetime

import heliosat

import streamlit as st
import functools 

import pickle as p
import numpy as np
import pandas as pds

import os


class Event:
    
    def __init__(self, begin, end, idd, sc):
        self.begin = begin
        self.end = end
        self.duration = self.end-self.begin
        self.id = idd 
        self.sc = sc
    def __str__(self):
        return self.id

def get_catevents(day):
    '''
    Returns from helioforecast.space the event list for a given day
    '''
    url='https://helioforecast.space/static/sync/icmecat/HELIO4CAST_ICMECAT_v21.csv'
    icmecat=pds.read_csv(url)
    starttime = icmecat.loc[:,'icme_start_time']
    idd = icmecat.loc[:,'icmecat_id']
    sc = icmecat.loc[:, 'sc_insitu']
    endtime = icmecat.loc[:,'mo_end_time']
    mobegintime = icmecat.loc[:, 'mo_start_time']
    
    evtList = []
    dateFormat="%Y/%m/%d %H:%M"
    begin = pds.to_datetime(starttime, format=dateFormat)
    mobegin = pds.to_datetime(mobegintime, format=dateFormat)
    end = pds.to_datetime(endtime, format=dateFormat)

    
    for i, event in enumerate(mobegin):
        if (mobegin[i].year == day.year and mobegin[i].month == day.month and mobegin[i].day == day.day):
            evtList.append(Event(mobegin[i], end[i], idd[i], sc[i]))
    if len(evtList) == 0:
        evtList = ['No events returned', ]
    
    return evtList


def load_cat(date):
    '''
    Returns from helioforecast.space the event list for a given day
    '''
    url='https://helioforecast.space/static/sync/icmecat/HELIO4CAST_ICMECAT_v21.csv'
    icmecat=pds.read_csv(url)
    starttime = icmecat.loc[:,'icme_start_time']
    idd = icmecat.loc[:,'icmecat_id']
    sc = icmecat.loc[:, 'sc_insitu']
    endtime = icmecat.loc[:,'mo_end_time']
    mobegintime = icmecat.loc[:, 'mo_start_time']
    
    evtList = []
    dateFormat="%Y/%m/%d %H:%M"
    begin = pds.to_datetime(starttime, format=dateFormat)
    mobegin = pds.to_datetime(mobegintime, format=dateFormat)
    end = pds.to_datetime(endtime, format=dateFormat)

    
    for i, event in enumerate(mobegin):
        if (mobegin[i].year == date.year and mobegin[i].month == date.month and mobegin[i].day == date.day and mobegin[i].hour == date.hour):
            return Event(mobegin[i], end[i], idd[i], sc[i])


def get_fitobserver(mag_coord_system, sc):
    
    if mag_coord_system == 'HEEQ':
        reference_frame = 'HEEQ'
        
    if sc == 'BepiColombo':
        if mag_coord_system == 'RTN':
            reference_frame = 'GSE'
        observer = 'Bepi'
    elif sc == 'DSCOVR':
        if mag_coord_system == 'RTN':
            reference_frame = 'GSE'
        observer = 'DSCOVR'
    elif sc == 'MESSENGER':
        if mag_coord_system == 'RTN':
            reference_frame = 'MSGR_RTN'
        observer = 'Mes'
    elif sc == 'PSP':
        if mag_coord_system == 'RTN':
            reference_frame = 'SPP_RTN'
        observer = 'PSP'
    elif sc == 'SolarOrbiter':
        if mag_coord_system == 'RTN':
            reference_frame = 'SOLO_SUN_RTN'
        observer = 'SolO'
    elif sc == 'STEREO-A':
        if mag_coord_system == 'RTN':
            reference_frame = 'STAHGRTN'
        observer = 'STA'
    elif sc == 'VEX-A':
        if mag_coord_system == 'RTN':
            reference_frame = 'VSO'
        observer = 'VEX'
    elif sc == 'Wind':
        if mag_coord_system == 'RTN':
            reference_frame = 'GSE'
        observer = 'Wind'
        
    return observer, reference_frame

@functools.lru_cache(maxsize=25)    
def get_insitudata(mag_coord_system, sc, insitubegin, insituend):
    
        
    if mag_coord_system == 'HEEQ':
        reference_frame = 'HEEQ'
        names = ['Bx', 'By', 'Bz']
    else:
        names = ['Br', 'Bt', 'Bn']
        
    if sc == 'BepiColombo':
        if mag_coord_system == 'RTN':
            reference_frame = 'GSE'
        observer = 'Bepi'
    elif sc == 'DSCOVR':
        if mag_coord_system == 'RTN':
            reference_frame = 'GSE'
        observer = 'DSCOVR'
    elif sc == 'MESSENGER':
        if mag_coord_system == 'RTN':
            reference_frame = 'MSGR_RTN'
        observer = 'Mes'
    elif sc == 'PSP':
        if mag_coord_system == 'RTN':
            reference_frame = 'SPP_RTN'
        observer = 'PSP'
    elif sc == 'SolarOrbiter':
        if mag_coord_system == 'RTN':
            reference_frame = 'SOLO_SUN_RTN'
        observer = 'SolO'
    elif sc == 'STEREO-A':
        if mag_coord_system == 'RTN':
            reference_frame = 'STAHGRTN'
        observer = 'STA'
    elif sc == 'VEX-A':
        if mag_coord_system == 'RTN':
            reference_frame = 'VSO'
        observer = 'VEX'
    elif sc == 'Wind':
        if mag_coord_system == 'RTN':
            reference_frame = 'GSE'
        observer = 'Wind'
            
            
    observer_obj = getattr(heliosat, observer)() 
    
    t, b = observer_obj.get([insitubegin,insituend], "mag", reference_frame=reference_frame, as_endpoints=True)
    dt = [datetime.datetime.utcfromtimestamp(ts) for ts in t]
    pos = observer_obj.trajectory(dt, reference_frame=reference_frame)
    
    
    return b, t, pos


def loadpickle(path=None, number=-1):

    """ Loads the filepath of a pickle file. """
    
    path = 'output/' + path + '/'

    # Get the list of all files in path
    dir_list = sorted(os.listdir(path))

    resfile = []
    respath = []
    # we only want the pickle-files
    for file in dir_list:
        if file.endswith(".pickle"):
            resfile.append(file) 
            respath.append(os.path.join(path,file))
            
    filepath = path + resfile[number]

    return filepath
                
class defaulttimer:
    
    def __init__(self, st, hours):
                 
        datetimestart = st.session_state.event_selected.begin + datetime.timedelta(hours = hours)
        datetimeend = st.session_state.event_selected.end + datetime.timedelta(hours = hours)
        self.dateA = datetime.date(datetimestart.year, datetimestart.month, datetimestart.day)
        self.dateB = datetime.date(datetimeend.year, datetimeend.month, datetimeend.day)
        self.timeA = datetime.time(datetimestart.hour, datetimestart.minute)
        self.timeB = datetime.time(datetimeend.hour, datetimeend.minute)
        
        
        
        
def get_iparams(st):
    
    t_launch = st.session_state.dt_launch
    
    model_kwargs = {
        "ensemble_size": int(1), #2**17
        "iparams": {
            "cme_longitude": {
                "distribution": "fixed",
                "default_value": st.session_state.longit
            },
            "cme_latitude": {
                "distribution": "fixed",
                "default_value": st.session_state.latitu
            },
            "cme_inclination": {
                "distribution": "fixed",
                "default_value": st.session_state.inc
            },
            "cme_diameter_1au": {
                "distribution": "fixed",
                "default_value": st.session_state.dia
            },
            "cme_aspect_ratio": {
                "distribution": "fixed",
                "default_value": st.session_state.asp
            },
            "cme_launch_radius": {
                "distribution": "fixed",
                "default_value": st.session_state.l_rad
            },
            "cme_launch_velocity": {
                "distribution": "fixed",
                "default_value": st.session_state.l_vel
            },
            "t_factor": {
                "distribution": "fixed",
                "default_value": st.session_state.t_fac
            },
            "cme_expansion_rate": {
                "distribution": "fixed",
                "default_value": st.session_state.exp_rat
            },
            "magnetic_decay_rate": {
                "distribution": "fixed",
                "default_value": st.session_state.mag_dec
            },
            "magnetic_field_strength_1au": {
                "distribution": "fixed",
                "default_value": st.session_state.mag_strength
            },
            "background_drag": {
                "distribution": "fixed",
                "default_value": st.session_state.b_drag
            },
            "background_velocity": {
                "distribution": "fixed",
                "default_value": st.session_state.bg_vel
            }
        }
    }
    
    return t_launch, model_kwargs

def get_iparams_exp(row):
    
    model_kwargs = {
        "ensemble_size": int(1), #2**17
        "iparams": {
            "cme_longitude": {
                "distribution": "fixed",
                "default_value": row['Longitude']
            },
            "cme_latitude": {
                "distribution": "fixed",
                "default_value": row['Latitude']
            },
            "cme_inclination": {
                "distribution": "fixed",
                "default_value": row['Inclination']
            },
            "cme_diameter_1au": {
                "distribution": "fixed",
                "default_value": row['Diameter 1 AU']
            },
            "cme_aspect_ratio": {
                "distribution": "fixed",
                "default_value": row['Aspect Ratio']
            },
            "cme_launch_radius": {
                "distribution": "fixed",
                "default_value": row['Launch Radius']
            },
            "cme_launch_velocity": {
                "distribution": "fixed",
                "default_value": row['Launch Velocity']
            },
            "t_factor": {
                "distribution": "fixed",
                "default_value": row['T_Factor']
            },
            "cme_expansion_rate": {
                "distribution": "fixed",
                "default_value": row['Expansion Rate']
            },
            "magnetic_decay_rate": {
                "distribution": "fixed",
                "default_value": row['Magnetic Decay Rate']
            },
            "magnetic_field_strength_1au": {
                "distribution": "fixed",
                "default_value": row['Magnetic Field Strength 1 AU']
            },
            "background_drag": {
                "distribution": "fixed",
                "default_value": row['Background Drag']
            },
            "background_velocity": {
                "distribution": "fixed",
                "default_value": row['Background Velocity']
            }
        }
    }
    
    return model_kwargs