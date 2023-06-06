import datetime

import heliosat

import streamlit as st

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
    
    return b, dt


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