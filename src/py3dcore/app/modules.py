import datetime

import numpy as np
import pandas as pds

import astropy.units as u

from py3dcore.app.utils import get_catevents, defaulttimer #, model_fittings
from py3dcore.app.config.config_sliders import sliders_dict as sd
from py3dcore.app.config.config_sliders import mag_sliders_dict as msd
from py3dcore.app.fitting import fitting_main

def date_and_event_selection(st):
    st.sidebar.markdown('## Date and event selection')
    col1, col2 = st.sidebar.columns(2)
    
    initialisation = st.sidebar.select_slider('How to initialize?',
                                        options=('Manual', 'Catalog', 'File'), value='Catalog')
    
    
    if initialisation == 'Catalog':
        day = st.sidebar.date_input('Select a day to process',
                          value=datetime.datetime(2023, 1, 17, 0, 0, 0),
                          min_value=datetime.datetime(2010, 1, 1, 0, 0, 0),
                          max_value=datetime.date.today())
        evtlist = get_catevents(day)
        evtlist.insert(0, 'Select event')
        event_selected = st.sidebar.selectbox('ICMEs', options=evtlist)
        if event_selected != 'Select event' and event_selected != 'No events returned':
            st.session_state.date_process = event_selected.begin
            st.session_state.event_selected = event_selected
            st.experimental_rerun()
        elif event_selected == 'No events returned':
            st.sidebar.warning('Initiate manually')

def fitting_and_slider_options_container(st):
    container = st.sidebar.container()
    
    with container.expander('Options'):
        col1, col2 = st.columns(2)
        col1.radio('Coordinate System', options=['HGS'], #, 'HGC'
                   #on_change=change_long_lat_sliders, 
                 args=[st], key='coord_system')    
        col2.radio('Geometrical model', options=['3DCORE'], 
                 args=[st], key='geo_model')    
        st.checkbox('View 3D Positions', value=False, key='3d_positions')
        st.checkbox('View Insitu Data', value=False, key='insitu_data')
        st.checkbox('View Remote Imaging', value=False, key='remote_imaging')
        st.checkbox('View Fitting Results', value=False, key='fitting_results')
        
        #st.experimental_rerun()
        
def fitting_sliders(st):
    options = {'3DCORE': {'Standard Representation': ['Inclination', 
                                       'Diameter 1 AU', 
                                       'Aspect Ratio',
                                       'Launch Radius', 
                                       'Launch Velocity', 
                                       'Expansion Rate', 
                                       'Background Drag',
                                       'Background Velocity']}}

    adjustments = 'Standard'
    magoptions = ['T_Factor','Magnetic Decay Rate','Magnetic Field Strength 1 AU']

    if st.session_state.geo_model == '3DCORE':
        gmodel = st.session_state.geo_model
        rmode = 'Standard Representation'
        sliders = options[gmodel][rmode]
        for slider in sliders:
            st.sidebar.slider(f'{slider} {sd[gmodel][slider]["unit"]}:',  
                              min_value=sd[gmodel][slider][adjustments]['min'],
                              max_value=sd[gmodel][slider][adjustments]['max'],
                              value=sd[gmodel][slider][adjustments]['def'],
                              step=sd[gmodel][slider][adjustments]['step'], key=sd[gmodel][slider]["variablename"])  
            
        for slider in magoptions:
            st.sidebar.slider(f'{slider} {msd[gmodel][slider]["unit"]}:',
                              min_value=msd[gmodel][slider][adjustments]['min'],
                              max_value=msd[gmodel][slider][adjustments]['max'],
                              value=msd[gmodel][slider][adjustments]['def'],
                              step=msd[gmodel][slider][adjustments]['step'],
                              key=msd[gmodel][slider]["variablename"])
            

def double_fitting_sliders(st):
    options = {'3DCORE': {'Standard Representation': ['Inclination',
                                                      'Diameter 1 AU',
                                                      'Aspect Ratio',
                                                      'Launch Radius',
                                                      'Launch Velocity',
                                                      'Expansion Rate',
                                                      'Background Drag',
                                                      'Background Velocity'
                                                     ]}}
    magoptions = ['T_Factor','Magnetic Decay Rate','Magnetic Field Strength 1 AU']
    
    if st.session_state.coord_system == 'HGC':
        long_val = [0., 360.]
    else:
        long_val = [0., 360.]
                                                      

    adjustments = 'Standard'

    if st.session_state.geo_model == '3DCORE':
        gmodel = st.session_state.geo_model
        rmode = 'Standard Representation'
        sliders = options[gmodel][rmode]
        range_container = st.container()
        range_container.info('Fitting might take a long time if the whole parameter space is searched. Try to limit the parameter range as much as possible.')
        longit_double = range_container.slider(f'{st.session_state.coord_system} Longitude [deg.]:',
                                               min_value=long_val[0],
                                               max_value=long_val[1],
                                               value=[long_val[0],long_val[1]],
                                               step=0.01, key='longit_double') * u.degree
        latitu_double = range_container.slider(f'{st.session_state.coord_system} Longitude [deg.]:',
                                               min_value=-90.,
                                               max_value=90.,
                                               value=[-90.,90.],
                                               step=0.01, key='latitu_double') * u.degree
        
        sliders = options[gmodel][rmode]
        for slider in sliders:
            range_container.slider(f'{slider} {sd[gmodel][slider]["unit"]}:',
                                   min_value=sd[gmodel][slider][adjustments]['min'],
                                   max_value=sd[gmodel][slider][adjustments]['max'],
                                   value=[sd[gmodel][slider][adjustments]['min'],sd[gmodel][slider][adjustments]['max']],
                                   step=sd[gmodel][slider][adjustments]['step'], key=sd[gmodel][slider]["variablename_double"])
        for slider in magoptions:
            range_container.slider(f'{slider} {msd[gmodel][slider]["unit"]}:',
                                   min_value=msd[gmodel][slider][adjustments]['min'],
                                   max_value=msd[gmodel][slider][adjustments]['max'],
                                   value=[msd[gmodel][slider][adjustments]['min'],msd[gmodel][slider][adjustments]['max']],
                                   step=msd[gmodel][slider][adjustments]['step'], key=msd[gmodel][slider]["variablename_double"])
        
        
            
def fitting_points(st):
    
    with st.sidebar.expander('Fitting'):
        st.info('Select the launch time for your event. You might want to check remote images to make an educated guess.')
        col1, col2 = st.columns(2)
        t_launch_date = col1.date_input('Launch Time:',
                              value=defaulttimer(st,-72).dateA,
                              min_value=defaulttimer(st,-(72*5)).dateA,
                              max_value=defaulttimer(st,-24).dateA)
        t_launch_time = col2.time_input('Launch Time Time',
                                   value = defaulttimer(st,-72).timeA,
                                   step = 1800,
                                   label_visibility = 'hidden')
        st.info('Select two reference points outside of the fluxrope (A before, B after). These are used to determine whether the CME hits.')
        col1, col2 = st.columns(2)
        t_s_date = col1.date_input('Reference A:',
                              value=defaulttimer(st,-8).dateA,
                              min_value=defaulttimer(st,-72).dateA,
                              max_value=defaulttimer(st,-2).dateA)

        t_s_time = col2.time_input('Reference A Time',
                                   value = defaulttimer(st,-8).timeA,
                                   step = 1800,
                                   label_visibility = 'hidden')
        
        st.session_state.dt_A = datetime.datetime.combine(t_s_date, t_s_time)
        
        t_e_date = col1.date_input('Reference B:',
                              value=defaulttimer(st,8).dateB,
                              min_value=defaulttimer(st,2).dateB,
                              max_value=defaulttimer(st,72).dateB)
        t_e_time = col2.time_input('Reference B Time',
                                   value = defaulttimer(st,8).timeB,
                                   step = 1800,
                                   label_visibility = 'hidden')
        
        st.session_state.dt_B = datetime.datetime.combine(t_e_date, t_e_time)
        
        st.info('Select a minimum of 1 or up to 5 fitting points inside of the fluxrope. These are used to determine the quality of a fit.')
        st.checkbox('t_1', value=True, key='t_1')
        if st.session_state.t_1:
            col1, col2 = st.columns(2)
            t_1_date = col1.date_input('Fitting Point 1:',
                                       value=defaulttimer(st,2).dateA,
                                       min_value=defaulttimer(st,0).dateA,
                                       max_value=defaulttimer(st,0).dateB)
            t_1_time = col2.time_input('t_1_time',
                                        value = defaulttimer(st,2).timeA,
                                        step = 1800,
                                        label_visibility = 'hidden')
            st.session_state.dt_1 = datetime.datetime.combine(t_1_date, t_1_time)

            st.checkbox('t_2', value=False, key='t_2')
            if st.session_state.t_2:
                col1, col2 = st.columns(2)
                t_2_date = col1.date_input('Fitting Point 2:',
                                           value=defaulttimer(st,4).dateA,
                                           min_value=defaulttimer(st,0).dateA,
                                           max_value=defaulttimer(st,0).dateB)
                t_2_time = col2.time_input('t_2_time',
                                            value = defaulttimer(st,4).timeA,
                                            step = 1800,
                                            label_visibility = 'hidden')
                st.session_state.dt_2 = datetime.datetime.combine(t_2_date, t_2_time)

                st.checkbox('t_3', value=False, key='t_3')
                if st.session_state.t_3:
                    col1, col2 = st.columns(2)
                    t_3_date = col1.date_input('Fitting Point 3:',
                                               value=defaulttimer(st,6).dateA,
                                               min_value=defaulttimer(st,0).dateA,
                                               max_value=defaulttimer(st,0).dateB)
                    t_3_time = col2.time_input('t_3_time',
                                                value = defaulttimer(st,6).timeA,
                                                step = 1800,
                                                label_visibility = 'hidden')
                    st.session_state.dt_3 = datetime.datetime.combine(t_3_date, t_3_time)

                    st.checkbox('t_4', value=False, key='t_4')
                    if st.session_state.t_4:
                        col1, col2 = st.columns(2)
                        t_4_date = col1.date_input('Fitting Point 4:',
                                                   value=defaulttimer(st,8).dateA,
                                                   min_value=defaulttimer(st,0).dateA,
                                                   max_value=defaulttimer(st,0).dateB)
                        t_4_time = col2.time_input('t_4_time',
                                                    value = defaulttimer(st,8).timeA,
                                                    step = 1800,
                                                    label_visibility = 'hidden')
                        st.session_state.dt_4 = datetime.datetime.combine(t_4_date, t_4_time)

                        st.checkbox('t_5', value=False, key='t_5')
                        if st.session_state.t_5:
                            col1, col2 = st.columns(2)
                            t_5_date = col1.date_input('Fitting Point 5:',
                                                       value=defaulttimer(st,10).dateA,
                                                       min_value=defaulttimer(st,0).dateA,
                                                       max_value=defaulttimer(st,0).dateB)
                            t_5_time = col2.time_input('t_5_time',
                                                        value = defaulttimer(st,10).timeA,
                                                        step = 1800,
                                                        label_visibility = 'hidden')
                            st.session_state.dt_5 = datetime.datetime.combine(t_5_date, t_5_time)
        double_fitting_sliders(st)                        
        fitting_form = st.form(key = 'fitting_form')
        fitting_form.info('Set up the fitting run. Fitting may take some time depending on the following parameters.')
        fitting_form.radio('Fitter', options=['ABC-SMC'],
                           args=[st], key='fitter')  
        fitting_form.checkbox('Multiprocessing', value=False, key='Multiprocessing')
        fitting_form.number_input('Number of Jobs', value=8, key='Nr_of_Jobs')
        fitting_form.slider('Number of Iterations',
                           min_value=1,
                           max_value=15,
                           value = [3,5],
                           step = 1)
        fitting_form.select_slider('Number of Particles',
                                   options=(265,512,1024,2048),
                                   value=512)
        fitting_form.select_slider('Ensemble Size',
                                   options = ('2**16','2**17','2**18'),
                                   value = '2**17')
        
        fitting_form.form_submit_button(label='Run fitting',
                                        on_click=fitting_main(st))