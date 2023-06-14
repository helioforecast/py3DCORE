import datetime

import numpy as np
import pandas as pds

import pickle as p

import streamlit as st



import astropy.units as u

from py3dcore.app.utils import get_catevents, load_cat, defaulttimer #, model_fittings
from py3dcore.app.config.config_sliders import sliders_dict as sd
from py3dcore.app.config.config_sliders import mag_sliders_dict as msd
from py3dcore.app.fitting import fitting_main, save_session_state

def date_and_event_selection(st):
    st.sidebar.markdown('## Date and event selection')
    col1, col2 = st.sidebar.columns(2)
    
    initialisation = st.sidebar.select_slider('How to initialize?',
                                        options=('Manual', 'Catalog', 'File'), value='Catalog')
    
    
    if initialisation == 'Catalog':
        day = st.sidebar.date_input('Select a day to process',
                          value=datetime.datetime(2022, 6, 22, 0, 0, 0),
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
    
    if initialisation == 'File':
        session_state_file = st.sidebar.file_uploader("Upload a session state file", type=["pickle","pkl","p"])

        if session_state_file is not None:
            try:
                # Load the session state from the uploaded file
                st.session_state.session_state_file = p.load(session_state_file)
                
                st.session_state.date_process = st.session_state.session_state_file['etc']['date_process']
                
                st.session_state.event_selected = load_cat(st.session_state.date_process)
                
                

                st.success("✅ Session State loaded successfully!")
                st.experimental_rerun()
            except p.UnpicklingError:
                st.error("Invalid session state file. Please upload a valid pickle file.")



def fitting_and_slider_options_container(st):
    container = st.sidebar.container()
    
    with container.expander('Options'):
        col1, col2 = st.columns(2)
        # Radio button for Coordinate System
        coord_system_options = ['HGS']  # Add other options if needed
        default_coord_system = 'HGS'  # Set a default value
        if 'session_state_file' in st.session_state and 'coord_system' in st.session_state.session_state_file['Options']:
            default_coord_system = st.session_state.session_state_file['Options']['coord_system']
        col1.radio('Coordinate System', options=coord_system_options,
                   index=coord_system_options.index(default_coord_system),
                   key='coord_system')

        # Radio button for Geometrical Model
        geo_model_options = ['3DCORE']  # Add other options if needed
        default_geo_model = '3DCORE'  # Set a default value
        if 'session_state_file' in st.session_state and 'geo_model' in st.session_state.session_state_file['Options']:
            default_geo_model = st.session_state.session_state_file['Options']['geo_model']
        col2.radio('Geometrical Model', options=geo_model_options,
                   index=geo_model_options.index(default_geo_model),
                   key='geo_model')

        # Checkbox for 3D Positions
        default_3d_positions = False
        if 'session_state_file' in st.session_state and '3d_positions' in st.session_state.session_state_file['Options']:
            default_3d_positions = st.session_state.session_state_file['Options']['3d_positions']
        st.checkbox('View 3D Positions', value=default_3d_positions, key='3d_positions')

        # Checkbox for Insitu Data
        default_insitu_data = False
        if 'session_state_file' in st.session_state and 'insitu_data' in st.session_state.session_state_file['Options']:
            default_insitu_data = st.session_state.session_state_file['Options']['insitu_data']
        st.checkbox('View Insitu Data', value=default_insitu_data, key='insitu_data')

        # Checkbox for Remote Imaging
        default_remote_imaging = False
        if 'session_state_file' in st.session_state and 'remote_imaging' in st.session_state.session_state_file['Options']:
            default_remote_imaging = st.session_state.session_state_file['Options']['remote_imaging']
        st.checkbox('View Remote Imaging', value=default_remote_imaging, key='remote_imaging')

        # Checkbox for Fitting Results
        default_fitting_results = False
        if 'session_state_file' in st.session_state and 'fitting_results' in st.session_state.session_state_file['Options']:
            default_fitting_results = st.session_state.session_state_file['Options']['fitting_results']
        st.checkbox('View Fitting Results', value=default_fitting_results, key='fitting_results')
        
        # Checkbox for Parameter Distribution
        default_parameter_distribution = False
        if 'session_state_file' in st.session_state and 'parameter_distribution' in st.session_state.session_state_file['Options']:
            default_parameter_distribution = st.session_state.session_state_file['Options']['parameter_distribution']
        st.checkbox('View Parameter Distribution', value=default_parameter_distribution, key='parameter_distribution')
        
        # Create the save button
        if st.button("Save Session State"):
            save_session_state(st)
    
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
        
        #print(st.session_state)
        for slider in sliders:
            
            if 'session_state_file' in st.session_state and sd[gmodel][slider]["variablename"] in st.session_state.session_state_file['Params']:
                slider_value = st.session_state.session_state_file['Params'][sd[gmodel][slider]["variablename"]]
            else:
                slider_value = sd[gmodel][slider][adjustments]['def']
            if 'selected_row' in st.session_state:
                
                slider_value = float(st.session_state.selected_row[slider])

            try:
                st.sidebar.slider(f'{slider} {sd[gmodel][slider]["unit"]}:',  
                                  min_value=sd[gmodel][slider][adjustments]['min'],
                                  max_value=sd[gmodel][slider][adjustments]['max'],
                                  value=slider_value,
                                  step=sd[gmodel][slider][adjustments]['step'], key=sd[gmodel][slider]["variablename"])  
            except:
                st.sidebar.slider(f'{slider} {sd[gmodel][slider]["unit"]}:',  
                                  min_value=sd[gmodel][slider][adjustments]['min'],
                                  max_value=sd[gmodel][slider][adjustments]['max'],
                                  value=int(slider_value),
                                  step=sd[gmodel][slider][adjustments]['step'], key=sd[gmodel][slider]["variablename"])  
            
        for slider in magoptions:
            if 'session_state_file' in st.session_state and slider in st.session_state.session_state_file['Params']:
                slider_value = st.session_state.session_state_file['Params'][slider]
            else:
                slider_value = msd[gmodel][slider][adjustments]['def']
            if 'selected_row' in st.session_state:
                slider_value = float(st.session_state.selected_row[slider])
            try:
                st.sidebar.slider(f'{slider} {msd[gmodel][slider]["unit"]}:',
                                  min_value=msd[gmodel][slider][adjustments]['min'],
                                  max_value=msd[gmodel][slider][adjustments]['max'],
                                  value=slider_value,
                                  step=msd[gmodel][slider][adjustments]['step'],
                                  key=msd[gmodel][slider]["variablename"])
            except:
                st.sidebar.slider(f'{slider} {msd[gmodel][slider]["unit"]}:',
                                  min_value=msd[gmodel][slider][adjustments]['min'],
                                  max_value=msd[gmodel][slider][adjustments]['max'],
                                  value=int(slider_value),
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
        
        # Longitude Slider
        default_longit_double = [long_val[0], long_val[1]]
        if 'session_state_file' in st.session_state and 'longit_double' in st.session_state.session_state_file['Fitting']:
            default_longit_double = st.session_state.session_state_file['Fitting']['longit_double']
        longit_double = range_container.slider(f'{st.session_state.coord_system} Longitude [deg.]:',
                                               min_value=long_val[0],
                                               max_value=long_val[1],
                                               value=default_longit_double,
                                               step=0.01, key='longit_double') * u.degree

        
        
        # Latitude Slider
        default_latitu_double = [-90., 90.]
        if 'session_state_file' in st.session_state and 'latitu_double' in st.session_state.session_state_file['Fitting']:
            default_latitu_double = st.session_state.session_state_file['Fitting']['latitu_double']
        latitu_double = range_container.slider(f'{st.session_state.coord_system} Latitude [deg.]:',
                                               min_value=-90.,
                                               max_value=90.,
                                               value=default_latitu_double,
                                               step=0.01, key='latitu_double') * u.degree
        
        fixedsliders = ['Launch Radius', 'Expansion Rate', 'Magnetic Decay Rate']

        for slider in sliders:
            if slider in fixedsliders:
                default_value = [sd[gmodel][slider][adjustments]['def'], sd[gmodel][slider][adjustments]['def']]
            else:
                default_value = [sd[gmodel][slider][adjustments]['min'], sd[gmodel][slider][adjustments]['max']]

            if 'session_state_file' in st.session_state and sd[gmodel][slider]["variablename_double"] in st.session_state.session_state_file['Fitting']:
                default_value = st.session_state.session_state_file['Fitting'][sd[gmodel][slider]["variablename_double"]]

            range_container.slider(f'{slider} {sd[gmodel][slider]["unit"]}:',
                                   min_value=sd[gmodel][slider][adjustments]['min'],
                                   max_value=sd[gmodel][slider][adjustments]['max'],
                                   value=default_value,
                                   step=sd[gmodel][slider][adjustments]['step'],
                                   key=sd[gmodel][slider]["variablename_double"])

        for slider in magoptions:
            if slider in fixedsliders:
                default_value = [msd[gmodel][slider][adjustments]['def'], msd[gmodel][slider][adjustments]['def']]
            else:
                default_value = [msd[gmodel][slider][adjustments]['min'], msd[gmodel][slider][adjustments]['max']]
                
            if 'session_state_file' in st.session_state and msd[gmodel][slider]["variablename_double"] in st.session_state.session_state_file['Fitting']:
                
                default_value = st.session_state.session_state_file['Fitting'][msd[gmodel][slider]["variablename_double"]]
            range_container.slider(f'{slider} {msd[gmodel][slider]["unit"]}:',
                                       min_value=msd[gmodel][slider][adjustments]['min'],
                                       max_value=msd[gmodel][slider][adjustments]['max'],
                                       value=default_value,
                                       step=msd[gmodel][slider][adjustments]['step'], 
                                       key=msd[gmodel][slider]["variablename_double"])
        
        
            
def fitting_points(st):
    
    with st.sidebar.expander('Fitting'):
        
        st.session_state.fitting_datetimes = []
        
        st.info('Select the launch time for your event. You might want to check remote images to make an educated guess.')
        col1, col2 = st.columns(2)
        
        default_launch_date = defaulttimer(st, -72).dateA
        if 'session_state_file' in st.session_state and 'dt_launch' in st.session_state.session_state_file['Fitting']:
            default_launch_date = st.session_state.session_state_file['Fitting']['dt_launch'].date()
        
        t_launch_date = col1.date_input('Launch Time:',
                              value=default_launch_date,
                              min_value=defaulttimer(st,-(72*5)).dateA,
                              max_value=defaulttimer(st,-24).dateA)
        
        default_launch_time = defaulttimer(st, -72).timeA
        if 'session_state_file' in st.session_state and 'dt_launch' in st.session_state.session_state_file['Fitting']:
            default_launch_time = st.session_state.session_state_file['Fitting']['dt_launch'].time()
        
        t_launch_time = col2.time_input('Launch Time Time',
                                   value = default_launch_time,
                                   step = 1800,
                                   label_visibility = 'hidden')
        st.session_state.dt_launch = datetime.datetime.combine(t_launch_date, t_launch_time)
        
        # Reference Points A and B
        st.info('Select two reference points outside of the fluxrope (A before, B after). These are used to determine whether the CME hits.')
        col1, col2 = st.columns(2)
        default_refA_date = defaulttimer(st, -8).dateA
        if 'session_state_file' in st.session_state and 'dt_A' in st.session_state.session_state_file['Fitting']:
            default_refA_date = st.session_state.session_state_file['Fitting']['dt_A'].date()
        t_s_date = col1.date_input('Reference A:', value=default_refA_date, min_value=defaulttimer(st, -72).dateA, max_value=defaulttimer(st, -2).dateA)

        default_refA_time = defaulttimer(st, -8).timeA
        if 'session_state_file' in st.session_state and 'dt_A' in st.session_state.session_state_file['Fitting']:
            default_refA_time = st.session_state.session_state_file['Fitting']['dt_A'].time()
        t_s_time = col2.time_input('Reference A Time', value=default_refA_time, step=1800, label_visibility='hidden')
        st.session_state.dt_A = datetime.datetime.combine(t_s_date, t_s_time)

        default_refB_date = defaulttimer(st, 8).dateB
        if 'session_state_file' in st.session_state and 'dt_B' in st.session_state.session_state_file['Fitting']:
            default_refB_date = st.session_state.session_state_file['Fitting']['dt_B'].date()
        t_e_date = col1.date_input('Reference B:', value=default_refB_date, min_value=defaulttimer(st, 2).dateB, max_value=defaulttimer(st, 72).dateB)
        t_e_time = col2.time_input('Reference B Time', value=defaulttimer(st, 8).timeB, step=1800, label_visibility='hidden')
        st.session_state.dt_B = datetime.datetime.combine(t_e_date, t_e_time)

        
        # Fitting Points
        st.info('Select a minimum of 2 or up to 5 fitting points inside of the fluxrope. These are used to determine the quality of a fit.')
        st.checkbox('t_1', value=True, key='t_1')
        if st.session_state.t_1:
            col1, col2 = st.columns(2)
            default_fit1_date = defaulttimer(st, 2).dateA
            if 'session_state_file' in st.session_state and 'dt_1' in st.session_state.session_state_file['Fitting']:
                default_fit1_date = st.session_state.session_state_file['Fitting']['dt_1'].date()
            t_1_date = col1.date_input('Fitting Point 1:', value=default_fit1_date, min_value=defaulttimer(st, 0).dateA, max_value=defaulttimer(st, 0).dateB)
            t_1_time = col2.time_input('t_1_time', value=defaulttimer(st, 2).timeA, step=1800, label_visibility='hidden')
            st.session_state.dt_1 = datetime.datetime.combine(t_1_date, t_1_time)
            st.session_state.fitting_datetimes.append(st.session_state.dt_1)

            st.checkbox('t_2', value=True, key='t_2')
            if st.session_state.t_2:
                col1, col2 = st.columns(2)
                default_fit2_date = defaulttimer(st, 4).dateA
                if 'session_state_file' in st.session_state and 'dt_2' in st.session_state.session_state_file['Fitting']:
                    default_fit2_date = st.session_state.session_state_file['Fitting']['dt_2'].date()
                t_2_date = col1.date_input('Fitting Point 2:', value=default_fit2_date, min_value=defaulttimer(st, 0).dateA, max_value=defaulttimer(st, 0).dateB)
                t_2_time = col2.time_input('t_2_time', value=defaulttimer(st, 4).timeA, step=1800, label_visibility='hidden')
                st.session_state.dt_2 = datetime.datetime.combine(t_2_date, t_2_time)
                st.session_state.fitting_datetimes.append(st.session_state.dt_2)

                st.checkbox('t_3', value=False, key='t_3')
                if st.session_state.t_3:
                    col1, col2 = st.columns(2)
                    default_fit3_date = defaulttimer(st, 6).dateA
                    if 'session_state_file' in st.session_state and 'dt_3' in st.session_state.session_state_file['Fitting']:
                        default_fit3_date = st.session_state.session_state_file['Fitting']['dt_3'].date()
                    t_3_date = col1.date_input('Fitting Point 3:', value=default_fit3_date, min_value=defaulttimer(st, 0).dateA, max_value=defaulttimer(st, 0).dateB)
                    t_3_time = col2.time_input('t_3_time', value=defaulttimer(st, 6).timeA, step=1800, label_visibility='hidden')
                    st.session_state.dt_3 = datetime.datetime.combine(t_3_date, t_3_time)
                    st.session_state.fitting_datetimes.append(st.session_state.dt_3)

                    st.checkbox('t_4', value=False, key='t_4')
                    if st.session_state.t_4:
                        col1, col2 = st.columns(2)
                        default_fit4_date = defaulttimer(st, 8).dateA
                        if 'session_state_file' in st.session_state and 'dt_4' in st.session_state.session_state_file['Fitting']:
                            default_fit4_date = st.session_state.session_state_file['Fitting']['dt_4'].date()
                        t_4_date = col1.date_input('Fitting Point 4:', value=default_fit4_date, min_value=defaulttimer(st, 0).dateA, max_value=defaulttimer(st, 0).dateB)
                        t_4_time = col2.time_input('t_4_time', value=defaulttimer(st, 8).timeA, step=1800, label_visibility='hidden')
                        st.session_state.dt_4 = datetime.datetime.combine(t_4_date, t_4_time)
                        st.session_state.fitting_datetimes.append(st.session_state.dt_4)

                        st.checkbox('t_5', value=False, key='t_5')
                        if st.session_state.t_5:
                            col1, col2 = st.columns(2)
                            default_fit5_date = defaulttimer(st, 10).dateA
                            if 'session_state_file' in st.session_state and 'dt_5' in st.session_state.session_state_file['Fitting']:
                                default_fit5_date = st.session_state.session_state_file['Fitting']['dt_5'].date()
                            t_5_date = col1.date_input('Fitting Point 5:', value=default_fit5_date, min_value=defaulttimer(st, 0).dateA, max_value=defaulttimer(st, 0).dateB)
                            t_5_time = col2.time_input('t_5_time', value=defaulttimer(st, 10).timeA, step=1800, label_visibility='hidden')
                            st.session_state.dt_5 = datetime.datetime.combine(t_5_date, t_5_time)
                            st.session_state.fitting_datetimes.append(st.session_state.dt_5)

        double_fitting_sliders(st)
        
        # Fitting Form
        fitting_form = st.form(key='fitting_form')
        fitting_form.info('Set up the fitting run. Fitting may take some time depending on the following parameters.')
        default_fitter = 'ABC-SMC'
        if 'session_state_file' in st.session_state and 'fitter' in st.session_state.session_state_file['Fitting']:
            default_fitter = st.session_state.session_state_file['Fitting']['fitter']
        fitting_form.radio('Fitter', options=['ABC-SMC'], args=[st], key='fitter')

        default_multiprocessing = False
        if 'session_state_file' in st.session_state and 'Multiprocessing' in st.session_state.session_state_file['Fitting']:
            default_multiprocessing = st.session_state.session_state_file['Fitting']['Multiprocessing']
        fitting_form.checkbox('Multiprocessing', value=default_multiprocessing, key='Multiprocessing')

        default_nr_of_jobs = 8
        if 'session_state_file' in st.session_state and 'Nr_of_Jobs' in st.session_state.session_state_file['Fitting']:
            default_nr_of_jobs = st.session_state.session_state_file['Fitting']['Nr_of_Jobs']
        fitting_form.number_input('Number of Jobs', value=default_nr_of_jobs, key='Nr_of_Jobs')

        default_iter = [3, 5]
        if 'session_state_file' in st.session_state and 'iter' in st.session_state.session_state_file['Fitting']:
            default_iter = st.session_state.session_state_file['Fitting']['iter']
        fitting_form.slider('Number of Iterations', min_value=1, max_value=15, value=default_iter, step=1, key='iter')

        default_n_particles = 512
        if 'session_state_file' in st.session_state and 'n_particles' in st.session_state.session_state_file['Fitting']:
            default_n_particles = st.session_state.session_state_file['Fitting']['n_particles']
        fitting_form.select_slider('Number of Particles', options=(265, 512, 1024, 2048), value=default_n_particles, key='n_particles')

        default_ensemble_size = '2**17'
        if 'session_state_file' in st.session_state and 'ensemble_size' in st.session_state.session_state_file['Fitting']:
            default_ensemble_size = st.session_state.session_state_file['Fitting']['ensemble_size']
        fitting_form.select_slider('Ensemble Size', options=('2**15', '2**16', '2**17', '2**18'), value=default_ensemble_size, key='ensemble_size')
        
        if 'session_state_file' in st.session_state and 'filename' in st.session_state.session_state_file['Fitting']:
            filename_value = st.session_state.session_state_file['Fitting']['filename']
        else:
            filename_value = st.session_state.event_selected

        fitting_form.text_input("Enter a filename", value=filename_value, key='filename')
        form_submitted = False
        submit_button = fitting_form.form_submit_button(label='Run fitting')
        
# Check if form button is clicked
    if submit_button:
        form_submitted = True
    if form_submitted:
        fitting_main(st)
        form_submitted = False

        
