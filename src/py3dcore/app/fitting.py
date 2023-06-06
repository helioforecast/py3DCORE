'''
Fitting
'''

import shutil as shutil
import streamlit as st
import os

import datetime

import pickle as p

import py3dcore
from py3dcore.app.utils import get_fitobserver


def fitting_main(st):
    
    t_launch = st.session_state.dt_launch
    
    t_s = st.session_state.dt_A
    t_e = st.session_state.dt_B
    
    t_fit = st.session_state.fitting_datetimes
    
    if st.session_state.ensemble_size == '2**16':
        ensemble_size = int(2**16)
    elif st.session_state.ensemble_size == '2**17':
        ensemble_size = int(2**17)
    elif st.session_state.ensemble_size == '2**18':
        ensemble_size = int(2**18)
        
        
    model_kwargs = {
        "ensemble_size": ensemble_size, #2**17
        "iparams": {
            "cme_longitude": {
                "maximum": st.session_state.longit_double[1],
                "minimum": st.session_state.longit_double[0]
            },
            "cme_latitude": {
                "maximum": st.session_state.latitu_double[1],
                "minimum": st.session_state.latitu_double[0]
            },
            "cme_inclination": {
                "distribution": "uniform",
                "maximum": st.session_state.inc_double[1],
                "minimum": st.session_state.inc_double[0]
            },
            "cme_diameter_1au": {
                "maximum": st.session_state.dia_double[1],
                "minimum": st.session_state.dia_double[0]
            },
            "cme_aspect_ratio": {
                "maximum": st.session_state.asp_double[1],
                "minimum": st.session_state.asp_double[0]
            },
            "cme_launch_radius": {
                "distribution": "uniform",
                "maximum": st.session_state.l_rad_double[1],
                "minimum": st.session_state.l_rad_double[0]
            },
            "cme_launch_velocity": {
                "maximum": st.session_state.l_vel_double[1],
                "minimum": st.session_state.l_vel_double[0]
            },
            "t_factor": {
                "maximum": st.session_state.t_fac_double[1],
                "minimum": st.session_state.t_fac_double[0]
            },
            "cme_expansion_rate": {
                "distribution": "uniform",
                "maximum": st.session_state.exp_rat_double[1],
                "minimum": st.session_state.exp_rat_double[0]
            },
            "magnetic_decay_rate": {
                "distribution": "uniform",
                "maximum": st.session_state.mag_dec_double[1],
                "minimum": st.session_state.mag_dec_double[0]
            },
            "magnetic_field_strength_1au": {
                "maximum": st.session_state.mag_strength_double[1],
                "minimum": st.session_state.mag_strength_double[0]
            },
            "background_drag": {
                "distribution": "uniform",
                "maximum": st.session_state.b_drag_double[1],
                "minimum": st.session_state.b_drag_double[0]
            },
            "background_velocity": {
                "distribution": "uniform",
                "maximum": st.session_state.bg_vel_double[1],
                "minimum": st.session_state.bg_vel_double[0]
            }
        }
    }
    
    for param, values in model_kwargs["iparams"].items():
        if values["maximum"] == values["minimum"]:
            values["distribution"] = "fixed"
            values["default_value"] = values["minimum"]
            del values["maximum"]
            del values["minimum"]
            
    output = st.session_state.filename
    
    # Deleting a non-empty folder
    try:
        shutil.rmtree('output/'+output, ignore_errors=True)
        logger.info("Successfully cleaned %s" , output)
    except:
        pass
    
    fitobserver, fit_coord_system = get_fitobserver(st.session_state.mag_coord_system, st.session_state.event_selected.sc)
    st.session_state.fitholder = st.empty()
    if st.session_state.fitter == 'ABC-SMC':
        fitter = py3dcore.streamlit_ABC_SMC()
        fitter.initialize(t_launch, py3dcore.ToroidalModel, model_kwargs)
        fitter.add_observer(fitobserver, t_fit, t_s, t_e)
        
        fitter.run(st = st,
                   iter_min = st.session_state.iter[0], 
                   iter_max = st.session_state.iter[1],
                   ensemble_size=st.session_state.n_particles, 
                   reference_frame=fit_coord_system, 
                   jobs=st.session_state.Nr_of_Jobs, 
                   workers=st.session_state.Nr_of_Jobs, 
                   sampling_freq=3600, 
                   output=output, 
                   noise_model="psd",
                   use_multiprocessing=st.session_state.Multiprocessing)
        
        st.session_state.model_fittings = True
        fitprocesscontainer = st.session_state.fitholder.container()
        fitprocesscontainer.success("✅ Reached maximum number of iterations")
        
    save_session_state(st,model_fittings = True)
    
    st.session_state.placeholder.success("✅ Session State saved to output folder!")
    
    return


        
# Function to save the session state
def save_session_state(st, model_fittings=False):
    
    output = str(st.session_state.event_selected)
    
    # Define the path to the session state file
    session_state_file = 'output/session_states/' + output + "_session_state.pkl"
    
    possible = {
        'Options':{
            'coord_system',
            'geo_model',
            '3d_positions',
            'insitu_data',
            'remote_imaging',
            'fitting_results',
            'parameter_distribution'
        },
        'Params':{
            'longit',
            'latitu',
            'inc',
            'dia',
            'asp',
            'l_rad',
            'l_vel',
            'exp_rat',
            'b_drag',
            'bg_vel',
            't_fac',
            'mag_dec',
            'mag_strength',
        },
        'Download_Options': {
            'insitu_list',
            'mag_coord_system',
            'insitu_time_before',
            'insitu_time_after'            
        },
        'Imaging_Options': {
            'view_legend_insitu',
            'view_catalog_insitu',
            'view_fitting_points',
            'view_fitting_results',
            'view_synthetic_insitu'                 
        },
        'Fitting': {
            'dt_launch',
            'dt_A',
            'dt_B',
            't_1',
            'dt_1',
            't_2',
            'dt_2',
            't_3',
            'dt_3',
            't_4',
            'dt_4',
            't_5',
            'dt_5',
            'fitting_datetimes',
            'longit_double',
            'latitu_double',
            'inc_double',
            'dia_double',
            'asp_double',
            'l_rad_double',
            'l_vel_double',
            'exp_rat_double',
            'b_drag_double',
            'bg_vel_double',
            't_fac_double',
            'mag_dec_double',
            'mag_strength_double',
            'fitter',
            'Multiprocessing',
            'Nr_of_Jobs',
            'iter',
            'n_particles',
            'ensemble_size',
            'filename',
            
        },
    
        'etc': {
            'model_fittings'
            't_data',
            'date_process',
            'insituplot',
            'b_data',
            'insituend',
            #'event_selected',
            'insitubegin',}
    }
    
    selected_items = ['Options', 'Params', 'Download_Options', 'Imaging_Options', 'Fitting', 'etc']
    
    session_state_dict = {}
    
    for item in selected_items:
        if item in possible:
            session_state_dict[item] = {}
            for key in possible[item]:
                if key in st.session_state:
                    session_state_dict[item][key] = st.session_state[key]
    if model_fittings == True:
        session_state_dict['etc']['model_fittings']= True
                    
                    
    # Save the dictionary as a file
    with open(session_state_file, 'wb') as file:
        p.dump(session_state_dict, file)
    
    st.success("✅ Session State saved to output folder!")
    print(session_state_dict)
    return 