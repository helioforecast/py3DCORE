'''
Fitting
'''

import shutil as shutil
import streamlit as st
import os

import pickle

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
    
    # Define the path to the session state file
    session_state_file = 'output/session_states/' + output + "_session_state.pkl"
    
    with open(session_state_file, "wb") as file:
        pickle.dump(st.session_state, file)
    
    st.session_state.placeholder.success("✅ Session State saved to output folder!")
    
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
    
    return