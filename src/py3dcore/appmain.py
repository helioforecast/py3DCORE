import streamlit as st
import datetime

import time

import astropy.units as u

from py3dcore import __version__ as version
from py3dcore.app.config import app_styles, selected_imagers #, config_sliders, selected_bodies
from py3dcore.app.modules import date_and_event_selection, fitting_and_slider_options_container, fitting_sliders, fitting_points #, final_parameters_gmodel, fitting_sliders, maps_clims
from py3dcore.app.utils import get_insitudata
from py3dcore.app.fitting import fitting_main
from py3dcore.app.plotting import *

import copy


import logging

logging.basicConfig(level=logging.INFO)
logging.getLogger("heliosat.spice").setLevel("WARNING")
logging.getLogger("heliosat.spacecraft").setLevel("WARNING")

logger = logging.getLogger(__name__)

def delete_from_state(vars):
    for var in vars:
        if var in st.session_state:
            del st.session_state[var]

def footer_text():
    st.subheader('About this application:')
    
    right, left = st.columns((1, 1))
    
    right.markdown("""
                   _3DCOREweb_  is an open-source software package based on 3DCORE, that can be used to
                   reconstruct the 3D structure of Coronal Mass Ejections (CMEs) and create synthetic 
                   insitu signatures. It can be fitted to insitu data from several spacecraft using an
                   Approximate Bayesian Computation Sequential Monte Carlo (ABC-SCM) algorithm, model 
                   their kinematics and compare to remote-sensing observations.
                   The 3DCORE model assumes an empirically motivated torus-like flux rope structure that 
                   expands self-similarly within the heliosphere, is influenced by a simplified interaction 
                   with the solar wind environment, and carries along an embedded analytical magnetic 
                   field. The tool also implements remote-sensing observations from multiple viewpoints 
                   such as the SOlar and Heliospheric Observatory (SOHO) and Solar Terrestrial Relations
                   Observatory (STEREO).
                """)
    right.markdown("""
                   **Github**: Find the latest version here
                               [![GitHub](https://img.shields.io/badge/github-%23121011.svg?style=for-the-badge&logo=github&logoColor=white)](https://github.com/helioforecast/py3DCORE) \n
                   **Paper**: Find the original Paper [![https://doi.org/10.48550/arXiv.2009.00327](http://img.shields.io/badge/astro-ph.SR-arXiv%2009.00327-B31B1B.svg)](https://doi.org/10.48550/arXiv.2009.00327) \n
                   """ #+
                   #f"""
#                   **Version**: {version} (latest release [![Version](https://img.shields.io/github/v/release/AthKouloumvakos/PyThea)](https://github.com/AthKouloumvakos/PyThea/releases))
                   #"""
                  )
    left.image('app/config/3dcore.jpeg')
    st.markdown('---')

def run():
    
    #############################################################
    # set page config
    st.set_page_config(page_title='py3DCORE', page_icon = ':satellite:', 
                       initial_sidebar_state='expanded', layout="wide") 
    
    # Possible other choices: # :rocket:, :sun:, :ringed_planet:, :star:, :telescope:, 
    
    #############################################################
    # HTML Styles
    app_styles.apply(st)
    
    #############################################################
    # Main page information text
    st.title('3DCOREweb: Reconstruct CMEs using the "3D Coronal Rope Ejection Model"')
    
    #############################################################
    # Startup Variables
    #if 'startup' not in st.session_state:
    #    st.session_state.startup = {'fitting': True}
        
    #############################################################
    # Date and Event selection
    
    form_submitted = False

    if 'date_process' not in st.session_state:
        date_and_event_selection(st)
        st.markdown(""" 
                   First choose how to initialize the tool: \n
                   **Manual:** \n
                   >_Choose a launch time for your event and start from scratch._ \n
                   **Catalog:** \n
                   >_Choose an event from the [helioforecast catalog](https://helioforecast.space/icmecat)._ \n
                   **File:**\n
                   >_Load from a previous fitting file._
                """)
    else:
        st.sidebar.markdown('## Processing Event:')
        st.sidebar.info(f'{st.session_state.event_selected}')

    if 'date_process' not in st.session_state:
        st.markdown('---')
        footer_text()
        st.stop()
    else:
        st.session_state.placeholder = st.empty()
        st.session_state.placeholder.markdown(""" 
                   You have several options to analyze the chosen event: \n
                   **View 2D Positions:** \n
                   >_Take a look at planet and spacecraft positions in 2D._ \n
                   **View 3D Positions:** \n
                   >_Take a look at planet and spacecraft positions and model the 3D shape of the CME._ \n
                   **View Insitu Data:** \n
                   >_Look at the insitu data measured by a specific observer, generate synthetic insitu data or fit the model numerically._ \n
                   **View Remote Imaging:**\n
                   >_Load images from various spacecraft and fit the 3D shape to them._ \n
                   **View Fitting Results:**\n
                   >_Scroll through the results of a fitting run._\n
                   **View Parameter Distribution:**\n
                   >_Visualize the parameter distributions of a fitting run._
                """)    
    
    st.sidebar.markdown('## 3D Fitting and Reconstruction')
    
    st.markdown('---')
    
    fitting_and_slider_options_container(st)
    
    if st.session_state.coord_system == 'HGC':
        long_val = [0., 360.]
    else:
        long_val = [-180., 180.]
    
    if 'session_state_file' in st.session_state and 'longit' in st.session_state.session_state_file['Params']:
        slider_value = st.session_state.session_state_file['Params']['longit']
    else:
        slider_value = 0.
    if 'selected_row' in st.session_state:
        slider_value = float(st.session_state.selected_row['Longitude'])
    longit = st.sidebar.slider(f'{st.session_state.coord_system} \
                               Longitude [deg.]:',
                               min_value=long_val[0],
                               max_value=long_val[1],
                               value=slider_value,
                               step=0.01, key='longit') * u.degree
    
    if 'session_state_file' in st.session_state and 'latitu' in st.session_state.session_state_file['Params']:
        slider_value = st.session_state.session_state_file['Params']['latitu']
    else:
        slider_value = 0.
    if 'selected_row' in st.session_state:
        slider_value = float(st.session_state.selected_row['Latitude'])
    latitu = st.sidebar.slider(f'{st.session_state.coord_system} \
                               Latitude [deg.]:',
                               min_value=-90.,
                               max_value=90.,
                               value=slider_value,
                               step=0.01, key='latitu') * u.degree
    
    fitting_sliders(st)
    

        
        
        
        #############################################################
    # 2D Positions
    
    # View the menus
    if st.session_state.twod_positions:
        st.sidebar.markdown('---')
        st.sidebar.markdown('## 2D Positions')
        with st.sidebar.expander('Download Options'):
            select_2d_form = st.form(key='select_2d_form')
            if 'event_selected' in st.session_state:
                default_observer = [st.session_state.event_selected.sc]
            else:
                default_observer = []
            body_list_2d = select_2d_form.multiselect('Select Bodies',
                                                   options=selected_imagers.planets_dict.keys(),
                                                   default=['Sun', 'Earth'],
                                                   key='body_list_2d')
            sc_list_2d = select_2d_form.multiselect('Select Spacecraft',
                                                   options=selected_imagers.bodies_dict.keys(),
                                                   default=default_observer,
                                                   key='sc_list_2d')
            
            select_2d_form.info('Select how many days should be plotted before/after the event.')
            col1, col2 = select_2d_form.columns(2)
            
            col1.slider('before [days]',
                        min_value=-30, max_value=-1,
                        value=-5, step=1,
                        key='twod_time_before')
            col2.slider('after [days]',
                        min_value=1, max_value=30,
                        value=5, step=1,
                        key='twod_time_after')
            
            st.session_state.twobegin = st.session_state.event_selected.begin + datetime.timedelta(days = st.session_state.twod_time_before)
            
            st.session_state.twodend = st.session_state.event_selected.end + datetime.timedelta(days = st.session_state.twod_time_after)
            
            select_2d_form.form_submit_button(label='Submit',
                                                  on_click=delete_from_state,
                                                  kwargs={'vars': ['twod_plot_base', ]})
            
        with st.sidebar.expander('Imaging Options'):            
            
            # Create a Streamlit dropdown widget to select the width
            width_options = ['80%', '100%', '120%', '140%', '160%', '180%', '200%']
            st.selectbox('Select Image Width', width_options, index=1, key='selected_width_2d')
            st.checkbox('View Legend', value=False, key='view_legend_2d')
            st.checkbox('View Trajectory', value=False, key='view_trajectory_2d')
            st.checkbox('View Catalog Events', value=False, key='view_catalog_2')

    #############################################################
    # Observer and 3D Positions
    
    # View the menus
    if st.session_state.threed_positions:
        st.sidebar.markdown('---')
        st.sidebar.markdown('## 3D Positions')
        with st.sidebar.expander('Download Options'):
            select_3d_form = st.form(key='select_3d_form')
            if 'event_selected' in st.session_state:
                default_observer = [st.session_state.event_selected.sc]
            else:
                default_observer = []
            body_list_3d = select_3d_form.multiselect('Select Bodies',
                                                   options=selected_imagers.planets_dict.keys(),
                                                   default=['Sun', 'Earth'],
                                                   key='body_list_3d')
            sc_list_3d = select_3d_form.multiselect('Select Spacecraft',
                                                   options=selected_imagers.bodies_dict.keys(),
                                                   default=default_observer,
                                                   key='sc_list_3d')
            
            select_3d_form.info('Select how many days should be plotted before/after the event.')
            col1, col2 = select_3d_form.columns(2)
            
            col1.slider('before [days]',
                        min_value=-6., max_value=-0.5,
                        value=-1., step=0.5,
                        key='threed_time_before')
            col2.slider('after [days]',
                        min_value=0.5, max_value=6.,
                        value=1., step=0.5,
                        key='threed_time_after')
            
            st.session_state.threedbegin = st.session_state.event_selected.begin + datetime.timedelta(days = st.session_state.threed_time_before)
            
            st.session_state.threedend = st.session_state.event_selected.end + datetime.timedelta(days = st.session_state.threed_time_after)
            
            select_3d_form.form_submit_button(label='Submit',
                                                  on_click=delete_from_state,
                                                  kwargs={'vars': ['threed_data', ]})
            
        with st.sidebar.expander('Imaging Options'):            
            
            # Create a Streamlit dropdown widget to select the width
            width_options = ['80%', '100%', '120%', '140%', '160%', '180%', '200%']
            st.selectbox('Select Image Width', width_options, index=1, key='selected_width_3d')
            st.checkbox('View Legend', value=False, key='view_legend_3d')
            st.checkbox('View Trajectory', value=False, key='view_trajectory_3d')
            st.checkbox('View Fluxrope', value=False, key='view_fluxrope')
            
            st.markdown('Show Bodies:')
            for i in st.session_state.body_list_3d:
                st.checkbox(i, value=False, key=i)
            for i in st.session_state.sc_list_3d:
                st.checkbox(i, value=False, key=i)
        
        st.session_state.twodplot = plot_2d_pos(st)

        
        #############################################################
    # Insitu Data
    
    # View the fittings table
    if st.session_state.insitu_data:
        st.sidebar.markdown('---')
        st.sidebar.markdown('## Insitu Data')
        with st.sidebar.expander('Download Options'):
            select_insitu_form = st.form(key='select_insitu_form')
            if 'event_selected' in st.session_state:
                default_observer = [st.session_state.event_selected.sc]
            else:
                default_observer = []
            insitu_list = select_insitu_form.multiselect('Select Additional Insitu Observers',
                                                          options=selected_imagers.insitu_dict,
                                                          default=default_observer,
                                                          key='insitu_list')
            select_insitu_form.radio('Magnetic Coordinate System', options=['HEEQ', 'RTN'], 
                     args=[st], key='mag_coord_system', index = 1)
            
            select_insitu_form.info('Select how many days should be plotted before/after the event.')
            col1, col2 = select_insitu_form.columns(2)
            
            col1.slider('before [days]',
                        min_value=-6., max_value=-0.5,
                        value=-1., step=0.5,
                        key='insitu_time_before')
            col2.slider('after [days]',
                        min_value=0.5, max_value=6.,
                        value=1., step=0.5,
                        key='insitu_time_after')
            
            st.session_state.insitubegin = st.session_state.event_selected.begin + datetime.timedelta(days = st.session_state.insitu_time_before)
            
            st.session_state.insituend = st.session_state.event_selected.end + datetime.timedelta(days = st.session_state.insitu_time_after)
            
            select_insitu_form.form_submit_button(label='Submit',
                                                  on_click=delete_from_state,
                                                  kwargs={'vars': ['b_data', ]})
            
        with st.sidebar.expander('Imaging Options'):
            # Create a Streamlit dropdown widget to select the width
            plottinglib = st.select_slider('Switch to matplotlib to speed-up!',
                                        options=('plotly', 'matplotlib'), value='plotly', key = "plottinglib", disabled = True, label_visibility="collapsed")
            
            width_options = ['80%', '100%', '120%', '140%', '160%', '180%', '200%']
            st.selectbox('Select Image Width', width_options, index=1, key='selected_width')
            st.checkbox('View Legend', value=False, key='view_legend_insitu')
            st.checkbox('View Catalog Event', value=False, key='view_catalog_insitu')
            st.checkbox('View Fitting Points', value=False, key='view_fitting_points')
            st.checkbox('View Fitting Results', value=False, key='view_fitting_results')
            st.checkbox('View Synthetic Insitu Data', value=False, key='view_synthetic_insitu')
            
            
            
        ## insitu plots
        
        st.markdown('### Insitu Data')
        
        st.session_state.insitucontainer = st.session_state.placeholder.container()
        
        if 'b_data' not in st.session_state:
            ph = st.session_state.insitucontainer.empty()
            ph.info("⏳ Downloading Insitu Data...")

            try:
                st.session_state.b_data, st.session_state.t_data, st.session_state.pos_data = get_insitudata(st.session_state.mag_coord_system, st.session_state.event_selected.sc, st.session_state.insitubegin, st.session_state.insituend)
                ph.success("✅ Successfully downloaded " + st.session_state.event_selected.sc + " Insitu Data")
                
            except:
                ph.info("❌ Failed to download " + st.session_state.event_selected.sc + " Insitu Data - Try downloading kernel manually or adding custom file in HelioSat Folder!")
        if 'insituplot_base' not in st.session_state:
            print('new')
            st.session_state.insituplot_base = plot_insitu(st)
                
        st.session_state.insituplot = copy.deepcopy(st.session_state.insituplot_base)
                
        
        
        if st.session_state.view_fitting_points:
            plot_fittinglines(st)
            
        if st.session_state.view_catalog_insitu:
            plot_catalog(st)
            
        if st.session_state.view_synthetic_insitu:
            plot_synthetic_insitu(st)
        if 'df' in st.session_state:
            plot_selected_synthetic_insitu(st)
        if st.session_state.view_fitting_results:
            plot_sigma(st)
        
        if st.session_state.plottinglib == 'plotly':
            st.write(st.session_state.insituplot)
        else:
            st.write('Not yet implemented. Please switch back to plotly!')
            #st.pyplot(st.session_state.insituplot)
            
        plot_additionalinsitu(st)
        
    #############################################################
    # Remote Sensing
    
    #############################################################
    # Download and Process the Images
    
    #############################################################
    # Fitting Results
    
    fitting_points(st)
    
    if st.session_state.fitting_results:
        st.markdown('### Fitting')
        if 'model_fittings' in st.session_state:
            plot_fitting_results(st)
        elif 'session_state_file' in st.session_state and 'model_fittings' in st.session_state.session_state_file['etc']:
            plot_fitting_results(st)
        else:
            st.info('Run fitting on insitu data to enable this feature.')
    





if __name__ == '__main__':
    run()