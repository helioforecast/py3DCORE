'''
Plotting
'''

from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly_express as px
import plotly.figure_factory as ff

import heliosat

import streamlit as st

import pickle as p
import seaborn as sns

import numpy as np
import pandas as pds
import datetime
import matplotlib.dates as mdates

from py3dcore.app.utils import get_insitudata, loadpickle, get_iparams, get_iparams_exp, generate_ensemble, get_fitobserver
import py3dcore

import traceback

'''
Plotting functions
'''

'''
###############################################################
########################### 2D PLOTS ###########################
###############################################################
'''

def plot_2d_pos(st):
    st.session_state.twodcontainer = st.session_state.placeholder.container()
    if 'twod_plot_base' in st.session_state:
        return st.session_state.twod_plot_base
    else:
        ph2d = st.session_state.twodcontainer.empty()
        ph2d.info("⏳ Downloading Position Data...")
        
        
        

'''
###############################################################
########################### 3D PLOTS ###########################
###############################################################
'''

#def plot_3d_pos(st):


'''
###############################################################
######################### INSITU PLOTS #########################
###############################################################
'''

def plot_insitu(st):
    
    #if st.session_state.plottinglib == 'matplotlib
    
    if st.session_state.mag_coord_system == 'HEEQ':
        names = ['Bx', 'By', 'Bz']
    else:
        names = ['Br', 'Bt', 'Bn']
        
    

    fig = make_subplots(rows=1, cols=1, shared_xaxes=True, subplot_titles = [st.session_state.event_selected.sc])
    fig.add_trace(
        go.Scatter(
            x=st.session_state.t_data,
            y=st.session_state.b_data[:, 0],
            name=names[0],
            line_color='red',
            line_width = 1,
            showlegend=st.session_state.view_legend_insitu
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=st.session_state.t_data,
            y=st.session_state.b_data[:, 1],
            name=names[1],
            line_color='green',
            line_width = 1,
            showlegend=st.session_state.view_legend_insitu
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=st.session_state.t_data,
            y=st.session_state.b_data[:, 2],
            name=names[2],
            line_color='blue',
            line_width = 1,
            showlegend=st.session_state.view_legend_insitu
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=st.session_state.t_data,
            y=np.sqrt(np.sum(st.session_state.b_data**2, axis=1)),
            name='Btot',
            line_color='black',
            line_width = 1,
            showlegend=st.session_state.view_legend_insitu
        ),
        row=1, col=1
    )

    # Extract the numerical value from the selected width option
    width_percentage = int(st.session_state.selected_width.strip('%'))
    # Calculate the width in pixels based on the selected percentage
    width_pixels = width_percentage / 100 * 800

    fig.update_layout(width=width_pixels)
    fig.update_yaxes(title_text='B [nT]', row=1, col=1)
    fig.update_yaxes(showgrid=True, zeroline=False, showticklabels=True,
                     showspikes=True, spikemode='across', spikesnap='cursor', showline=False, spikedash='solid', spikethickness=1)
    fig.update_xaxes(showgrid=True, zeroline=False, showticklabels=True, rangeslider_visible=False,
                     showspikes=True, spikemode='across', spikesnap='cursor', showline=False, spikedash='solid', spikethickness=1)

    
    return fig



def plot_selected_synthetic_insitu(st):

    dt_0 = st.session_state.dt_launch
    
    for index in st.session_state.selected_indices:
        if index != "mean":
            row = st.session_state.df.loc[index]
        else:
            row = st.session_state.mean_df.loc['Mean']
            
        iparams = get_iparams_exp(row)
        model_obj = py3dcore.ToroidalModel(dt_0, **iparams) # model gets initialized
        model_obj.generator()
        # Create ndarray with dtype=object to handle ragged nested sequences
        outa = np.array(model_obj.simulator(st.session_state.t_data, st.session_state.pos_data), dtype=object)
        outa = np.squeeze(outa[0])
        outa[outa==0] = np.nan


        st.session_state.insituplot.add_trace(
            go.Scatter(
                x=st.session_state.t_data,
                y=outa[:, 0],
                line=dict(color='red', width=2, dash='dot'),
                showlegend=False
            ),
            row=1, col=1
        )

        st.session_state.insituplot.add_trace(
            go.Scatter(
                x=st.session_state.t_data,
                y=outa[:, 1],
                line=dict(color='green', width=2, dash='dot'),
                showlegend=False
            ),
            row=1, col=1
        )

        st.session_state.insituplot.add_trace(
            go.Scatter(
                x=st.session_state.t_data,
                y=np.sqrt(np.sum(outa**2, axis=1)),
                line=dict(color='black', width=2, dash='dot'),
                showlegend=False
            ),
            row=1, col=1
        )

        st.session_state.insituplot.add_trace(
            go.Scatter(
                x=st.session_state.t_data,
                y=outa[:, 2],
                line=dict(color='blue', width=2, dash='dot'),
                showlegend=False
            ),
            row=1, col=1
        )
    
    return 


def plot_additionalinsitu(st):
    
    
    if st.session_state.mag_coord_system == 'HEEQ':
        names = ['Bx', 'By', 'Bz']
    else:
        names = ['Br', 'Bt', 'Bn']
        
    traceplots = []
    titles = []
        
    if len(st.session_state.insitu_list) > 1:
        for sc in st.session_state.insitu_list:
            if sc == st.session_state.event_selected.sc:
                pass
            else:
                ph = st.session_state.insitucontainer.empty()
                try:
                    ph.info("⏳ Downloading Insitu Data...")
                    b_data, t_data, _ = get_insitudata(st.session_state.mag_coord_system, sc, st.session_state.insitubegin, st.session_state.insituend)
                
                    trace1 = go.Scatter(
                        x=t_data,
                        y=b_data[:, 0],
                        name=names[0],
                        line_color='red',
                        line_width = 1,
                        showlegend=st.session_state.view_legend_insitu
                    )

                    trace2 = go.Scatter(
                            x=t_data,
                            y=b_data[:, 1],
                            name=names[1],
                            line_color='green',
                            line_width = 1,
                            showlegend=st.session_state.view_legend_insitu
                        )

                    trace3 = go.Scatter(
                            x=t_data,
                            y=b_data[:, 2],
                            name=names[2],
                            line_color='blue',
                            line_width = 1,
                            showlegend=st.session_state.view_legend_insitu
                        )

                    trace4 = go.Scatter(
                            x=t_data,
                            y=np.sqrt(np.sum(b_data**2, axis=1)),
                            name='Btot',
                            line_color='black',
                            line_width = 1,
                            showlegend=st.session_state.view_legend_insitu
                        )
                    
                    ph.success("✅ Successfully downloaded " + sc + " Insitu Data")
                    traceplots.append([trace1,trace2,trace3,trace4])
                    titles.append(sc)
                except Exception as e:
                    ph.info("❌ Failed to download " + sc + " Insitu Data - Try downloading kernel manually or adding custom file in HelioSat folder! See terminal output for more information.")
                    # Print the traceback information
                    print(sc + " ---- ERROR:")
                    print(str(e))
                
        fig = make_subplots(rows=len(titles), cols=1, shared_xaxes=True, subplot_titles = titles)
        
        for i, title in enumerate(titles):
            for trace in traceplots[i]:
                fig.add_trace(trace, row=i+1, col=1)
    
                            
        fig.update_yaxes(title_text='B [nT]')
        fig.update_yaxes(showgrid=True, zeroline=False, showticklabels=True,
                         showspikes=True, spikemode='across', spikesnap='cursor', showline=False, spikedash='solid', spikethickness=1)
        fig.update_xaxes(showgrid=True, zeroline=False, showticklabels=True, rangeslider_visible=False,
                         showspikes=True, spikemode='across', spikesnap='cursor', showline=False, spikedash='solid', spikethickness=1)
        # Set the height of each subplot to match the typical height of a single plot
        subplot_height = 400
        total_height = len(titles) * subplot_height
        fig.update_layout(height=total_height)
            

        st.write(fig)
    return

def plot_fittinglines(st):
    
    st.session_state.insituplot.add_shape(type="line",
                                                  x0=st.session_state.dt_A, 
                                                  y0=-30, 
                                                  x1=st.session_state.dt_A, 
                                                  y1=30,
                                                  line=dict(color="Red",width=1)
                                                 )
    st.session_state.insituplot.add_shape(type="line",
                                                  x0=st.session_state.dt_B, 
                                                  y0=-30, 
                                                  x1=st.session_state.dt_B, 
                                                  y1=30,
                                                  line=dict(color="Red",width=1)
                                                 )
    for fitline in st.session_state.fitting_datetimes:
        st.session_state.insituplot.add_shape(type="line",
                                                  x0=fitline, 
                                                  y0=-30, 
                                                  x1=fitline, 
                                                  y1=30,
                                                  line=dict(color="Black",width=1)
                                                 )
   
    return


def plot_catalog(st):
    
    st.session_state.insituplot.add_vrect(
        x0=st.session_state.event_selected.begin,
        x1=st.session_state.event_selected.end,
        fillcolor="LightSalmon", 
        opacity=0.5,
        layer="below",
        line_width=0
)
                                                 
    return

def plot_synthetic_insitu(st):

    dt_0, iparams = get_iparams(st)
    model_obj = py3dcore.ToroidalModel(dt_0, **iparams) # model gets initialized
    model_obj.generator()
    # model_obj = fp.returnfixedmodel(self.filename, fixed_iparams_arr='mean')
    # Create ndarray with dtype=object to handle ragged nested sequences
    outa = np.array(model_obj.simulator(st.session_state.t_data, st.session_state.pos_data), dtype=object)
    outa = np.squeeze(outa[0])

    outa[outa==0] = np.nan
    
    st.session_state.insituplot.add_trace(
        go.Scatter(
            x=st.session_state.t_data,
            y=outa[:, 0],
            line=dict(color='red', width=4, dash='dash'),
            showlegend=False
        ),
        row=1, col=1
    )
    
    st.session_state.insituplot.add_trace(
        go.Scatter(
            x=st.session_state.t_data,
            y=outa[:, 1],
            line=dict(color='green', width=4, dash='dash'),
            showlegend=False
        ),
        row=1, col=1
    )
    
    st.session_state.insituplot.add_trace(
        go.Scatter(
            x=st.session_state.t_data,
            y=np.sqrt(np.sum(outa**2, axis=1)),
            line=dict(color='black', width=4, dash='dash'),
            showlegend=False
        ),
        row=1, col=1
    )
    
    st.session_state.insituplot.add_trace(
        go.Scatter(
            x=st.session_state.t_data,
            y=outa[:, 2],
            line=dict(color='blue', width=4, dash='dash'),
            showlegend=False
        ),
        row=1, col=1
    )
    
    return 

    


def plot_fitting_results(st):
    
    filepath = loadpickle(st.session_state.filename)
    
    # read from pickle file
    file = open(filepath, "rb")
    data = p.load(file)
    file.close()
    
    model_objt = data["model_obj"]
    
    iparams_arrt = model_objt.iparams_arr
    
    df = pds.DataFrame(iparams_arrt)
    cols = df.columns.values.tolist()

    # drop first column
    df.drop(df.columns[[0]], axis=1, inplace=True)

    # rename columns
    df.columns = ['Longitude', 'Latitude', 'Inclination', 'Diameter 1 AU', 'Aspect Ratio', 'Launch Radius', 'Launch Velocity', 'T_Factor', 'Expansion Rate', 'Magnetic Decay Rate', 'Magnetic Field Strength 1 AU', 'Background Drag', 'Background Velocity']
    
    # Reorder columns
    ordered_cols = ['Longitude', 'Latitude', 'Inclination', 'Diameter 1 AU', 'Aspect Ratio', 'Launch Radius', 'Launch Velocity', 'Expansion Rate', 'Background Drag', 'Background Velocity', 'T_Factor', 'Magnetic Decay Rate', 'Magnetic Field Strength 1 AU']
    df = df[ordered_cols]
    
     # Add 'eps' column from data["epses"]
    epses = data["epses"]
    num_rows = min(len(epses), len(df))
    df.insert(0, 'RMSE Ɛ', epses[:num_rows])
    
    # Calculate statistics
    mean_values = df.mean()
    std_values = df.std()
    median_values = df.median()
    min_values = df.min()
    max_values = df.max()
    q1_values = df.quantile(0.25)
    q3_values = df.quantile(0.75)
    skewness_values = df.skew()
    kurtosis_values = df.kurt()

    # Create mean_row DataFrame with desired statistics
    mean_row = pds.DataFrame(
        [mean_values, std_values, median_values, min_values, max_values,
         q1_values, q3_values, skewness_values,
         kurtosis_values],
        index=["Mean", "Standard Deviation", "Median", "Minimum", "Maximum",
               "Q1", "Q3", "Skewness", "Kurtosis"],
        columns=df.columns
    )
    
    
    # Select the fit
    
    
    options = [""] + ["mean"] + df.index.tolist()
    st.info("Please note that once the sliders have been manually adjusted, it is currently not possible to select a specific fit from the dataframe within the app. To choose a fit, please reload the app by refreshing the page or using the reload functionality in your web browser.")
    
    col1, col2 = st.columns([1,4])
    with col1:
        selected_index = st.selectbox('Select fit:', options)
    options = ["mean"] + df.index.tolist()
    with col2:
        selected_indices = st.multiselect('Plot:', options)
    previous_index = st.session_state.get("selected_index")
    previous_indices = st.session_state.get("selected_indices") 
            
    if selected_index != "":
        
        if selected_index != previous_index:
            st.session_state.selected_index = selected_index
            if selected_index == "mean":
                st.session_state.selected_row = mean_row.loc['Mean']
            else:
                st.session_state.selected_row = df.loc[selected_index]
            # Rerun the app from the beginning
            st.experimental_rerun()
                
    if selected_indices != "":
         
        if selected_indices != previous_indices:
            st.session_state.selected_indices = selected_indices
            # Rerun the app from the beginning
            st.experimental_rerun()
        
        
        rounded_df = df.round(2)
        st.session_state.df = rounded_df
        rounded_df_styled = rounded_df.style.format("{:.2f}")

        rounded_mean_row = mean_row.round(2)
        st.session_state.mean_df = rounded_mean_row
        rounded_mean_row_styled = rounded_mean_row.style.format("{:.2f}")

        # Apply green background to the selected rows
        def highlight_selected_row(x):
            return ['background-color: #b3e6b3' if x.name in st.session_state.selected_indices else '' for _ in x]

        styled_df = rounded_df_styled.apply(highlight_selected_row, axis=1)


        def highlight_mean(x):
            return ['background-color: #b3e6b3' if x.name == 'Mean' else '' for _ in x]
        if "mean" in st.session_state.selected_indices:
            styled_mean = rounded_mean_row_styled.apply(highlight_mean, axis = 1)
        else: styled_mean = rounded_mean_row_styled
        
        
        

        # Render the styled DataFrame
        st.write('###### Statistics')
        st.write(styled_mean)
        st.write('###### Accepted Fits')
        st.write(styled_df)
        
        
    else:
        rounded_df = df.round(2)
        st.session_state.df = rounded_df
        rounded_df_styled = rounded_df.style.format("{:.2f}")

        rounded_mean_row = mean_row.round(2)
        st.session_state.mean_df = rounded_mean_row
        rounded_mean_row_styled = rounded_mean_row.style.format("{:.2f}")
        
        st.write('###### Statistics')
        st.write(rounded_mean_row_styled)
        st.write('###### Accepted Fits')
        st.write(rounded_df_styled)
    
    if st.session_state.parameter_distribution:
        st.write('###### Parameter Distribution')

        g = sns.pairplot(df, 
                         corner=True,
                         height= 2,
                         plot_kws=dict(marker="+", linewidth=1)
                        )
        g.map_lower(sns.kdeplot, levels=[0.05, 0.32], color=".2",warn_singular=False) #  levels are 2-sigma and 1-sigma contours

        st.pyplot(g.fig)
    

    return


def plot_fitting_process(st, reached = False):
    fitprocesscontainer = st.session_state.fitholder.container()
    if reached == True:
        fitprocesscontainer.success("✅ Reached target RMSE after iteration " + str(st.session_state.current_iter))
    else:
        fitprocesscontainer.info("⏳ Running iteration " + str(st.session_state.current_iter))
    col1, col2 = fitprocesscontainer.columns([3,1])
    
    if reached == True:
        st.session_state.progressbar = col1.progress(100)
    else: 
        st.session_state.progressbar = col1.progress(0)

        
    try:
        currenteps = round(st.session_state.currenteps[0],3)
    except:
        currenteps = round(st.session_state.currenteps,3)
        
    try:
        epsdiff = round(st.session_state.epsdiff[0],3)
    except:
        epsdiff = round(st.session_state.epsdiff,3)
        
        
    st.session_state.epsmetric = col2.metric(label = "RMSE Ɛ",
                                             value = currenteps,
                                             delta = epsdiff,
                                             delta_color="inverse")
                                             
                                             
    
    return


def plot_sigma(st):
    
    if 'filename' in st.session_state:
        path = str(st.session_state.filename)
        
    else:
        path = str(st.session_state.event_selected)
           
    # read from pickle file
    filepath = loadpickle(path)
    
    file = open(filepath, "rb")
    data = p.load(file)
    file.close()
    
    _, fit_coord_system = get_fitobserver(st.session_state.mag_coord_system, st.session_state.event_selected.sc)
    
    ed = generate_ensemble(filepath, st.session_state.t_data, reference_frame=data['data_obj'].reference_frame, reference_frame_to=fit_coord_system, max_index=data['model_obj'].ensemble_size)
    
    # black shadow
    st.session_state.insituplot.add_trace(
        go.Scatter(x=st.session_state.t_data,
                    y=ed[0][3][0], 
                    fill=None, 
                    mode='lines', 
                    line_color='black', 
                    line_width=0,
                    showlegend=False),
         row=1, col=1
    )
    st.session_state.insituplot.add_trace(
        go.Scatter(x=st.session_state.t_data,
                    y=ed[0][3][1], 
                    fill='tonexty', 
                    mode='lines', 
                    line_color='black',
                    line_width=0,
                    fillcolor='rgba(0, 0, 0, 0.25)',
                    showlegend=False),
         row=1, col=1
    )
    # red shadow
    st.session_state.insituplot.add_trace(
        go.Scatter(x=st.session_state.t_data,
                    y=ed[0][2][0][:, 0], 
                    fill=None, 
                    mode='lines', 
                    line_color='black', 
                    line_width=0,
                    showlegend=False),
         row=1, col=1
    )
    st.session_state.insituplot.add_trace(
        go.Scatter(x=st.session_state.t_data,
                    y=ed[0][2][1][:, 0], 
                    fill='tonexty', 
                    mode='lines', 
                    line_color='black',
                    line_width=0,
                    fillcolor='rgba(255, 0, 0, 0.25)',
                    showlegend=False),
         row=1, col=1
    )
    # green shadow
    st.session_state.insituplot.add_trace(
        go.Scatter(x=st.session_state.t_data,
                    y=ed[0][2][0][:, 1], 
                    fill=None, 
                    mode='lines', 
                    line_color='green', 
                    line_width=0,
                    showlegend=False),
         row=1, col=1
    )
    st.session_state.insituplot.add_trace(
        go.Scatter(x=st.session_state.t_data,
                    y=ed[0][2][1][:, 1], 
                    fill='tonexty', 
                    mode='lines', 
                    line_color='green',
                    line_width=0,
                    fillcolor='rgba(0, 255, 0, 0.25)',
                    showlegend=False),
         row=1, col=1
    )
    # blue shadow
    st.session_state.insituplot.add_trace(
        go.Scatter(x=st.session_state.t_data,
                    y=ed[0][2][0][:, 2], 
                    fill=None, 
                    mode='lines', 
                    line_color='blue', 
                    line_width=0,
                    showlegend=False),
         row=1, col=1
    )
    st.session_state.insituplot.add_trace(
        go.Scatter(x=st.session_state.t_data,
                    y=ed[0][2][1][:, 2], 
                    fill='tonexty', 
                    mode='lines', 
                    line_color='blue',
                    line_width=0,
                    fillcolor='rgba(0,  0, 255, 0.25)',
                    showlegend=False),
         row=1, col=1
    )
    
    return