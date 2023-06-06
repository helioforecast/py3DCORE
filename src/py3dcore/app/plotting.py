'''
Plotting
'''

from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly_express as px
import plotly.figure_factory as ff

import heliosat

import pickle as p
import seaborn as sns

import numpy as np
import pandas as pds
import datetime
import matplotlib.dates as mdates

from py3dcore.app.utils import get_insitudata, loadpickle
import py3dcore

import traceback

'''
Plotting functions
'''

'''
###############################################################
########################### 3D PLOTS ###########################
###############################################################
'''




'''
###############################################################
######################### INSITU PLOTS #########################
###############################################################
'''

def plot_insitu(st):
    
    if st.session_state.mag_coord_system == 'HEEQ':
        names = ['Bx', 'By', 'Bz']
    else:
        names = ['Br', 'Bt', 'Bn']
        
    

    fig = make_subplots(rows=1, cols=1, shared_xaxes=True)
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
    
    
    fig.update_yaxes(title_text='B [nT]', row=1, col=1)
    fig.update_yaxes(showgrid=True, zeroline=False, showticklabels=True,
                     showspikes=True, spikemode='across', spikesnap='cursor', showline=False, spikedash='solid', spikethickness=1)
    fig.update_xaxes(showgrid=True, zeroline=False, showticklabels=True, rangeslider_visible=False,
                     showspikes=True, spikemode='across', spikesnap='cursor', showline=False, spikedash='solid', spikethickness=1)

    
    return fig

def plot_additionalinsitu(st,insitucontainer):
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
                try:
                    b_data, t_data = get_insitudata(st.session_state.mag_coord_system, sc, st.session_state.insitubegin, st.session_state.insituend)
                
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
                    
                    insitucontainer.success("✅ Successfully downloaded " + sc + " Insitu Data")
                    traceplots.append([trace1,trace2,trace3,trace4])
                    titles.append(sc)
                except Exception as e:
                    insitucontainer.info("❌ Failed to download " + sc + " Insitu Data - Try downloading kernel manually or adding custom file in HelioSat folder! See terminal output for more information.")
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

import scipy.stats as kde

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
    
    # Select the fit
    options = [""] + df.index.tolist()
    selected_index = st.selectbox('Select fit:', options)
    
    previous_index = st.session_state.get("selected_index") 
        
    if selected_index != "":
        
            if selected_index != previous_index:
                st.session_state.selected_index = selected_index
                st.session_state.selected_row = df.loc[selected_index]
                # Rerun the app from the beginning
                st.experimental_rerun()
        
            # Get the index of the selected row
            index_position = df.index.get_loc(st.session_state.selected_index)

            # Apply green background to the selected row
            def highlight_selected_row(x):
                if x.name == index_position:
                    return ['background-color: #b3e6b3'] * len(x)
                else:
                    return [''] * len(x)

            styled_df = df.style.apply(highlight_selected_row, axis=1)

            # Render the styled DataFrame
            st.write(styled_df)
        
        
    else:
        st.write(df)
    
    if st.session_state.parameter_distribution:
        st.write('##### Parameter Distribution')

        g = sns.pairplot(df, 
                         corner=True,
                         height= 2,
                         plot_kws=dict(marker="+", linewidth=1)
                        )
        g.map_lower(sns.kdeplot, levels=[0.05, 0.32], color=".2",warn_singular=False) #  levels are 2-sigma and 1-sigma contours

        st.pyplot(g.fig)
    

    return
    
''''   
    iparams_arrt = iparams_arrt[:,1:]
    # Ensure the number of columns matches the expected size
    expected_columns = ['lon', 'lat', 'inc', 'D1AU', 'delta', 'launch radius', 'launch speed', 't factor', 'expansion rate', 'B decay rate', 'B1AU', 'gamma', 'vsw']
    if len(iparams_arrt[0]) != len(expected_columns):
        raise ValueError(f"Number of columns in the data ({len(iparams_arrt[0])}) does not match the expected size ({len(expected_columns)})")

    df = pds.DataFrame(iparams_arrt, columns=expected_columns)

    fig = make_subplots(rows=len(df.columns), cols=len(df.columns), shared_xaxes=False, shared_yaxes=False)
    
    
    # Create scatter plots and histograms
    for i in range(len(df.columns)):
        for j in range(i+1):
            if i == j:
                # Add histogram on diagonal
                fig.add_trace(go.Histogram(x=df.iloc[:, i], name=df.columns[i], nbinsx=30), row=i+1, col=i+1)
            else:
                # Add scatter plot in lower triangle
                fig.add_trace(go.Scatter(
                    x=df.iloc[:, j],
                    y=df.iloc[:, i],
                    mode='markers',
                    marker=dict(symbol="cross", size=4, opacity=0.5),
                    showlegend=False
                ), row=i+1, col=j+1)

                fig.add_trace(go.Contour(
                    x=df.iloc[:, j],
                    y=df.iloc[:, i],
                    colorscale='Greys',
                    contours=dict(coloring='lines', showlabels=False, start=0.05, end=0.95, size=0.1),
                    showscale=False,
                    hoverinfo='none'
                ), row=i+1, col=j+1)            )

    # Update subplot axes and layout
    for i, col in enumerate(df.columns):
        fig.update_xaxes(title_text=col, row=len(df.columns), col=i+1)
        fig.update_yaxes(title_text=col, row=i+1, col=1)

    fig.update_layout(
        title="Pairwise Scatter Plot",
        width=1800,
        height=1800,
    )
    
    #fig.show()
    st.write(fig)
    
    return
    
'''


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