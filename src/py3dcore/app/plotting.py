'''
Plotting
'''

from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly_express as px

import heliosat

import numpy as np
import pandas as pds
import datetime
import matplotlib.dates as mdates

from py3dcore.app.utils import get_insitudata

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
                    print(titles)
                    titles.append(sc)
                except:
                    insitucontainer.info("❌ Failed to download " + sc + " Insitu Data - Try downloading kernel manually or adding custom file in HelioSat Folder!")
                
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
        total_height = (len(st.session_state.insitu_list) - 1) * subplot_height
        fig.update_layout(height=total_height)
            

        st.write(fig)
    return

def plot_fittinglines(st):
    fittinglines = ['t_1','t_2','t_3','t_4','t_5']
    fittingdatetimes = ['dt_1','dt_2','dt_3','dt_4','dt_5']
    
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
    
    for i, fittingline in enumerate(fittinglines):
        if fittingline in st.session_state:
            if st.session_state[fittingline]:
                st.session_state.insituplot.add_shape(type="line",
                                                  x0=st.session_state[fittingdatetimes[i]], 
                                                  y0=-30, 
                                                  x1=st.session_state[fittingdatetimes[i]], 
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