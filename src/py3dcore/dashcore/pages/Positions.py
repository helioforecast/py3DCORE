from dash import dcc, html, Input, Output, State, callback, register_page
from dash.exceptions import PreventUpdate
import dash
import dash_mantine_components as dmc
import plotly.express as px
import numpy as np
        
import plotly.tools as tls
        
import datetime

import re
import functools
import asyncio

import plotly.graph_objs as go
from plotly.subplots import make_subplots
import plotly_express as px
import plotly.figure_factory as ff

import py3dcore
from py3dcore.dashcore.assets.config_sliders import modelstate
from py3dcore.dashcore.utils.utils import *
from py3dcore.dashcore.utils.maputils import *

from dash_iconify import DashIconify
import dash_bootstrap_components as dbc

import matplotlib.pyplot as plt
import matplotlib

from sunpy.coordinates import get_horizons_coord
import os
import urllib.request

from sunpy.net import Fido, attrs as a
from sunpy.time import TimeRange
from sunpy.map import Map
from sunpy.visualization import axis_labels_from_ctype


register_page(__name__, icon="iconoir:position", order=4)

app = dash.get_app()
    

matplotlib.use('agg')


reload_icon = dmc.ThemeIcon(
                     DashIconify(icon='zondicons:refresh', width=18, color="black"),
                     size=40,
                     radius=40,
                     style={"backgroundColor": "#eaeaea", "marginRight": "12px"},
                    
                 )

success_icon = dmc.ThemeIcon(
                     DashIconify(icon='mdi:check-bold', width=18, color="black"),
                     size=40,
                     radius=40,
                     style={"backgroundColor": "#eaeaea", "marginRight": "12px"},
                 )

fail_icon = dmc.ThemeIcon(
                     DashIconify(icon='ph:x-bold', width=18, color="black"),
                     size=40,
                     radius=40,
                     style={"backgroundColor": "#eaeaea", "marginRight": "12px"},
                 )


plotoptionrow = dbc.Row([
            dbc.Col(
                dmc.CheckboxGroup(
                    id="plotoptions_posfig",
                    label="Options for plotting",
                    orientation="horizontal",
                    withAsterisk=False,
                    offset="md",
                    mb=10,
                    children=[
                        dmc.Checkbox(label="Title", value="title", color="green"),
                        dmc.Checkbox(label="Latitudinal Grid", value="latgrid", color="green"),
                        dmc.Checkbox(label="Longitudinal Grid", value="longgrid", color="green"),
                        dmc.Checkbox(label="Trajectories", value="trajectories", color="green", disabled = False),
                        dmc.Checkbox(label="CME", value="cme", color="green"),
                    ],
                    value=["longgrid"],
                ),
                width=8,
            ),
                dbc.Col(
                [
                    dmc.SegmentedControl(
                        id = "segmented",
                        value="2D",
                        data = [
                            {"value": "2D", "label": "2D"},
                            {"value": "3D", "label": "3D"},
                            {"value": "coronograph", "label": "📹", "disabled": True},
                        ],
                        style={"marginRight": "12px"},
                    )
                ],
                width=4,  # Adjust the width of the right-aligned columns
                style={"display": "flex", "justify-content": "flex-end"},
            ),
        ], justify="between",  # Distribute the columns
           align="end",  # Align content at the bottom of the row
        )





bodyoptionrow = dbc.Row([
            dbc.Col(
                dmc.CheckboxGroup(
                    id="bodyoptions_posfig",
                    label="Bodies",
                    orientation="horizontal",
                    withAsterisk=False,
                    offset="md",
                    mb=10,
                    children=[
                        dmc.Checkbox(label="Sun", value="showsun", color="green"),
                        dmc.Checkbox(label="Mercury", value="mercury", color="green"),
                        dmc.Checkbox(label="Venus", value="venus", color="green"),
                        dmc.Checkbox(label="Earth", value="earth", color="green"),
                        #dmc.Checkbox(label="Mars", value="mars", color="green"),
                    ],
                    value=["showsun", "earth"],
                ),
                width=8,  # Adjust the width of the CheckboxGroup column
            ),
            #dbc.Col(
            #            [
                            #dcc.Dropdown(
                            #    id = "focusdropdown",
                            #    value="2D",
                            #    options=[
                            #        {"label": "Sun", "value": "Sun"},
                            #        {"label": "Earth", "value": "Earth"},
                                    # Add more objects as needed
                            #    ],
                            #    placeholder="Select focus object",
                            #    style={"min-width": "250px", "marginLeft": "0px", "marginRight": "12px"},
                            #)
                        #],
                        #width=4,  # Adjust the width of the right-aligned columns
                        #style={"display": "flex", "justify-content": "flex-end"},
                   # ),
        ], justify="between",  # Distribute the columns
           align="end",  # Align content at the bottom of the row
        )

spacecraftoptionrow = dbc.Row([
            dbc.Col(
                dmc.CheckboxGroup(
                    id="spacecraftoptions_posfig",
                    label="Spacecraft",
                    orientation="horizontal",
                    withAsterisk=False,
                    offset="md",
                    mb=10,
                    children=[
                    ],
                ),
                width=6,  # Adjust the width of the CheckboxGroup column
            ),
            dbc.Col(
                [
                    html.Div(id = "loadscspinner", 
                             children = "",
                            ),
                    dcc.Dropdown(
                        id="additional-sc",
                        multi=False,
                        options=[
                            {"label": "SolarOrbiter", "value": "SOLO"},
                            {"label": "ParkerSolarProbe", "value": "PSP"},
                            {"label": "BepiColombo", "value": "BEPI"},
                            #{"label": "Wind", "value": "Wind"},
                            {"label": "STEREO-A", "value": "STEREO-A"},
                        ],
                        placeholder="Select additional spacecraft",
                        style={"min-width": "250px", "marginLeft": "0px", "marginRight": "12px"},
                    ),
                    dbc.Button(
                        children=reload_icon,
                        id="reload-sc-button",
                        style={
                            "backgroundColor": "transparent",
                            "border": "none",
                            "padding": "0",
                            "marginLeft": "12px"
                        },
                    ),
                ],
                width=6,  # Adjust the width of the right-aligned columns
                style={"display": "flex", "justify-content": "flex-end"},
            ),
        ], justify="between",  # Distribute the columns
           align="end",  # Align content at the bottom of the row
        )


layout = html.Div(
    [
        plotoptionrow,
        bodyoptionrow,
        spacecraftoptionrow,
        dcc.Graph(id="posfig", style={'display': 'none'}),  # Hide the figure by default
        dbc.Row(
            [
                dmc.Text("Δt"), #,style={"font-size": "12px"}),
                dcc.Slider(
                    id="time_slider",
                    min=0,
                    max=168,
                    step=0.5,
                    value=5,
                    marks = {i: '+' + str(i)+'h' for i in range(0, 169, 12)},
                    persistence=True,
                    persistence_type='session',
                ),
            ],
            id = 'timesliderdiv'
        ),
        dbc.Row(
            [
                dmc.Text("Δt"), #,style={"font-size": "12px"}),
                dcc.Slider(
                    id="corono_slider",
                    min=0,
                    max=12,
                    step=0.5,
                    value=5,
                    marks = {i: '+' + str(i)+'h' for i in range(0, 13)},
                    persistence=True,
                    persistence_type='session',
                ),
            ],
            id = 'coronosliderdiv'
        )
    ]
)


@callback(
    Output("plotoptions_posfig", "children"),
    Output("timesliderdiv", "style"),
    Output("coronosliderdiv", "style"),
    Output("reload-sc-button", "style"),
    Output("spacecraftoptions_posfig", "style"),
    Output("additional-sc", "style"),
    Output("bodyoptions_posfig", "style"),
    Input("segmented", "value"),
)
def updateplotoptions(dim):
    if dim == "2D":
        new_options = [
            dmc.Checkbox(label="Title", value="title", color="green"),
            dmc.Checkbox(label="AU axis", value="axis", color="green"),
            dmc.Checkbox(label="Trajectories", value="trajectories", color="green", disabled=False),
            dmc.Checkbox(label="Parker Spiral", value="parker", color="green", disabled=False),
        ]
        focus = {"visibility": "hidden"}
        focus2 = {"visibility": "hidden"}
        body_options_visibility = {"visibility": "visible", 
                            "backgroundColor": "transparent",
                            "border": "none",
                            "padding": "0",
                            "marginLeft": "12px"}
        spacecraft_options_visibility = {"visibility": "visible"}
        plot_options_visibility = {"visibility": "visible","min-width": "250px", "marginLeft": "0px", "marginRight": "12px"}

    elif dim == "3D":
        new_options = [
            dmc.Checkbox(label="Title", value="title", color="green"),
            dmc.Checkbox(label="Latitudinal Grid", value="latgrid", color="green"),
            dmc.Checkbox(label="Datetime", value="datetime", color="green"),
            dmc.Checkbox(label="Longitudinal Grid", value="longgrid", color="green"),
            dmc.Checkbox(label="Trajectories", value="trajectories", color="green", disabled=False),
            dmc.Checkbox(label="CME", value="cme", color="green"),
        ]
        focus = {"visibility": "visible"}
        focus2 = {"visibility": "hidden"}
        body_options_visibility = {"visibility": "visible", 
                            "backgroundColor": "transparent",
                            "border": "none",
                            "padding": "0",
                            "marginLeft": "12px"}
        spacecraft_options_visibility = {"visibility": "visible"}
        plot_options_visibility = {"visibility": "visible","min-width": "250px", "marginLeft": "0px", "marginRight": "12px"}

    elif dim == "coronograph":
        new_options = []
        focus = {"visibility": "hidden"}
        focus2 = {"visibility": "visible"}
        body_options_visibility = {"visibility": "hidden"}
        spacecraft_options_visibility = {"visibility": "hidden"}
        plot_options_visibility = {"visibility": "hidden"}

    else:
        new_options = []
        focus = {"visibility": "hidden"}
        focus2 = {"visibility": "visible"}
        body_options_visibility = {"visibility": "hidden"}
        spacecraft_options_visibility = {"visibility": "visible"}
        plot_options_visibility = {"visibility": "visible","min-width": "250px", "marginLeft": "0px", "marginRight": "12px"}

    return (
        new_options,
        focus,
        focus2,
        body_options_visibility,
        spacecraft_options_visibility,
        plot_options_visibility,
        spacecraft_options_visibility
    )
    

@app.long_callback(
    output=[
        Output("loadscspinner", "children"),
        Output("spacecraftoptions_posfig", "children"),
        Output("additional-sc", "options"),  
        Output("posstore", "data"),
    ],
    inputs=[
        State("event-alert-div","children"),
        Input("reload-sc-button", "n_clicks"),
        State("spacecraftoptions_posfig", "children"),
        State("additional-sc", "value"),
        State("additional-sc", "options"),  
        State("posstore", "data"),
    ],
    running=[
        (Output("reload-sc-button", "disabled"), True, False),
        (Output("loadscspinner", "children"),
            dbc.Spinner(), " "
        ), 
    ],
    prevent_initial_call=True,
)
def download_scdata(alertdiv, n_clicks, kids, addbodies, current_options, posstore):
   
    if alertdiv == None:
        return no_update, no_update, no_update, {}
    
    changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]
    
    if (n_clicks == None) or (n_clicks == 0):
        return no_update, no_update, no_update, {}

    
    alertdiv = alertdiv[0]
    # Find the indices of the first and second underscores
    first_underscore_index = alertdiv.index('_')
    second_underscore_index = alertdiv.index('_', first_underscore_index + 1)
    sc = alertdiv[first_underscore_index + 1:second_underscore_index]
    
    # Extract the date using regular expression
    date_pattern = r'(\d{8})'
    match = re.search(date_pattern, alertdiv)
    if match:
        extracted_date = match.group(1)
        extracted_datetime = datetime.datetime.strptime(extracted_date, '%Y%m%d')
    
    #if (n_clicks == None) or (n_clicks == 0):
    #    addbodies = [sc]
    #el
    if addbodies == None:
        raise PreventUpdate

    try:
        traces = get_posdata('HEEQ',addbodies, extracted_datetime)
        #traces3d = get_posdata('HEEQ',addbodies, extracted_datetime, True)
    except:
        return fail_icon, no_update, no_update, no_update
    
    #else:
    new_checkbox = dmc.Checkbox(label=addbodies, value=addbodies, color="green")

    # Update the existing checkboxes with the new checkbox
    updated_options = kids + [new_checkbox]

    # Remove the selected option from the options list
    new_options = [opt for opt in current_options if opt["value"] != addbodies]
    
    
    data = load_pos_data('HEEQ', addbodies, extracted_datetime)

    # Update posstore using a dictionary
    traj_data = {addbodies: {'traces': traces, 'data': data}}
    if posstore == None:
        posstore = traj_data
    else:
        posstore.update(traj_data)

    return success_icon, updated_options, new_options, posstore



@callback(
    Output("posfig", "figure"),
    Output("posfig", "style"),  # Add this Output to control the display style
    State("posstore", "data"),
    #Input("focusdropdown", "value"),  # Add this input
    Input("time_slider", "value"),
    Input("segmented", "value"),
    Input("graphstore", "data"),
    Input("event-info", "data"),
    Input("launch-label", "children"),
    Input("plotoptions_posfig", "value"),
    Input("spacecraftoptions_posfig", "value"),
    Input("bodyoptions_posfig", "value"),
    Input("reference_frame", "value"),
    *[
            Input(id, "value") for id in modelstate
        ],
    
)
def update_posfig(posstore, timeslider, dim, graph, infodata, launchlabel,plotoptions, spacecraftoptions, bodyoptions, refframe, *modelstatevars):
    
    #if (graph is {}) or (graph is None):  # This ensures that the function is not executed when no figure is present
    #    fig = {}
    #    return fig
    
    ################################################################
    ############################## 3D ##############################
    ################################################################
    fig = go.Figure()
    datetime_format = "Launch Time: %Y-%m-%d %H:%M"
    t_launch = datetime.datetime.strptime(launchlabel, datetime_format)
    roundedlaunch = round_to_hour_or_half(t_launch) 
    if dim == "3D":
        



        if "cme" in plotoptions:
            iparams = get_iparams_live(*modelstatevars)
            model_obj = py3dcore.ToroidalModel(t_launch, **iparams) # model gets initialized
            model_obj.generator()
            model_obj.propagator(roundedlaunch + datetime.timedelta(hours=timeslider))
            
            wf_model = model_obj.visualize_shape(iparam_index=0)  
            
            wf_array = np.array(wf_model)

            # Extract x, y, and z data from wf_array
            x = wf_array[:,:,0].flatten()
            y = wf_array[:,:,1].flatten()
            z = wf_array[:,:,2].flatten()

            # Create a 3D wireframe plot using plotly
            fig.add_trace(go.Scatter3d(x=x, y=y, z=z, mode='lines',
                           line=dict(width=1, color='rgba(100, 100, 100, 0.8)'),
                           showlegend=False))

            # Transpose the wf_array to extract wireframe points along the other direction
            x_wire = wf_array[:,:,0].T.flatten()
            y_wire = wf_array[:,:,1].T.flatten()
            z_wire = wf_array[:,:,2].T.flatten()

            # Create another 3D wireframe plot using plotly
            fig.add_trace(go.Scatter3d(x=x_wire, y=y_wire, z=z_wire, mode='lines',
                           line=dict(width=1, color='rgba(100, 100, 100, 0.8)'),
                           showlegend=False))



        if "showsun" in bodyoptions:

            # Create data for the Sun
            sun_trace = go.Scatter3d(
                x=[0], y=[0], z=[0],
                mode='markers',
                marker=dict(size=8, color='yellow'),
                name='Sun'
            )

            fig.add_trace(sun_trace)

        if "earth" in bodyoptions:

            # Create data for the Earth
            earth_trace = go.Scatter3d(
                x=[1], y=[0], z=[0],
                mode='markers',
                marker=dict(size=4, color='mediumseagreen'),
                name='Earth'
            )

            fig.add_trace(earth_trace)
                        
        if "mercury" in bodyoptions:
            fig.add_trace(plot_body3d(graph['bodydata']['Mercury']['data'], roundedlaunch + datetime.timedelta(hours=timeslider), 'slategrey', 'Mercury')[0])
            
            
        if "venus" in bodyoptions:
            fig.add_trace(plot_body3d(graph['bodydata']['Venus']['data'], roundedlaunch + datetime.timedelta(hours=timeslider), 'darkgoldenrod', 'Venus')[0])
            
        if "mars" in bodyoptions:
            fig.add_trace(plot_body3d(graph['bodydata']['Mars']['data'], roundedlaunch + datetime.timedelta(hours=timeslider), 'red', 'Mars')[0])
            
        if spacecraftoptions is not None:
            for scopt in spacecraftoptions:
                
                traces = process_coordinates(posstore[scopt]['data']['data'], roundedlaunch, roundedlaunch + datetime.timedelta(hours=timeslider), posstore[scopt]['data']['color'], scopt)
            
                if "trajectories" in plotoptions:
                    fig.add_trace(traces[0])
                    fig.add_trace(traces[1])
                    
                fig.add_trace(traces[2])


        if "longgrid" in plotoptions:
            # Create data for concentrical circles
            circle_traces = []
            radii = [0.3, 0.5, 0.8]  # Radii for the concentrical circles
            for r in radii:
                theta = np.linspace(0, 2 * np.pi, 100)
                x = r * np.cos(theta)
                y = r * np.sin(theta)
                z = np.zeros_like(theta)
                circle_trace = go.Scatter3d(
                    x=x, y=y, z=z,
                    mode='lines',
                    line=dict(color='gray'),
                    showlegend=False,
                    hovertemplate = None, 
                    hoverinfo = "skip", 
                )
                fig.add_trace(circle_trace)

                # Add labels for the circles next to the line connecting Sun and Earth
                label_x = r  # x-coordinate for label position
                label_y = 0  # y-coordinate for label position
                label_trace = go.Scatter3d(
                    x=[label_x], y=[label_y], z=[0],
                    mode='text',
                    text=[f'{r} AU'],
                    textposition='middle left',
                    textfont=dict(size=8),
                    showlegend=False,
                    hovertemplate = None, 
                    hoverinfo = "skip", 
                )
                fig.add_trace(label_trace)

            # Create data for the AU lines and their labels
            num_lines = 8
            for i in range(num_lines):
                angle_degrees = -180 + (i * 45)  # Adjusted angle in degrees (-180 to 180)
                angle_radians = np.deg2rad(angle_degrees)
                x = [0, np.cos(angle_radians)]
                y = [0, np.sin(angle_radians)]
                z = [0, 0]
                au_line = go.Scatter3d(
                    x=x, y=y, z=z,
                    mode='lines',
                    line=dict(color='gray'),
                    name=f'{angle_degrees}°',
                    showlegend=False,
                    hovertemplate = None, 
                    hoverinfo = "skip", 
                )
                fig.add_trace(au_line)

                # Add labels for the AU lines
                label_x = 1.1 * np.cos(angle_radians)
                label_y = 1.1 * np.sin(angle_radians)
                label_trace = go.Scatter3d(
                    x=[label_x], y=[label_y], z=[0],
                    mode='text',
                    text=[f'+/{angle_degrees}°' if angle_degrees == -180 else f'{angle_degrees}°'],
                    textposition='middle center',
                    textfont=dict(size=8),
                    showlegend=False,
                    hovertemplate = None, 
                    hoverinfo = "skip", 
                )
                fig.add_trace(label_trace)
                
        if "latgrid" in plotoptions:
            # Create data for concentrical circles
            circle_traces = []
            radii = [0.3, 0.5, 0.8]  # Radii for the concentrical circles
            for r in radii:
                theta = np.linspace(0, 1/2 * np.pi, 100)
                x = r * np.cos(theta)
                y = np.zeros_like(theta)
                z = r * np.sin(theta)
                circle_trace = go.Scatter3d(
                    x=x, y=y, z=z,
                    mode='lines',
                    line=dict(color='gray'),
                    showlegend=False,
                    hovertemplate = None, 
                    hoverinfo = "skip", 
                )
                fig.add_trace(circle_trace)

                # Add labels for the circles next to the line connecting Sun and Earth
                label_x = r  # x-coordinate for label position
                label_y = 0  # y-coordinate for label position
                label_trace = go.Scatter3d(
                    x=[0], y=[0], z=[r],
                    mode='text',
                    text=[f'{r} AU'],
                    textposition='middle left',
                    textfont=dict(size=8),
                    showlegend=False,
                    hovertemplate = None, 
                    hoverinfo = "skip", 
                )
                fig.add_trace(label_trace)

            # Create data for the AU lines and their labels
            num_lines = 10
            for i in range(num_lines):
                angle_degrees = (i * 10)  # Adjusted angle in degrees (0 to 90)
                angle_radians = np.deg2rad(angle_degrees)
                x = [0, np.cos(angle_radians)]
                y = [0, 0]
                z = [0, np.sin(angle_radians)]
                au_line = go.Scatter3d(
                    x=x, y=y, z=z,
                    mode='lines',
                    line=dict(color='gray'),
                    name=f'{angle_degrees}°',
                    showlegend=False,
                    hovertemplate = None, 
                    hoverinfo = "skip", 
                )
                fig.add_trace(au_line)

                # Add labels for the AU lines
                label_x = 1.1 * np.cos(angle_radians)
                label_y = 1.1 * np.sin(angle_radians)
                label_trace = go.Scatter3d(
                    x=[label_x], y=[0], z=[label_y],
                    mode='text',
                    text=[f'{angle_degrees}°'],
                    textposition='middle center',
                    textfont=dict(size=8),
                    showlegend=False,
                    hovertemplate = None, 
                    hoverinfo = "skip", 
                )
                fig.add_trace(label_trace)

        # Set the layout
        fig.update_layout(
            scene=dict(
                xaxis=dict(showticklabels=False, showgrid=False, zeroline=False, showline=False, title = ''),
                yaxis=dict(showticklabels=False, showgrid=False, zeroline=False, showline=False, title = ''),
                zaxis=dict(showticklabels=False, showgrid=False, zeroline=False, showline=False, title = ''),
                aspectmode='cube'
            ),
        )

        if "title" in plotoptions:
            fig.update_layout(title=str(roundedlaunch + datetime.timedelta(hours=timeslider)))
            
        if "datetime" in plotoptions:
            fig.add_annotation(text=f"t_launch + {timeslider} h", xref="paper", yref="paper", x=0.5, y=1.05, showarrow=False)
            

        # Set the layout
        fig.update_layout(
            scene=dict(
                xaxis=dict(showticklabels=False, showgrid=False, zeroline=False, showline=False, title = ''),
                yaxis=dict(showticklabels=False, showgrid=False, zeroline=False, showline=False, title = ''),
                zaxis=dict(showticklabels=False, showgrid=False, zeroline=False, showline=False, title = '', range=[-1, 1]),  # Adjust the range as needed
                aspectmode='cube',
            ),
        )
        
        #fig.update_scenes(camera_center=center)
        
        
    ################################################################
    ############################## 2D ##############################
    ################################################################
    
    elif dim == "2D":
        fig = make_subplots(rows=1, cols=1, specs=[[{'type': 'polar'}]])
        
        if "axis" in plotoptions:
            showticks = True
        else:
            showticks = False
            
            
        if "parker" in plotoptions:
                
            res_in_days=1 #/48.
            AUkm=149597870.7   
            sun_rot=26.24
            theta=np.arange(0,180,0.01)
            omega=2*np.pi/(sun_rot*60*60*24) #solar rotation in seconds

            v=300/AUkm #km/s
            r0=695000/AUkm
            r=v/omega*theta+r0*7
            
            # Create Parker spiral traces
            for q in np.arange(0, 12):
                omega = 2 * np.pi / (sun_rot * 60 * 60)  # Solar rotation in radians per second
                r = v / omega * theta + r0 * 7
                trace = go.Scatterpolar(
                    r=r,
                    theta=-theta + (0 + (360 / sun_rot) * res_in_days + 360 / 12 * q),
                    mode='lines',
                    line=dict(width=1, color='rgba(128, 128, 128, 0.3)'),
                    showlegend=False,
                    hovertemplate="Parker Spiral" +
                    "<extra></extra>",
                )
                fig.add_trace(trace)
                
                
                
        fig.update_layout(
            template="seaborn",
            polar=dict(
                angularaxis=dict(
                    tickmode='array',  # Set tick mode to 'array'
                    tickvals=[0, 45, 90, 135, 180, 225, 270, 315],  # Specify tick values for angles
                    ticktext=[ '0°', '45°', '90°', '135°', '+/-180°', '-135°', '-90°', '-45°',],  # Specify tick labels
                    showticklabels=True,  # Show tick labels
                    #rotation=90  # Rotate tick labels
                ),
                radialaxis=dict(
                    tickmode='array',  # Set tick mode to 'array'
                    tickvals=[0.2,0.4, 0.6,0.8,1, 1.2],  # Provide an empty list to remove tick labels
                    ticktext=['0.2 AU', '0.4 AU', '0.6 AU', '0.8 AU', '1 AU', '1.2 AU'],  # Specify tick labels
                    tickfont=dict(size=10),
                    showticklabels=showticks,  # Hide tick labels
                    range=[0, 1.2]  # Adjust the range of the radial axis,
                )
            )
        )
        
        if spacecraftoptions is not None:
            for scopt in spacecraftoptions:
            
                if "trajectories" in plotoptions:
                    fig.add_trace(posstore[scopt]['traces'][0])
                    fig.add_trace(posstore[scopt]['traces'][1])
                    
                fig.add_trace(posstore[scopt]['traces'][2])

        if "showsun" in bodyoptions:
            # Add the sun at the center
            fig.add_trace(go.Scatterpolar(r=[0], theta=[0], mode='markers', marker=dict(color='yellow', size=10, line=dict(color='black', width=1)), showlegend=False, hovertemplate="%{r:.1f} AU<br>%{theta:.1f}°"+
                    "<extra></extra>"))
            # Add label "Sun" next to the sun marker
            fig.add_trace(go.Scatterpolar(r=[0.03], theta=[15], mode='text', text=['Sun'],textposition='top right', showlegend=False, hovertemplate = None, hoverinfo = "skip", textfont=dict(color='black', size=14)))


        if "earth" in bodyoptions:# Add Earth at radius 1
            fig.add_trace(go.Scatterpolar(r=[1], theta=[0], mode='markers', marker=dict(color='mediumseagreen', size=10), showlegend=False, hovertemplate="%{r:.1f} AU<br>%{theta:.1f}°"+
                    "<extra></extra>"))
            fig.add_trace(go.Scatterpolar(r=[1.03], theta=[1], mode='text', text=['Earth'],textposition='top right', name = 'Earth', showlegend=False, hovertemplate = None, hoverinfo = "skip",  textfont=dict(color='mediumseagreen', size=14)))
        
        
        if "mercury" in bodyoptions:
            fig.add_trace(graph['bodytraces'][0][0])
            fig.add_trace(graph['bodytraces'][1])
            #fig.add_trace(go.Scatterpolar(r=[1.03], theta=[1], mode='text', text=['Earth'],textposition='top right', showlegend=False, textfont=dict(color='mediumseagreen', size=14)))
        if "venus" in bodyoptions:
            fig.add_trace(graph['bodytraces'][2][0])
            fig.add_trace(graph['bodytraces'][3])
            
        if "mars" in bodyoptions:
            fig.add_trace(graph['bodytraces'][4][0])
            fig.add_trace(graph['bodytraces'][5])
            
            
        if "title" in plotoptions:
            titledate = datetime.datetime.strptime(infodata['processday'][0], "%Y-%m-%dT%H:%M:%S%z")
            datetitle = datetime.datetime.strftime(titledate, "%Y-%m-%d")
            fig.update_layout(title=datetitle)
        

        # Adjust the subplot size
        fig.update_layout(height=800, width=800)
        
    else:
        # Specify the time range for downloading
        start_time = "2022-08-01"
        end_time = "2022-08-02"

        # Define the time range
        time_range = a.Time(start_time, end_time)

        # Create a query for SOHO LASCO images
        lasco_query = Fido.search(time_range, a.Instrument.lasco)
        
        print(type(lasco_query))
        print(lasco_query)

        # Download the data
        #files = Fido.fetch(lasco_query, max_conn=1)
        # Plot the image
        fig = plt.figure()
        
        x = np.arange(0,4*np.pi,0.1)   # start,stop,step
        y = np.sin(x)
        
        
        plt.plot(x,y)
    
        
        fig = tls.mpl_to_plotly(fig)

    
    
    return fig, {'display': 'block'}

        