import dash
from dash import dcc, html, Output, Input, State, callback, no_update
import dash_mantine_components as dmc
from dash_iconify import DashIconify
import dash_bootstrap_components as dbc

import datetime
import functools

import pandas as pd
import pickle as p
import numpy as np

import os

import plotly.graph_objects as go
import plotly.express as px
import seaborn as sns 

import astrospice
from astropy.time import Time, TimeDelta
import astropy.units as u
from sunpy.coordinates import HeliographicStonyhurst, HeliocentricEarthEcliptic
from astrospice.net.reg import RemoteKernel, RemoteKernelsBase

import urllib.request
from urllib.request import urlopen
from bs4 import BeautifulSoup

import heliosat

from py3dcore.methods.method import BaseMethod


def create_nav_link(icon, label, href):
    
    '''
    Used to create the link to pages.
    '''
    
    return dcc.Link(
        dmc.Group(
            [
                dmc.ThemeIcon(
                    DashIconify(icon=icon, width=18),
                    size=40,
                    radius=40,
                    variant="light",
                    style={"backgroundColor": "#eaeaea", "color": "black"}
                ),
                dmc.Text(label, size="l", color="gray", weight=500),
            ],
            style={"display": "flex", 
                   "alignItems": "center", 
                   "justifyContent": "flex-end", 
                   #"border": "1px solid black", 
                   "padding": "28px"
                  },
        ),
        href=href,
        style={"textDecoration": "none",
               #"marginTop": 40
              },
    )

def create_double_slider(mins, maxs, values, step, label, ids, html_for, marks=None):
    '''
    Creates a double slider with label.
    '''
    
    return html.Div(
    [
        dbc.Label(label, html_for=html_for, style={"font-size": "12px"}),
        dcc.RangeSlider(id=ids, min=mins, max=maxs, step=step, value=values, marks=marks, persistence=True),
    ],
    className="mb-3",
)

def create_single_slider(mins, maxs, values, step, label, ids, html_for, marks, unit):
    '''
    Creates a single slider with label.
    '''
    
    slider_label = dbc.Label(f"{label}: {values}{unit}", id=html_for, style={"font-size": "12px"})
    if marks == None:
        slider = dcc.Slider(id=ids, min=mins, max=maxs, step=step, value=values, persistence=True)
    else:
        slider = dcc.Slider(id=ids, min=mins, max=maxs, step=step, value=values, marks=marks, persistence=True)
    return html.Div([slider_label, slider], className="mb-3")



def custom_indicator(val, prev):
    '''
    creates a custom indicator to visualize the error.
    '''
    
    if isinstance(val, list):
        val = val[0]
    if isinstance(prev, list):
        prev = prev[0]

    if isinstance(val, np.ndarray):
        val = val[0]
    if isinstance(prev, np.ndarray):
        prev = prev[0]

    if prev is None:
        color = 'lightgrey'
        symbol = '-'
        prev = 0
    elif val - prev == 0:
        color = 'lightgrey'
        symbol = '-'
    elif val - prev > 0:
        color = 'red'
        symbol = '▲'
    else:
        color = 'green'
        symbol = '▼'

    val = np.around(val, 3)
    prev = np.around(prev, 3)
    diff = np.around(val - prev,3)

    card = dbc.Card(
        body=True,
        className='text-center',
        children=[
            dbc.CardBody([
                html.H5('RMSE', className='text-dark font-medium'),
                html.H2(val, className='text-black font-large'),
                html.H6(f'{symbol} {diff}', className='text-small', style={'color': color})
            ])
        ]
    )
    return card
    
    


def make_progress_graph(progress, total, rmse, rmse_prev, iteration, status):
    '''
    creates the card that visualizes the fitting progress
    '''
    
    progressperc = (progress / total) * 100
    if progressperc > 100:
        progressperc = 100
        
    if status == 0:
         progress_graph = html.Div([
            html.Div(
                id="alert-div",
                className="alert alert-primary",
                children=f"ⓘ Run fitting process or load existing fit!",
                style={
                    "overflowX": "auto",
                    "whiteSpace": "nowrap",
                },
            )
         ]
         )
        
    elif status == 1:
        # Secondary alert with progress bar and metric card
        progress_graph = html.Div([
            
            dbc.Row([
                dbc.Col([
                    html.Br(),
                    html.Div(
                        id="alert-div",
                        className="alert alert-dark",
                        children=f" Running iteration: {iteration +1 }",
                        style={
                            "overflowX": "auto",
                            "whiteSpace": "nowrap",
                        },                    
                    ),
                    #html.Br(),
                    html.Div(f"{progress}/{total}"),
                    dbc.Progress(
                        id='progress-bar',
                        value=progressperc,
                        max=100,
                        style={"height": "30px"}
                    ),
                ], width=8),  # Set width to half of the row's width (12-column grid system)
                dbc.Col([custom_indicator(rmse, rmse_prev)
                    
                ], width=4)  # Set width to half of the row's width (12-column grid system)
            ]),
        ], style={"max-height": "250px"#, "overflow-y": "auto"
                 })
    elif status == 2:
        progress_graph = html.Div([
                html.Div(
                    id="alert-div",
                    className="alert alert-dark",
                    children=f"✅ Reached target RMSE!",
                    style={
                        "overflowX": "auto",
                        "whiteSpace": "nowrap",
                    },
                ),
                dbc.Row([
                    dbc.Col([
                        html.Div(f"{progress}/{total}"),
                        dbc.Progress(
                            id='progress-bar',
                            value=progressperc,
                            max=100,
                            style={"height": "30px"}
                        ),
                    ], width=8),  
                    dbc.Col([custom_indicator(rmse, rmse_prev)

                    ], width=4) 
                ]),
            ], style={"max-height": "250px"
                     })
        
    elif status == 3:
        progress_graph = html.Div([
                html.Div(
                    id="alert-div",
                    className="alert alert-dark",
                    children=f"✅ Reached maximum number of iterations!",
                    style={
                        "overflowX": "auto",
                        "whiteSpace": "nowrap",
                    },
                ),
                dbc.Row([
                    dbc.Col([
                        html.Div(f"{progress}/{total}"),
                        dbc.Progress(
                            id='progress-bar',
                            value=progressperc,
                            max=100,
                            style={"height": "30px"}
                        ),
                    ], width=8),  # Set width to half of the row's width (12-column grid system)
                    dbc.Col([custom_indicator(rmse, rmse_prev)

                    ], width=4)  # Set width to half of the row's width (12-column grid system)
                ]),
            ], style={"max-height": "250px"#, "overflow-y": "auto"
                     })

    # Combine the secondary alert and progress graph components
    return html.Div([progress_graph])


@functools.lru_cache()    
def get_insitudata(mag_coord_system, sc, insitubegin, insituend):
    
    '''
    used to generate the insitudata for the graphstore (app.py)
    '''
        
    if mag_coord_system == 'HEEQ':
        reference_frame = 'HEEQ'
        names = ['Bx', 'By', 'Bz']
    else:
        names = ['Br', 'Bt', 'Bn']
        
    if (sc == 'BepiColombo') or (sc == 'BEPI'):
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
    pos = observer_obj.trajectory(dt, reference_frame=reference_frame)
    
    
    return b, dt, pos



class BepiPredict(RemoteKernelsBase):
    '''
    enable handling Bepi Positions
    '''
    
    body = 'mpo'
    type = 'predict'

    def get_remote_kernels(self):
        """
        Returns
        -------
        list[RemoteKernel]
        """
        page = urlopen('https://naif.jpl.nasa.gov/pub/naif/BEPICOLOMBO/kernels/spk/')
        soup = BeautifulSoup(page, 'html.parser')

        kernel_urls = []
        for link in soup.find_all('a'):
            href = link.get('href')
            if href is not None and href.startswith('bc'):
                fname = href.split('/')[-1]
                matches = self.matches(fname)
                if matches:
                    kernel_urls.append(
                        RemoteKernel(f'https://naif.jpl.nasa.gov/pub/naif/BEPICOLOMBO/kernels/spk/{href}', *matches[1:]))

        return kernel_urls

    @staticmethod
    def matches(fname):
        """
        Check if the given filename matches the pattern of this kernel.

        Returns
        -------
        matches : bool
        start_time : astropy.time.Time
        end_time : astropy.time.Time
        version : int
        """
        # Example filename: bc_mpo_fcp_00154_20181020_20251102_v01.bsp 
        fname = fname.split('_')
        if (len(fname) != 7 or
                fname[0] != 'bc' or
                fname[1] != 'mpo' or
                fname[2] != 'fcp'):
            return False

        start_time = Time.strptime(fname[4], '%Y%m%d')
        end_time = Time.strptime(fname[5], '%Y%m%d')
        version = int(fname[6][1:3])
        return True, start_time, end_time, version


def plot_body3d(data_list, nowdate, color, sc):
    '''
    plots the current 3d position for a body
    '''
    
    
    data = np.array(data_list, dtype=object)
    df_columns = ['time', 'r', 'lon', 'lat', 'x', 'y', 'z']
    df = pd.DataFrame(data, columns=df_columns)
    
    df['time'] = pd.to_datetime(df['time'])  # Convert time column to datetime objects
    
    # Filter data based on date and nowdate
    now_data = df[df['time']== nowdate]
    
    x_now, y_now, z_now, now_time = now_data['x'], now_data['y'], now_data['z'], now_data['time']
    
    r_now, lon_now, lat_now = now_data['r'], now_data['lon'], now_data['lat']
    
    times_now_list = now_time.tolist()

    now_time_str = [time.strftime("%Y-%m-%d %H:%M:%S") for time in times_now_list]
    
    trace = go.Scatter3d(x=x_now, y=y_now, z=z_now,
                         mode='markers', 
                         marker=dict(size=4, 
                                     #symbol='square',
                                     color=color),
                         name=sc, 
                         customdata=np.vstack((r_now, lon_now, lat_now )).T,  # Custom data for r, lat, lon values
                         showlegend=True, 
                         hovertemplate="%{text}<br><b>(x, y, z):</b> (%{x:.2f} AU, %{y:.2f} AU, %{z:.2f} AU)<br><b>(r, lon, lat):</b> (%{customdata[0]:.2f} AU, %{customdata[2]:.2f}°, %{customdata[1]:.2f}°)<extra>" 
                         + sc + "</extra>", text=[now_time_str]),
            

    return trace



def load_body_data(mag_coord_system, date):
    '''
    Used in generate_graphstore to load the body data for later 3d plotting
    '''
    dt = TimeDelta(0.5 * u.hour)
    delta = datetime.timedelta(days=10)
    now_time = Time(date, scale='utc')
    start_time, end_time = now_time - delta, now_time + delta
    frame = HeliographicStonyhurst()
    
    times = Time(np.arange(start_time, end_time, dt))
    
    planets = [1, # Mercury
              2, #Venus
              4, #Mars
              ]
    colors = ['slategrey',
             'darkgoldenrod',
             'red']
    names = ['Mercury', 'Venus', 'Mars']
    
    dicc = {}
    
    ########## BODY
    
    for i, planet in enumerate(planets):
        coords = astrospice.generate_coords(planet, times)
        coords = coords.transform_to(frame)
        data = np.zeros(np.size(times),dtype=[('time',object),('r','f8'),('lon','f8'),('lat','f8'),('x','f8'),('y','f8'),('z','f8')])
        data = data.view(np.recarray)
        data.time = times.to_datetime()
        data.r = coords.radius.to(u.au).value
        data.lon = coords.lon.value #degrees
        data.lat = coords.lat.value
        [data.x, data.y, data.z] = sphere2cart(data.r, np.deg2rad(-data.lat+90), np.deg2rad(data.lon))
        
        dicc[names[i]] = {'data': data, 'color': colors[i]}

    
    return dicc
    
def load_pos_data(mag_coord_system, sc, date):
    
    '''
    used to load the data of a specific spacecraft (manually)
    '''
    
    dt = TimeDelta(0.5 * u.hour)
    delta = datetime.timedelta(days=10)
    now_time = Time(date, scale='utc')
    start_time, end_time = now_time - delta, now_time + delta
    frame = HeliographicStonyhurst()
    
    times = Time(np.arange(start_time, end_time, dt))
    
    ########## SPACECRAFT
    
    if (sc == 'SolarOrbiter') or (sc == "SOLO"):
        solo_kernel = astrospice.registry.get_kernels('solar orbiter', 'predict')[0]
        
        color = 'coral'
        coords = astrospice.generate_coords('Solar orbiter', times).transform_to(frame)
        
    elif (sc == "PSP"):
        kernels_psp = astrospice.registry.get_kernels('psp', 'predict')[0]
        color = 'black'
        coords = astrospice.generate_coords('Solar probe plus', times).transform_to(frame)
    
    elif (sc == "STEREO-A"):
        kernels_sta = astrospice.registry.get_kernels('stereo-a', 'predict')[0]
        color = 'darkred'
        coords = astrospice.generate_coords('Stereo ahead', times).transform_to(frame)
        
    elif (sc == "BEPI"):
        kernels_bepi = astrospice.registry.get_kernels('mpo', 'predict')[0]
        color = 'blue'
        coords = astrospice.generate_coords('Bepicolombo mpo', times).transform_to(frame)
        
    data = np.zeros(np.size(times),dtype=[('time',object),('r','f8'),('lon','f8'),('lat','f8'),('x','f8'),('y','f8'),('z','f8')])
    data = data.view(np.recarray)
    data.time = times.to_datetime()
    data.r = coords.radius.to(u.au).value
    data.lon = coords.lon.value #degrees
    data.lat = coords.lat.value
    [data.x, data.y, data.z] = sphere2cart(data.r, np.deg2rad(-data.lat+90), np.deg2rad(data.lon))

    
    return {'data': data, 'color': color}
    
def get_posdata(mag_coord_system, sc, date, threed = False):
    '''
    used to load the traces for the 2d plot for a manually selected sc
    '''
    dt = TimeDelta(24 * u.hour)
    delta = datetime.timedelta(days=100)
    now_time = Time(date, scale='utc')
    start_time, end_time = now_time - delta, now_time + delta
    frame = HeliographicStonyhurst()
    
    times_past = Time(np.arange(start_time, now_time, dt))
    times_future = Time(np.arange(now_time, end_time, dt))
    
    ########## SOLAR ORBITER
    
    if (sc == 'SolarOrbiter') or (sc == "SOLO"):
        solo_kernel = astrospice.registry.get_kernels('solar orbiter', 'predict')[0]
        
        color = 'coral'
        coords_past = astrospice.generate_coords('Solar orbiter', times_past).transform_to(frame)
        coords_future = astrospice.generate_coords('Solar orbiter', times_future).transform_to(frame)
        coords_now = astrospice.generate_coords('Solar orbiter', now_time).transform_to(frame)
        
    elif (sc == "PSP"):
        kernels_psp = astrospice.registry.get_kernels('psp', 'predict')[0]
        color = 'black'
        coords_past = astrospice.generate_coords('Solar probe plus', times_past).transform_to(frame)
        coords_future = astrospice.generate_coords('Solar probe plus', times_future).transform_to(frame)
        coords_now = astrospice.generate_coords('Solar probe plus', now_time).transform_to(frame)
    
    elif (sc == "STEREO-A"):
        kernels_sta = astrospice.registry.get_kernels('stereo-a', 'predict')[0]
        color = 'darkred'
        coords_past = astrospice.generate_coords('Stereo ahead', times_past).transform_to(frame)
        coords_future = astrospice.generate_coords('Stereo ahead', times_future).transform_to(frame)
        coords_now = astrospice.generate_coords('Stereo ahead', now_time).transform_to(frame)
        
        
    elif (sc == "BEPI"):
        kernels_bepi = astrospice.registry.get_kernels('mpo', 'predict')[0]
        color = 'blue'
        coords_past = astrospice.generate_coords('Bepicolombo mpo', times_past).transform_to(frame)
        coords_future = astrospice.generate_coords('Bepicolombo mpo', times_future).transform_to(frame)
        coords_now = astrospice.generate_coords('Bepicolombo mpo', now_time).transform_to(frame)
        
    
        
        
    r_past, lon_past, lat_past = coords_past.radius.to(u.au).value, coords_past.lon.value, coords_past.lat.value
    r_future, lon_future, lat_future = coords_future.radius.to(u.au).value, coords_future.lon.value, coords_future.lat.value
    r_now, lon_now, lat_now = coords_now.radius.to(u.au).value, coords_now.lon.value, coords_now.lat.value
    
    
     # Convert Time objects to formatted strings
    times_past_str = [time.datetime.strftime("%Y-%m-%d %H:%M:%S") for time in times_past]
    times_future_str = [time.datetime.strftime("%Y-%m-%d %H:%M:%S") for time in times_future]
    now_time_str = now_time.datetime.strftime("%Y-%m-%d %H:%M:%S")

    if threed == False:
        traces = [
            go.Scatterpolar(r=r_past, theta=lon_past, mode='lines', line=dict(color=color), name=sc + '_past_100', showlegend=False, hovertemplate="%{text}<br>%{r:.1f} AU<br>%{theta:.1f}°<extra>" + sc + "</extra>", text=times_past_str),
            go.Scatterpolar(r=r_future, theta=lon_future, mode='lines', line=dict(color=color, dash='dash'), name=sc + '_future_100', showlegend=False, hovertemplate="%{text}<br>%{r:.1f} AU<br>%{theta:.1f}°<extra>" + sc + "</extra>", text=times_future_str),
            go.Scatterpolar(r=r_now, theta=lon_now, mode='markers', marker=dict(size=8, symbol='square', color=color), name=sc, showlegend=False, hovertemplate="%{text}<br>%{r:.1f} AU<br>%{theta:.1f}°<extra>" + sc + "</extra>", text=[now_time_str]),
        ]
        
        
    else:
        x_past,y_past,z_past = sphere2cart(r_past, np.deg2rad(-lat_past+90), np.deg2rad(lon_past))
        x_future,y_future,z_future = sphere2cart(r_future, np.deg2rad(-lat_future+90), np.deg2rad(lon_future))
        x_now,y_now,z_now = sphere2cart(r_now, np.deg2rad(-lat_now+90), np.deg2rad(lon_now))
        
        traces = [
            go.Scatter3d(x=x_past, y=y_past, z=z_past,
                         mode='lines', 
                         line=dict(color=color), 
                         name=sc + '_past_100', 
                         customdata=np.vstack((r_past, lat_past, lon_past)).T,  # Custom data for r, lat, lon values
                         showlegend=False, 
                         hovertemplate="%{text}<br><b>(x, y, z):</b> (%{x:.2f} AU, %{y:.2f} AU, %{z:.2f} AU)<br><b>(r, lon, lat):</b> (%{customdata[0]:.2f} AU, %{customdata[2]:.2f}°, %{customdata[1]:.2f}°)<extra>" 
                         + sc + "</extra>", text=times_past_str),
            go.Scatter3d(x=x_future, y=y_future, z=z_future,
                         mode='lines', 
                         line=dict(color=color), 
                         name=sc + '_future_100', 
                         customdata=np.vstack((r_past, lat_past, lon_past)).T,  # Custom data for r, lat, lon values
                         showlegend=False, 
                         hovertemplate="%{text}<br><b>(x, y, z):</b> (%{x:.2f} AU, %{y:.2f} AU, %{z:.2f} AU)<br><b>(r, lon, lat):</b> (%{customdata[0]:.2f} AU, %{customdata[2]:.2f}°, %{customdata[1]:.2f}°)<extra>" 
                         + sc + "</extra>", text=times_future_str),
            go.Scatter3d(x=x_now, y=y_now, z=z_now,
                         mode='markers', 
                         marker=dict(size=3, 
                                     symbol='square',
                                     color=color),
                         name=sc, 
                         customdata=np.vstack((r_past, lat_past, lon_past)).T,  # Custom data for r, lat, lon values
                         showlegend=True, 
                         hovertemplate="%{text}<br><b>(x, y, z):</b> (%{x:.2f} AU, %{y:.2f} AU, %{z:.2f} AU)<br><b>(r, lon, lat):</b> (%{customdata[0]:.2f} AU, %{customdata[2]:.2f}°, %{customdata[1]:.2f}°)<extra>" 
                         + sc + "</extra>", text=[now_time_str]),
            
        ]
    
    return traces


def round_to_hour_or_half(dt):
    
    '''
    round launch datetime
    '''
    
    remainder = dt.minute % 30
    if remainder < 15:
        dt -= datetime.timedelta(minutes=remainder)
    else:
        dt += datetime.timedelta(minutes=30 - remainder)
    return dt.replace(second=0, microsecond=0)

def process_coordinates(data_list, date, nowdate, color, sc):
    '''
    plot spacecraft 3d position from previously loaded data
    '''
    
    data = np.array(data_list, dtype=object)
    df_columns = ['time', 'r', 'lon', 'lat', 'x', 'y', 'z']
    df = pd.DataFrame(data, columns=df_columns)
    
    df['time'] = pd.to_datetime(df['time'])  # Convert time column to datetime objects
    
    # Filter data based on date and nowdate
    filtered_data = df[(df['time'] >= date) & (df['time'] <= date + datetime.timedelta(days=7))]



    # Split data into past, future, and now coordinates
    past_data = filtered_data[filtered_data['time'] < nowdate]
    future_data = filtered_data[filtered_data['time'] > nowdate]
    now_data = filtered_data[filtered_data['time'] == nowdate]

    # Extract coordinates for each category
    x_past, y_past, z_past, times_past = past_data['x'], past_data['y'], past_data['z'], past_data['time']
    x_future, y_future, z_future, times_future = future_data['x'], future_data['y'], future_data['z'], future_data['time']
    x_now, y_now, z_now, now_time = now_data['x'], now_data['y'], now_data['z'], now_data['time']
    
    r_past, lon_past, lat_past = past_data['r'], past_data['lon'], past_data['lat']
    r_future, lon_future, lat_future = future_data['r'], future_data['lon'], future_data['lat']
    r_now, lon_now, lat_now = now_data['r'], now_data['lon'], now_data['lat']
    
    # Convert Timestamp Series to a list of Timestamps
    times_past_list = times_past.tolist()
    times_future_list = times_future.tolist()
    times_now_list = now_time.tolist()

    # Convert Timestamps to formatted strings
    times_past_str = [time.strftime("%Y-%m-%d %H:%M:%S") for time in times_past_list]
    times_future_str = [time.strftime("%Y-%m-%d %H:%M:%S") for time in times_future_list]
    now_time_str = [time.strftime("%Y-%m-%d %H:%M:%S") for time in times_now_list]
    
    traces = [
            go.Scatter3d(x=x_past, y=y_past, z=z_past,
                         mode='lines', 
                         line=dict(color=color), 
                         name=sc + '_past_100', 
                         customdata=np.vstack((r_past, lat_past, lon_past)).T,  # Custom data for r, lat, lon values
                         showlegend=False, 
                         hovertemplate="%{text}<br><b>(x, y, z):</b> (%{x:.2f} AU, %{y:.2f} AU, %{z:.2f} AU)<br><b>(r, lon, lat):</b> (%{customdata[0]:.2f} AU, %{customdata[2]:.2f}°, %{customdata[1]:.2f}°)<extra>" 
                         + sc + "</extra>", text=times_past_str),
            go.Scatter3d(x=x_future, y=y_future, z=z_future,
                         mode='lines', 
                         line=dict(color=color, dash='dash'), 
                         name=sc + '_future_100', 
                         customdata=np.vstack((r_past, lat_past, lon_past)).T,  # Custom data for r, lat, lon values
                         showlegend=False, 
                         hovertemplate="%{text}<br><b>(x, y, z):</b> (%{x:.2f} AU, %{y:.2f} AU, %{z:.2f} AU)<br><b>(r, lon, lat):</b> (%{customdata[0]:.2f} AU, %{customdata[2]:.2f}°, %{customdata[1]:.2f}°)<extra>" 
                         + sc + "</extra>", text=times_future_str),
            go.Scatter3d(x=x_now, y=y_now, z=z_now,
                         mode='markers', 
                         marker=dict(size=3, 
                                     symbol='square',
                                     color=color),
                         name=sc, 
                         customdata=np.vstack((r_past, lat_past, lon_past)).T,  # Custom data for r, lat, lon values
                         showlegend=True, 
                         hovertemplate="%{text}<br><b>(x, y, z):</b> (%{x:.2f} AU, %{y:.2f} AU, %{z:.2f} AU)<br><b>(r, lon, lat):</b> (%{customdata[0]:.2f} AU, %{customdata[2]:.2f}°, %{customdata[1]:.2f}°)<extra>" 
                         + sc + "</extra>", text=[now_time_str]),
            
        ]

    return traces

def getbodytraces(mag_coord_system, sc, date, threed = False):
    
    '''
    This function is used at startup to obtain the 2d traces.
    '''
    
    
    date_object = datetime.datetime.strptime(date, "%Y-%m-%dT%H:%M:%S%z")
    now_time = Time(date_object, scale='utc')
    frame = HeliographicStonyhurst()
    
    traces = []
    
    planets = [1, # Mercury
              2, #Venus
              4, #Mars
              ]
    colors = ['slategrey',
             'darkgoldenrod',
             'red']
    names = ['Mercury', 'Venus', 'Mars']
    
    for i, planet in enumerate(planets):
        coords = astrospice.generate_coords(planet, now_time)
        coords = coords.transform_to(frame)
    
        r = coords.radius.to(u.au).value
        lon = coords.lon.value #degrees
        lat = coords.lat.value
        
        if threed == False:
            
            trace = go.Scatterpolar(r=r, theta=lon, mode='markers', marker=dict(size=10, color=colors[i]), name = names[i], showlegend=False, hovertemplate="%{r:.1f} AU<br>%{theta:.1f}°"+
                        "<extra></extra>"),
            traces.append(trace)

            nametrace = go.Scatterpolar(r=r + 0.03, theta=lon + 0.03, mode='text', text=names[i],textposition='top right', showlegend=False, hovertemplate = None, hoverinfo = "skip", textfont=dict(color=colors[i], size=14))
            traces.append(nametrace)
        else:
            x,y,z = sphere2cart(r, np.deg2rad(-lat+90), np.deg2rad(lon))
            
            scs = names[i]

            trace = go.Scatter3d(
                x=x, y=y, z=z,
                mode='markers',
                marker=dict(size=4, color=colors[i]),
                name=names[i],
                customdata=np.vstack((r, lat, lon)).T,  # Custom data for r, lat, lon values
                hovertemplate="<b>(x, y, z):</b> (%{x:.2f} AU, %{y:.2f} AU, %{z:.2f} AU)<br><b>(r, lon, lat):</b> (%{customdata[0]:.2f} AU, %{customdata[2]:.2f}°, %{customdata[1]:.2f}°)<extra>" + sc + "</extra>",
                text=names[i]  # Text to display in the hover label
            )
            traces.append(trace)

        
    return traces

def cart2sphere(x,y,z):
    r = np.sqrt(x**2+ y**2 + z**2)           
    theta = np.arctan2(z,np.sqrt(x**2+ y**2))
    phi = np.arctan2(y,x)                    
    return (r, theta, phi)
    

def sphere2cart(r,lat,lon):
    x = r * np.sin( lat ) * np.cos( lon )
    y = r * np.sin( lat ) * np.sin( lon )
    z = r * np.cos( lat )
    return (x, y,z)


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
    Returns from helioforecast.space the event list for a given day.
    Used during startup.
    '''
    url='https://helioforecast.space/static/sync/icmecat/HELIO4CAST_ICMECAT_v21.csv'
    icmecat=pd.read_csv(url)
    starttime = icmecat.loc[:,'icme_start_time']
    idd = icmecat.loc[:,'icmecat_id']
    sc = icmecat.loc[:, 'sc_insitu']
    endtime = icmecat.loc[:,'mo_end_time']
    mobegintime = icmecat.loc[:, 'mo_start_time']
    
    evtList = []
    dateFormat="%Y/%m/%d %H:%M"
    begin = pd.to_datetime(starttime, format=dateFormat)
    mobegin = pd.to_datetime(mobegintime, format=dateFormat)
    end = pd.to_datetime(endtime, format=dateFormat)
    
    
    for i, event in enumerate(mobegin):
        if (mobegin[i].year == day.year and mobegin[i].month == day.month and mobegin[i].day == day.day):
            evtList.append(str(Event(mobegin[i], end[i], idd[i], sc[i])))
    
    if len(evtList) == 0:
        evtList = ['No events returned', ]
    
    return evtList

def load_cat_id(idd):
    '''
    Returns from helioforecast.space the event with a given ID.
    '''
    
    url='https://helioforecast.space/static/sync/icmecat/HELIO4CAST_ICMECAT_v21.csv'
    icmecat=pd.read_csv(url)
    starttime = icmecat.loc[:,'icme_start_time']
    idds = icmecat.loc[:,'icmecat_id']
    sc = icmecat.loc[:, 'sc_insitu']
    endtime = icmecat.loc[:,'mo_end_time']
    mobegintime = icmecat.loc[:, 'mo_start_time']
    
    dateFormat="%Y/%m/%d %H:%M"
    begin = pd.to_datetime(starttime, format=dateFormat)
    mobegin = pd.to_datetime(mobegintime, format=dateFormat)
    end = pd.to_datetime(endtime, format=dateFormat)
    
    i = np.where(idds == idd)[0]
    
    
    return Event(mobegin[i], end[i], idds[i], sc[i])
        
def load_cat(date):
    '''
    Returns from helioforecast.space the event list for a given day
    '''
    url='https://helioforecast.space/static/sync/icmecat/HELIO4CAST_ICMECAT_v21.csv'
    icmecat=pd.read_csv(url)
    starttime = icmecat.loc[:,'icme_start_time']
    idd = icmecat.loc[:,'icmecat_id']
    sc = icmecat.loc[:, 'sc_insitu']
    endtime = icmecat.loc[:,'mo_end_time']
    mobegintime = icmecat.loc[:, 'mo_start_time']
    
    evtList = []
    dateFormat="%Y/%m/%d %H:%M"
    begin = pd.to_datetime(starttime, format=dateFormat)
    mobegin = pd.to_datetime(mobegintime, format=dateFormat)
    end = pd.to_datetime(endtime, format=dateFormat)

    
    for i, event in enumerate(mobegin):
        if (mobegin[i].year == date.year and mobegin[i].month == date.month and mobegin[i].day == date.day and mobegin[i].hour == date.hour):
            return Event(mobegin[i], end[i], idd[i], sc[i])
        
        
        
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


        
def load_fit(name):
    
    filepath = loadpickle(name)
    
    # read from pickle file
    file = open(filepath, "rb")
    data = p.load(file)
    file.close()
    
    observers = data['data_obj'].observers
    
    ###### tablenew
    
    t0 = data['t_launch']
    
    for i, observer in enumerate(observers):
        
        obs_dic = {
                "spacecraft": [""],
                "ref_a": [""],
                "ref_b": [""],
                "t_1": [""],
                "t_2": [""],
                "t_3": [""],
                "t_4": [""],
                "t_5": [""],
                "t_6": [""],
            }
        
        if observer[0] == "SOLO":
            obs_dic['spacecraft'] = ['SolarOrbiter']
        elif observer[0] == "BEPI":
            obs_dic['spacecraft'] = ['BepiColombo']
        else:
            obs_dic['spacecraft'] = [observer[0]]
            
        obs_dic['ref_a'] = [observer[2].strftime("%Y-%m-%d %H:%M")]
        obs_dic['ref_b'] = [observer[3].strftime("%Y-%m-%d %H:%M")]
        
        for j, dt in enumerate(observer[1]):
            t_key = "t_" + str(j + 1)
            obs_dic[t_key] = [dt.strftime("%Y-%m-%d %H:%M")]         
        df = pd.DataFrame(obs_dic)
        
        if i == 0:
            tablenew = df
        else:
            pd.concat([tablenew, df_new_row])
            
    ####### modelsliders
    
    long_new = [data['model_kwargs']['iparams']['cme_longitude']['minimum'], data['model_kwargs']['iparams']['cme_longitude']['maximum']]
    
    # Create a dictionary to store the results
    result_dict = []
    mag_result_dict = []

    # Iterate through each key in iparams
    for key, value in data['model_kwargs']['iparams'].items():
        # Get the maximum and minimum values if they exist, otherwise use the default_value
        maximum = value.get('maximum', value.get('default_value'))
        minimum = value.get('minimum', value.get('default_value'))
        # Create the long_new list
        var_new = [minimum, maximum]
        
        # Check if the key starts with 'mag' or 't_fac' and append it to the corresponding dictionary
        if key.startswith('mag') or key.startswith('t_fac'):
            mag_result_dict.append(var_new)
        else:
            result_dict.append(var_new)
            
            
    ######## particle slider
    
    given_values = [265, 512, 1024, 2048]
    closest_value = min(given_values, key=lambda x: abs(x - len(data['epses'])))

    
    partslid_new = given_values.index(closest_value)
    
    
    
    ######## reference frame
    
    refframe_new = data['data_obj'].reference_frame
    
    ####### "fitter-radio", 'n_jobs' no update
    ####### n_iter
    
    n_iternew = [len(data['hist_eps'])+1,len(data['hist_eps'])+1]
    
    #######ensemble size
    
    ens = data['model_kwargs']['ensemble_size']
    
    if ens == int(2**16):
        ens_new = 16
    elif ens == int(2**17):
        ens_new = 17
    elif ens == int(2**18):
        ens_new = 18
        
    ####### resulttab
    
    iparams_arrt = data["model_obj"].iparams_arr
    resdf = pd.DataFrame(iparams_arrt)
    
    rescols = resdf.columns.values.tolist()
    
    # drop first column
    resdf.drop(resdf.columns[[0]], axis=1, inplace=True)
    # rename columns
    resdf.columns = ['Longitude', 'Latitude', 'Inclination', 'Diameter 1 AU', 'Aspect Ratio', 'Launch Radius', 'Launch Velocity', 'T_Factor', 'Expansion Rate', 'Magnetic Decay Rate', 'Magnetic Field Strength 1 AU', 'Background Drag', 'Background Velocity']
    
    
    # Scatter plot matrix using go.Splom
    statsfig = go.Figure(data=go.Splom(
        dimensions=[dict(label=col, values=resdf[col]) for col in resdf.columns],
        diagonal_visible=False,
        marker=dict(size=5, symbol='cross', line=dict(width=1, color='black'),
                   ),
    showupperhalf = False))
    
    # Add histograms on the diagonal
    #for i in range(len(resdf.columns)):
    #    statsfig.add_trace(go.Histogram(x=resdf.iloc[:, i], xaxis=f"x{i + 1}", yaxis=f"y{i + 1}"))


    # Customizing the axis and grid styles
    statsfig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
    statsfig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')

    # Update layout to control height and width
    statsfig.update_layout(height=2500, width=2500)
    

    # Add 'eps' column from data["epses"]
    resepses = data["epses"]
    num_rows = min(len(resepses), len(resdf))
    resdf.insert(0, 'RMSE Ɛ', resepses[:num_rows])
    
    # Calculate statistics
    mean_values = resdf.mean()
    std_values = resdf.std()
    median_values = resdf.median()
    min_values = resdf.min()
    max_values = resdf.max()
    q1_values = resdf.quantile(0.25)
    q3_values = resdf.quantile(0.75)
    skewness_values = resdf.skew()
    kurtosis_values = resdf.kurt()
    
    mean_row = pd.DataFrame(
        [mean_values, std_values, median_values, min_values, max_values,
         q1_values, q3_values, skewness_values,
         kurtosis_values],
        columns=resdf.columns
    )

    
    # Add the index column
    resdf.insert(0, 'Index', range(0, num_rows ))

    # Round all values to 2 decimal points
    resdfnew = round_dataframe(resdf)
    
    ###### stattab
    
   # Add the index column
    mean_row.insert(0, 'Index', ["Mean", "Standard Deviation", "Median", "Minimum", "Maximum", "Q1", "Q3", "Skewness", "Kurtosis"],)

    # Round all values to 2 decimal points
    mean_rownew = round_dataframe(mean_row)
    
    mean_row_df = pd.DataFrame([mean_rownew.iloc[0]], columns=resdf.columns)
    # Concatenate resdf and mean_row_df along the rows (axis=0) and reassign it to resdf
    resdffinal = pd.concat([resdfnew, mean_row_df], axis=0)

    return tablenew, *result_dict, *mag_result_dict, partslid_new, refframe_new, no_update, no_update, no_update, n_iternew, ens_new, resdffinal.to_dict("records"), t0, mean_rownew.to_dict("records"), statsfig

def round_dataframe(df):
    return df.applymap(lambda x: round(x, 2) if isinstance(x, float) else x)




def get_iparams(row):
    
    model_kwargs = {
        "ensemble_size": int(1), #2**17
        "iparams": {
            "cme_longitude": {
                "distribution": "fixed",
                "default_value": row['Longitude']
            },
            "cme_latitude": {
                "distribution": "fixed",
                "default_value": row['Latitude']
            },
            "cme_inclination": {
                "distribution": "fixed",
                "default_value": row['Inclination']
            },
            "cme_diameter_1au": {
                "distribution": "fixed",
                "default_value": row['Diameter 1 AU']
            },
            "cme_aspect_ratio": {
                "distribution": "fixed",
                "default_value": row['Aspect Ratio']
            },
            "cme_launch_radius": {
                "distribution": "fixed",
                "default_value": row['Launch Radius']
            },
            "cme_launch_velocity": {
                "distribution": "fixed",
                "default_value": row['Launch Velocity']
            },
            "t_factor": {
                "distribution": "fixed",
                "default_value": row['T_Factor']
            },
            "cme_expansion_rate": {
                "distribution": "fixed",
                "default_value": row['Expansion Rate']
            },
            "magnetic_decay_rate": {
                "distribution": "fixed",
                "default_value": row['Magnetic Decay Rate']
            },
            "magnetic_field_strength_1au": {
                "distribution": "fixed",
                "default_value": row['Magnetic Field Strength 1 AU']
            },
            "background_drag": {
                "distribution": "fixed",
                "default_value": row['Background Drag']
            },
            "background_velocity": {
                "distribution": "fixed",
                "default_value": row['Background Velocity']
            }
        }
    }
    
    return model_kwargs



def get_iparams_live(*modelstatevars):
    
    model_kwargs = {
        "ensemble_size": int(1), #2**17
        "iparams": {
            "cme_longitude": {
                "distribution": "fixed",
                "default_value": modelstatevars[0]
            },
            "cme_latitude": {
                "distribution": "fixed",
                "default_value": modelstatevars[1]
            },
            "cme_inclination": {
                "distribution": "fixed",
                "default_value": modelstatevars[2]
            },
            "cme_diameter_1au": {
                "distribution": "fixed",
                "default_value": modelstatevars[3]
            },
            "cme_aspect_ratio": {
                "distribution": "fixed",
                "default_value": modelstatevars[4]
            },
            "cme_launch_radius": {
                "distribution": "fixed",
                "default_value": modelstatevars[5]
            },
            "cme_launch_velocity": {
                "distribution": "fixed",
                "default_value": modelstatevars[6]
            },
            "t_factor": {
                "distribution": "fixed",
                "default_value": modelstatevars[10]
            },
            "cme_expansion_rate": {
                "distribution": "fixed",
                "default_value": modelstatevars[7]
            },
            "magnetic_decay_rate": {
                "distribution": "fixed",
                "default_value": modelstatevars[11]
            },
            "magnetic_field_strength_1au": {
                "distribution": "fixed",
                "default_value": modelstatevars[12]
            },
            "background_drag": {
                "distribution": "fixed",
                "default_value": modelstatevars[8]
            },
            "background_velocity": {
                "distribution": "fixed",
                "default_value": modelstatevars[9]
            }
        }
    }
    
    return model_kwargs
    

    



def generate_ensemble(path: str, dt: datetime.datetime, posdata, reference_frame: str="HCI", reference_frame_to: str="HCI", perc: float=0.95, max_index=None) -> np.ndarray:
    
    """
    Generates an ensemble from a Fitter object.
    
    Arguments:
        path                where to load from
        dt                  time axis used for fitting
        reference_frame     reference frame used for fitter object
        reference_frame_to  reference frame for output data
        perc                percentage of quantile to be used
        max_index           how much of ensemble is kept
    Returns:
        ensemble_data 
    """

    
    ensemble_data = []
    

    ftobj = BaseMethod(path) # load Fitter from path
    
    # simulate flux ropes using iparams from loaded fitter
    ensemble = np.squeeze(np.array(ftobj.model_obj.simulator(dt, posdata)[0]))
        
    # how much to keep of the generated ensemble
    if max_index is None:
        max_index = ensemble.shape[1]

    ensemble = ensemble[:, :max_index, :]
    
    print(ensemble)

    # transform frame
    if reference_frame != reference_frame_to:
        for k in range(0, ensemble.shape[1]):
            ensemble[:, k, :] = transform_reference_frame(dt, ensemble[:, k, :], reference_frame, reference_frame_to)

    ensemble[np.where(ensemble == 0)] = np.nan

    # generate quantiles
    b_m = np.nanmean(ensemble, axis=1)

    b_s2p = np.nanquantile(ensemble, 0.5 + perc / 2, axis=1)
    b_s2n = np.nanquantile(ensemble, 0.5 - perc / 2, axis=1)

    b_t = np.sqrt(np.sum(ensemble**2, axis=2))
    b_tm = np.nanmean(b_t, axis=1)

    b_ts2p = np.nanquantile(b_t, 0.5 + perc / 2, axis=1)
    b_ts2n = np.nanquantile(b_t, 0.5 - perc / 2, axis=1)

    ensemble_data.append([None, None, (b_s2p, b_s2n), (b_ts2p, b_ts2n)])
        
    return ensemble_data