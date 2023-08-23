import dash
from dash import dcc, html, Output, Input, State, callback, long_callback, ctx
import dash_mantine_components as dmc
from dash_iconify import DashIconify
import dash_bootstrap_components as dbc
from dash.long_callback import DiskcacheLongCallbackManager
from dash.exceptions import PreventUpdate

import plotly.graph_objs as go

import pickle
import os 
import sys
import pandas as pd

import re
import time
import datetime

from py3dcore.dashcore.utils.utils import *
from py3dcore.dashcore.utils.plotting import *
from py3dcore.dashcore.utils.main_fitting import *

from py3dcore.dashcore.assets.config_sliders import *

from py3dcore.methods.abc_smc import abc_smc_worker
from py3dcore.model import SimulationBlackBox, set_random_seed
from py3dcore.methods.data import FittingData

from heliosat.util import sanitize_dt
from typing import Any, Optional, Sequence, Tuple, Union

import multiprocessing as mp
import diskcache
cache = diskcache.Cache("./cache")
long_callback_manager = DiskcacheLongCallbackManager(cache)

app = dash.Dash(__name__, use_pages=True,external_stylesheets=[dbc.themes.BOOTSTRAP, dbc.icons.FONT_AWESOME],long_callback_manager=long_callback_manager)

app.config.suppress_callback_exceptions = True

################# components
############################

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


launchdatepicker = html.Div(
    [
    html.Label(id="launch-label", 
               children="Launch Time:", 
               style={"font-size": "12px", 
                      #"opacity": "0.6"
                     }),
    html.Br(),
    dcc.Slider(
        id="launch_slider",
        min=-168,
        max=-24,
        step=0.5,
        value=-120,
        marks = {i: str(i)+'h' for i in range(-24, -169, -24)},
        persistence=True,
        persistence_type='session',
    ),
    
    ],
    style= {"margin": "20px",
                    "maxWidth": "310px",
                    "overflowX": "auto",
                    "whiteSpace": "nowrap",
                },
)




reference_frame = html.Div(
    [
        dbc.Row(
            [
                html.Label(children="Reference Frame:", style={"font-size": "12px"}),
                dbc.Col(
                    [
                        dcc.Dropdown(
                            id="reference_frame",
                            options=[
                                {"label": "HEEQ", "value": "HEEQ"},
                                {"label": "RTN", "value": "RTN"},
                            ],
                            value="HEEQ",
                        ),
                    ],
                    width={"size": 6},  # Adjust the width as needed
                ),
                dbc.Col(
                    [
                        dcc.Dropdown(
                            id="other_dropdown",
                            options=[
                                {"label": "HGC", "value": "HGC"},
                                {"label": "HGS", "value": "HGS"},
                            ],
                            value="HGC",
                            disabled=True,
                        ),
                    ],
                    width={"size": 4},  # Adjust the width as needed
                ),
                dbc.Col(
                    html.Div(id = "loadgraphstoreicon", 
                             children = "",
                            ),
                    #dbc.Spinner(id="loadgraphstorespinner"),
                    width={"size": 2},  # Adjust the width as needed
                    #style={"textAlign": "right"},  # Align the button to the right within its column
                ),
            ],
            justify="between",  # Distribute the columns to the edges of the row
            #align="center",
        ),
    ],
    style={"margin": "20px", "maxWidth": "310px", "whiteSpace": "nowrap"},
    className="mb-3",
)



modelsliders = html.Div(
    [create_single_slider(
        var['min'],
        var['max'],
        var['def'],
        var['step'],
        var['var_name'],
        var['variablename'],
        var['variablename']+ 'label',
        var['marks'],
        var['unit']
    ) for var in modelslidervars],
    style= {"margin": "20px",
                    "maxWidth": "310px",
                    "overflowX": "auto",
                    "whiteSpace": "nowrap",
                },
)


magsliders = html.Div(
    [create_single_slider(
        var['min'],
        var['max'],
        var['def'],
        var['step'],
        var['var_name'],
        var['variablename'],
        var['variablename']+ 'label',
        var['marks'],
        var['unit']
    ) for var in magslidervars],
    style= {"margin": "20px",
                    "maxWidth": "310px",
                    "overflowX": "auto",
                    "whiteSpace": "nowrap",
                },
)

##################### Layout
############################


topbar = dmc.Navbar(
    # fixed=True,
    # width={"base": 350},
    height=110,
    style={"backgroundColor": "#f8f9fa"},
    children=[
        html.Div(
            style={"display": "flex", "gap": "20px", "justifyContent": "flex-end" },
            children=[
                    create_nav_link(
                        icon=page["icon"], label=page["name"], href=page["path"]
                    )
                    for page in dash.page_registry.values() 
                    #if page["path"].startswith("/subpages")
                ],
        ),
    ],
)

sidebar = dmc.Navbar(
    fixed=True,
    width={"base": 350},
    position={"top": 1, "left": 1},
    height=2500,
    style={"backgroundColor": "#f8f9fa","maxHeight": "calc(100vh - 0px)", "overflowY": "auto"},  # Set maximum height and enable vertical overflow
            
    children=[
        dmc.ScrollArea(
            offsetScrollbars=True,
            type="scroll",
            style={"height": "100%"},  # Set the height of the scroll area
            children=[
                html.H2(
                    "3DCOREweb",
                    className="display-4",
                    style={"marginBottom": 20, "marginLeft": 20, "marginTop": 20},
                ),
                html.Hr(style={"marginLeft": 20}),
                html.P(
                    "Reconstruct CMEs using the 3D Coronal Rope Ejection Model",
                    className="lead",
                    style={"marginLeft": 20},
                ),
                dmc.Divider(
                    label="CME Event",
                    style={"marginBottom": 20, "marginTop": 20, "marginLeft": 20},
                ),
                html.Div(
                    id="event-alert-div",
                    className="alert alert-primary",
                    children="No event selected",
                    style={
                        "margin": "20px",
                        "maxWidth": "310px",
                        "overflowX": "auto",
                        "whiteSpace": "nowrap",
                    },
                ),
                reference_frame,
                launchdatepicker,
                dmc.Divider(
                    label="Model Parameters",
                    style={"marginBottom": 20, "marginTop": 20, "marginLeft": 20},
                ),
                modelsliders,
                dmc.Divider(
                    label="Magnetic Field Parameters",
                    style={"marginBottom": 20, "marginTop": 20, "marginLeft": 20},
                ),
                magsliders,
            ],
        )
    ],
)

app.layout = dmc.Container(
    [dcc.Location(id='url', refresh=False),
     topbar,
     sidebar,
     dmc.Container(
         dash.page_container,
         size="lg",
         pt=20,
         style={"marginLeft": 340,
                "marginTop": 20},
     ),
    # dcc.Store stores the event info
    dcc.Store(id='event-info', storage_type='local'),
    dcc.Store(id='graphstore', storage_type='local'),
    dcc.Store(id='prevhash', storage_type='local'),
    dcc.Store(id='posstore', storage_type='local'),
    #dcc.Store(id='coronostore', storage_type='local'),
    html.Label(id="try-label", 
               children="Launch Time:", 
               style={"font-size": "12px", 
                      #"opacity": "0.6"
                     }),
    ],
    fluid=True,
)

################## callbacks
############################


@callback(
    Output("launch-label", "children"),
    Output("event-alert-div", "children"),
    Input("event-info", "data"),
    Input("launch_slider", "value"),
)
def update_launch_label(data, slider_value):
    if data == None:
        return "Launch Time:", 'No event selected'
    
    else:
        datetime_value = data['begin'][0]
        dateFormat = "%Y-%m-%dT%H:%M:%S%z"
        dateFormat2 = "%Y-%m-%d %H:%M:%S"
        try:
            input_datetime = datetime.datetime.strptime(datetime_value, dateFormat2)
        except ValueError:
            input_datetime = datetime.datetime.strptime(datetime_value, dateFormat)
            
        hours = slider_value
        launch = input_datetime + datetime.timedelta(hours=hours)
        launch_formatted = launch.strftime("%Y-%m-%d %H:%M")
        return f"Launch Time: {launch_formatted}", data['id']
    
    
    
##slider callbacks

def create_callback(var, index):
    html_for = var['variablename'] + 'label'
    ids = var['variablename']
    unit = var['unit']
    func_name = f"update_slider_label{index}"
    
    def callback_func(value, label=f"{var['var_name']}"):
        return f"{label}: {value} {unit}"
    
    callback_func.__name__ = func_name
    app.callback(Output(html_for, "children"), [Input(ids, "value")])(callback_func)
    
callbacks = []

for i, var in enumerate(modelslidervars):
    create_callback(var, i)
    
for i, var in enumerate(magslidervars):
    create_callback(var, i + len(modelslidervars))
##############long callbacks
############################

def starmap(func, args):
    return [func(*_) for _ in args]

def save(path: str, extra_args: Any) -> None:
    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))
    
    with open(path, "wb") as fh:
        pickle.dump(extra_args, fh)        
        

        
@app.long_callback(
    Output("insitufitgraph", "figure"),
    Input("spacecrafttable", "cellValueChanged"),
    Input("graphstore", "data"),
    Input("plotoptions", "value"),
    Input("spacecrafttable", "rowData"),
    Input("event-info", "data"),
    State("restable", "selectedRows"),
    Input("plt-synresults", "n_clicks"),
    State("launch-label", "children"),
    State("fit_dropdown", "value"),
    running = [
        (Output("insitufitspinner", "style"),
            {"visibility": "visible"},
            {"visibility": "hidden"}
        ), 
    ]
)
def plot_insitufig(_, graph, plotoptions, tabledata, infodata, selectedrows, nclicks, launchlabel, name):
    
    if (graph is {}) or (graph is None):  # This ensures that the function is not executed when no figure is present
        fig = {}
        return fig
    
    fig = go.Figure(graph['fig'])
    
    triggered_id = ctx.triggered_id
    
    if triggered_id == None:
        raise PreventUpdate
        
    if "fittingresults" in plotoptions:
        
        filepath = loadpickle(name)
    
        # read from pickle file
        file = open(filepath, "rb")
        data = p.load(file)
        file.close()
        
        refframe = data['data_obj'].reference_frame
        
        ed = generate_ensemble(filepath, graph['t_data'], graph['pos_data'], reference_frame=refframe, reference_frame_to=refframe, max_index=data['model_obj'].ensemble_size)
        
        shadow_data = [
            (ed[0][3][0], None, 'black'),
            (ed[0][3][1], 'rgba(0, 0, 0, 0.15)', 'black'),
            (ed[0][2][0][:, 0], None, 'red'),
            (ed[0][2][1][:, 0], 'rgba(255, 0, 0, 0.15)', 'red'),
            (ed[0][2][0][:, 1], None, 'green'),
            (ed[0][2][1][:, 1], 'rgba(0, 255, 0, 0.15)', 'green'),
            (ed[0][2][0][:, 2], None, 'blue'),
            (ed[0][2][1][:, 2], 'rgba(0, 0, 255, 0.15)', 'blue')
        ]

        for i in range(0, len(shadow_data), 2):
            y1, fill_color, line_color = shadow_data[i]
            y2, _, _ = shadow_data[i + 1]

            fig.add_trace(
                go.Scatter(
                    x=graph['t_data'],
                    y=y1,
                    fill=None,
                    mode='lines',
                    line_color=line_color,
                    line_width=0,
                    showlegend=False
                )
            )

            fig.add_trace(
                go.Scatter(
                    x=graph['t_data'],
                    y=y2,
                    fill='tonexty',
                    mode='lines',
                    line_color=line_color,
                    line_width=0,
                    fillcolor=fill_color,
                    showlegend=False
                )
            )
        
        

    if "ensemblemembers" in plotoptions:
        datetime_format = "Launch Time: %Y-%m-%d %H:%M"
        t_launch = datetime.datetime.strptime(launchlabel, datetime_format)
        
        for row in selectedrows:
            iparams = get_iparams(row)
            rowind = row['Index']
            model_obj = py3dcore.ToroidalModel(t_launch, **iparams) # model gets initialized
            model_obj.generator()
            # Create ndarray with dtype=object to handle ragged nested sequences
            outa = np.array(model_obj.simulator(graph['t_data'], graph['pos_data']), dtype=object)
            outa = np.squeeze(outa[0])
            outa[outa==0] = np.nan
            
            names = graph['names']
                            
            fig.add_trace(
                go.Scatter(
                    x=graph['t_data'],
                    y=outa[:, 0],
                    line=dict(color='red', width=3, dash='dot'),
                    name=names[0]+'_'+str(rowind),
                )
            )
            
            fig.add_trace(
                go.Scatter(
                    x=graph['t_data'],
                    y=outa[:, 1],
                    line=dict(color='green', width=3, dash='dot'),
                    name=names[1]+'_'+str(rowind),
                )
            )
            
            fig.add_trace(
                go.Scatter(
                    x=graph['t_data'],
                    y=outa[:, 2],
                    line=dict(color='blue', width=3, dash='dot'),
                    name=names[2]+'_'+str(rowind),
                )
            )
            
            fig.add_trace(
                go.Scatter(
                    x=graph['t_data'],
                    y=np.sqrt(np.sum(outa**2, axis=1)),
                    line=dict(color='black', width=3, dash='dot'),
                    name='Btot_'+str(rowind),
                )
            )
                            
                                      
        
        
    if "fittingpoints" in plotoptions:
        
        t_s, t_e, t_fit = extract_t(tabledata[0])
        
        fig.add_vrect(
                x0=t_s,
                x1=t_s,
                line=dict(color="Red", width=.5),
                name="Ref_A",  # Add label "Ref_A" for t_s
        )

        fig.add_vrect(
            x0=t_e,
            x1=t_e,
            line=dict(color="Red", width=.5),
            name="Ref_B",  # Add label "Ref_B" for t_e
        )

        for idx, line in enumerate(t_fit):
            fig.add_vrect(
                x0=line,
                x1=line,
                line=dict(color="Black", width=.5),
                name=f"t_{idx + 1}",  # Add labels for each line in t_fit
            )
    if "title" in plotoptions:
        fig.update_layout(title=tabledata[0]['spacecraft'])
    
    if "catalogevent" in plotoptions:
        sc = infodata['sc'][0]
        begin = infodata['begin'][0]
        end = infodata['end'][0]
        
        if infodata['id'][0] == 'I':
            opac = 0
        else:
            opac = 0.3

        dateFormat = "%Y-%m-%dT%H:%M:%S%z"
        dateFormat2 = "%Y-%m-%d %H:%M:%S"
        dateFormat3 = "%Y-%m-%dT%H:%M:%S"
    
        try:
            begin = datetime.datetime.strptime(begin, dateFormat2)
        except ValueError:
            try:
                begin = datetime.datetime.strptime(begin, dateFormat)
            except:
                try:
                    begin = datetime.datetime.strptime(begin, dateFormat3)
                except:
                    pass

        try:
            end = datetime.datetime.strptime(end, dateFormat2)
        except ValueError:
            try:
                end = datetime.datetime.strptime(end, dateFormat)
            except:
                try:
                    end = datetime.datetime.strptime(end, dateFormat3)
                except:
                    pass
                
        fig.add_vrect(
                x0=begin,
                x1=end,
                fillcolor="LightSalmon", 
                opacity=opac,
                layer="below",
                line_width=0
        )
    return fig


@app.long_callback(
    Output("graphstore", "data"),
    Output("prevhash","data"),
    Output("loadgraphstoreicon", "children"),
    Input("event-info","data"),
    Input("reference_frame","value"),
    State("prevhash", "data"),
    running = [
        (Output("loadgraphstoreicon", "children"),
            dbc.Spinner(), " "
        ), 
    ]
)
def generate_graphstore(infodata, reference_frame, prevhash):
    newhash = infodata['id']

    
    #if (clicks is None) and (newhash == prevhash):  # This ensures that the function is not executed on page load
        #raise PreventUpdate
        
    changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]
    
    if (newhash == "No event selected") or (newhash == None):
        return {}, newhash, " "
    
    #### intermediate stuff
    if reference_frame == "HEEQ":
        names = ['Bx', 'By', 'Bz']
    elif reference_frame == "RTN": 
        names = ['Br', 'Bt', 'Bn']
    sc = infodata['sc'][0]
    begin = infodata['begin'][0]
    end = infodata['end'][0]
    
    dateFormat = "%Y-%m-%dT%H:%M:%S%z"
    dateFormat2 = "%Y-%m-%d %H:%M:%S"
    dateFormat3 = "%Y-%m-%dT%H:%M:%S"
    
    try:
        begin = datetime.datetime.strptime(begin, dateFormat2)
    except ValueError:
        try:
            begin = datetime.datetime.strptime(begin, dateFormat)
        except:
            try:
                begin = datetime.datetime.strptime(begin, dateFormat3)
            except:
                pass
        
    try:
        end = datetime.datetime.strptime(end, dateFormat2)
    except ValueError:
        try:
            end = datetime.datetime.strptime(end, dateFormat)
        except:
            try:
                end = datetime.datetime.strptime(end, dateFormat3)
            except:
                pass

    insitubegin = begin - datetime.timedelta(hours=24)
    insituend = end + datetime.timedelta(hours=24)
    
    try:
        print(sc)
        b_data, t_data, pos_data = get_insitudata(reference_frame, sc, insitubegin, insituend)
        
        view_legend_insitu = True
        fig = plot_insitu(names, t_data, b_data, view_legend_insitu) 
    except Exception as e:
        print("An error occurred:", e)
        return {}, newhash, fail_icon
    
    bodytraces = getbodytraces(reference_frame, sc, infodata['processday'][0])
    
    # Extract the date using regular expression
    date_pattern = r'(\d{8})'
    
    match = re.search(date_pattern, newhash[0])
    if match:
        extracted_date = match.group(1)
        extracted_datetime = datetime.datetime.strptime(extracted_date, '%Y%m%d')
    else:
        match = re.search(date_pattern, newhash)
        extracted_date = match.group(1)
        extracted_datetime = datetime.datetime.strptime(extracted_date, '%Y%m%d')
    
    bodydata = load_body_data(reference_frame, extracted_datetime)
    
    
    return {'fig': fig,  'b_data': b_data, 't_data': t_data, 'pos_data': pos_data, 'names': names, 'bodytraces': bodytraces, 'bodydata': bodydata}, newhash, success_icon


@app.long_callback(
    output = Output("statusplaceholder", "children"),
    inputs = Input("run_button", "n_clicks"),
    state=[
        State("launch-label", "children"),
        State("spacecrafttable", "rowData"),
        *[
            State(id, "value") for id in fittingstate
            if id != "launch-label" and id != "spacecrafttable"
        ],
        State("event-info", "data")
    ],
    running = [        
        (Output("run_button", "disabled"), True, False),
        (Output("cancel_button", "disabled"), False, True),
    ],
    cancel=[Input("cancel_button", "n_clicks")],
    progress=Output("statusplaceholder", "children"),
    progress_default=make_progress_graph(0, 512, 0, 0, 0, 1),
    prevent_initial_call=True,
    interval=1000,
)
def main_fit(set_progress, n_clicks, *fittingstate_values):
    
    iter_i = 0 # keeps track of iterations
    hist_eps = [] # keeps track of epsilon values
    hist_time = [] # keeps track of time
    
    balanced_iterations = 3
    time_offsets = [0]
    eps_quantile = 0.25
    epsgoal = 0.3
    kernel_mode = "cm"
    random_seed = 42
    summary_type = "norm_rmse"
    print(fittingstate_values)
    
    base_fitter, fit_coord_system, multiprocessing, itermin, itermax, n_particles, outputfile, njobs, model_kwargs, t_launch  = extract_fitvars(fittingstate_values)
    
    t_launch = sanitize_dt(t_launch)
    
    
    if multiprocessing == True:
        mpool = mp.Pool(processes=njobs) # initialize Pool for multiprocessing
    
    data_obj = FittingData(base_fitter.observers, fit_coord_system)
    data_obj.generate_noise("psd",300)
    
    kill_flag = False
    pcount = 0
    timer_iter = None
    
    try:
        for iter_i in range(iter_i, itermax):
            # We first check if the minimum number of 
            # iterations is reached.If yes, we check if
            # the target value for epsilon "epsgoal" is reached.
            reached = False
            print(str(iter_i) + ' iter_i')
                
            if iter_i >= itermin:
                if hist_eps[-1] < epsgoal:
                    try:
                        return make_progress_graph(pcount, n_particles, hist_eps[-1], hist_eps[-2], iter_i, 2)
                    except:
                        pass
                    kill_flag = True
                    break    
            timer_iter = time.time()
                
            # correct observer arrival times

            if iter_i >= len(time_offsets):
                _time_offset = time_offsets[-1]
            else:
                _time_offset = time_offsets[iter_i]
                
            data_obj.generate_data(_time_offset)
            
            if len(hist_eps) == 0:
                eps_init = data_obj.sumstat(
                    [np.zeros((1, 3))] * len(data_obj.data_b), use_mask=False
                )[0]
                # returns summary statistic for a vector of zeroes for each observer                
                hist_eps = [eps_init, eps_init * 0.98]
                #hist_eps gets set to the eps_init and 98% of it
                print(str(hist_eps) + 'first round')
                hist_eps_dim = len(eps_init) # number of observers
                    
                model_obj_kwargs = dict(model_kwargs)
                model_obj_kwargs["ensemble_size"] = n_particles
                model_obj = base_fitter.model(t_launch, **model_obj_kwargs) # model gets initialized
            sub_iter_i = 0 # keeps track of subprocesses 
            
            _random_seed = random_seed + 100000 * iter_i # set random seed to ensure reproducible results
            # worker_args get stored
            
            worker_args = (
                    iter_i,
                    t_launch,
                    base_fitter.model,
                    model_kwargs,
                    model_obj.iparams_arr,
                    model_obj.iparams_weight,
                    model_obj.iparams_kernel_decomp,
                    data_obj,
                    summary_type,
                    hist_eps[-1],
                    kernel_mode,
                )
            
            try:
                set_progress(make_progress_graph(0, n_particles, hist_eps[-1], hist_eps[-2], iter_i-1, 1))
            except:
                set_progress(make_progress_graph(0, n_particles, hist_eps[-1], eps_init, iter_i-1, 1))
                pass
            
            if multiprocessing == True:
                _results = mpool.starmap(abc_smc_worker, [(*worker_args, _random_seed + i) for i in range(njobs)]) # starmap returns a function for all given arguments
            else:
                _results = starmap(abc_smc_worker, [(*worker_args, _random_seed + i) for i in range(njobs)]) # starmap returns a function for all given arguments
            
            # the total number of runs depends on the ensemble size set in the model kwargs and the number of jobs
            total_runs = njobs * int(model_kwargs["ensemble_size"])  #
            # repeat until enough samples are collected
            while True:
                pcounts = [len(r[1]) for r in _results] # number of particles collected per job 
                _pcount = sum(pcounts) # number of particles collected in total
                dt_pcount = _pcount - pcount # number of particles collected in current iteration
                pcount = _pcount # particle count gets updated

                # iparams and according errors get stored in array
                particles_temp = np.zeros(
                    (pcount, model_obj.iparams_arr.shape[1]), model_obj.dtype
                )
                epses_temp = np.zeros((pcount, hist_eps_dim), model_obj.dtype)
                try:
                    set_progress(make_progress_graph(pcount, n_particles, hist_eps[-1], hist_eps[-2], iter_i, 1))
                except:
                    set_progress(make_progress_graph(pcount, n_particles, hist_eps[-1], eps_init, iter_i, 1))
                    pass

                for i in range(0, len(_results)):
                    particles_temp[
                        sum(pcounts[:i]) : sum(pcounts[: i + 1])
                    ] = _results[i][0] # results of current iteration are stored
                    epses_temp[sum(pcounts[:i]) : sum(pcounts[: i + 1])] = _results[
                        i
                    ][1] # errors of current iteration are stored
            
                if pcount > n_particles:
                    print(str(pcount) + 'reached particles')
                    break
                # if ensemble size isn't reached, continue
                # random seed gets updated

                _random_seed = (
                    random_seed + 100000 * iter_i + 1000 * (sub_iter_i + 1)
                )
            
                if multiprocessing == True:
                    _results_ext = mpool.starmap(abc_smc_worker, [(*worker_args, _random_seed + i) for i in range(njobs)]) # starmap returns a function for all given arguments
                else:
                    _results_ext = starmap(abc_smc_worker, [(*worker_args, _random_seed + i) for i in range(njobs)]) # starmap returns a function for all given arguments
            
                _results.extend(_results_ext) #results get appended to _results
                sub_iter_i += 1
                # keep track of total number of runs
                total_runs += njobs * int(model_kwargs["ensemble_size"])  #

                if pcount == 0:
                    print("no hits, aborting")
                    kill_flag = True
                    break
                
            if kill_flag:
                break

            if pcount > n_particles: # no additional particles are kept
                particles_temp = particles_temp[:n_particles]
            
            # if we're in the first iteration, the weights and kernels have to be initialized. Otherwise, they're updated. 
            if iter_i == 0:
                model_obj.update_iparams(
                    particles_temp,
                    update_weights_kernels=False,
                    kernel_mode=kernel_mode,
                ) # replace iparams_arr by particles_temp
                model_obj.iparams_weight = (
                    np.ones((n_particles,), dtype=model_obj.dtype) / n_particles
                )
                model_obj.update_kernels(kernel_mode=kernel_mode)
            else:
                model_obj.update_iparams(
                    particles_temp,
                    update_weights_kernels=True,
                    kernel_mode=kernel_mode,
                )
            if isinstance(eps_quantile, float):
                new_eps = np.quantile(epses_temp, eps_quantile, axis=0)
                # after the first couple of iterations, the new eps gets simply set to the its maximum value instead of choosing a different eps for each observer
            
                if balanced_iterations > iter_i:
                    new_eps[:] = np.max(new_eps)
            
                hist_eps.append(new_eps)
            elif isinstance(eps_quantile, list) or isinstance(
                eps_quantile, np.ndarray
            ):
                eps_quantile_eff = eps_quantile ** (1 / hist_eps_dim)  #
                _k = len(eps_quantile_eff)  #
                new_eps = np.array(
                    [
                        np.quantile(epses_temp, eps_quantile_eff[i], axis=0)[i]
                        for i in range(_k)
                    ]
                )
                hist_eps.append(new_eps)

            hist_time.append(time.time() - timer_iter)
            iter_i = iter_i + 1  # iter_i gets updated

            # save output to file 
            if outputfile:
                output_file = os.path.join(
                    outputfile, "{0:02d}.pickle".format(iter_i - 1)
                )

                extra_args = {"t_launch": t_launch,
                  "model_kwargs": model_kwargs,
                  "hist_eps": hist_eps,
                  "hist_eps_dim": hist_eps_dim,
                  "base_fitter": base_fitter,
                  "model_obj": model_obj,
                  "data_obj": data_obj,
                  "epses": epses_temp,
                 }

                save(output_file, extra_args)
    finally:
        pass

                
            
        
    
    #print(data_obj)
    
    #set_progress(make_progress_graph(10, 512, 0, 0, 0, 1))
    #time.sleep(3)
    
    
    return make_progress_graph(pcount, n_particles, hist_eps[-1], hist_eps[-2], iter_i, 3)


