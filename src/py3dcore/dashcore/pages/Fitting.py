import dash
from dash import dcc, html, Input, Output, State, callback, register_page, no_update, ctx, long_callback
import dash_mantine_components as dmc
import plotly.express as px
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc
import dash_ag_grid as dag

from py3dcore.dashcore.assets.config_sliders import modelslidervars, magslidervars, fittingstate, modelstate
from py3dcore.dashcore.utils.utils import load_fit

from dash.long_callback import CeleryLongCallbackManager

from celery import Celery

import diskcache
import time
import os

import pandas as pd

import datetime
import functools 

from py3dcore.dashcore.utils.utils import create_double_slider, get_insitudata, make_progress_graph
from py3dcore.dashcore.utils.plotting import plot_insitu

register_page(__name__, icon="mdi:chart-histogram", order=2)

df = pd.DataFrame({
            "spacecraft": [""],
            "ref_a": [""],
            "ref_b": [""],
            "t_1": [""],
            "t_2": [""],
            "t_3": [""],
            "t_4": [""],
            "t_5": [""],
            "t_6": [""],
        })

################ COMPONENTS
###########################



fitterradio = html.Div(
                            [
                                dbc.Label("Fitter", style={"font-size": "12px"}),
                                dcc.RadioItems(
                            options=[
                                {"label": " ABC-SMC", "value": "abc-smc"},
                            ],
                            id="fitter-radio",
                            value="abc-smc",
                            inline=True,
                        ),
                            ],
                            className="form-group",
                        )

multiprocesscheck = html.Div(
    [
        dbc.Label("Number of Jobs", style={"font-size": "12px"}),
        html.Br(),
        html.Div(
            [
                dcc.Input(
                    id='n_jobs',
                    type='number',
                    min=1,
                    max=300,
                    step=1,
                    value=8,
                    style={"margin-right": "10px"}
                ),
                dcc.Checklist(
                    id="multiprocesscheck",
                    options=[{"label": " Multiprocessing", "value": "multiprocessing"}],
                    value=["multiprocessing"]
                )
            ],
            style={"display": "inline-flex", "align-items": "center"}
        )
    ],
    className="form-group",
)



numiter = create_double_slider(1,15, [3,5], 1, 'Number of Iterations', 'n_iter', 'n_iter', {i: str(i) for i in range(1, 16, 1)})

particlenum = html.Div(
                            [
                                dbc.Label("Number of Particles",style={"font-size": "12px"}),
                                dcc.Slider(
                                    id="particle-slider",
                                    min=0,
                                    max=3,
                                    marks={
                                        0: "265",
                                        1: "512",
                                        2: "1024",
                                        3: "2048"
                                    },
                                    value=1,
                                    step=None,
                                    included=False,
                                    updatemode="drag"
                                ),
                            ],
                            className="form-group",
                        )

ensemblenum = html.Div(
                            [
                                dbc.Label("Ensemble Size",style={"font-size": "12px"}),
                                dcc.Slider(
                                    id="ensemble-slider",
                                    min=16,
                                    max=18,
                                    marks={
                                        16: "2^16",
                                        17: "2^17",
                                        18: "2^18"
                                    },
                                    value=17,
                                    step=None,
                                    included=False,
                                    updatemode="drag"
                                ),
                            ],
                            className="form-group",
                        )


tabform = dbc.Card(
    dbc.CardBody(
        [
            html.Div(
                dbc.Form(
                    [
                        fitterradio,
                        html.Br(),
                        multiprocesscheck,
                        html.Br(),
                        numiter,
                        particlenum,
                        html.Br(),
                        ensemblenum,
                        html.Br(),
                    ]
                ),
                style={"max-height": "400px", "overflow-y": "auto"},
            ),
            dbc.ButtonGroup(
                [
                    dbc.Button("Run", id = 'run_button', color="primary"),
                    dbc.Button("Cancel", id = 'cancel_button', color="secondary"),
                ],
                className="mr-2",
            ),
        ]
    ),
    className="mt-3",
)


modelsliders_double = html.Div(
    [create_double_slider(
        var['min'],
        var['max'],
        [var['min'], var['max']],
        var['step'],
        var['var_name'],
        var['variablename_double'],
        var['variablename_double']+ 'label',
        var['marks'],
    ) for var in modelslidervars],
    style= {"margin": "20px",
                    "maxWidth": "310px",
                    "overflowX": "auto",
                    "whiteSpace": "nowrap",
                },
)


magsliders_double = html.Div(
    [create_double_slider(
        var['min'],
        var['max'],
        [var['min'], var['max']],
        var['step'],
        var['var_name'],
        var['variablename_double'],
        var['variablename_double']+ 'label',
        var['marks']
    ) for var in magslidervars],
    style= {"margin": "20px",
                    "maxWidth": "310px",
                    "overflowX": "auto",
                    "whiteSpace": "nowrap",
                },
)


tabparam = dbc.Card(
    dbc.CardBody(
        [
           dmc.Divider(
                    label="Model Parameters",
                    #style={"marginBottom": 20, "marginTop": 20, "marginLeft": 20},
                ),
                modelsliders_double,
                dmc.Divider(
                    label="Magnetic Field Parameters",
                    #style={"marginBottom": 20, "marginTop": 20, "marginLeft": 20},
                ),
                magsliders_double,
        ]
    ),
    className="mt-3",style={"max-height": "400px", "overflow-y": "auto"},
)

tabload = dbc.Card(
    dbc.CardBody(
        [
            html.Div(
                dbc.Form(
                    [
                        dcc.Dropdown(id="fit_dropdown", options=[], placeholder="Select a fit", value=None,persistence=True, persistence_type='session',)
                    ]
                ),
                #style={"max-height": "400px", "overflow-y": "auto"},
            ),
            html.Br(),
            dbc.Button("Load Results", id = 'loadfit_button', color="primary", disabled=False),
        ]
    ),
    className="mt-3",
)


# Create the Accordion component
accordion = dbc.Accordion(
    [
        dbc.AccordionItem(tabparam, title="Parameters"),
        dbc.AccordionItem(tabform, title="Fitting"),
        dbc.AccordionItem(tabload, title="Load"),
    ]

)

# Create the Accordion component
statusplaceholder = html.Div(
            [make_progress_graph(0, 512, 0, 0, 0, 0)
            ],id="statusplaceholder",
)

# Create the Spacecraft Table

columnDefs = [
    {
        "headerName": "Spacecraft",
        "field": "spacecraft", 
        "checkboxSelection": True,
        "cellEditor": "agSelectCellEditor",
        "cellEditorParams": {
            'values': ["BepiColombo",
                       "DSCOVR",
                       "PSP",
                       "SolarOrbiter",
                       "STEREO A",
                       "STEREO B",
                       "Wind"]}
        },
    {
        "headerName": "Reference A",
        "field": "ref_a",
    },
    {
        "headerName": "Reference B",
        "field": "ref_b",
    },
    {
        "headerName": "t_1",
        "field": "t_1",
    },
    {
        "headerName": "t_2",
        "field": "t_2",
    },
    {
        "headerName": "t_3",
        "field": "t_3",
    },
    {
        "headerName": "t_4",
        "field": "t_4",
    },
    {
        "headerName": "t_5",
        "field": "t_5",
    },
    {
        "headerName": "t_6",
        "field": "t_6",
    },
]

defaultColDef = {
    "resizable": True,
    "editable": True,
    "minWidth": 180,
    "flex":1,
}

rowData = df.to_dict("records")



spacecrafttable = dag.AgGrid(
    id="spacecrafttable",
    className="ag-theme-alpine",
    columnDefs=columnDefs,
    rowData=rowData,
    defaultColDef=defaultColDef,
    dashGridOptions={"undoRedoCellEditing": True},
    style={"max-height": "200px", "overflow-y": "auto"},
    persistence=True,
    persistence_type='session',
)

table = dbc.Card(
                            [
                                dbc.CardBody(
                                    [
                                        spacecrafttable,
                                        html.Span(
                                            [
                                                dbc.Button(
                                                    id="delete-sc",
                                                    children="Delete Spacecraft",
                                                    color="secondary",
                                                    size="md",
                                                    className='mt-3 me-1',
                                                    disabled=True,
                                                ),
                                                dbc.Button(
                                                    id="add-sc",
                                                    children="Add Spacecraft",
                                                    color="primary",
                                                    size="md",
                                                    className='mt-3 me-1',
                                                    disabled=True,
                                                ),
                                                dbc.Button(
                                                    id="ld-evt",
                                                    children="Load from event data",
                                                    color="primary",
                                                    size="md",
                                                    className='mt-3',
                                                    disabled=False,
                                                ),
                                            ]
                                        ),
                                    ]
                                ),
                            ],
                        )

# Create the fittingform layout
fittingform = dbc.Container(
    [
        dbc.Row(
            [
                dbc.Col(accordion, width=6),  # Set the accordion width to half of the screen (6 out of 12 columns)
                dbc.Col(statusplaceholder,width=6),  # Set the Markdown width to half of the screen (6 out of 12 columns)
            ]
        )
    ]
)
    

insitufitgraphcontainer = html.Div(
    [
        dmc.CheckboxGroup(
            id="plotoptions",
            label="Options for plotting",
            #description="This is anonymous",
            orientation="horizontal",
            withAsterisk=False,
            offset="md",
            mb=10,
            children=[
                dmc.Checkbox(label="Fitting Points", value="fittingpoints", color="green"),
                dmc.Checkbox(label="Catalog Event", value="catalogevent", color="green"),
                dmc.Checkbox(label="Title", value="title", color="green"),
                dmc.Checkbox(label="Fitting Results", value="fittingresults", 
                             color="green", disabled=True),
                dmc.Checkbox(label="Ensemble Members", value="ensemblemembers", color="green", disabled=True),
            ],
            value=[ "catalogevent"],
        ),
        dbc.Spinner(id="insitufitspinner"),
        dcc.Graph(id="insitufitgraph"),
    ],
    id = "insitufitgraphcontainer",
)



# Create the Spacecraft Table

resultstatcolumnDefs = [
    {
        "headerName": "Statistic",
        "field": "Index",
    },
    {
        "headerName": 'RMSE Ɛ',
        "field": 'RMSE Ɛ',
    },
    {
        "headerName": 'Longitude',
        "field": 'Longitude',
    },
    {
        "headerName": 'Latitude',
        "field": 'Latitude',
    },
    {
        "headerName": 'Inclination',
        "field": 'Inclination',
    },
    {
        "headerName": 'Diameter 1 AU',
        "field": 'Diameter 1 AU',
    },
    {
        "headerName": 'Aspect Ratio',
        "field": 'Aspect Ratio',
    },
    {
        "headerName": 'Launch Radius',
        "field": 'Launch Radius',
    },
    {
        "headerName": 'Launch Velocity',
        "field": 'Launch Velocity',
    },
    {
        "headerName": 'T_Factor',
        "field": 'T_Factor',
    },
    {
        "headerName": 'Expansion Rate',
        "field": 'Expansion Rate',
    },
    {
        "headerName": 'Magnetic Decay Rate',
        "field": 'Magnetic Decay Rate',
    },
    {
        "headerName": 'Magnetic Field Strength 1 AU',
        "field": 'Magnetic Field Strength 1 AU',
    },
    {
        "headerName": 'Background Drag',
        "field": 'Background Drag',
    },
    {
        "headerName": 'Background Velocity',
        "field": 'Background Velocity',
    },
]


resulttabcolumnDefs = [
    {
        "headerName": "Index",
        "field": "Index",
        "checkboxSelection": True,
        "rowSelection": "multiple",  # Enable multiple row selection
        "sortable": True,  # The index column should not be sortable
        "minWidth": 100,  # Adjust the width of the index column as needed
        "rowMultiSelectWithClick": True,
        "resizable": False,  # Disable resizing of the index column
        "suppressSizeToFit": True  # Avoid the index column from participating in sizeToFit calculations
    },
    {
        "headerName": 'RMSE Ɛ',
        "field": 'RMSE Ɛ',
        "sortable": True,  # Enable sorting for this column
    },
    {
        "headerName": 'Longitude',
        "field": 'Longitude',
        "sortable": True,  # Enable sorting for this column
    },
    {
        "headerName": 'Latitude',
        "field": 'Latitude',
        "sortable": True,  # Enable sorting for this column
    },
    {
        "headerName": 'Inclination',
        "field": 'Inclination',
        "sortable": True,  # Enable sorting for this column
    },
    {
        "headerName": 'Diameter 1 AU',
        "field": 'Diameter 1 AU',
        "sortable": True,  # Enable sorting for this column
    },
    {
        "headerName": 'Aspect Ratio',
        "field": 'Aspect Ratio',
        "sortable": True,  # Enable sorting for this column
    },
    {
        "headerName": 'Launch Radius',
        "field": 'Launch Radius',
        "sortable": True,  # Enable sorting for this column
    },
    {
        "headerName": 'Launch Velocity',
        "field": 'Launch Velocity',
        "sortable": True,  # Enable sorting for this column
    },
    {
        "headerName": 'T_Factor',
        "field": 'T_Factor',
        "sortable": True,  # Enable sorting for this column
    },
    {
        "headerName": 'Expansion Rate',
        "field": 'Expansion Rate',
        "sortable": True,  # Enable sorting for this column
    },
    {
        "headerName": 'Magnetic Decay Rate',
        "field": 'Magnetic Decay Rate',
        "sortable": True,  # Enable sorting for this column
    },
    {
        "headerName": 'Magnetic Field Strength 1 AU',
        "field": 'Magnetic Field Strength 1 AU',
        "sortable": True,  # Enable sorting for this column
    },
    {
        "headerName": 'Background Drag',
        "field": 'Background Drag',
        "sortable": True,  # Enable sorting for this column
    },
    {
        "headerName": 'Background Velocity',
        "field": 'Background Velocity',
        "sortable": True,  # Enable sorting for this column
    },
]

defaultresColDef = {
    "resizable": True,
    "editable": False,
    "flex":1,
    "sortable": True,
    "minWidth": 120,
}

resdf = pd.DataFrame(columns = ['Index','RMSE Ɛ','Longitude', 'Latitude', 'Inclination', 'Diameter 1 AU', 'Aspect Ratio', 'Launch Radius', 'Launch Velocity', 'T_Factor', 'Expansion Rate', 'Magnetic Decay Rate', 'Magnetic Field Strength 1 AU', 'Background Drag', 'Background Velocity'] )

rowresData = resdf.to_dict("records")

defaultstatColDef = {
    "resizable": True,
    "editable": False,
    "flex":1,
    "sortable": True,
    "minWidth": 150,
}

restable = dag.AgGrid(
    id="restable",
    className="ag-theme-alpine",
    columnDefs=resulttabcolumnDefs,
    rowData=rowresData,
    defaultColDef=defaultresColDef,
    style={"max-height": "250px", "overflow-y": "auto"},
    persistence=True,
    persistence_type='session',
    dashGridOptions={"rowSelection":"multiple"},
)

statstable = dag.AgGrid(
    id="statstable",
    className="ag-theme-alpine",
    columnDefs=resultstatcolumnDefs,
    rowData=rowresData,
    defaultColDef=defaultstatColDef,
    style={"max-height": "250px", "overflow-y": "auto"},
    persistence=True,
    persistence_type='session',
)

dbc.ButtonGroup(
                [
                    dbc.Button("Run", id = 'run_button', color="primary"),
                    dbc.Button("Cancel", id = 'cancel_button', color="secondary"),
                ],
                className="mr-2",
            ),


restabtable = html.Div(
    [
        html.H5("Results", className="display-10"),
        html.Br(),
        dbc.Card(
            [
                dbc.CardBody(
                    [
                        html.H6("Ensemble Members", className="display-10"),
                        restable,
                        html.Div(  # Create a new container to hold the button and apply the style
                            [ 
                                dbc.ButtonGroup(
                                    [
                                        dbc.Button(
                                            id="ld-synresults",
                                            children="Adjust Sliders",
                                            color="primary",
                                            size="md",
                                            className="mt-3",
                                            disabled=True,
                                        ),
                                        dbc.Button(
                                            id="plt-synresults",
                                            children="Plot Synthetic Insitu",
                                            color="primary",
                                            size="md",
                                            className="mt-3",
                                            disabled=True,
                                        )
                                    ],
                                    className="mr-2",
                                )
                            ],
                            style={"text-align": "right"},  # Align the button to the right
                        ),
                        html.H6("Statistics", className="display-10"),
                        statstable,
                        dcc.Graph(id="statsgraph"),
                    ]
                ),
            ]
        ),
    ],
    id="restabtable-card",
    style={"display": "none"},
)

#################### LAYOUT
###########################

# Define the app layout
layout = dbc.Container(
    [html.Br(),
     html.H2("Numerical Fitting", 
             className="display-10"),
     html.Br(),
     html.Hr(),
     html.Br(),
     fittingform,
     html.Br(),
     html.Hr(),
     html.Br(),
     html.H5("Observers", 
             className="display-10"),
     html.Br(),
     table,
     html.Br(),
     html.Hr(),
     html.Br(),
     html.H5("Insitu Data", 
             className="display-10"),
     html.Br(),
     insitufitgraphcontainer,
     html.Hr(),
     html.Br(),
     restabtable,
     ]
)


################# FUNCTIONS
###########################




################# CALLBACKS
###########################

@callback(
    *[
            Output(id, "value") for id in modelstate
        ],
    Input("ld-synresults", "n_clicks"),
    State("restable", "selectedRows"),
)
def update_buttons(nlicks, selected_rows):
    if (nlicks == None) or (nlicks) == 0:
            raise PreventUpdate
    
    row = selected_rows[0]
    values_to_return = []
    for id in modelstate:
        key = {
            'longit': 'Longitude',
            'latitu': 'Latitude',
            'inc': 'Inclination',
            'dia': 'Diameter 1 AU',
            'asp': 'Aspect Ratio',
            'l_rad': 'Launch Radius',
            'l_vel': 'Launch Velocity',
            'exp_rat': 'Expansion Rate',
            'b_drag': 'Background Drag',
            'bg_vel': 'Background Velocity',
            't_fac': 'T_Factor',
            'mag_dec': 'Magnetic Decay Rate',
            'mag_strength': 'Magnetic Field Strength 1 AU',
        }.get(id)
        
        if key is not None:
            value = row.get(key, no_update)
            values_to_return.append(value)
        else:
            values_to_return.append(no_update)

    return values_to_return



@callback(
    Output("plt-synresults", "disabled"),
    Output("ld-synresults", "disabled"),
    Input("restable", "selectedRows"),
)
def update_buttons(selected_rows):
    plot_button_disabled = True
    another_button_disabled = True

    if selected_rows:
        plot_button_disabled = False
        if len(selected_rows) == 1:
            another_button_disabled = False

    return plot_button_disabled, another_button_disabled


@callback(
    Output("fit_dropdown", "options"),
    Input("event-info", "data"),
    Input("reference_frame", "value"),
)
def update_fit_dropdown(data, refframe):
    ids = data['id'][0] + refframe[0]
    options = [f for f in os.listdir('output/') if os.path.isdir(os.path.join('output/', f)) and f.startswith(ids)],
    return options[0]

# add or delete rows of table
@callback(
    output = [
        Output("spacecrafttable", "deleteSelectedRows"),
        Output("spacecrafttable", "rowData"),
        *[
            Output(id, "value") for id in fittingstate
            if id != "launch-label" and id != "spacecrafttable"
        ],
        Output("restabtable-card", "style"),
        Output("restable", "rowData"),
        Output("launch_slider", "value"),
        Output("statstable", "rowData"),
        Output("statsgraph", "figure"),
        Output("plotoptions", "children"),
    ],
    inputs = [
        Input("ld-evt","n_clicks"),
        Input("loadfit_button", "n_clicks"),
    Input("delete-sc", "n_clicks"),
    Input("add-sc", "n_clicks"),
    #Input("event-info", "data"),
    ],
    state = [
        State("event-info", "data"),
        State("spacecrafttable", "rowData"),
        State("fit_dropdown", "value"),],
    prevent_initial_call = True
)
def update_table_or_load(n_ldevt, n_load, n_dlt, n_add, infodata, data, name):
    

    triggered_id = ctx.triggered_id
    
    
    if triggered_id == "loadfit_button":
        
        if (n_load == None) or (n_load) == 0:
            raise PreventUpdate
         
        tablenew, *fitting_values, resdfdic, t0, mean_row, statfig = load_fit(name)

        dtval = infodata['begin'][0]
        dateFormat = "%Y-%m-%dT%H:%M:%S%z"
        dateFormat2 = "%Y-%m-%d %H:%M:%S"
        
        try:
            dtval_in = datetime.datetime.strptime(dtval, dateFormat2)
        except ValueError:
            dtval_in = datetime.datetime.strptime(dtval, dateFormat)
        
        ltval = (t0-dtval_in).total_seconds() // 3600
        
        plotchildren = [
            dmc.Checkbox(label="Fitting Points", value="fittingpoints", color="green"),
            dmc.Checkbox(label="Catalog Event", value="catalogevent", color="green"),
            dmc.Checkbox(label="Title", value="title", color="green"),
            dmc.Checkbox(label="Fitting Results", value="fittingresults", color="green"),
            dmc.Checkbox(label="Ensemble Members", value="ensemblemembers", color="green"),
        ]
        
        
        return False, tablenew.to_dict("records"), *fitting_values, {"display": "block"}, resdfdic, ltval, mean_row, statfig, plotchildren
    
    
    
    if triggered_id == "add-sc":
        new_row = {
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
        df_new_row = pd.DataFrame(new_row)
        updated_table = pd.concat([pd.DataFrame(data), df_new_row])
        return False, updated_table.to_dict("records"), *[no_update] * 20, {"display": "none"}, rowresData, no_update,rowresData, no_update, no_update

    elif triggered_id == "delete-sc":
        return True, *[no_update] * 21, {"display": "none"}, rowresData, no_update,rowresData, no_update, no_update
    
    else:
        try:
            begin = datetime.datetime.strptime(infodata["begin"][0], "%Y-%m-%d %H:%M:%S")
        except:
            begin = datetime.datetime.strptime(infodata["begin"][0], "%Y-%m-%dT%H:%M:%S%z")
        try:
            end = datetime.datetime.strptime(infodata["end"][0], "%Y-%m-%dT%H:%M:%S")
        except:
            end = datetime.datetime.strptime(infodata["end"][0], "%Y-%m-%dT%H:%M:%S%z")
        refa = begin - datetime.timedelta(hours = 6)
        refb = end + datetime.timedelta(hours = 6)
        t_1 = begin + datetime.timedelta(hours = 2)
        t_2 = begin + datetime.timedelta(hours = 4)
        row = {
            "spacecraft": infodata["sc"],
            "ref_a": refa.strftime("%Y-%m-%d %H:%M"),
            "ref_b": refb.strftime("%Y-%m-%d %H:%M"),
            "t_1": t_1.strftime("%Y-%m-%d %H:%M"),
            "t_2": t_2.strftime("%Y-%m-%d %H:%M"),
            "t_3": [""],
            "t_4": [""],
            "t_5": [""],
            "t_6": [""],
        }
        df_row = pd.DataFrame(row)
        return False, df_row.to_dict("records"), *[no_update] * 20, {"display": "none"},rowresData, no_update,rowresData, no_update, no_update
    
    
