from dash import dcc, html, Input, Output, State, callback, register_page
import dash_mantine_components as dmc
import plotly.express as px
import numpy as np

import datetime

import plotly.graph_objs as go

import py3dcore
from py3dcore.dashcore.assets.config_sliders import modelstate
from py3dcore.dashcore.utils.utils import *

from dash_iconify import DashIconify
import dash_bootstrap_components as dbc

register_page(__name__, icon="streamline:money-graph-analytics-business-product-graph-data-chart-analysis", order=3)

layout = html.Div(
    [
        
        dbc.Col(
                    dmc.CheckboxGroup(
                        id="plotoptions_sliderfig",
                        label="Options for plotting",
                        orientation="horizontal",
                        withAsterisk=False,
                        offset="md",
                        mb=10,
                        children=[
                            dmc.Checkbox(label="Catalog Event", value="catalogevent", color="green"),
                            dmc.Checkbox(label="Synthetic Insitu", value="synthetic", color="green", disabled=False),
                            dmc.Checkbox(label="Title", value="title", color="green"),
                        ],
                        value=["synthetic", "catalogevent", "title"],
                    ),
                    width=8,  # Adjust the width of the CheckboxGroup column
                ),
        dcc.Graph(id="sliderfiginsitu"),
    
        ]
)



@callback(
    Output("sliderfiginsitu", "figure"),
    Input("graphstore", "data"),
    Input("event-info", "data"),
    State("launch-label", "children"),
    Input("plotoptions_sliderfig", "value"),
    Input("reference_frame", "value"),
    *[
            Input(id, "value") for id in modelstate
        ],
    
)
def update_sliderfiginsitu(graph, infodata, launchlabel,plotoptions, refframe, *modelstatevars):
    
    if (graph is {}) or (graph is None):  # This ensures that the function is not executed when no figure is present
        fig = {}
        return fig
    
    fig = go.Figure(graph['fig'])
    
    if "title" in plotoptions:
        fig.update_layout(title=infodata['id'][0])
    
    if "catalogevent" in plotoptions:
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
                
        fig.add_vrect(
                x0=begin,
                x1=end,
                fillcolor="LightSalmon", 
                opacity=0.5,
                layer="below",
                line_width=0
        )
    if "synthetic" in plotoptions:
        iparams = get_iparams_live(*modelstatevars)
        datetime_format = "Launch Time: %Y-%m-%d %H:%M"
        t_launch = datetime.datetime.strptime(launchlabel, datetime_format)
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
                name=names[0]+'_synth',
            )
        )

        fig.add_trace(
            go.Scatter(
                x=graph['t_data'],
                y=outa[:, 1],
                line=dict(color='green', width=3, dash='dot'),
                name=names[1]+'_synth',
            )
        )

        fig.add_trace(
            go.Scatter(
                x=graph['t_data'],
                y=outa[:, 2],
                line=dict(color='blue', width=3, dash='dot'),
                name=names[2]+'_synth',
            )
        )

        fig.add_trace(
            go.Scatter(
                x=graph['t_data'],
                y=np.sqrt(np.sum(outa**2, axis=1)),
                line=dict(color='black', width=3, dash='dot'),
                name='Btot_synth',
            )
        )
        
        
    return fig