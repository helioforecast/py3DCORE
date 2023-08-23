import dash
from dash import dcc, html, Input, Output, State, callback, register_page, no_update, ctx
import dash_mantine_components as dmc
import dash_bootstrap_components as dbc
from dash_iconify import DashIconify

import datetime

from py3dcore.dashcore.utils.utils import get_catevents, load_cat_id

import json
import os

from PIL import Image
pil_image = Image.open("assets/3dcore.png")



register_page(__name__, path="/", icon="fa-solid:home", order=1)


################ COMPONENTS
###########################

#catalogform

datepicker = html.Div(
    dbc.Row(
        [
            dbc.Col(
                dbc.Label("Select a day to process", html_for="date_picker"),
                width={"size": 6, "order": "first"},  # Left-aligned label
            ),
            dbc.Col(
                dcc.DatePickerSingle(
                    id="date_picker",
                    min_date_allowed=datetime.date(2010, 1, 1),
                    max_date_allowed=datetime.date.today(),
                    initial_visible_month=datetime.date(2022, 6, 22),
                    date=datetime.date(2022, 6, 22),
                    persistence=True,
                    persistence_type='session',
                ),
                width={"size": 6, "order": "last"},  # Right-aligned date picker
                style={"display": "flex", "justify-content": "flex-end"},  # Right-align the content
            ),
        ],
        className="mb-3",
    )
)

customdatepicker = html.Div(
    dbc.Row(
        [
            dbc.Col(
                dbc.Label("Select a day to process", html_for="date_picker"),
                width={"size": 6, "order": "first"},  # Left-aligned label
            ),
            dbc.Col(
                dcc.DatePickerSingle(
                    id="custom_date_picker",
                    min_date_allowed=datetime.date(2010, 1, 1),
                    max_date_allowed=datetime.date.today(),
                    initial_visible_month=datetime.date(2022, 6, 22),
                    date=datetime.date(2022, 6, 22),
                    persistence=True,
                    persistence_type='session',
                ),
                width={"size": 6, "order": "last"},  # Right-aligned date picker
                style={"display": "flex", "justify-content": "flex-end"},  # Right-align the content
            ),
        ],
        className="mb-3",
    )
)

event_dropdown = dcc.Dropdown(id="event_dropdown", options=[], placeholder="Select an event", value=None,persistence=True, persistence_type='session',)

#fileform

loadpicker = html.Div(
                dcc.Dropdown(
                id='file-dropdown',
                options=[{'label': f, 'value': f} for f in os.listdir('output/') if os.path.isdir(os.path.join('output/', f)) and not f.startswith('.')],
                placeholder='Select a folder',
                persistence=True,
                persistence_type='session',
            )
)

#manualform

#forms


catalogform = dbc.Card(
    dbc.CardBody(
        [
            datepicker,
            event_dropdown,
            dbc.Button("Submit", id="submitcatalog", color="primary", style={"marginTop": "12px"}),
        ]
    ),
    className="mt-3",
)

fileform = dbc.Card(
    dbc.CardBody(
        [
            loadpicker,
            dbc.Button("Submit", id="submitfile", color="primary", style={"marginTop": "12px"}),
        ]
    ),
    className="mt-3",
)

manualform = dbc.Card(
    dbc.CardBody(
        [
            customdatepicker,
            dcc.Dropdown(
                id='sc-dropdown',
                options=["BepiColombo",
                       "DSCOVR",
                       "PSP",
                       "SolarOrbiter",
                       "STEREO A",
                       "STEREO B",
                       "Wind"],
                placeholder='Select a spacecraft'
            ),
            dbc.Button("Submit", id="submitmanual", color="primary", style={"marginTop": "12px"}),
            ]
    ),
    className="mt-3",
)



# Create the Accordion component
accordion = dbc.Accordion(
    [
        dbc.AccordionItem(catalogform, title="Catalog"),
        dbc.AccordionItem(fileform, title="File", 
                          style={"pointer-events": "none", "opacity": 0.5}
                         ),
        dbc.AccordionItem(manualform, title="Manual", 
                          style={"pointer-events": "none", "opacity": 0.5}
                         ),
        dbc.AccordionItem(fileform, title="Realtime", 
                          style={"pointer-events": "none", "opacity": 0.5}
                         ),
    ],
    persistence=True,
    persistence_type='session',
)


#################### LAYOUT
###########################

layout = dbc.Container(
    [
     html.Br(),
     html.H2("Get started!", 
             className="display-10"),
     html.Br(),
     html.Hr(),
     html.Br(),
     dbc.Row([
         dbc.Col([
             dcc.Markdown(
                 """ 
                 First choose how to initialize the tool: \n
                 **Catalog:** \n
                 >_Choose an event from the [helioforecast catalog](https://helioforecast.space/icmecat)._ \n
                 **File:**\n
                 >_Load from file._\n
                 **Manual:** \n
                 >_Choose a launch time for your event and start from scratch._ \n
                 **Realtime:** \n
                 >_Choose a spacecraft and work on the latest available data._ \n
                 """
             ),
         ], width = 6),
         dbc.Col([html.Div(
         accordion)
                 ], width=6)
     ]),
     
    
    ]
)


    
################# FUNCTIONS
########################### 

def save_widget_state(widget_states, filename):
    with open('output/' + filename, 'w') as file:
        json.dump(widget_states, file)

def load_widget_state(filename):
    with open('output/' + filename, 'r') as file:
        widget_states = json.load(file)
    return widget_states

def get_alternative_sc(sc):
    if sc == 'BepiColombo':
        return "BEPI"
    elif sc == 'DSCOVR':
        return "DSCOVR"
    elif sc == 'PSP':
        return "PSP"
    elif sc == 'SolarOrbiter':
        return "SOLO"
    elif sc == 'STEREO A':
        return "STEREO_A"
    elif sc == 'STEREO B':
        return "STEREO_B"
    elif sc == 'Wind':
        return "Wind"
    
def create_event_info(processday, 
                      begin, 
                      end, 
                      sc,
                      ids,
                      loaded = False, 
                     ):
    return {"processday": processday,
            "begin": begin,
            "end": end,
            "sc": sc,
            "id": ids,
            "loaded": loaded,
            "changed": True
           }
    

    
################# CALLBACKS
###########################


@callback(
    Output("event-info", "data"),
    Input('submitcatalog', 'n_clicks'),
    Input('submitfile', 'n_clicks'),
    Input('submitmanual', 'n_clicks'),
    State("event_dropdown", "value"),
    State('file-dropdown', "value"),
    State("custom_date_picker", "date"),
    State('sc-dropdown',"value"),
    prevent_initial_call=True
)
def update_alert_for_init(cat_clicks, file_clicks, manual_clicks, cat_event, file_event, manual_date, manual_sc):
    changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]
    
    event_info = None
    
    if 'submitcatalog' in changed_id:
        if cat_event == None:
            event = "No event selected"
            event_info = create_event_info(['2022-06-22T18:24:00+00:00'],
                                           ['2022-06-22T18:24:00+00:00'],
                                           ['2022-06-23T05:07:00+00:00'],
                                           'SolarOrbiter',
                                           event,
                                          )
        
        else:
            event_obj = load_cat_id(cat_event)
            event = f"{cat_event}"
            event_info = create_event_info(event_obj.begin,
                                           event_obj.begin,
                                           event_obj.end,
                                           event_obj.sc,
                                           event_obj.id,
                                          )
            
    elif 'submitfile' in changed_id:
        event = f"{file_event}"
        
    elif 'submitmanual' in changed_id:
        sc = get_alternative_sc(manual_sc)
        event = f"ICME_{sc}_CUSTOM_{manual_date.replace('-', '')}"
        dateFormat = "%Y-%m-%d"
        input_datetime = datetime.datetime.strptime(manual_date, dateFormat)
        endtime_formatted = input_datetime + datetime.timedelta(hours=20)
        input_datetime_formatted = input_datetime.strftime("%Y-%m-%d %H:%M:%S%z")
        event_info = create_event_info([input_datetime_formatted],
                                       [input_datetime_formatted],
                                       [endtime_formatted],
                                       [sc],
                                       f"ICME_{sc}_CUSTOM_{manual_date.replace('-', '')}"
                                      )
        
    
    return event_info
    
    
    
    
@callback(
    Output("event_dropdown", "options"),
    Input("date_picker", "date")
)
def update_event_dropdown(date):
    selected_date = datetime.datetime.strptime(date, "%Y-%m-%d").date()
    options = get_catevents(selected_date)
    return options