from dash import html, dcc
import plotly.express as px
import numpy as np
import dash_bootstrap_components as dbc


def get_chat_layout2():
    default_message = html.P(className='llm-msg', 
                             style={'height': '200px', 'overflowY': 'scroll'},
                             children=["LLM: Hello! How can I assist you today?"])
    chat_layout = [
        dbc.CardHeader("CALMS-LLM"),
        dbc.CardBody(
            html.Div([
                html.Div(id='chat-container', children=[default_message]),
                dcc.Input(id='user-input', type='text', 
                          placeholder='Message to LLM', 
                          style={'width': '90%'}),
                html.Button('Send', id='send-button', n_clicks=0)
            ], 
            style={'height': 'relative'}
            # style={'position': 'relative'}
            )
        )
    ]
    return chat_layout

def get_chat_layout():
    default_message = html.P(className='llm-msg', 
                             children=["LLM: Hello! How can I assist you today?"])
    chat_layout = [
        dbc.CardHeader("CALMS-LLM"),
        dbc.CardBody(
            [
                dbc.Row(
                    html.Div(id='chat-container', children=[default_message]),
                    style={'height': '200px','overflowY': 'scroll'},
                    className="h-75",
                    ),
                dbc.Row(
                    html.Div([
                        dcc.Input(id='user-input', type='text', 
                                placeholder='Message to LLM', 
                                style={'width': '90%'}),
                        html.Button('Send', id='send-button', n_clicks=0)
                            ], 
                    ),
                    className="h-75",
                ),
            ],

        )
    ]
    return chat_layout