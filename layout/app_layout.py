from dash import html, dcc
import dash_bootstrap_components as dbc
from layout.header import app_header
from layout.visualization import get_plot_layout, get_pdf_layout
from layout.chat import get_chat_layout
# from layout.controls import get_controls_layout
# from layout.scaler import get_scaler_layout
# from layout.scan import get_scan_layout


def get_app_layout(
    src_app_logo="assets/aps_logo.png",
    logo_height="60px",
    app_title="Advanced Photon Source | X-ray Science Division",
):
    ##### DEFINE LAYOUT #####
    layout = html.Div(
        [
            app_header(src_app_logo, logo_height, app_title),
            dbc.Container(
                [
                    dbc.Row(
                        [
                            # dbc.Col(get_controls_layout(component_gui), width=4),
                            dbc.Col(
                                chat(), width=6,
                            ),
                            dbc.Col(
                                visualization_collaspe(), width=6,
                            ),
                        ]
                    ),
                    dcc.Interval(id="refresh-interval"),
                    dcc.Store(id="plot-cache", data=None),
                    dcc.Store(id="chat-history", data=None),
                    dcc.Store(id="pdf_src", data='./assets/Chen et al 2014.pdf'),
                ],
                fluid=True,
            ),
        ]
    )
    return layout


# Visualization output (numpy array plotting or pdf)
def visualization():
    vis_layout = [
        dbc.Card(children=get_plot_layout()),
        dbc.Card(children=get_pdf_layout())
        # dbc.Card(children=get_scaler_layout(dropdown_scalers)),
        # dbc.Card(children=get_scan_layout(component_list)),
    ]
    return vis_layout


def visualization_collaspe():
    vis_layout = [
        dbc.Card(children=[
                  dbc.Accordion(get_plot_layout(), always_open=False, start_collapsed=True),
                  dbc.Accordion(get_pdf_layout(), always_open=False, start_collapsed=True)
                  ]
                ),
    ]
    return vis_layout

## Chat layout
def chat():
    chatlayout = [
        dbc.Card(children=get_chat_layout()),
    ]
    return chatlayout