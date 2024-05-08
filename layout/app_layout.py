from dash import html, dcc
import dash_bootstrap_components as dbc
from layout.header import app_header
from layout.visualization import get_plot_layout, get_pdf_layout
# from layout.controls import get_controls_layout
# from layout.scaler import get_scaler_layout
# from layout.scan import get_scan_layout


def get_app_layout(
    # component_list,
    # component_gui,
    # dropdown_scalers,
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
                                visualization(), width=8,
                            ),
                        ]
                    ),
                    dcc.Interval(id="refresh-interval"),
                    dcc.Store(id="scan-cache", data=None),
                    dcc.Store(id="livescaler-cache", data=None),
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