from dash import html, dcc
import plotly.express as px
import numpy as np
import dash_bootstrap_components as dbc


def get_plot_layout():
    plot_layout = [
        dbc.CardHeader("Data Viewer"),
        dbc.CardBody(
            [
                dbc.Col(
                            html.Div(
                                [
                                    dcc.Graph(
                                        id="bl-cam",
                                        figure=px.imshow(img=np.zeros((1024, 1024))),
                                    )
                                ]
                            ),
                        ),
            ]
        ),
    ]
    return plot_layout

def get_pdf_layout():
    pdf_layout = [
        dbc.CardHeader("PDF Viewer"),
        dbc.CardBody(
            [
                dbc.Col(
                        html.Div(
                            [html.Iframe(src="../assets/Chen et al 2014.pdf", 
                                         style={"width": "100%", "height": "500px"})]
                                         )
                        ),
            ]
        ),
    ]
    return pdf_layout
