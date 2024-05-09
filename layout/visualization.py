from dash import html, dcc
import plotly.express as px
import numpy as np
import dash_bootstrap_components as dbc


def get_plot_layout():
    plot_layout = [
        dbc.AccordionItem(
            [
                dbc.Card(
                    id={"base": "plot_card", "type": "datavis"},
                    children=[
                        dbc.CardBody(
                            [
                                dbc.Col(
                                    html.Div(
                                        [
                                            dcc.Graph(
                                                id="graph",
                                                figure=px.imshow(img=np.zeros((1024, 1024))),
                                            )
                                        ]
                                    ),
                                ),
                            ]
                        )
                    ]
                )
            ],
            title="Data Viewer",
        )
    ]
    return plot_layout


def get_pdf_layout():
    pdf_layout = [
        dbc.AccordionItem(
            [
                dbc.Card(
                    id={"base": "pdf_card", "type": "datavis"},
                    children=[
                        dbc.CardBody(
                            [
                                dbc.Col(
                                    html.Div(
                                            [
                                                html.Iframe(id ='iframe-src-input',
                                                            src="../assets/Chen et al 2014.pdf", 
                                                            style={"width": "100%", "height": "500px"})
                                            ]
                                            )
                                    ),
                            ]
                        )
                    ]
                )
            ],
            title="PDF Viewer",
        )
    ]
    return pdf_layout
