from dash import html
import dash_bootstrap_components as dbc
import dash_daq as daq


def app_header(src_app_logo, logo_height, app_title):
    header = dbc.Navbar(
        dbc.Container(
            [
                dbc.Row(
                    [
                        dbc.Col(
                            html.Img(
                                id="app-logo", src=src_app_logo, height=logo_height
                            ),
                            md="auto",
                        ),
                        dbc.Col(
                            html.Div(
                                id="app-title",
                                children=[html.H3(app_title)],
                            ),
                            md=True,
                            align="center",
                        ),
                    ]
                ),
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                dbc.NavbarToggler(id="navbar-toggler"),
                                html.Div(
                                    dbc.Nav(
                                        [
                                            dbc.NavItem(
                                                dbc.Button(
                                                    className="fa fa-github",
                                                    style={
                                                        "font-size": 40,
                                                        "margin-right": "1rem",
                                                        "color": "#00313C",
                                                        "background-color": "white",
                                                    },
                                                    href="https://github.com/als-computing/beamline531",
                                                )
                                            ),
                                            dbc.NavItem(
                                                dbc.Button(
                                                    className="fa fa-question-circle-o",
                                                    style={
                                                        "font-size": 40,
                                                        "color": "#00313C",
                                                        "background-color": "white",
                                                    },
                                                    href="https://github.com/als-computing/beamline531",
                                                )
                                            ),
                                        ],
                                        navbar=True,
                                    )
                                ),
                            ]
                        )
                    ]
                ),
            ],
            fluid=True,
        ),
        dark=True,
        color = "black",
        # color="#00313C",
        sticky="top",
    )
    return header
