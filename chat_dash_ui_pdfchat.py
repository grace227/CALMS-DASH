import dash
from dash import dcc, html, Input, Output, State, Dash
import dash_bootstrap_components as dbc
import plotly.express as px
import pandas as pd
import random
from layout.app_layout import get_app_layout
from callback.chat_prompt import update_chat, update_PDFchat
from llm_init import get_llm, get_embeddings
from Chat import Chat, ToolChat, PDFChat


class CalmsDashApp:
    def __init__(self, title="LLMicroscopy", favicon="llm_logo.png") -> None:
        # Set up app
        self.setup_app(title=title, favicon=favicon)
        self.assign_layout()
        self.llm = get_llm()
        self.embeddings = get_embeddings()
        self.chat = PDFChat(self.llm, self.embeddings, doc_store=None)
        self.chat.update_pdf_docstore(["./assets/Chen et al 2014.pdf"])
        update_PDFchat(self.app, self.chat)

    def assign_layout(
        self,
        src_app_logo="assets/llm_logo.png",
        logo_height="60px",
        app_title="LLMicroscopy",
    ):
        layout = get_app_layout(
            src_app_logo=src_app_logo,
            logo_height=logo_height,
            app_title=app_title,
        )
        self.app.layout = layout

    def setup_app(
        self, title="LLMicroscopy", favicon="llm_logo.png",
    ):
        #### SETUP DASH APP ####
        external_stylesheets = [
            dbc.themes.BOOTSTRAP,
            "../assets/style.css",
            "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/4.7.0/css/font-awesome.min.css",
        ]
        self.app = Dash(__name__, external_stylesheets=external_stylesheets)
        self.app.title = title
        self.app._favicon = favicon

if __name__ == "__main__":
    calms = CalmsDashApp()
    calms.app.run_server(debug=True, host="0.0.0.0", port="8050")

# # Dummy function to simulate chatbot response
# def chatbot_response(prompt):
#     responses = [
#         "Hello! How can I assist you today?",
#         "Sure, I can help with that.",
#         "Could you please provide more details?",
#         "Thank you for asking, here's what I know."
#     ]
#     return random.choice(responses)

# # Dummy data for visualization
# def generate_data():
#     data = pd.DataFrame({
#         "Category": ["A", "B", "C"],
#         "Values": [random.randint(10, 100) for _ in range(3)]
#     })
#     fig = px.bar(data, x='Category', y='Values', title="Sample Data Visualization")
#     return fig

# app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# app.layout = dbc.Container([
#     dbc.Row([
#         dbc.Col([
#             dcc.Textarea(id='user-input', style={'width': '100%', 'height': 200}),
#             html.Button('Submit', id='submit-btn', n_clicks=0)
#         ], width=6),
        
#         dbc.Col([
#             dcc.Textarea(id='chatbot-response', style={'width': '100%', 'height': 200}),
#             dcc.Graph(id='data-visualization')
#         ], width=6)
#     ])
# ])

# @app.callback(
#     [Output('chatbot-response', 'value'),
#      Output('data-visualization', 'figure')],
#     [Input('submit-btn', 'n_clicks')],
#     [State('user-input', 'value')]
# )
# def update_output(n_clicks, user_input):
#     if n_clicks > 0:
#         response = chatbot_response(user_input)
#         fig = generate_data()
#         return response, fig
#     return "", px.figure()

# if __name__ == '__main__':
#     app.run_server(debug=False, port=8050, host="0.0.0.0")


