from dash.dependencies import Input, Output, State, MATCH
from dash import html

def update_chat(app, chat):
    @app.callback(
        Output('chat-container', 'children'),
        Output('user-input', 'value'),
        # Output('graph', 'figure'),
        # Output('chat-history', 'data'),
        Input('send-button', 'n_clicks'),
        # Input('plot-cache', 'data'),
        [State('user-input', 'value'), State('chat-container', 'children')],
        # Input('chat-history', 'data'),
        prevent_initial_call=True,
    )
    def _chat_update(n_clicks, user_input, chat_container):
        if n_clicks > 0 and user_input:
            chat_container.append(html.P(className='user-message', children=["You: " + user_input]))
            r = chat.generate_response([[user_input, '']], None)
            # Placeholder response, you can replace this with your own logic
            response = r[0][-1]
            print(response)
            chat_container.append(html.P(className='llm-msg', children=["LLM: " + response]))
            user_input = ''
            # chat_history.append(r)
        return chat_container, user_input
    
    # Callback to handle pressing enter key
    @app.callback(
        Output('send-button', 'n_clicks'),
        [Input('user-input', 'n_submit')],
        [State('send-button', 'n_clicks')]
    )
    def _update_chat_on_enter_key_press(n_submit, n_clicks):
        if n_submit:
            return n_clicks + 1
        else:
            return n_clicks


def update_PDFchat(app, chat):
    @app.callback(
        Output('chat-container', 'children'),
        Output('user-input', 'value'),
        Output('pdf_src', 'data'),
        Input('send-button', 'n_clicks'),
        [State('user-input', 'value'), State('chat-container', 'children')],
        prevent_initial_call=True,
    )
    def _chat_update(n_clicks, user_input, chat_container):
        if n_clicks > 0 and user_input:
            chat_container.append(html.P(className='user-message', children=["You: " + user_input]))
            r = chat.generate_response([[user_input, '']], None)
            response = r[0][-1]
            print(response)
            chat_container.append(html.P(className='llm-msg', children=["LLM: " + response]))
            user_input = ''
            # chat_history.append(r)
        return chat_container, user_input, './bboxed_Chen et al 2014.pdf'
    


    # Callback to handle pressing enter key
    @app.callback(
        Output('send-button', 'n_clicks'),
        [Input('user-input', 'n_submit')],
        [State('send-button', 'n_clicks')]
    )
    def _update_chat_on_enter_key_press(n_submit, n_clicks):
        if n_submit:
            return n_clicks + 1
        else:
            return n_clicks

