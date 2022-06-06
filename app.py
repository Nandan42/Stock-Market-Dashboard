# import dash

# import dash_core_components as dcc

# import dash_html_components as html

# from dash.dependencies import Input, Output, State

# from datetime import datetime as dt

# from dash import dcc,html

# import yfinance as yf

# import pandas as pd

# import plotly.graph_objs as go

# import plotly.express as px

# from dash.exceptions import PreventUpdate

# import dash_bootstrap_components as dbc


# app = dash.Dash(__name__)

# server = app.server

# item1 = html.Div(
#           [
#             html.P("Welcome to the Stock Dash App!", className="start"),
#             html.Div([
#                 html.Label("Input Stock Data:"),
#                 html.Br(),
#                dcc.Input(
#                 id="stock-code",
#                 type="text",
#                 placeholder="",
#                 ),
#                 html.Button('Submit', id='submit-val', n_clicks=0),
#                 # stock code input
#             ], className="stock-code"),
#             html.Div([
#                     dcc.DatePickerRange(
#                     id='my-date-picker-range',
#                     initial_visible_month=dt(2021, 4, 1),
#                     end_date=dt(2021, 4, 1)
#                 ),
#                 html.Div(id='output-container-date-picker-range')# Date range picker input
#             ]),
#             html.Div([
#                 html.Button('Stock Price', id='stock-price', n_clicks=0),# Stock price button
#                 html.Button('Indicators', id='indicator', n_clicks=0),# Indicators button
#                 html.Br(),
#                 dcc.Input(
#                 id="forecast-input",
#                 type="text",
#                 placeholder="number of days",
#                 ),  # Number of days of forecast input

#                 html.Button('Forecast', id='forecast', n_clicks=0)  # Forecast button
#             ]),
#           ],
#         className="nav")
# item2 = html.Div(
#           [
#             html.Div(
#                   [html.Div([],id="Logo"), # Logo,
#                   html.Div([],id="company_name")],  #company_name ,
#                 className="header"),
#             html.Div( #Description
#               id="description", className="decription_ticker"),
#             dcc.Graph(id="graphs-content"),
#             dcc.Graph(id='indicator-graph'),# Indicator plot,
#             html.Div([
#                 # Forecast plot
#             ], id="forecast-content")
#           ],
#         className="content")

# app.layout = html.Div([item1, item2], className="container inputs")

# app = dash.Dash(external_stylesheets=[dbc.themes.BOOTSTRAP])

# controls = dbc.Card(
#     [
#         html.Div(
#             [
#                 dbc.Label("X variable"),
#                 dcc.Dropdown(
#                     id="x-variable",
#                     options=[
#                         {"label": col, "value": col} for col in iris.columns
#                     ],
#                     value="sepal length (cm)",
#                 ),
#             ]
#         ),
#         html.Div(
#             [
#                 dbc.Label("Y variable"),
#                 dcc.Dropdown(
#                     id="y-variable",
#                     options=[
#                         {"label": col, "value": col} for col in iris.columns
#                     ],
#                     value="sepal width (cm)",
#                 ),
#             ]
#         ),
#         html.Div(
#             [
#                 dbc.Label("Cluster count"),
#                 dbc.Input(id="cluster-count", type="number", value=3),
#             ]
#         ),
#     ],
#     body=True,
# )

# app.layout = dbc.Container(
#     [
#         html.H1("Iris k-means clustering"),
#         html.Hr(),
#         dbc.Row(
#             [
#                 dbc.Col(controls, md=4),
#                 dbc.Col(dcc.Graph(id="cluster-graph"), md=8),
#             ],
#             align="center",
#         ),
#     ],
#     fluid=True,
# )


# @app.callback(
#     Output("description", "children"),
#     Output("Logo", "children"),
#     Output("company_name", "children"),
#     Input("stock-code", "value"),
# )
# def update_output(input_value):
#     if not any(input_value):
#         raise PreventUpdate
#     else:
#         val = input_value
#         ticker = yf.Ticker(val)
#         inf = ticker.info
#         df = pd.DataFrame().from_dict(inf, orient="index").T
#         # print(df)
#         return df.longBusinessSummary, df.logo_url, df.shortName # df's first element of 'longBusinessSummary', df's first element value of 'logo_url', df's first element value of 'shortName'

# @app.callback(
#     Output('graphs-content', 'figure'),
#     State('stock-code', 'value'),
#     Input('my-date-picker-range', 'start_date'),
#     Input('my-date-picker-range', 'end_date'),
#     Input('stock-price', 'n_clicks'),
# )
# def update_graph(value, start_date, end_date, n_clicks):
#         print(start_date,end_date)
#         if n_clicks is None or not any(value):  #start_date is None or end_date is None or
#             raise PreventUpdate
#         else:
#             symbol = value
#             end_date=dt.fromisoformat(end_date)
#             df = yf.download(symbol, start = start_date, end = end_date)
#             df.reset_index(inplace=True)
#             fig = get_stock_price_fig(df)
#             return fig


# def get_stock_price_fig(df):
#     fig = px.line(
#             df,
#             x="Date",
#             y=["Close", "Open"],
#             title="Open Close Price",
#             height=500, width=800
#         )
#     return fig


# @app.callback(
#     Output('indicator-graph', 'figure'),
#     State('stock-code', 'value'),
#     Input('indicator', 'n_clicks'),
# )
# def update_ema_graph(value, n_clicks):
#         if n_clicks is None or not any(value):
#             raise PreventUpdate
#         else:
#             symbol = value
#             df = yf.download(symbol, start='2018-01-01')
#             df.reset_index(inplace=True)
#             fig = get_stock_price_fig(df)
#             df['EWA_20'] = df['Close'].ewm(span=20, adjust=False).mean()
#             fig = get_stock_ema_price_fig(df)
#             return fig

# def get_stock_ema_price_fig(df):
#     fig = px.line(
#             df,
#             x="Date",
#             y=["EWA_20"],
#             title="Exponential Moving Average vs Date",
#             height=500, width=800
#         )
#     return fig

from pydoc import classify_class_attrs, classname
import dash
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.graph_objs as go
from dash import Input, Output, dcc, html, State
from sklearn import datasets
from datetime import datetime as dt
from dash.exceptions import PreventUpdate
import yfinance as yf
import plotly.express as px

iris_raw = datasets.load_iris()
iris = pd.DataFrame(iris_raw["data"], columns=iris_raw["feature_names"])

app = dash.Dash(external_stylesheets=[dbc.themes.DARKLY])

server = app.server

controls = dbc.Card(html.Div(
    [
        dbc.Row([dbc.Col([
        dbc.Input(id="stock-code", placeholder="name",style={"margin-top": "15px"}),
    ]),
        dbc.Col(dbc.Button("Submit", id="submit-val", n_clicks=0, style={"margin-top": "15px"}))]),

        dbc.Row(dbc.Col([
        html.Br(),
        dcc.DatePickerRange(
        id='my-date-picker-range',
        min_date_allowed=dt(1995, 8, 5),
        max_date_allowed=dt(2022, 4, 1),
        initial_visible_month=dt(2022, 4, 1),
        end_date=dt(2021, 4, 1),
    ),
    html.Div(id='output-container-date-picker-range')
    ])),

        dbc.Row(
            [  
                dbc.Col(dbc.Button("Stock Price", id="stock-price", n_clicks=0,style={"margin-top": "15px"})),
                dbc.Col(dbc.Button("Indicators", id="indicator", n_clicks=0,style={"margin-top": "15px"})),
            ],className="stock-buttons"
        ),

        dbc.Row(
            [dbc.Col([
            dbc.Input(id="no-of-days", placeholder="number of days",style={"margin-top": "15px","margin-bottom": "15px"}),
            ]),
            dbc.Col(dbc.Button("Forecast", id="forecast", n_clicks=0,style={"margin-top": "15px","margin-bottom": "15px"}))] 
            )
    ]
))

information_output = html.Div(
            [
            html.Div(
                  [html.Img([],id="Logo", src=""), # Logo,
                  html.Div([],id="company_name")],  #company_name ,
                className="header"),
            html.Div( #Description
              id="description", className="decription_ticker"),
            dcc.Graph(id="graphs-content", style={"margin-top": "10px"}),
            dcc.Graph(id='indicator-graph', style={"margin-top": "10px"}),# Indicator plot,
            html.Div([
                # Forecast plot
            ], id="forecast-content")
          ],
        className="content")

app.layout = dbc.Container(
    [
        html.H1("Welcome to the Stock Dash App!"),
        html.Hr(),
        dbc.Row(
            [
                dbc.Col(controls, md=4),
                dbc.Col(information_output, md=8),
            ],
            align="center",
        ),
    ],
    fluid=True,
)

@app.callback(
    Output('graphs-content', 'figure'),
    State('stock-code', 'value'),
    Input('my-date-picker-range', 'start_date'),
    Input('my-date-picker-range', 'end_date'),
    Input('stock-price', 'n_clicks'),
)
def update_graph(value, start_date, end_date, n_clicks):
        print(start_date,end_date)
        if n_clicks is None or not any(value):  #start_date is None or end_date is None or
            raise PreventUpdate
        else:
            symbol = value
            end_date=dt.fromisoformat(end_date)
            df = yf.download(symbol, start = start_date, end = end_date)
            df.reset_index(inplace=True)
            fig = get_stock_price_fig(df)
            return fig

@app.callback(
    Output("description", "children"),
    Output("Logo", "src"),
    Output("company_name", "children"),
    Input("stock-code", "value"),
)

def update_output(input_value):
    if not any(input_value):
        raise PreventUpdate
    else:
        val = input_value
        ticker = yf.Ticker(val)
        inf = ticker.info
        df = pd.DataFrame().from_dict(inf, orient="index").T
        # print(df)
        return df.longBusinessSummary, df.logo_url, df.shortName # df's first element of 'longBusinessSummary', df's first element value of 'logo_url', df's first element value of 'shortName'

@app.callback(
    Output('indicator-graph', 'figure'),
    State('stock-code', 'value'),
    Input('my-date-picker-range', 'start_date'),
    Input('my-date-picker-range', 'end_date'),
    Input('indicator', 'n_clicks'),
)
def update_ema_graph(value, start_date, end_date, n_clicks):
        if n_clicks is None or not any(value):
            raise PreventUpdate
        else:
            symbol = value
            df = yf.download(symbol, start = start_date, end = end_date)
            df.reset_index(inplace=True)
            fig = get_stock_price_fig(df)
            df['EWA_20'] = df['Close'].ewm(span=20, adjust=False).mean()
            fig = get_stock_ema_price_fig(df)
            return fig

def get_stock_ema_price_fig(df):
    fig = px.line(
            df,
            x="Date",
            y=["EWA_20"],
            title="Exponential Moving Average vs Date",
            height=500, width=1200
        )
    return fig



def get_stock_price_fig(df):
    fig = px.line(
            df,
            x="Date",
            y=["Close", "Open"],
            title="Open Close Price",
            height=500, width=1200
        )
    return fig

# make sure that x and y values can't be the same variable
def filter_options(v):
    """Disable option v"""
    return [
        {"label": col, "value": col, "disabled": col == v}
        for col in iris.columns
    ]


# functionality is the same for both dropdowns, so we reuse filter_options
app.callback(Output("x-variable", "options"), [Input("y-variable", "value")])(
    filter_options
)
app.callback(Output("y-variable", "options"), [Input("x-variable", "value")])(
    filter_options
)


if __name__ == '__main__':
    app.run_server(debug=True)

#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