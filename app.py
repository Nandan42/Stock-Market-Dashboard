### Import Packages ########################################
import imp
from pydoc import classify_class_attrs, classname
import dash
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.graph_objs as go
from dash import Input, Output, dcc, html, State
from sklearn import datasets
from datetime import datetime as dt,timedelta
from datetime import date
from json import dumps
from datetime import date
from dash.exceptions import PreventUpdate
import yfinance as yf
import plotly.express as px
from sklearn.svm import SVR
import numpy as np
import pandas_datareader.data as web
import pickle
from prophet import Prophet


### Setup ###################################################
iris_raw = datasets.load_iris()
iris = pd.DataFrame(iris_raw["data"], columns=iris_raw["feature_names"])

app = dash.Dash(external_stylesheets=[dbc.themes.DARKLY])

server = app.server

### load ML model ###########################################
with open('stock_model.pickle', 'rb') as f:
    clf = pickle.load(f)

### App Layout ###############################################
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
            html.Div(id="graphs-content", style={"margin-top": "10px"}),
            html.Div(id='indicator-graph', style={"margin-top": "10px"}),# Indicator plot,
            html.Div(id="forecast-content", style={"margin-top": "10px"})#forecast plot
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
    Output('graphs-content', 'children'),
    State('stock-code', 'value'),
    State('my-date-picker-range', 'start_date'),
    State('my-date-picker-range', 'end_date'),
    Input('stock-price', 'n_clicks'),
)
def update_graph(value, start_date, end_date, n_clicks):
        graphs = []
        print(start_date,end_date)
        if n_clicks is None or not any(value):  #start_date is None or end_date is None or
            raise PreventUpdate
        else:
            symbol = value
            end_date=dt.fromisoformat(end_date)
            df = yf.download(symbol, start = start_date, end = end_date)
            df.reset_index(inplace=True)
            graphs.append(dcc.Graph(
                id='open_close_graph',
                figure= get_stock_price_fig(df)))
            # fig = get_stock_price_fig(df)
            return graphs

def get_stock_price_fig(df):
    fig = px.line(
            df,
            x="Date",
            y=["Close", "Open"],
            title="Open Close Price",
            height=500, width=1200
        )
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
    Output('indicator-graph', 'children'),
    State('stock-code', 'value'),
    State('my-date-picker-range', 'start_date'),
    State('my-date-picker-range', 'end_date'),
    Input('indicator', 'n_clicks'),
)
def update_ema_graph(value, start_date, end_date, n_clicks):
        graphs = []
        if n_clicks is None or not any(value):
            raise PreventUpdate
        else:
            symbol = value
            df = yf.download(symbol, start = start_date, end = end_date)
            df.reset_index(inplace=True)
            # fig = get_stock_price_fig(df)
            df['EWA_20'] = df['Close'].ewm(span=20, adjust=False).mean()
            # fig = get_stock_ema_price_fig(df)
            # return fig
            graphs.append(dcc.Graph(
            id='ema_graph',
            figure= get_stock_ema_price_fig(df)))
            return graphs

def get_stock_ema_price_fig(df):
    fig = px.line(
            df,
            x="Date",
            y=["EWA_20"],
            title="Exponential Moving Average vs Date",
            height=500, width=1200
        )
    return fig
    

@app.callback(
    Output('forecast-content', 'children'),
    State('stock-code', 'value'),
    State('no-of-days', 'value'),
    Input('forecast', 'n_clicks'),
)
def forecast(value, n_days, n_clicks):
    graphs = []
    if n_clicks is None or not any(value):
            raise PreventUpdate
    else:
        symbol = value
        # start = date.today() - timedelta(60)
        # end = date.today()
        start = date.today() - timedelta(60)
        # start =  dumps(start, indent=4, sort_keys=True, default=str)
        end = date.today()
        # end = dumps(end, indent=4, sort_keys=True, default=str)
        df = web.DataReader(symbol, 'yahoo', start, end)  # Collects data#prices in USD
        df.reset_index(inplace=True)
        data=df[["Date","Adj Close"]]
        data=data.rename(columns={"Date": "ds", "Adj Close": "y"})
        print(data.head())
        df_train=data[0:54]
        df_test=data[54:60]
        m = Prophet()
        m.fit(df_train)
        future = m.make_future_dataframe(periods=int(n_days))
        forecast = m.predict(future)
        graphs.append(dcc.Graph(
        id='forecast_graph',
        figure = get_forecast_graph(forecast)))
        return graphs

def get_forecast_graph(df):
    fig = px.line(
            df,
            x="ds",
            y=["yhat"],
            title="Forecast Graph",
            height=500, width=1200
        )
    return fig
    


# def get_forecast_graph(dates, y_pred):



# make sure that x and y values can't be the same variable
# def filter_options(v):
#     """Disable option v"""
#     return [
#         {"label": col, "value": col, "disabled": col == v}
#         for col in iris.columns
#     ]


# # functionality is the same for both dropdowns, so we reuse filter_options
# app.callback(Output("x-variable", "options"), [Input("y-variable", "value")])(
#     filter_options
# )
# app.callback(Output("y-variable", "options"), [Input("x-variable", "value")])(
#     filter_options
# )


if __name__ == '__main__':
    app.run_server(debug=True)

#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