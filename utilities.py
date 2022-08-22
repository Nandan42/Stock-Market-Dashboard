import plotly.express as px

def get_stock_price_fig(df):
    fig = px.line(
            df,
            x="Date",
            y=["Close", "Open"],
            title="Open Close Price",
            height=500, width=1200
        )
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

def get_forecast_graph(df):
    fig = px.line(
            df,
            x="ds",
            y=["yhat"],
            title="Forecast Graph",
            height=500, width=1200
        )
    return fig