import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from scipy import stats
import statsmodels.api as sm
from sklearn.neighbors.kde import KernelDensity
from sklearn.model_selection import GridSearchCV
import functools




def read_ohlcv(coin):
    csv_file = 'data/{0}.csv'.format(coin)
    dateparse = lambda x: pd.datetime.strptime(x, '%Y-%m-%d %H:%M:%S')
    return pd.read_csv(csv_file, index_col='date', parse_dates=True, date_parser=dateparse)

def resample(data_df, frequency):
    mappings = {'close':'last','high':'max','low':'min','open':'first','volumefrom':'sum','volumeto':'sum'}
    data_df = data_df.resample(frequency).agg(mappings)
    return data_df

def calculate_emas(data_df):

    data_df['ema10'] =  data_df['close'].ewm(span=10).mean()
    data_df['ema20'] = data_df['close'].ewm(span=20).mean()
    data_df['ema50'] = data_df['close'].ewm(span=50).mean()
    data_df['ema100'] = data_df['close'].ewm(span=100).mean()
    data_df['ema200'] = data_df['close'].ewm(span=200).mean()
    data_df['ema300'] = data_df['close'].ewm(span=300).mean()
    data_df['ema400'] = data_df['close'].ewm(span=400).mean()
    
    return data_df

def process_df(data_df):
    if pd.infer_freq(data_df.index).split('H')[0] == '':
        freq=1
    else:
        freq=float(pd.infer_freq(data_df.index).split('H')[0])#in hours

    #INFLECTION METRICS
    #take second derivative
    data_df['ema10_inflection'] = 1/np.log(data_df['ema10']).diff().diff()
    data_df['ema20_inflection'] = 1/np.log(data_df['ema20']).diff().diff()
    data_df['ema50_inflection'] = 1/np.log(data_df['ema50']).diff().diff()
    data_df['ema100_inflection'] = 1/np.log(data_df['ema100']).diff().diff()
    data_df['ema200_inflection'] = 1/np.log(data_df['ema200']).diff().diff()
    data_df['ema300_inflection'] = 1/np.log(data_df['ema300']).diff().diff()
    data_df['ema400_inflection'] = 1/np.log(data_df['ema400']).diff().diff()

    #convert to binary signal
    MIN_THRESHOLD = 1e6
    data_df['ema10_inflection_signal'] = abs(data_df['ema10_inflection']) > MIN_THRESHOLD
    data_df['ema20_inflection_signal'] = abs(data_df['ema20_inflection']) > MIN_THRESHOLD
    data_df['ema50_inflection_signal'] = abs(data_df['ema50_inflection']) > MIN_THRESHOLD
    data_df['ema100_inflection_signal'] = abs(data_df['ema100_inflection']) > MIN_THRESHOLD
    data_df['ema200_inflection_signal'] = abs(data_df['ema200_inflection']) > MIN_THRESHOLD
    data_df['ema300_inflection_signal'] = abs(data_df['ema300_inflection']) > MIN_THRESHOLD
    data_df['ema400_inflection_signal'] = abs(data_df['ema400_inflection']) > MIN_THRESHOLD


    #CONSOLIDATION METRICS, trend exhaustion

    #Metric 1, variance between EMAs, lower = ideal for position building
    data_df['ema_dispersion'] = data_df[['ema10', 'ema20', 'ema50']].var(axis=1) / data_df['close']
    data_df['ema_dispersion_chg'] = (data_df['ema_dispersion'].diff()/freq)

    #Metric 2, dispersion of EMA slopes to predict inflection points
    data_df['ema10_slope'] = data_df['ema10'].pct_change()/freq
    data_df['ema20_slope'] = data_df['ema20'].pct_change()/freq
    data_df['ema50_slope'] = data_df['ema50'].pct_change()/freq
    data_df['price_slope'] = ((data_df['close']/data_df['open'])-1)/freq

    data_df['ema_slope_dispersion'] = data_df[['ema10_slope', 'ema20_slope', 'ema50_slope']].var(axis=1) / data_df['price_slope']
    data_df['ema_slope_dispersion'] = np.log(abs(data_df['ema_slope_dispersion']))*np.sign(data_df['ema_slope_dispersion'])

    #BIAS METRICS

    #Metric 1, inverse log distance from ema weighted by slope of ema

    data_df['ema_inv_dist'] = abs(1/(np.log(data_df['close']/data_df['ema10']))) * data_df['ema10_slope'] + \
                              abs(1/(np.log(data_df['close']/data_df['ema20']))) * data_df['ema20_slope'] + \
                              abs(1/(np.log(data_df['close']/data_df['ema50']))) * data_df['ema50_slope']

    #Metric 2
    data_df['total_dist'] = abs(data_df['close']-data_df['ema10']) + abs(data_df['close']-data_df['ema20']) + abs(data_df['ema50']-data_df['close'])
    data_df['ema10_weight'] = (abs(data_df['close']-data_df['ema20']) + abs(data_df['ema50']-data_df['close'])) / data_df['total_dist']
    data_df['ema20_weight'] = (abs(data_df['close']-data_df['ema10']) + abs(data_df['ema50']-data_df['close'])) / data_df['total_dist']
    data_df['ema50_weight'] = (abs(data_df['close']-data_df['ema10']) + abs(data_df['ema20']-data_df['close'])) / data_df['total_dist']



    data_df['ema_weighted_slope'] = (data_df['ema10_weight']*data_df['ema10_slope']) + \
                                (data_df['ema20_weight']*data_df['ema20_slope']) + \
                                (data_df['ema50_weight']*data_df['ema50_slope'])

    #Metric3
    total_inv_distance = (1/abs(data_df['close']-data_df['ema10'])) + (1/abs(data_df['close']-data_df['ema20'])) + \
                         (1/abs(data_df['close']-data_df['ema50']))

    data_df['ema10_weight2'] = (1/abs(data_df['close']-data_df['ema10'])) / total_inv_distance
    data_df['ema20_weight2'] = (1/abs(data_df['close']-data_df['ema20'])) / total_inv_distance
    data_df['ema50_weight2'] = (1/abs(data_df['close']-data_df['ema50'])) / total_inv_distance

    data_df['ema_weighted_slope2'] = (data_df['ema10_weight2'] *data_df['ema10_slope']) + \
                                    (data_df['ema20_weight2']*data_df['ema20_slope']) + \
                                    (data_df['ema50_weight2']*data_df['ema50_slope'])

    
    return data_df







def plot2(plot_df):

    dates = plot_df.index.strftime("%Y-%m-%d %H:%M")
    data_df.index[data_df['ema10_inflection_signal']].strftime("%Y")

    fig = go.Figure()

    #Add traces
    fig.add_trace(go.Candlestick(x=dates,open=plot_df['open'], high=plot_df['high'],
                    low=plot_df['low'], close=plot_df['close'], name='Price', yaxis='y1'))
    fig.add_trace(go.Scatter(x=dates, y=plot_df['ema10'], name='EMA10', yaxis='y1', fillcolor='Indigo'))
    fig.add_trace(go.Scatter(x=dates, y=plot_df['ema20'], name='EMA20', yaxis='y1', fillcolor='Black'))
    fig.add_trace(go.Scatter(x=dates, y=plot_df['ema50'], name='EMA50', yaxis='y1', fillcolor='Blue'))
    fig.add_trace(go.Scatter(x=dates, y=plot_df['ema100'], name='EMA100', yaxis='y1', fillcolor='Yellow'))
    fig.add_trace(go.Scatter(x=dates, y=plot_df['ema200'], name='EMA200', yaxis='y1', fillcolor='Cyan'))
    fig.add_trace(go.Scatter(x=dates, y=plot_df['ema300'], name='EMA300', yaxis='y1', fillcolor='Teal'))
    fig.add_trace(go.Scatter(x=dates, y=plot_df['ema400'], name='EMA400', yaxis='y1', fillcolor='Fuchsia'))

    fig.add_trace(
        go.Scatter(
            mode='markers',
            x=plot_df.index[plot_df['ema10_inflection_signal']].strftime("%Y-%m-%d %H:%M"),
            y=plot_df['ema10'][plot_df['ema10_inflection_signal']],
            name='EMA10 Inflection Pt',
            yaxis='y1',
            marker=dict(
                color='Indigo',
                size=10
            )
        )
    )

    fig.add_trace(
        go.Scatter(
            mode='markers',
            x=plot_df.index[plot_df['ema20_inflection_signal']].strftime("%Y-%m-%d %H:%M"),
            y=plot_df['ema20'][plot_df['ema20_inflection_signal']],
            name='EMA20 Inflection Pt',
            yaxis='y1',
            marker=dict(
                color='Black',
                size=10
            )
        )
    )

    fig.add_trace(
        go.Scatter(
            mode='markers',
            x=plot_df.index[plot_df['ema50_inflection_signal']].strftime("%Y-%m-%d %H:%M"),
            y=plot_df['ema50'][plot_df['ema50_inflection_signal']],
            name='EMA50 Inflection Pt',
            yaxis='y1',
            marker=dict(
                color='Blue',
                size=10
            )
        )
    )

    fig.add_trace(
        go.Scatter(
            mode='markers',
            x=plot_df.index[plot_df['ema100_inflection_signal']].strftime("%Y-%m-%d %H:%M"),
            y=plot_df['ema100'][plot_df['ema100_inflection_signal']],
            name='EMA100 Inflection Pt',
            yaxis='y1',
            marker=dict(
                color='Yellow',
                size=10
            )
        )
    )

    fig.add_trace(
        go.Scatter(
            mode='markers',
            x=plot_df.index[plot_df['ema200_inflection_signal']].strftime("%Y-%m-%d %H:%M"),
            y=plot_df['ema200'][plot_df['ema200_inflection_signal']],
            name='EMA200 Inflection Pt',
            yaxis='y1',
            marker=dict(
                color='Cyan',
                size=10
            )
        )
    )

    fig.add_trace(
        go.Scatter(
            mode='markers',
            x=plot_df.index[plot_df['ema300_inflection_signal']].strftime("%Y-%m-%d %H:%M"),
            y=plot_df['ema300'][plot_df['ema300_inflection_signal']],
            name='EMA300 Inflection Pt',
            yaxis='y1',
            marker=dict(
                color='Teal',
                size=10
            )
        )
    )
    
    fig.add_trace(
        go.Scatter(
            mode='markers',
            x=plot_df.index[plot_df['ema400_inflection_signal']].strftime("%Y-%m-%d %H:%M"),
            y=plot_df['ema400'][plot_df['ema400_inflection_signal']],
            name='EMA400 Inflection Pt',
            yaxis='y1',
            marker=dict(
                color='Fuchsia',
                size=10
            )
        )
    )



    fig.add_trace(go.Scatter(x=dates, y=plot_df['ema_dispersion_chg'], name='EMA Dispersion Chg', yaxis='y2'))


    fig.update_layout(

        xaxis=go.layout.XAxis(
            autorange=True,
            rangeslider=dict(
                autorange=True,

            ),
        type="date"
        ), 

        yaxis1 = go.layout.YAxis(
            anchor="x",
            range=[3000, 14000],
            fixedrange=False,
            autorange=False,
            domain=[0.3, 1]
        ),

        yaxis2 = go.layout.YAxis(
            anchor="x",
            range=[0, 100],
            autorange=True,
            domain=[0, 0.29]
        ),
        height=1600, width=1400
    )

    fig.show()





if __name__ == "__main__":
    data_df = read_ohlcv('BTC')
    data_df = resample(data_df, '1H')
    data_df = calculate_emas(data_df)
    data_df = process_df(data_df)

    plot2(data_df[data_df.index.year==2019])






























#POTENTIALLY USE LATER
def kde_sklearn(x, x_grid, bandwidth, **kwargs):
    """Kernel Density Estimation with Scikit-learn"""
    kde_skl = KernelDensity(bandwidth=bandwidth, **kwargs)
    kde_skl.fit(x[:, np.newaxis])
    # score_samples() returns the log-likelihood of the samples
    log_pdf = kde_skl.score_samples(x_grid[:, np.newaxis])
    return np.exp(log_pdf)

def optimize_bandwidth(X):
    #Returns kernel density functions for different params
    #sm.nonparametric.KDEUnivariate
    # use grid search cross-validation to optimize the bandwidth
    grid = GridSearchCV(KernelDensity(kernel='exponential'),{'bandwidth': np.linspace(0.1, 1.0, 30)},cv=20)
    grid.fit(X)
    return grid

def create_ema_dispersion_chg_kde(data_df, bandwidth= 0.08):
    X = data_df['ema_dispersion_chg'].dropna().values
    kde = KernelDensity(kernel='exponential', bandwidth=bandwidth)
    kde.fit(X)
    return kde










