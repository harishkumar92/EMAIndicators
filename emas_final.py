import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import functools
import scipy.stats as st
import load_ohlcv as load




def read_ohlcv(coin):
    csv_file = 'data/{0}.csv'.format(coin)
    dateparse = lambda x: pd.datetime.strptime(x, '%Y-%m-%d %H:%M:%S')
    return pd.read_csv(csv_file, index_col='date', parse_dates=True, date_parser=dateparse)

def resample(data_df, frequency):
    mappings = {'close':'last','high':'max','low':'min','open':'first','volumefrom':'sum','volumeto':'sum'}
    data_df = data_df.resample(frequency).agg(mappings)
    return data_df

def calculate_mas(data_df, periods):

    for period in periods:
        data_df['ema'+str(period)] = data_df['close'].ewm(span=period).mean()
        data_df['ma'+str(period)] = data_df['close'].rolling(period).mean()


    if pd.infer_freq(data_df.index).split('H')[0] == '':
        freq=1
    else:
        freq=float(pd.infer_freq(data_df.index).split('H')[0])#in hours
    
    data_df['200dma'] = data_df['close'].rolling(int(200*24.0/freq)).mean()
    data_df['bull'] = data_df['close'] > data_df['200dma']
    return data_df

def process_df(data_df, periods):

    
    for period in periods:

        ####################################################      SIGNAL 1     #################################################################
        #INFLECTION METRICS
        #take second derivative
        MIN_PERCENTILE=0.995
        data_df['ema{0}_inflection'.format(period)] = 1/np.log(data_df['ema{0}'.format(period)]).diff().diff()
        threshold = abs(data_df['ema{0}_inflection'.format(period)]).quantile(MIN_PERCENTILE)
        data_df['ema{0}_inflection_signal'.format(period)] = abs(data_df['ema{0}_inflection'.format(period)]) > threshold


        ####################################################     SIGNAL 2     #################################################################
        #DISTANCE BETWEEN PRICE AND EMAS
        data_df['ema{0}_exhaustion'.format(period)] = np.log(data_df['close']/data_df['ema{0}'.format(period)])

        #now normalize over 100d lookback
        lookback_hrs = 100*24
        rolling_max = data_df['ema{0}_exhaustion'.format(period)].rolling(lookback_hrs).max() 
        rolling_min = data_df['ema{0}_exhaustion'.format(period)].rolling(lookback_hrs).min()
        data_df['ema{0}_exhaustion'.format(period)] = (data_df['ema{0}_exhaustion'.format(period)]-rolling_min) / (rolling_max-rolling_min)
        data_df['ema{0}_exhaustion'.format(period)] *=  10
        
        ####################################################     SIGNAL 3     #################################################################
        #distance between ema and ma
        #eg1, if 200d sma catches up to 200d ema from below(bear market signal)
        #data_df['ema{0}_exhaustion2'.format(period)] = 1/abs((data_df['ema{0}'.format(period)]/data_df['ma{0}'.format(period)]) - 1.0)
        #threshold2 = data_df['ema{0}_exhaustion2'.format(period)].quantile(.95)
        #data_df['ema{0}_exhaustion2_signal'.format(period)] = data_df['ema{0}_exhaustion2'.format(period)] > threshold2
        data_df['ema{0}_exhaustion2'.format(period)]  = np.log(data_df['ema{0}'.format(period)]/data_df['ma{0}'.format(period)])
        

        #now normalize over 100d lookback
        lookback_hrs = 100*24
        rolling_max2 = data_df['ema{0}_exhaustion2'.format(period)].rolling(lookback_hrs).max() 
        rolling_min2 = data_df['ema{0}_exhaustion2'.format(period)].rolling(lookback_hrs).min()
        #data_df['ema{0}_exhaustion2'.format(period)] = (data_df['ema{0}_exhaustion2'.format(period)]-rolling_min) / (rolling_max-rolling_min)

    return data_df

def test_distribution(test_data, dist):
    args = dist.fit(test_data)
    pspace = np.linspace(test_data.min(), test_data.max(), 1000)
    fig,ax = plt.subplots()
    nbins=1000
    ax.hist(test_data, nbins)
    ax.plot(pspace, dist.pdf(pspace, *args) * len(test_data) * test_data.max() / nbins, '-r', lw=3)
    return args


def plot2(plot_df):

    dates = plot_df.index.strftime("%Y-%m-%d %H:%M")

    fig = go.Figure()

    #Add traces
    fig.add_trace(go.Candlestick(x=dates,open=plot_df['open'], high=plot_df['high'],
                    low=plot_df['low'], close=plot_df['close'], name='Price', yaxis='y1'))
    
    colors = ['red','navy', 'lightseagreen', 'darkgreen', 'purple', 'saddlebrown', 'darkslategray']
    for period, color in zip(periods, colors):
        fig.add_trace(go.Scatter(x=dates, y=plot_df['ema{0}'.format(period)], 
                                 name='EMA{0}'.format(period), yaxis='y1', line=dict(color=color)))
        
        fig.add_trace(go.Scatter(mode='markers', x=plot_df.index[plot_df['ema{0}_inflection_signal'.format(period)]].strftime("%Y-%m-%d %H:%M"), \
                                 y=plot_df['close'][plot_df['ema{0}_inflection_signal'.format(period)]],
                                 name='EMA{0} Inflection Pt'.format(period), yaxis='y1', marker=dict(color=color, size=12)))

        fig.add_trace(go.Scatter(x=dates, y=plot_df['ema{0}_exhaustion'.format(period)], 
                                 name='EMA{0} Exhaustion'.format(period), yaxis='y2', line=dict(color=color)))
        
        fig.add_trace(go.Scatter(x=dates, y=plot_df['ema{0}_exhaustion2'.format(period)], 
                                 name='EMA{} Exhaustion 2'.format(period), yaxis='y3', line=dict(color=color)))

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
            range=[1000, 14000],
            fixedrange=False,
            autorange=True,
            domain=[0.31, 1]
        ),

        yaxis2 = go.layout.YAxis(
            anchor="x",
            range=[0, 100],
            autorange=True,
            domain=[0.1, 0.3]
        ),

        yaxis3 = go.layout.YAxis(
            anchor="x",
            range=[0, 1],
            autorange=True,
            domain=[0, 0.09]
        ),
        height=1600, width=1400
    )

    fig.show()


def save_figs(data_df, days):
    pass



if __name__ == "__main__":
    load.update_with_latest('BTC')
    periods = [20,50,100,200,400,800]
    data_df = read_ohlcv('BTC')
    data_df = resample(data_df, '1H')
    data_df = calculate_mas(data_df, periods)
    data_df = process_df(data_df, periods)
    date1, date2 = '2019-05-01', '2019-12-31'
    date_sel = (data_df.index >= date1) & (data_df.index <= date2)
    plot2(data_df[date_sel])

    #test_data = abs(data_df.tail(1000)['ema40_exhaustion'])
    #test_distribution(test_data, st.lomax)



