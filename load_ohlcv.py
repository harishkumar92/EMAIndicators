import requests, sys
from datetime import datetime
import pandas as pd

def get_coin_list(n):
    req_url = 'https://min-api.cryptocompare.com/data/top/mktcapfull?limit={0}&tsym=USD'.format(n)
    result = requests.get(req_url)
    coins = [x['CoinInfo']['Name'] for x in result.json()['Data']]
    return coins

def make_request(coin, toTs=None):
    ohlcv_url = 'https://min-api.cryptocompare.com/data/histohour?fsym={0}&tsym={1}&limit={2}'
    quote_pair = 'BTC' if coin != 'BTC' else 'USD'
    req_url = ohlcv_url.format(coin, quote_pair, 2000)
    if toTs:
        req_url = req_url + '&toTs=' + str(toTs)
    result = requests.get(req_url)
    return result

def make_requests(coin):
    results = []
    results.append(make_request(coin, None))
    for i in range(20):
        print ('Getting batch {0}/20'.format(i+1))
        toTs = results[-1].json()['TimeFrom']
        toTs = toTs - (1*60*60)
        curr_result = make_request(coin, toTs)
        results.append(curr_result)

    results = results[::-1]
    return results

def process_result(result):
    if result.json()['Response'] != 'Success':
        print (result['Response'])

    data = pd.DataFrame(result.json()['Data'])
    data['date'] = pd.to_datetime(data.time.apply(lambda x: datetime.utcfromtimestamp(x).strftime('%Y-%m-%d %H:%M:%S')))
    data = data.set_index('date')
    data = data.drop(['time'], axis=1)
    data = data[~((data.close==0) & (data.high==0) & (data.low==0) & (data.open==0))]
    return data

def process_results(results):
    data_df = pd.concat(map(process_result, results))
    return data_df

def save_df(coin, df):
    output_file = 'data/{0}.csv'.format(coin)
    df.to_csv(output_file)


def update_with_latest(coin):
    csv_file = 'data/{0}.csv'.format(coin)
    dateparse = lambda x: pd.datetime.strptime(x, '%Y-%m-%d %H:%M:%S')
    ohlcv_df = pd.read_csv(csv_file, index_col='date', parse_dates=True, date_parser=dateparse)

    latest_df = process_result(make_request(coin))

    first_time = latest_df.index.min()
    
    ohlcv_df = ohlcv_df[(ohlcv_df.index < first_time)]
    ohlcv_df = ohlcv_df.append(latest_df)
    save_df(coin, ohlcv_df)


def update_historical(coins):
    if coins == None:
        coins = get_coin_list(num_coins)

    for coin in coins:
        print ("Getting {0}...".format(coin))
        results = make_requests(coin)
        df = process_results(results)
        save_df(coin, df)


if __name__ == '__main__':
    update_with_latest(coin='BTC')
    #update_historical(coins=['BTC'])
    pass



