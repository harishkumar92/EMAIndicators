import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys



def read_ohlcv(coin):
	csv_file = 'data/{0}.csv'.format(coin)
	dateparse = lambda x: pd.datetime.strptime(x, '%Y-%m-%d')
	return pd.read_csv(csv_file, index_col='date', parse_dates=True, date_parser=dateparse)


def cross_correlation(a, b):
	a = (a - np.mean(a)) / (np.std(a) * len(a))
	b = (b - np.mean(b)) / (np.std(b))
	c = np.correlate(a, b, 'full')
	return c



def filter_df(df, year_range):
	year1, year2 = year_range
	return df[(df.index.year >= year1) & (df.index.year <= year2)]

def compute_df(df, top_n):

	df['returns'] = df.close.pct_change(periods=1)
	return df.returns.nlargest(top_n)



def plot(btc_ret, alt_ret, alt):
	for quarter in [[1,3], [4,6], [7,9], [10,12]]:
		temp = btc_ret.index.month

    
	plt.scatter(btc_ret.index, btc_ret.values, marker='x')
	plt.scatter(alt_ret.index, alt_ret.values, marker='o')
	plt.title('Best {0} and best BTC Returns'.format(alt))
	plt.savefig('output/{0}'.format(alt))
	plt.clf()


if __name__ == "__main__":
	for alt in ['ETH', 'XMR', 'DASH', 'LTC', 'XRP', 'XLM', 'BNB', 'DCR', 'DGB', 'NEO']:
		alt_df = read_ohlcv(alt)
		btc_df = read_ohlcv('BTC')

		year_range=[2017, 2017]

		alt_df = filter_df(alt_df, year_range)
		btc_df = filter_df(btc_df, year_range)

		top_n=30
		alt_ret = compute_df(alt_df, top_n)
		btc_ret = compute_df(btc_df, top_n)


		#plot(btc_ret, alt_ret, alt)


	#merged_df = btc_df.join(dash_df, how='inner', lsuffix='.btc', rsuffix='.dash')
	#merged_df['btc_returns'] = (merged_df['close.btc']/merged_df['open.btc'])-1
	#merged_df['dashbtc_returns'] = ((merged_df['close.dash'] / merged_df['close.btc']) / (merged_df['open.dash'] / merged_df['open.btc']))-1
	#c = cross_correlation(merged_df['btc_returns'], merged_df['dashbtc_returns'])




	
