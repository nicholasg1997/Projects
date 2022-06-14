#!/usr/bin/env python
# coding: utf-8

import os
import alpaca_trade_api as tradeapi
import numpy as np 
import pandas as pd
import time

stock_list = ['AAPL', 'KO', 'IVV', 'BP','CSCO','PG','GOLD']
commodity_list = ['USO', 'UNG', 'GLD', 'PPLT', 'DBA']

high_price = {'AAPL': 0, 'KO' : 0, 'IVV': 0, 'BP' : 0,'CSCO' : 0,'PG' : 0,'GOLD' : 0,
             'USO' : 0, 'UNG' : 0, 'GLD' : 0, 'PPLT' : 0, 'DBA' : 0}

def trading_algorithm(stock):
    
    #setting up trading enviroment
    os.environ['APCA_API_BASE_URL'] = 'https://paper-api.alpaca.markets'
    api = tradeapi.REST('PKHVVIU1XV6F5ZIJPYD1', 'Xk83BUqY5awBaPmVpjj1eziWXBvwYRcC7EcfxUxj', api_version='v2')
    account=api.get_account()
    
    #build data
    days = 365
    stock = stock
    
    stock_barset = api.get_barset(stock, '15Min', limit=days)
    stock_bars = stock_barset[stock]
    
    #put data into array
    data = []
    open_prices = []
    times = []
    volume = []
    
    for i in range(days):
        stock_close = stock_bars[i].c
        stock_open = stock_bars[i].o
        stock_time = stock_bars[i].t
        vol = stock_bars[i].v
        data.append(stock_close)
        open_prices.append(stock_open)
        volume.append(vol)
        times.append(stock_time)
        
        
    data = pd.DataFrame(data, columns=['close'])
    data['open'] = open_prices
    data['volume'] = volume
    
    #indicators
    time_period = 30
    data['log returns'] = np.log(data['close']/data['close'].shift(1))
    #z score for price
    
    #data['20_sma'] = data['open'].rolling(time_period).mean()
    #data['20_sma_std'] = data['open'].rolling(time_period).std()  * np.sqrt((60*5.5)*253)
    #data['z_score'] = (data['open'] - data['20_sma']) / data['20_sma_std']
    
    #z score for returns
    data['20_day_ret_mavg'] = data['log returns'].rolling(time_period).mean()
    data['20_day_ret_std'] = data['log returns'].rolling(time_period).std()
    data['z_score'] = (data['log returns'] - data['20_day_ret_mavg'])/data['20_day_ret_std']
    
    #data.loc[(data['z_score'] > (data['z_score'].mean() + data['z_score'].std())), 'buy_sell'] = 'buy'
    #data.loc[(data['z_score'] < (data['z_score'].mean() - data['z_score'].std())), 'buy_sell'] = 'sell'
    #data.loc[(data['z_score'] > data['z_score'].mean()*-1) & (data['z_score'] < data['z_score'].mean()), 'buy_sell'] = 'hold'
    
    data.dropna(inplace=True, axis=0)
    
    #trade execution
    portfolio = api.list_positions()
    clock = api.get_clock()
    
    #calculate num of shares
    cash = float(account.buying_power)
    leverage = 1 / len(stock_list)
    spendable_cash = cash * leverage
    num_shares = (spendable_cash // data['open'][-1:])
    num_shares = int(num_shares)
    print(f'{stock} has {num_shares} shares orderable')
    
    
    #dont trade if less then $2000
    if cash < 2000:
        print('not enough cash')
        num_shares = 0
    
    times_long = 0
    times_short = 0
    
    positions = api.list_positions()
    
    open_positions = []
    
    for position in positions:
        #print(position.symbol)
        open_positions.append(position.symbol)
    
    
    print('current positions : {}'.format(open_positions))
    
    cur_pos = 'none'
    cur_pl = 0
    cur_price = 0
    shares_held = 0
    
    
    for position in positions:
        if stock == position.symbol:
            cur_pos = position.side
            cur_pl = float(position.unrealized_pl)
            total_pl.append(float(position.unrealized_pl))
            cur_price = float(position.current_price)
            shares_held = int(position.qty)
            
            
    #setting trailing stop loss
    if high_price[stock] < cur_price:
        high_price[stock] = cur_price
    
    
    print(f'{stock} current pos: {cur_pos}')
    print(f'{stock} unrealized PL : {cur_pl}')
    if clock.is_open:
        if stock not in open_positions:
            print('no position')
            # score greater then 1(long)
            if (data.z_score[-1:].values < (data['z_score'].mean() - data['z_score'].std()) ):
                if num_shares != 0:
                    #api.submit_order(symbol=stock, qty=num_shares, side='buy', type='market', time_in_force='day')
                    print('Going long {}'.format(stock))
                    times_long += 0
                else:
                    print(f"order for {stock} 0 shares")
                
            #sell stock if z score less then -1    
            elif data.z_score[-1:].values > (data['z_score'].mean() + data['z_score'].std() ):
                if num_shares != 0:
                    #api.submit_order(symbol=stock, qty=num_shares, side='sell', type='market', time_in_force='day')
                    print('Going short {}'.format(stock))
                    times_short += 0
                else:
                    print(f"order for {stock} 0 shares")
                         
        else:
            print('already hold {}'.format(stock))
            # if long sell stock if z score goes below mean
            if (cur_pos == 'long') and (data.z_score[-1:].values > data['z_score'].mean() ):
                print('closing long: {}'.format(stock))
                #api.close_position(stock)
                open_positions.remove(stock)
                high_price[stock] = 0
            #if short close position in z score > mean
            elif (cur_pos == 'short') and (data.z_score[-1:].values < data['z_score'].mean() ):
                print('closing short: {}'.format(stock))
                #api.close_position(stock)
                open_positions.remove(stock)
                high_price[stock] = 0
            #stop trailing loss
            elif ((cur_pos != 'none') and ((cur_price - high_price[stock]) * shares_held) < -50):
                print('stop loss hit closing {}: {}'.format(cur_pos ,stock))
                #api.close_position(stock)
                open_positions.remove(stock)
                high_price[stock] = 0
                
    
    
    #print('trailing stop: {}'.format((cur_price - high_price[stock]) * shares_held))
    return data


    
def execute_algo(stock_list):
    global total_pl
    total_pl = []
    for stock in stock_list:
        trading_algorithm(stock)
        
    print('total_pl: ${}'.format(round(sum(total_pl),2)))
    
 
    


# In[ ]:


def trading_algorithm(stock):
    
    #setting up trading enviroment
    os.environ['APCA_API_BASE_URL'] = 'https://paper-api.alpaca.markets'
    api = tradeapi.REST('PKHVVIU1XV6F5ZIJPYD1', 'Xk83BUqY5awBaPmVpjj1eziWXBvwYRcC7EcfxUxj', api_version='v2')
    account=api.get_account()
    
    #build data
    days = 20
    stock = stock
    
    stock_barset = api.get_barset(stock, '15Min', limit=days)
    stock_bars = stock_barset[stock]
    
    #put data into array
    data = []
    open_prices = []
    times = []
    volume = []
    
    for i in range(days):
        stock_close = stock_bars[i].c
        stock_open = stock_bars[i].o
        stock_time = stock_bars[i].t
        vol = stock_bars[i].v
        data.append(stock_close)
        open_prices.append(stock_open)
        volume.append(vol)
        times.append(stock_time)
        
        
    data = pd.DataFrame(data, columns=['close'])
    data['open'] = open_prices
    data['volume'] = volume
    
    #indicators
    time_period = 30
    data['log returns'] = np.log(data['close']/data['close'].shift(1))
    #z score for price
    
    #data['20_sma'] = data['open'].rolling(time_period).mean()
    #data['20_sma_std'] = data['open'].rolling(time_period).std() * np.sqrt(60*5.5)
    #data['z_score'] = (data['open'] - data['20_sma']) / data['20_sma_std']
    
    #z score for returns
    data['20_day_ret_mavg'] = data['log returns'].rolling(time_period).mean()
    data['20_day_ret_std'] = data['log returns'].rolling(time_period).std()
    data['z_score'] = (data['log returns'] - data['20_day_ret_mavg'])/data['20_day_ret_std']
    
    #data.loc[(data['z_score'] > (data['z_score'].mean() + data['z_score'].std())), 'buy_sell'] = 'buy'
    #data.loc[(data['z_score'] < (data['z_score'].mean() - data['z_score'].std())), 'buy_sell'] = 'sell'
    #data.loc[(data['z_score'] > data['z_score'].mean()*-1) & (data['z_score'] < data['z_score'].mean()), 'buy_sell'] = 'hold'
    
    data.dropna(inplace=True, axis=0)
    
    #trade execution
    portfolio = api.list_positions()
    clock = api.get_clock()
    
    #calculate num of shares
    cash = float(account.buying_power)
    leverage = 1 / len(stock_list)
    spendable_cash = cash * leverage
    #num_shares = (spendable_cash // data['open'][-1:])
    #num_shares = int(num_shares)
    
    #dont trade if less then $2000
    if cash < 2000:
        print('not enough cash')
        num_shares = 0
    
    times_long = 0
    times_short = 0
    
    positions = api.list_positions()
    
    open_positions = []
    
    for position in positions:
        #print(position.symbol)
        open_positions.append(position.symbol)
    
    
    print('current positions : {}'.format(open_positions))
    
    cur_pos = 'none'
    cur_pl = 0
    cur_price = 0
    shares_held = 0
    
    
    for position in positions:
        if stock == position.symbol:
            cur_pos = position.side
            cur_pl = float(position.unrealized_pl)
            #total_pl.append(float(position.unrealized_pl))
            cur_price = float(position.current_price)
            shares_held = int(position.qty)
            
            
    #setting trailing stop loss
    #if high_price[stock] < cur_price:
        #high_price[stock] = cur_price
    
    
    print(f'{stock} current pos: {cur_pos}')
    print(f'{stock} unrealized PL : {cur_pl}')
    if clock.is_open:
        if stock not in open_positions:
            print('no position')
            # score greater then 1(long)
            if (data.z_score[-1:].values > (data['z_score'].mean() + data['z_score'].std()) ):
                #api.submit_order(symbol=stock, qty=num_shares, side='buy', type='market', time_in_force='day')
                print('Going long {}'.format(stock))
                times_long += 0
                
            #sell stock if z score less then -1    
            elif data.z_score[-1:].values < (data['z_score'].mean() - data['z_score'].std() ):
                #api.submit_order(symbol=stock, qty=num_shares, side='sell', type='market', time_in_force='day')
                print('Going short {}'.format(stock))
                times_short += 0
        
        else:
            print('already hold {}'.format(stock))
            # if long sell stock if z score goes below mean
            if (cur_pos == 'long') and (data.z_score[-1:].values < data['z_score'].mean() ):
                print('closing long: {}'.format(stock))
                #api.close_position(stock)
                open_positions.remove(stock)
            #if short close position in z score > mean
            elif (cur_pos == 'short') and (data.z_score[-1:].values > data['z_score'].mean() ):
                print('closing short: {}'.format(stock))
                #api.close_position(stock)
                open_positions.remove(stock)
            #stop trailing loss
            elif ((cur_pos != 'none') and ((cur_price - high_price[stock]) * shares_held) < - 50):
                print('stop loss hit closing {}: {}'.format(cur_pos ,stock))
                #api.close_position(stock)
                open_positions.remove(stock)
                
    
    
    #print('trailing stop: {}'.format((cur_price - high_price[stock]) * shares_held))
    return data


# In[ ]:





# In[ ]:


def execute_algo(stock_list):
    global total_pl 
    total_pl = []
    for stock in stock_list:
        trading_algorithm(stock)
        
    print('total_pl: ${}'.format(round(sum(total_pl),2)))


# In[ ]:


execute_algo(stock_list)
execute_algo(commodity_list)


# In[9]:


total_pl = []
trade = trading_algorithm('KO')


# In[ ]:


#stop loss
float(trade[0].unrealized_pl) < -200


# In[10]:


trade.describe()


# In[ ]:


trade[0]


# In[ ]:


trade


# In[6]:


import matplotlib.pyplot as plt


# In[13]:


total_pl = []
trade = trading_algorithm('IVV')


# In[14]:


data1 = trade['z_score']
data2 = trade['close']


# In[15]:


fig, ax1 = plt.subplots()

ax1.set_xlabel('date')
ax1.set_ylabel('price', color='red')
ax1.plot(trade['close'], color='red')
ax1.tick_params(axis='y', labelcolor='red')

ax2 = ax1.twinx()

ax2.set_ylabel('zscore', color='blue')
ax2.plot(trade['z_score'], color='blue', alpha=0.7)
ax2.tick_params(axis='y', labelcolor='blue')
ax2.axhline(y=trade['z_score'].mean())
ax2.axhline(y=trade['z_score'].mean() + (trade['z_score'].std() * 2), color='green')
ax2.axhline(y=trade['z_score'].mean() - (trade['z_score'].std() * 2), color='red')

fig.tight_layout()
plt.plot()


# In[19]:


import talib


# In[54]:


import os
import alpaca_trade_api as tradeapi
import numpy as np 
import pandas as pd
import time

stock_list = ['AAPL', 'KO', 'IVV', 'BP','CSCO','PG','GOLD']
commodity_list = ['USO', 'UNG', 'GLD', 'PPLT', 'DBA']

high_price = {'AAPL': 0, 'KO' : 0, 'IVV': 0, 'BP' : 0,'CSCO' : 0,'PG' : 0,'GOLD' : 0,
             'USO' : 0, 'UNG' : 0, 'GLD' : 0, 'PPLT' : 0, 'DBA' : 0}

def trading_algorithm(stock):
    
    #setting up trading enviroment
    os.environ['APCA_API_BASE_URL'] = 'https://paper-api.alpaca.markets'
    api = tradeapi.REST('PKHVVIU1XV6F5ZIJPYD1', 'Xk83BUqY5awBaPmVpjj1eziWXBvwYRcC7EcfxUxj', api_version='v2')
    account=api.get_account()
    
    #build data
    days = 365
    stock = stock
    
    stock_barset = api.get_barset(stock, '15Min', limit=days)
    stock_bars = stock_barset[stock]
    
    #put data into array
    data = []
    open_prices = []
    high = []
    low = []
    times = []
    volume = []
    
    for i in range(days):
        stock_close = stock_bars[i].c
        stock_open = stock_bars[i].o
        stock_high = stock_bars[i].h
        stock_low = stock_bars[i].l
        
        stock_time = stock_bars[i].t
        vol = stock_bars[i].v
        data.append(stock_close)
        open_prices.append(stock_open)
        high.append(stock_high)
        low.append(stock_low)
        volume.append(vol)
        times.append(stock_time)
        
        
    data = pd.DataFrame(data, columns=['close'])
    data['open'] = open_prices
    data['volume'] = volume
    data['time'] = times
    data['low'] = low
    data['high'] = high
    #data.set_index('time', inplace=True)
    
    #indicators
    time_period = 30
    data['log returns'] = np.log(data['close']/data['close'].shift(1))
    #z score for price
    
    #data['20_sma'] = data['open'].rolling(time_period).mean()
    #data['20_sma_std'] = data['open'].rolling(time_period).std()  * np.sqrt((60*5.5)*253)
    #data['z_score'] = (data['open'] - data['20_sma']) / data['20_sma_std']
    
    #z score for returns
    data['20_day_ret_mavg'] = data['log returns'].rolling(time_period).mean()
    data['20_day_ret_std'] = data['log returns'].rolling(time_period).std()
    data['z_score'] = (data['log returns'] - data['20_day_ret_mavg'])/data['20_day_ret_std']
    
    #data.loc[(data['z_score'] > (data['z_score'].mean() + data['z_score'].std())), 'buy_sell'] = 'buy'
    #data.loc[(data['z_score'] < (data['z_score'].mean() - data['z_score'].std())), 'buy_sell'] = 'sell'
    #data.loc[(data['z_score'] > data['z_score'].mean()*-1) & (data['z_score'] < data['z_score'].mean()), 'buy_sell'] = 'hold'
    
    data.dropna(inplace=True, axis=0)
    
    #trade execution
    portfolio = api.list_positions()
    clock = api.get_clock()
    
    #calculate num of shares
    cash = float(account.buying_power)
    leverage = 1 / len(stock_list)
    spendable_cash = cash * leverage
    num_shares = (spendable_cash // data['open'][-1:])
    num_shares = int(num_shares)
    print(f'{stock} has {num_shares} shares orderable')
    
    
    #dont trade if less then $2000
    if cash < 2000:
        print('not enough cash')
        num_shares = 0
    
    times_long = 0
    times_short = 0
    
    positions = api.list_positions()
    
    open_positions = []
    
    for position in positions:
        #print(position.symbol)
        open_positions.append(position.symbol)
    
    
    print('current positions : {}'.format(open_positions))
    
    cur_pos = 'none'
    cur_pl = 0
    cur_price = 0
    shares_held = 0
    
    
    for position in positions:
        if stock == position.symbol:
            cur_pos = position.side
            cur_pl = float(position.unrealized_pl)
            total_pl.append(float(position.unrealized_pl))
            cur_price = float(position.current_price)
            shares_held = int(position.qty)
            
            
    #setting trailing stop loss
    if high_price[stock] < cur_price:
        high_price[stock] = cur_price
    
    
    print(f'{stock} current pos: {cur_pos}')
    print(f'{stock} unrealized PL : {cur_pl}')
    if clock.is_open:
        if stock not in open_positions:
            print('no position')
            # score greater then 1(long)
            if (data.z_score[-1:].values < (data['z_score'].mean() - data['z_score'].std()) ):
                if num_shares != 0:
                    #api.submit_order(symbol=stock, qty=num_shares, side='buy', type='market', time_in_force='day')
                    print('Going long {}'.format(stock))
                    times_long += 0
                else:
                    print(f"order for {stock} 0 shares")
                
            #sell stock if z score less then -1    
            elif data.z_score[-1:].values > (data['z_score'].mean() + data['z_score'].std() ):
                if num_shares != 0:
                    #api.submit_order(symbol=stock, qty=num_shares, side='sell', type='market', time_in_force='day')
                    print('Going short {}'.format(stock))
                    times_short += 0
                else:
                    print(f"order for {stock} 0 shares")
                         
        else:
            print('already hold {}'.format(stock))
            # if long sell stock if z score goes below mean
            if (cur_pos == 'long') and (data.z_score[-1:].values > data['z_score'].mean() ):
                print('closing long: {}'.format(stock))
                #api.close_position(stock)
                open_positions.remove(stock)
                high_price[stock] = 0
            #if short close position in z score > mean
            elif (cur_pos == 'short') and (data.z_score[-1:].values < data['z_score'].mean() ):
                print('closing short: {}'.format(stock))
                #api.close_position(stock)
                open_positions.remove(stock)
                high_price[stock] = 0
            #stop trailing loss
            elif ((cur_pos != 'none') and ((cur_price - high_price[stock]) * shares_held) < -50):
                print('stop loss hit closing {}: {}'.format(cur_pos ,stock))
                #api.close_position(stock)
                open_positions.remove(stock)
                high_price[stock] = 0
                
    
    
    #print('trailing stop: {}'.format((cur_price - high_price[stock]) * shares_held))
    return data


    
def execute_algo(stock_list):
    global total_pl
    total_pl = []
    for stock in stock_list:
        trading_algorithm(stock)
        
    print('total_pl: ${}'.format(round(sum(total_pl),2)))
    
 
    


# In[55]:


total_pl = []
trade = trading_algorithm('IVV')


# In[56]:


trade


# In[57]:


trade['ADX'] = talib.ADX(trade['high'], trade['low'], trade['close'], timeperiod=14)


# In[58]:


trade.dropna(inplace=True)


# In[59]:


data1 = trade['z_score']
data2 = trade['close']
indicator = trade['ADX']


# In[66]:


fig, ax1 = plt.subplots()

ax1.set_xlabel('date')
ax1.set_ylabel('price', color='red')
ax1.plot(trade['close'], color='red')
ax1.tick_params(axis='y', labelcolor='red')

ax2 = ax1.twinx()

ax2.set_ylabel('indicator', color='blue')
ax2.plot(indicator, color='blue', alpha=0.7)
ax2.tick_params(axis='y', labelcolor='blue')
ax2.axhline(y=indicator.mean())
ax2.axhline(y=indicator.mean() + (indicator.std()), color='green')
#ax2.axhline(y=trade['z_score'].mean() - (trade['z_score'].std() * 2), color='red')

fig.tight_layout()
plt.plot()


# In[ ]:




