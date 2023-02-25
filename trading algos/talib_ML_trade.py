#!/usr/bin/env python
# coding: utf-8

# In[11]:


import os
import alpaca_trade_api as tradeapi
import numpy as np 
import pandas as pd
import time
import talib
from sklearn.svm import SVC

stock_list = ['AAPL', 'KO', 'IVV', 'BP','CSCO','PG','GOLD', 'USO', 'UNG', 'GLD', 'PPLT', 'DBA']

high_price = {'AAPL': 0, 'KO' : 0, 'IVV': 0, 'BP' : 0,'CSCO' : 0,'PG' : 0,'GOLD' : 0,
             'USO' : 0, 'UNG' : 0, 'GLD' : 0, 'PPLT' : 0, 'DBA' : 0}

def trading_algorithm(stock):
    
    #setting up trading enviroment
    os.environ['APCA_API_BASE_URL'] = 'https://paper-api.alpaca.markets'
    api = tradeapi.REST(API, API_TOKEN, api_version='v2')
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
    #data['time'] = times
    data['low'] = low
    data['high'] = high
    #data.set_index('time', inplace=True)
    
    #indicators
    time_period = 30
    data['log returns'] = np.log(data['close']/data['close'].shift(1))
    
    timeperiod = 14

    data['ADX'] = talib.ADX(data['high'], data['low'], data['close'], timeperiod=timeperiod)
    data['ADX'] = np.sign(np.log(data['ADX']/data['ADX'].shift(1)))

    data['AROONOSC'] = talib.AROONOSC(data['high'], data['low'], timeperiod=timeperiod)
    data['AROONOSC'] = np.sign(np.log(data['AROONOSC']/data['AROONOSC'].shift(1)))

    data['ROCP'] = talib.ROCP(data['close'], timeperiod=timeperiod)
    data['ROCP'] = np.sign(np.log(data['ROCP']/data['ROCP'].shift(1)))
    
    data['tomorrow'] = np.sign(data['log returns'].shift(-1))
    
    
    
    
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
    
    #Machine Learning
    model = SVC(gamma='auto')
    
    signals = data[['ADX', 'AROONOSC', 'ROCP']]
    signals = pd.DataFrame(signals)
    
    model.fit(signals, data['tomorrow'])
    
    data['predictions'] = model.predict(signals)
    
    
    
    
    
    if clock.is_open:
        if stock not in open_positions:
            print('no position')
            # score greater then 1(long)
            if (data.predictions[-1:].values == 1):
                if num_shares != 0:
                    api.submit_order(symbol=stock, qty=num_shares, side='buy', type='market', time_in_force='day')
                    print('Going long {}'.format(stock))
                    times_long += 0
                else:
                    print(f"order for {stock} 0 shares")
                
            #sell stock if z score less then -1    
            #elif data.predictions[-1:].values == -1:
                #if num_shares != 0:
                    #api.submit_order(symbol=stock, qty=num_shares, side='sell', type='market', time_in_force='day')
                    #print('Going short {}'.format(stock))
                    #times_short += 0
                #else:
                    #print(f"order for {stock} 0 shares")
                         
        else:
            print('already hold {}'.format(stock))
            # if long sell stock if z score goes below mean
            if (cur_pos == 'long') and ((data.predictions[-1:].values == 0) or (data.predictions[-1:].values == -1) ):
                print('closing long: {}'.format(stock))
                api.close_position(stock)
                open_positions.remove(stock)
                high_price[stock] = 0
            #if short close position in z score > mean
            elif (cur_pos == 'short') and ((data.predictions[-1:].values == 0) or (data.predictions[-1:].values == 1) ):
                print('closing short: {}'.format(stock))
                api.close_position(stock)
                open_positions.remove(stock)
                high_price[stock] = 0
            #stop trailing loss
            elif ((cur_pos != 'none') and ((cur_price - high_price[stock]) * shares_held) < -50):
                print('stop loss hit closing {}: {}'.format(cur_pos ,stock))
                api.close_position(stock)
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
    
 
    


# In[13]:


a = time.time()
runs = 0
while runs <26:
    if (time.time() - a) >= 900:
        print('ran trading algo')
        a = time.time()
        runs += 1
        execute_algo(stock_list)


# In[12]:


execute_algo(stock_list)


# In[ ]:


#######


# In[ ]:


trade.dropna(inplace=True)


# In[ ]:


trade


# In[ ]:


trade.dropna(inplace=True)


# In[ ]:


trade['tomorrow'] = np.sign(trade['log returns'].shift(-1))


# In[ ]:


trade.dropna(inplace=True)


# In[ ]:


from sklearn.svm import SVC


# In[ ]:


model = SVC(gamma='auto')


# In[ ]:


signals = trade[['ADX', 'AROONOSC', 'ROCP']]
signals = pd.DataFrame(signals)


# In[ ]:


model.fit(signals, trade['tomorrow'])


# In[ ]:


trade['predictions'] = model.predict(signals)


# In[ ]:


trade


# In[ ]:


trade['strategy'] = trade['predictions'] * trade['log returns'].shift(-1)


# In[ ]:


trade[['log returns', 'strategy']].cumsum().plot(figsize=(10,6))


# In[ ]:


a = 5
a == 4 or a == 6


# In[ ]:




