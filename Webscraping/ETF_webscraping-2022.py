#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from selenium import webdriver
from bs4 import BeautifulSoup
import pandas as pd
import time
from urllib.request import urlopen
from pylab import plt, mpl
import numpy as np
from tqdm import tqdm
from selenium.webdriver.common.action_chains import ActionChains
from datetime import date
import concurrent.futures


# In[ ]:


driver = webdriver.Chrome(executable_path='/Users/nickgault/Desktop/Python_code/chromedriver')


# In[ ]:


driver.get("https://www.ishares.com/us/products/etf-investments#!type=ishares&fc=44049%7C43541%7C43632&fac=43511&view=keyFacts")


# In[ ]:


driver.execute_script("window.scrollTo(0, 2200)") 
time.sleep(1)


# In[ ]:


driver.find_element_by_xpath('//*[@id="screener-funds"]/div/screener-show-all-button').click()


# In[ ]:


page_source = driver.page_source
soup = BeautifulSoup(page_source, 'lxml')


# In[ ]:


links = soup.find_all('a', class_ = 'link-to-product-page ng-star-inserted', href=True)


# In[ ]:


column_names = []
df_columns = []
errors=[]
first_run = True

for link in tqdm(links[:2]):
    fund = link.find('div').text
    action = ActionChains(driver)
    driver.get("https://www.ishares.com/" + link['href'])
    time.sleep(2)
    #expand list of stock
    try:
        button = driver.find_element_by_xpath("//*[@id='allHoldingsTable_wrapper']/div[2]/div[3]/a")
        action.move_to_element(button).perform()
        button.click()
    except:
        driver.execute_script("window.scrollTo(0, 5000)")
    
    page_source = driver.page_source
    soup = BeautifulSoup(page_source, 'html.parser')
    #find table with holding on page
    try:
        table = soup.find('div', class_='dataTables_scrollBody')
    except:
        continue
    #close ad if it pops up
    try:
        ad = driver.find_element_by_xpath("//button[@class='QSIWebResponsiveDialog-Layout1-SI_9QDC1Oi7YaUJskR_close-btn']")
        ad.click()
    except:
        pass
    
    #for first run get column names
    if first_run:
        first_run = False
        columns = table.find_all('th')
        for column in columns:
            column_names.append(column.get_text()[1:-1])
        df = pd.DataFrame(columns=column_names)
        
    #get list of the stocks
    stocks = table.find_all('tr', role='row', class_=['odd', 'even'])
    for stock in stocks[:100]:
        df_data = []
        data = stock.find_all('td')
        for datum in data:
            df_data.append(datum.get_text())
        if len(data) == 12:
            df.loc[len(df)] = df_data


# In[ ]:


driver.close()
driver.quit()


# In[ ]:


df = df[df['Asset Class'] == 'Equity']


# In[ ]:


raw_df = df[['Ticker', 'Notional Value', 'Shares']]


# In[ ]:


def fix_num(value):
    value = value.replace(',','')
    value = float(value)
    return value


# In[ ]:


raw_df['Notional Value'] = raw_df['Notional Value'].apply(fix_num)
raw_df['Shares'] = raw_df['Shares'].apply(fix_num)


# In[ ]:


def get_market_value(stock):
    try:
        url = f'https://finance.yahoo.com/quote/{stock.upper()}/key-statistics'
        page = urlopen(url)
        soup = BeautifulSoup(page, 'lxml')
        mk = soup.find('td', class_='Fw(500) Ta(end) Pstart(10px) Miw(60px)')
        mk = mk.get_text()
        multiplier = mk[-1:]
        if multiplier == 'T':
            multiplier = 1000000000000
        elif multiplier == 'B':
            multiplier = 1000000000
        elif multiplier == 'M':
            multiplier = 1000000
        mk = float(mk[:-1])
        mk = mk * multiplier
    except:
        print(stock)
        mk = np.nan
    print(mk)
    mc[stock] = mk  
    

tickers = raw_df['Ticker'].unique()
    
#mc = dict()
#no_threads = 7

#with concurrent.futures.ThreadPoolExecutor(max_workers=no_threads) as executor:
    #for stock in tickers:
        #executor.submit(get_market_value, stock)


# In[ ]:


grouped = raw_df.groupby(by='Ticker', as_index=False).sum()


# In[ ]:


#grouped['Market Cap'] = grouped['Ticker'].apply(lambda x: mc[x])
grouped['Market Cap'] = grouped['Ticker'].apply(lambda x: get_market_value(x))


# In[ ]:





# In[ ]:


grouped["% of market cap owned"] = (grouped['Notional Value'] / grouped['Market Cap']) * 100


# In[ ]:


grouped.dropna(axis=0, inplace=True)


# In[ ]:


grouped.sort_values(by='% of market cap owned', ascending=False, inplace=True)


# In[ ]:


grouped


# In[ ]:





# In[ ]:


#save to csv


# In[ ]:


today = date.today()


# In[ ]:


grouped.to_csv(f'{today}_etf_data', index=False)


# In[ ]:




