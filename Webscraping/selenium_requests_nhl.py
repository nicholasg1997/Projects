#!/usr/bin/env python
# coding: utf-8

# In[2]:


from selenium import webdriver
from bs4 import BeautifulSoup
import pandas as pd
import time
from urllib.request import urlopen
from pylab import plt, mpl
import numpy as np


# In[3]:


driver = webdriver.Chrome(executable_path='/Users/nickgault/Desktop/Python_code/chromedriver')


# In[4]:


driver.get('https://www.quanthockey.com/nhl/seasons/2020-21-nhl-players-stats.html')


# In[5]:


#get column names and create dataframe using columns
page_source = driver.page_source
soup = BeautifulSoup(page_source, 'lxml')

columns = soup.find(class_='orange')
columns = columns.find_all('th')

column_names = []

for column in columns:
    column_names.append(column.get_text())
    
df = pd.DataFrame(columns=column_names)

#get player stats on all 20 pages
for _ in range(2):
    #get page links
    pages = driver.find_elements_by_xpath("//a[contains(@href, 'javascript:PaginateStats')]")
    page_source = driver.page_source
    soup = BeautifulSoup(page_source, 'lxml')
    
    #get list of players on page
    player_odd = soup.find_all(class_='odd')
    player_even = soup.find_all(class_='even')
    players = player_odd + player_even
    
    for player in players:
        skater_stats = []
        stats = player.find_all('td')
        for stat in stats:
            skater_stats.append(stat.get_text())
        len_df = len(df)
        try:
            df.loc[len_df] = skater_stats
        except:
            pass
    
    #click on arrow to go to next page
    pages[-1].click()
    time.sleep(1)
    
#df = df.sort_values(by='P', ascending=False)   


# In[6]:


driver.close()
driver.quit()


# In[7]:


df = df.sort_values(by='P', ascending=False)   


# In[8]:


df['P'] = df['P'].apply(lambda x: int(x))


# In[9]:


def fix_name(name):
    name = name.replace(' ', '_')
    return name

def get_age(name):
    try:
        #print(name)
        url = urlopen(f'https://en.wikipedia.org/wiki/{name}')
        soup = BeautifulSoup(url.read(), 'lxml')
        age = soup.find(class_='noprint ForceAgeToShow')
        age = age.get_text()[6:-1]
        #print(age)
    except:
        age = None
    return age


# In[10]:


df['Name'] = df['Name'].apply(fix_name)


# In[11]:


df['Age'] = 0


# In[12]:


df['Age'] = df['Name'].apply(get_age)


# In[13]:


df = df[df['Age'] != None]
df['Age'] = df['Age'].astype(float)


# In[14]:


df


# In[59]:


#####
df_grouped = df.groupby(['Age']).mean()


# In[60]:


age = df_grouped.index.values


# In[61]:


points = df_grouped['P']


# In[62]:


plt.plot(age, points, 'ro')


# In[63]:


reg = np.polyfit(age, points, deg=1)


# In[64]:


p = np.polyval(reg, age)


# In[65]:


plt.plot(age, points, 'ro', label='sample data')
plt.plot(age, points, '--', label='regression')
plt.legend();


# In[ ]:





# In[ ]:




