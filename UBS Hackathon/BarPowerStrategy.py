import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
import os
from datetime import datetime,timedelta
import talib as ta
import copy
import math
import matplotlib.gridspec as gridspec
from collections import OrderedDict
import scipy.interpolate as interp
import scipy.stats as sci
from scipy.stats import norm
from scipy import interpolate
from numpy import array
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import get_hist_refinitiv as get_h
import Pricing_Engine as pe
import warnings
warnings.filterwarnings('ignore')

getter = get_h.getter()

data = get_h.get_hist('JPY')

#get data
#连续时间
df = []
for index, row in data.iterrows():
    bar = {}
    bar['open'] = (row.OPEN_BID+row.OPEN_ASK)/2          
    bar['close'] = (row.BID+row.ASK)/2
    bar['high'] = (row.BID_HIGH_1+row.ASK_HIGH_1)/2
    bar['low'] = (row.BID_LOW_1+row.ASK_LOW_1)/2
    bar['date'] = row.DATE
    df.append(bar)
df = pd.DataFrame(df)
df.index = df.date
df.drop('date',axis=1,inplace=True)
order = ['open','high','low','close']
df = df[order]
df.sort_index(inplace=True)
data = df

#indicator function
def cross(s1,s2):
    dif=s1-s2
    dif[dif>0]=1
    dif[dif<0]=0
    dif=dif.diff()
    dif[dif==0]=np.nan
    return dif

def greater(s1,s2):
    dif=s1-s2
    dif[dif>0]=1
    dif[dif<0]=0
    return dif

def status(s1):
    dif=s1-s1.shift(5)
    dif[dif>0]=1
    dif[dif<0]=-1
    return dif
def rsiCross(s1,low,up):
    l=copy.copy(s1)
    l[(s1>low)&(s1.shift(1)<low)]=1
    l[(s1<up)&(s1.shift(1)>up)]=-1
    l[(l!=1)&(l!=-1)]=0
    return l

#indicator
data['ATR_10']=ta.ATR(data.high,data.low,data.close,10)
data['ATR_50']=ta.ATR(data.high,data.low,data.close,50)
data['ATR_trade']=greater(data['ATR_10'],1*data['ATR_50'])
data['EMA_10']=ta.EMA(data.close,10)
data['EMA_20']=ta.EMA(data.close,20)
data['EMA_40']=ta.EMA(data.close,40)
data['EMA']=ta.EMA(data['EMA_20'],10)
data['EMA_cross']=greater(data['EMA_20'],data['EMA_40'])
data['EMA_trade']=greater(data['EMA_10'],data['EMA_20'])
data['EMA_status']=status(data['EMA'])
data['BP']=ta.EMA(data.close-data.open,20)
data['MBP']=ta.EMA(data.BP,3)
data['up'],_,data['low']=ta.BBANDS(data.close,15)
data['RSI']=ta.RSI(data.close,14)
data['RSI_cross']=rsiCross(data['RSI'],30,70)

#strategy
#entry
data['entry']=0
data['entry'][(data['RSI_cross']==1)|((data['ATR_trade']==1)&((data['close']<data['low'])|\
            ((data['close']<data['up'])&(data['BP']>data['MBP'])&(data['EMA_cross']==1)&(data['EMA_status']==1))))]=1
data['entry'][(data['RSI_cross']==-1)|((data['ATR_trade']==1)&((data['close']>data['up'])|\
            ((data['close']>data['low'])&(data['BP']<data['MBP'])&(data['EMA_cross']==0)&(data['EMA_status']==-1))))]=-1
data['entry']=data['entry'].shift(1)

class Account:
    balance = 0
    value = 0
    asset = ''
    commission = 0
    position = {}
    time = '2000-01-01'
    price = pd.read_csv('USDJPY.csv',index_col='date')
    def __init__(self, balance):
        self.balance = balance
    def setAsset(self, asset):
        self.asset = asset
        self.position[self.asset]=0
    def setCommission(self, commission):
        self.commission = commission
    def setTime(self,time):
        self.time = time
    def setData(self,data):
        self.data = data
    def buy(self, asset, price, volume):
        self.position[asset] += volume
        self.balance -= price*volume
        self.balance -= price*volume*self.commission
    def sell(self, asset, price, volume):
        if volume>self.position[asset]:
            print('You cannot sell more than you have!!!')
        else:
            self.position[asset] -= volume
            self.balance += price*volume
            self.balance -= price*volume*self.commission
    def short(self, asset, price, volume):
        self.position[asset] -= volume
        self.balance += price*volume
        self.balance -= price*volume*self.commission
    def cover(self, asset, price, volume):
        if volume>self.position[asset]:
            print('You cannot cover more than you have!!!')
        else:
            self.position[asset] -= volume
            self.balance += price*volume
            self.balance -= price*volume*self.commission
    def cal_value(self,price):
        self.value = self.balance
        for asset in self.position.keys():
            self.value += self.position[asset]*price

#backtest
#No slide point so far
account = Account(10_000)
asset = 'USDJPY_forward'
account.setAsset(asset)
account.setCommission(0)
account.setData(data)
account_series = []
action = np.nan
action_series = []
last_price = 0
entry = False

lot = 100
transaction = 0

stoploss = 0.01
takeprofit = 0.015
add = False
add_point = 0.005
add_TIME = 2
stop = False
max_price = 0
min_price = 999999

TIME = 90
expire = False
drawdown = []
peak = 0

for time in data.index:
    
    account.setTime(time)
    #No consideration about stop loss and entry at the same day
    action = np.nan
    
    #init
    if account.position[asset] == 0:
        add_time = add_TIME
        t = TIME
    else:
        t -= 1
 
    price = pe.get_forward(time, t)
        
    if expire:
        if account.position[asset] > 0:
            #sell buy position
            account.sell(asset, price, account.position[asset])            
            expire = False
            action = 0
            transaction = 0
            t= TIME

        elif account.position[asset] < 0:
            #cover short position
            account.cover(asset, price, account.position[asset])            
            expire = False
            action = 0
            transaction = 0
            t= TIME
    
    #stop loss & take profit
    if stop and data.loc[time]['entry']==0:
        if account.position[asset] > 0:
            #sell buy position
            account.sell(asset, price, account.position[asset])            
            stop = False
            action = 0

            transaction = 0

        elif account.position[asset] < 0:
            #cover short position
            account.cover(asset, price, account.position[asset])            
            stop = False
            action = 0

            transaction = 0
    
        
    #entry    
    if data.loc[time]['entry']==1 and account.position[asset] <= 0:
        if account.position[asset] < 0:
            #cover short position
            account.cover(asset, price, account.position[asset])
                
        # open long position
        transaction = data.loc[time]['open'] #transaction price
        max_price = transaction
        account.buy(asset, price,lot)
        entry = True
        action = 1
        count += 1
    
    elif data.loc[time]['entry']==-1 and account.position[asset] >= 0:
        if account.position[asset] > 0:
            #sell buy position
            account.sell(asset, price, account.position[asset])
            
        # open short position
        transaction = data.loc[time]['open'] #transaction price
        min_price = transaction
        account.short(asset, price, lot)
        entry = True
        action = -1
        count += 1
    
    #add position
    if add:
        if account.position[asset]>0:
            add_lot = lot
            account.buy(asset, price, add_lot)
            action = 1
            count += 1
            add = False

        elif account.position[asset]<0:
            add_lot = lot
            account.short(asset, price, add_lot)        
            action = -1
            count += 1
            add = False
        
    #calculate balance at the end of days
    
    account.cal_value(price)
    account_series.append(account.value)
    action_series.append(action)
    
    #drawdown
    peak = max(account.value,peak)
    drawdown.append(peak - account.value)
    
    #stop loss
    #ceiling stop
    if account.position[asset] > 0:
        max_price = max(data.loc[time]['close'],max_price)
        if (data.loc[time]['close']-max_price)/max_price < -stoploss:
            stop = True

    elif account.position[asset] < 0:
        min_price = max(data.loc[time]['close'],min_price)
        if (data.loc[time]['close']-min_price)/min_price > stoploss:
            stop = True

    #take profit
    #general
    if account.position[asset] > 0:
        if (data.loc[time]['close']-transaction)/transaction > takeprofit:
            stop = True

    elif account.position[asset] < 0:
        if (data.loc[time]['close']-transaction)/transaction < -takeprofit:
            stop = True
    
    #expire
    if t==2:
        expire = True
    
    #add position
    if add_time>0:
        if account.position[asset]>0:
            if (data.loc[time]['close']-transaction)/transaction > add_point:
                add = True
                add_time -=1

        elif account.position[asset]<0:
            if (data.loc[time]['close']-transaction)/transaction < -add_point:
                add = True
                add_time -=1
                
result = pd.DataFrame(zip(account_series,action_series,drawdown),index=data.index,columns=['balance','action','drawdown'])

idxLong = result.index[result['action']==1]
data['Long']=np.nan
data['Long'][idxLong]=data['close'][idxLong]
idxShort = result.index[result['action']==-1]
data['Short']=np.nan
data['Short'][idxShort]=data['close'][idxShort]
idxCover = result.index[result['action']==0]
data['Cover']=np.nan
data['Cover'][idxCover]=data['close'][idxCover]
fig, (ax1,ax2,ax3)=plt.subplots(3,1)
result[['balance']].plot(
        ax=ax1,
        figsize=[12, 12],
        style=['k-'],
        linewidth=1)
data[['close','Long','Short','Cover']].plot(
        ax=ax2,
        style=['k-','ro','go','yo'],
        linewidth=1)
result[['drawdown']].plot(
        ax=ax3,
        style=['k-'],
        linewidth=1)

ax1.grid(axis="y")
plt.autoscale(enable=True, axis='x', tight=True)
plt.autoscale(enable=True, axis='y', tight=True)
plt.show()

# get annual yield
timePeriod = (pd.to_datetime(result.index[-1])-pd.to_datetime(result.index[0])).days/365
expected_return = (result['balance'][-1]-result['balance'][0])/result['balance'][0]
annual_yield = (expected_return+1)**(1/timePeriod)-1

#get annual standard deviation
return_series = ((result['balance']-result['balance'].shift(1))/result['balance'].shift(1)).dropna()
annual_std = return_series.std() * np.sqrt(252)

#define risk-free rate
riskLess = 0

#get sharpe ratio
sharpe_ratio = (annual_yield-riskLess)/annual_std

print('Sharpe Ratio: ' + str(round(sharpe_ratio,2)))