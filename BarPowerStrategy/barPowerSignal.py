import talib as ta
import numpy as np
import pandas as pd

"""
将kdj策略需要用到的信号生成器抽离出来
"""

class barPowerSignal():

    def __init__(self):
        self.author = 'Haolin Pan'

    def maEnvironment(self, am, paraDict):
        envFastPeriod = paraDict["envFastPeriod"]
        envSlowPeriod = paraDict["envSlowPeriod"]

        sma = ta.MA(am.close, envFastPeriod)
        lma = ta.MA(am.close, envSlowPeriod)

        envSignal = 0
        if sma[-1]>lma[-1]:
            envSignal = 1
        else:
            envSignal = -1
        return envSignal, sma, lma

    def maCross(self,am,paraDict):
        fastPeriod = paraDict["fastPeriod"]
        slowPeriod = paraDict["slowPeriod"]

        sma = ta.MA(am.close, fastPeriod)
        lma = ta.MA(am.close, slowPeriod)
        goldenCross = sma[-1]>lma[-1] and sma[-2]<=lma[-2]
        deathCross = sma[-1]<lma[-1] and sma[-2]>=lma[-2]

        maCrossSignal = 0
        if goldenCross:
            maCrossSignal = 1
        elif deathCross:
            maCrossSignal = -1
        else:
            maCrossSignal = 0
        return maCrossSignal, sma, lma
    
    def maStatus(self,am,paraDict):
        fastPeriod = paraDict["fastPeriod"]
        slowPeriod = paraDict["slowPeriod"]

        sma = ta.MA(am.close, fastPeriod)
        lma = ta.MA(am.close, slowPeriod)
        goldenCross = sma[-1]>lma[-1] and sma[-2]<=lma[-2]
        deathCross = sma[-1]<lma[-1] and sma[-2]>=lma[-2]

        maCrossSignal = 0
        if sma[-1]>lma[-1]:
            maCrossSignal = 1
        else:
            maCrossSignal = -1
        return maCrossSignal, sma, lma
    
    def atrCross(self,am,paraDict):
        atrFastPeriod = paraDict["atrFastPeriod"]
        atrSlowPeriod = paraDict["atrSlowPeriod"]

        fa = ta.ATR(am.high, am.low, am.close, atrFastPeriod)
        sa = ta.ATR(am.high, am.low, am.close, atrSlowPeriod)
        atrDif = fa-sa*1.2
        
        atrCrossSignal = 0
        if atrDif[-1]>0:
            atrCrossSignal = 1
        else:
            atrCrossSignal = 0
        return atrCrossSignal, atrDif
    
    def rsiStatus(self,am,paraDict):
        rsiPeriod = paraDict["rsiPeriod"]

        rsi = ta.RSI(am.close, rsiPeriod)
        mrsi = ta.MA(rsi, 5)

        overSell = rsi[-1]>30 and rsi[-2]<30
        overBuy = rsi[-1]<70 and rsi[-2]>70

        rsiSignal = 0
        if overSell:
            rsiSignal = 1
        elif overBuy:
            rsiSignal = -1

        return rsiSignal, rsi
    
    def rsiExit(self,am,paraDict):
        rsiPeriod = paraDict["rsiPeriod"]

        rsi = ta.RSI(am.close, rsiPeriod)

        overSell = rsi[-1]<20
        overBuy = rsi[-1]>80

        rsiSignal = 0
        if overSell:
            rsiSignal = 1
        elif overBuy:
            rsiSignal = -1
        else:
            rsiSignal = 0        
        return rsiSignal, rsi
    
    def priceStatus(self,am,paraDict):
        price = am.close
        priceSignal = 0
        if price[-1]>price[-3]:
            priceSignal = 1
        else:
            priceSignal = -1
        return priceSignal, price
    
    def barStatus(self,am,paraDict):
        barSignal = 0
        if am.close[-1]>am.open[-1]:
            barSignal = 1
        else:
            barSignal = -1
        return barSignal
    
    def cciCross(self,am,paraDict):
        cciPeriod = paraDict["cciPeriod"]
        cci = ta.CCI(am.high, am.low, am.close, cciPeriod)

        cciSignal = 0
        if cci[-1]>100:
            cciSignal = 1
        elif cci[-1]<-100:
            cciSignal = -1
        else:
            cciSignal = 0
        return cciSignal, cci
    
    def cciExit(self,am,paraDict):
        cciPeriod = paraDict["cciPeriod"]
        cci = ta.CCI(am.high, am.low, am.close, cciPeriod)

        cciExitSignal = 0
        if cci[-1]>-100 and cci[-2]<-100:
            cciExitSignal = 1
        elif cci[-1]<100 and cci[-2]>100:
            cciExitSignal = -1
        else:
            cciExitSignal = 0
        return cciExitSignal, cci
    
    def bollStatus(self,am,paraDict):
        bollPeriod = paraDict["bollPeriod"]
        up, mid, low = ta.BBANDS(am.close, bollPeriod)
        
        bollExitSignal = 0
        if am.close[-1]<low[-1]:
            bollExitSignal = 1
        elif am.close[-1]>up[-1]:
            bollExitSignal = -1
        else:
            bollExitSignal = 0
        return bollExitSignal, up, mid, low
    
    def barPower(self,am,paraDict):
        bpPeriod = paraDict["bpPeriod"]
        o = am.open
        h = am.high
        l = am.low
        c = am.close
        v = am.volume
        
        spread = c-o
        rv = v/ta.SMA(v,bpPeriod)
        rv[spread==0]=0
        rv[np.isnan(rv)]=0
        bp = ta.EMA(spread,bpPeriod)
        mbp = ta.EMA(bp,3)

        mbpSignal = 0
        if bp[-1]>mbp[-1]:
            mbpSignal = 1
        else:
            mbpSignal = -1
        return mbpSignal, bp
        