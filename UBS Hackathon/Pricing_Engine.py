import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from collections import OrderedDict
import scipy.interpolate as interp
import scipy.stats as sci
from scipy.stats import norm
from scipy import interpolate
from numpy import array
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import warnings
warnings.filterwarnings('ignore')

import get_hist_refinitiv as get_h

getter = get_h.getter()

class Conrad:
    """Vanna-Volga Implied Volatility method used to price an FX option"""

    def __init__(self, S0, strike, t,date,**kwarg):
        '''
        S0: current fx rate
        strike: strike rate
        t: tenor
        kwarg: absort additional info, such as typ for call/put option type
            option_type = 'c'/'p' default = 'c'
        importfile: imported file name
        '''
        self.dc = 360  # day count convention
        self.S0 = S0
        self.strike = strike
        self.t = t
        self.date=date
        self.kwarg=kwarg
#        self.importfile = importfile

    def foreign_rate(self, fp, Rates,T):
        '''
        cal foreign rates with forward points and local rates
        generate local rates corresponding to forward periods
        return local and foreign rates
        是否在1d interp时x轴用term更好？
        是否时间匹配错误？
        '''
        xf = [] * len(fp) # ???
        Rates_array = np.array(Rates).squeeze() # 2D -> 1D array
        # interplot, input any data within range
        Rates_interp = interp.interp1d(np.arange(Rates_array.size), Rates_array, kind='cubic') # rate
        Rates_compress = Rates_interp(np.linspace(0, Rates_array.size - 1, fp.size))
        for i in range(len(Rates_compress)):
            xf.append((((fp[i] / self.S0) + 1) * (1 + Rates_compress[i]*T[i]) - 1)/T[i])
#        print('xd:',Rates_compress, '\nxf',xf)
        return Rates_compress, xf

    def k25_data(self, T, sig25DeltaPut, sigATM, sig25DeltaCall, xd, xf):
        """
        Calculate the strikes K25DeltaPut, KATM, K25DeltaCall
        """
        # Pricing Inputs
        Dd = np.exp(-np.multiply(xd, T))
        Df = np.exp(-np.multiply(xf, T))
        delta = 0.25
        alpha = -sci.norm.ppf(delta * np.reciprocal(Df))
        mu = xd - xf
        F = self.S0 * np.exp(np.multiply(mu, T))

        # Strikes
        KATM = np.array([])
        K25DeltaCall = np.array([])
        K25DeltaPut = np.array([])

        for i in range(len(T)):
            K25DeltaPut = np.append(K25DeltaPut, F[i] * math.exp(
                -(alpha[i] * sig25DeltaPut[i] * math.sqrt(T[i])) + (xd[i] - xf[i] + 0.5 * sig25DeltaPut[i] ** 2) * T[i]))
            KATM = np.append(KATM, F[i] * math.exp(xd[i] - xf[i] + 0.5 * T[i] * (sigATM[i]) ** 2))
            K25DeltaCall = np.append(K25DeltaCall, F[i] * math.exp(
                alpha[i] * sig25DeltaCall[i] * math.sqrt(T[i]) + (xd[i] - xf[i] + 0.5 * sig25DeltaCall[i] ** 2) * T[i]))

        return K25DeltaPut, KATM, K25DeltaCall, F

    def d1(self, F, K, sig, T, xd, xf):
        d1_val = (np.log(F / K) + (xd - xf + 0.5 * (sig ** 2)) * T) / (sig * np.sqrt(T))
        return d1_val

    def d2(self, F, K, sig, T, xd, xf):
        d2_val = self.d1(F, K, sig, T, xd, xf) - sig * np.sqrt(T)
        return d2_val

    def bs_price(self, F, K, T, xd, xf, sig):
        ''' 
        price for call 
        put = call + K*e(-rd t) - S0*e(-rf t)
        '''
        a = norm.cdf(self.d1(F, K, sig, T, xd, xf))
        b = self.d1(F, K, sig, T, xd, xf)
        if 'option_type' in self.kwarg.keys():
            option_type = self.kwarg['option_type']
        else:
            option_type = 'c'
        if option_type=='p':
            bs_val = - norm.cdf(- self.d1(F, K, sig, T, xd, xf)) * F * math.exp(-xf * T) + norm.cdf(
                - self.d2(F, K, sig, T, xd, xf)) * F * math.exp(-xd * T)
        else:   
            bs_val = norm.cdf(self.d1(F, K, sig, T, xd, xf)) * F * math.exp(-xf * T) - norm.cdf(
                self.d2(F, K, sig, T, xd, xf)) * F * math.exp(-xd * T)
        return bs_val
    
    def vega(self, F, K, T, xd, xf, sig):
        np = math.exp(-self.d1(F, K, sig, T, xd, xf) ** 2 / 2) / math.sqrt(2 * math.pi)
        vega_val = (F * math.exp(-xf * T) * math.sqrt(T) * np) / 100

        return vega_val

    def delta(self, F, K, T, xd, xf, sig):
        delta_val = math.exp(xf * T) * norm.cdf(self.d1(F, K, sig, T, xd, xf))

        return delta_val
    
    def implied_price(self, F, K, T, K1, K2, K3, sig1, sig2, sig3, xd, xf):
        v_v1 = self.vega(F, K, T, xd, xf, sig2) / self.vega(F, K1, T, xd, xf, sig1)
        v_v2 = self.vega(F, K, T, xd, xf, sig2) / self.vega(F, K2, T, xd, xf, sig2)
        v_v3 = self.vega(F, K, T, xd, xf, sig2) / self.vega(F, K3, T, xd, xf, sig3)

        x1 = v_v1 * ((math.log(K2 / K) * math.log(K3 / K)) / (math.log(K2 / K1) * math.log(K3 / K1)))
        x2 = v_v2 * ((math.log(K / K1) * math.log(K3 / K)) / (math.log(K2 / K1) * math.log(K3 / K2)))
        x3 = v_v3 * ((math.log(K / K1) * math.log(K / K2)) / (math.log(K3 / K1) * math.log(K3 / K2)))

        a = self.bs_price(F, K2, T, xd, xf, sig2)

        vv_price = self.bs_price(F, K2, T, xd, xf, sig2) + x1 * (
        self.bs_price(F, K1, T, xd, xf, sig1) - self.bs_price(F, K1, T, xd, xf, sig2))
        + x3 * (self.bs_price(F, K3, T, xd, xf, sig3) - self.bs_price(F, K3, T, xd, xf, sig2))

        return vv_price

    def market_price(self, F, K, T, K1, K2, K3, sig1, sig2, sig3, xd, xf, j):
        sig = self.implied_vol(F, K, T, K1, K2, K3, sig1, sig2, sig3, xd, xf, j)
        market_val = self.bs_price(F, K, T, xd, xf, sig)

        return market_val

    def implied_vol(self, F, K, T, K1, K2, K3, sig1, sig2, sig3, xd, xf, j):
        x1 = (math.log(K2 / K) * math.log(K3 / K)) / (math.log(K2 / K1) * math.log(K3 / K1))
        x2 = (math.log(K / K1) * math.log(K3 / K)) / (math.log(K2 / K1) * math.log(K3 / K2))
        x3 = (math.log(K / K1) * math.log(K / K2)) / (math.log(K3 / K1) * math.log(K3 / K2))

        D1 = (x1 * sig1 + x2 * sig2 + x3 * sig3) - sig2

        D2 = (x1 * self.d1(F, K1, sig1, T, xd, xf) * self.d2(F, K1, sig1, T, xd, xf) * (sig1 - sig2) ** 2) \
             + (x2 * self.d1(F, K2, sig2, T, xd, xf) * self.d2(F, K2, sig2, T, xd, xf) * (sig2 - sig2) ** 2) \
             + (x3 * self.d1(F, K3, sig3, T, xd, xf) * self.d2(F, K3, sig3, T, xd, xf) * (sig3 - sig2) ** 2)

        sig = sig2 + (-sig2 + np.sqrt(
            sig2 ** 2 + self.d1(F, K, sig2, T, xd, xf) * self.d2(F, K, sig2, T, xd, xf) * (2 * sig2 * D1 + D2))) \
                     / (self.d1(F, K, sig2, T, xd, xf) * self.d2(F, K, sig2, T, xd, xf))

        return sig

    def interp_vv(self, row_find, col_find, flat_row_array, flat_col_array, z_matrix):
        GD = interpolate.griddata((flat_row_array, flat_col_array), z_matrix,
                                  ([row_find], [col_find]), method='cubic')

        return GD

    def create_plots(self, F, T, K25DeltaPut, KATM, K25DeltaCall, sig25DeltaPut, sigATM, sig25DeltaCall, xd,
                     xf):
        # segmentation according to tenor
        m = len(T[:-1])
        K_array = np.linspace(0.8 * self.S0, 1.2 * self.S0, m) # saving K derived from implied vol
        
        # gen implied vol
        implied_matrix = np.zeros((m, m), dtype=float)
        for i in range(K_array.size):
            for j in range(m):
                implied_matrix[i][j] = self.implied_vol(F[j], K_array[i], T[j], K25DeltaPut[j], KATM[j],
                                                        K25DeltaCall[j],
                                                        sig25DeltaPut[j], sigATM[j], sig25DeltaCall[j], xd[j], xf[j], j)
        implied_matrix = pd.DataFrame(implied_matrix)
        implied_matrix.fillna(method='ffill', axis=1, inplace=True)
        
        # gen price, market price, delta
        price_matrix = np.zeros((m, m), dtype=float)
        market_price_matrix = np.zeros((m, m), dtype=float)
        delta_matrix = np.zeros((m, m), dtype=float)
        for i in range(K_array.size):
            for j in range(m):
                price_matrix[i][j] = self.implied_price(F[j], K_array[i], T[j], K25DeltaPut[j], KATM[j],
                                                        K25DeltaCall[j],
                                                        sig25DeltaPut[j], sigATM[j], sig25DeltaCall[j], xd[j], xf[j])
                market_price_matrix[i][j] = self.market_price(F[j], K_array[i], T[j], K25DeltaPut[j], KATM[j],
                                                              K25DeltaCall[j],
                                                              sig25DeltaPut[j], sigATM[j], sig25DeltaCall[j], xd[j],
                                                              xf[j], j)
                delta_matrix[i][j] = self.delta(F[j], K_array[i], T[j], xd[j], xf[j], implied_matrix[i][j])

        #  Fill NAs
        implied_matrix = pd.DataFrame(implied_matrix)
        implied_matrix.fillna(method='ffill', axis=1, inplace=True)
        price_matrix = pd.DataFrame(price_matrix)
        price_matrix.fillna(method='ffill', axis=1, inplace=True)
        market_price_matrix = pd.DataFrame(market_price_matrix)
        market_price_matrix.fillna(method='ffill', axis=1, inplace=True)
        delta_matrix = pd.DataFrame(delta_matrix)
        delta_matrix.fillna(method='ffill', axis=1, inplace=True)

        #  Flatten the matrices for use with griddata
        implied_flat = implied_matrix.as_matrix().ravel()
        price_flat = price_matrix.as_matrix().ravel()
        market_price_flat = market_price_matrix.as_matrix().ravel()
        delta_flat = delta_matrix.as_matrix().ravel()
        
        # meshgrid = generate 2D matrix 
        # ravel = flatten , 2D->1D array, except that ravel adjust on view
        # griddata = get 3D plot rst. input = 2D array (x,y), value z, (x1,y1),method=''
        KK, TT = np.meshgrid(np.array(K_array), np.array(T[:-1]), indexing='ij')

        option_vol = interpolate.griddata((KK.ravel(), TT.ravel()), implied_flat,
                                  ([self.t], [self.strike]), method='nearest')

        option_premium = interpolate.griddata((KK.ravel(), TT.ravel()), price_flat,
                                  ([self.t], [self.strike]), method='nearest')

        option_delta = interpolate.griddata((KK.ravel(), TT.ravel()), delta_flat,
                                  ([self.t], [self.strike]), method='nearest')
        
        option_market_price = interpolate.griddata((KK.ravel(), TT.ravel()), market_price_flat,
                                  ([self.t], [self.strike]), method='nearest')
        
        return option_premium, option_vol, option_delta, option_market_price

    def get_data(self):
        ''' get data '''
        date=self.date # for easy testing
        tenor_dic={'ON':'1',
                   '1W':'7','2W':'14','3W':'21',
                   '1M':'30','2M':'60','3M':'90'}
        typ_dic={'O':'ATM','RR':'25 Delta Risk Reversal','BF':'25 Delta Butterfly',
                 'R10':'10 Delta Risk Reversal','B10':'10 Delta Butterfly'}
        # Import Excel file, a excelfile type
        # use parse(sheet name) to extrace DF
        # get Option
        vol_data = []
        tenor_l=['ON','1M','2M','3M']
        typ_l=['O','RR','BF','B10','R10']
        for tenor in tenor_l:
            for typ in typ_l:
                mark='JPY'+tenor+typ
                temp=getter.get_hist_d(mark,date)
                temp['Term']=tenor_dic[tenor]
                temp['Strike']=typ_dic[typ]
                temp['Value']=(temp['BID']+temp['ASK'])/2
                vol_data.append(temp[['Term','Strike','Value']])
        vol_data=pd.concat(vol_data)
        
        # get US rate
        swaps_futures = []     
        for tenor in tenor_l:
            # mark='US'+tenor+'RP'
            mark='USD'+tenor+'FSR'
            temp=getter.get_hist_d(mark,date)
            temp['Value']=temp['FIXING_1']
            temp['Term']=tenor_dic[tenor]
            swaps_futures.append(temp[['Term','Value']])
        swaps_futures=pd.concat(swaps_futures)
        
        # get fp rate
        fp_data = []
        for tenor in tenor_l:
            mark='JPY'+tenor
            temp=getter.get_hist_d(mark,date)
            temp['Value']=(temp['BID']+temp['ASK'])/2
            temp['Term']=tenor_dic[tenor]
            fp_data.append(temp[['Term','Value']])
        fp_data=pd.concat(fp_data)
        
        return vol_data, swaps_futures, fp_data
    
    def FXOptionPrice(self):
        '''
        main
        '''
        vol_data, swaps_futures, fp_data = self.get_data()
        ''' clear data '''
        # dictionary for excel tabulated terms converted to day/360 format
        # Modify the terms to daycount and store the vols in numpy arrays
        # change str into formula
        # exec + change to yr
        dc= self.dc
        # Volatility Data
        T = pd.DataFrame({'X': vol_data['Term'].unique()})  # terms
#        T = T.replace(super_dict, regex=True) # modify term
        T = T.X.apply(lambda x: eval(str(x))) / dc # change time to yr

        # US Yield Data
        Rates = np.array(swaps_futures['Value']) / 100
        T_Rates = pd.DataFrame({'X': swaps_futures['Term']}) # term
#        T_Rates = T_Rates.replace(super_dict, regex=True) 
        T_Rates = T_Rates.X.apply(lambda x: eval(str(x))) / dc

        # Forward Points data
        fp = np.array(fp_data['Value']) / 100 # forward points, JPYvsUSD use 100
        T_fp = pd.DataFrame({'X': fp_data['Term']}) # fw term
#        T_fp = T_fp.replace(super_dict, regex=True) 
        T_fp = T_fp.X.apply(lambda x: eval(str(x))) / dc

        ''' cal Delta '''
        # Delta cal
        sigATM = np.array(vol_data[vol_data['Strike'] == 'ATM']['Value']) / 100  # ATM volatilities
        sig25DeltaRR = np.array(
            vol_data[vol_data['Strike'] == '25 Delta Risk Reversal']['Value']) / 100  # ATM volatilities
        sig25DeltaBF = np.array(vol_data[vol_data['Strike'] == '25 Delta Butterfly']['Value']) / 100  # ATM volatilities
        
        sig25DeltaCall = sig25DeltaBF + sigATM + sig25DeltaRR / 2
        sig25DeltaPut = sig25DeltaBF + sigATM - sig25DeltaRR / 2

        xd, xf = self.foreign_rate(fp, Rates,T) # cal foreign rates
        K25DeltaPut, KATM, K25DeltaCall, F = self.k25_data(T, sig25DeltaPut, sigATM, sig25DeltaCall, xd, xf) # get K from delta
        option_data = self.create_plots(F, T, K25DeltaPut, KATM, K25DeltaCall, sig25DeltaPut, sigATM, sig25DeltaCall, xd, xf)
        return option_data, xd,xf,T

class Options(object):
    def __init__(self, S, K, T, r, q, sigma):
        self.S = float(S)
        self.K = float(K)
        self.T = float(T)
        self.r = float(r)
        self.q = float(q)
        self.sigma = float(sigma)

class EuropeanOptions(Options):
    def value_BS(self):
        d1 = ((np.log(self.S / self.K)
            + (self.r + 0.5 * self.sigma ** 2) * self.T)
            / (self.sigma * np.sqrt(self.T)))
        d2 = ((np.log(self.S / self.K)
            + (self.r - 0.5 * self.sigma ** 2) * self.T)
            / (self.sigma * np.sqrt(self.T)))
        c = self.S * np.exp(-self.q * self.T) * norm.cdf(d1, 0.0, 1.0) \
            - self.K * np.exp(-self.r * self.T) * norm.cdf(d2, 0.0, 1.0)
        p = self.K * np.exp(-self.r * self.T) * norm.cdf(-d2, 0.0, 1.0) \
            - self.S * np.exp(-self.q * self.T) * norm.cdf(-d1, 0.0, 1.0)
        return {'c': c, 'p': p}
    
def get_option_price(S0,strike,t,date,option_type='c'):
    '''
    S0: FX spot rate, float
    strike: FX strike price, float
    t: day to expriation, int
    date: current date, str
    option_type: 'c','p'
    '''
    # Example Input
#    S0=108.2
#    strike=108.2
#    t=33
#    date='2018-08-06'
#    option_type='p'
    (option_premium, option_vol, option_delta, option_market_price),xd,xf,T = Conrad(
            S0=S0, strike=strike, t=t,date=date,option_type=option_type).FXOptionPrice()
    T_interp = interp.interp1d(x=T, y=xd, kind='linear') # rate
    xd_int = T_interp(np.array([t/360]))[0]
    T_interp = interp.interp1d(x=T, y=xf, kind='linear') # rate
    xf_int = T_interp(np.array([t/360]))[0]
    option = EuropeanOptions(S0,strike,t/360,xd_int,xf_int,option_vol).value_BS()
#    print(option_vol)
    return option[option_type]

def get_forward(date, t):
    '''
    get forward
    date = str
    t = int
    '''
    def get_data(date):
        ''' get data '''
        tenor_dic={'ON':'1',
                   '1W':'7','2W':'14','3W':'21',
                   '1M':'30','2M':'60','3M':'90'}
        
        tenor_l = ['ON','1M','2M','3M']
        # get spot
        spot_data = []
        for tenor in tenor_l:
            mark='JPY'
            temp=getter.get_hist_d(mark,date)
            temp['Value']=(temp['BID']+temp['ASK'])/2
            temp['Term']=tenor_dic[tenor]
            spot_data.append(temp[['Term','Value']])
        spot_data=pd.concat(spot_data)
        
        # get fp rate
        fp_data = []
        for tenor in tenor_l:
            mark='JPY'+tenor
            temp=getter.get_hist_d(mark,date)
            temp['Value']=(temp['BID']+temp['ASK'])/2
            temp['Term']=tenor_dic[tenor]
            fp_data.append(temp[['Term','Value']])
        fp_data=pd.concat(fp_data)
        
        return spot_data, fp_data
    assert type(date)==str, "date={} type error, date should be like '2018-08-06'".format(date)
    assert 1<=t<=90, "date {} should be within range 1-90 days".format(t)
    # get data
    spot_data, fp_data = get_data(date)
    try:
        # inter plot
        interp_spot = interp.interp1d(x=spot_data['Term'].apply(int), 
                                      y=spot_data['Value'], kind='linear') # rate
        spot = interp_spot(np.array([t]))[0]
        interp_fp = interp.interp1d(x=fp_data['Term'].apply(int), 
                                      y=fp_data['Value'], kind='linear') # rate
        fp = interp_fp(np.array([t]))[0]
    except:
        print(date,t,spot_data,fp_data)
        raise ValueError

    return spot + fp/100

if __name__ == "__main__":
    S0=108
    strike=108.2
    t=2
    date='2018-08-06'
    option_type='c'
    print(get_option_price(S0,strike,t,date,option_type))
    forward = get_forward(date, t)
    print(forward)



    

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    



