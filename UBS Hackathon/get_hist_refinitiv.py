# -*- coding: utf-8 -*-
"""
UBS Hackathon 2019
Get hist data from refinitiv

Created on Tue Oct  8 14:09:37 2019

@author: yello
"""

import pandas as pd
import requests
import json

# Data Science Accelerator Credentials 
access_token = 'PUEtXpOJ974oPBhU2RkQ91HVwtFRjBf7LEZoJd7e'  # your personal key for Data Science Accelerator access to Historical Pricing Data
#access_token = 'CCYtO2gn9U96DYMQY3RAu8L43fqO4rNr9F0ICyyt' # Kohr
def get_data_request(url, requestData):
    """
    HTTP GET request to Refinitiv API
    
    There is more information in the returned dict (i.e. json) object from the API, we store 
    the data in a DataFrame.
    
    :param url: str, the url of the API endpoint
    :param requestData: dict, contains user-defined variables
    :return: DataFrame, containing the historical pricing data. 
        Returned field list order and content can vary depending on the exchange / instrument.
        Therefore returned data must be treated using appropriate parsing techniques.
    """
    dResp = requests.get(url, headers = {'X-api-key': access_token}, params = requestData);       

    if dResp.status_code != 200:
        raise ValueError("Unable to get data. Code %s, Message: %s" % (dResp.status_code, dResp.text));
    else:
        print("Data access successful")
        jResp = json.loads(dResp.text);
        data = jResp[0]['data']
        headers = jResp[0]['headers']  
        names = [headers[x]['name'] for x in range(len(headers))]
        df = pd.DataFrame(data, columns=names )
        return df

def get_hist(ric='EURONO',interval='P1D',start='2016-11-01',end='2018-10-31',fields=None):
    '''
    ric : The RIC of the asset for which you want to retrieve data
    start_date, end_date : These need to be within the range permissioned by the Data Science 
        Accelerator
    interval : The bar width
    For intraday, 1 to 60 minutes, PT1M, PT5M, PT10M, PT30M, PT60M
    For interday, 1 day to 1 year, P1D, P1W (or P7D), P1M, P3M, P12M (or P1Y)
    fields : A comma separated list of case-sensitive field names (i.e. header names), to filter 
        the returned data to a set of fields. By default all available fields are returned. If you 
        are unsure of which field names are available for the asset you are requesting, we recommed
        leaving fields empty.
    '''
    # ric = 'EURONO=' # put the RIC of the asset you want to retrieve data
    requestData = {
        'interval': interval,
        'start': start,
        'end': end,
    }
    if fields != None:
        requestData.update({'fields':fields})
        #"fields": 'TRDPRC_1' # Uncomment this line if you wish to specify which fields to be 
        # returned, e.g. TRDPRC_1 is an available field for AAPL.O
    
    RESOURCE_ENDPOINT = "https://dsa-stg-edp-api.fr-nonprod.aws.thomsonreuters.com/data/historical-pricing/beta1/views/summaries/" 
    resource_endpoint_ric = RESOURCE_ENDPOINT + ric  +'='
    df = get_data_request(resource_endpoint_ric, requestData)  
    return df

class getter:
    ''' for loading data in session '''
    data_dict={}
    def get_hist_d(self, ric, date):
        '''
        get daily data
        '''
        if ric in getter.data_dict.keys():
            return pd.DataFrame(getter.data_dict[ric].loc[date]).T
        else:
            data = get_hist(ric)
            if len(data) > 0:
                data.set_index('DATE',inplace=True)
                getter.data_dict[ric]=data
                return pd.DataFrame(getter.data_dict[ric].loc[date]).T
            else:
                return data

if __name__=='__main__':
    data = getter().get_hist_d('JPYTWBF','2017-07-05')

        
        