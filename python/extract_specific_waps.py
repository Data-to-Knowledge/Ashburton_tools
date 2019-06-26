# -*- coding: utf-8 -*-

'''
Extacts time-series of abstraction for a list of pre-given waps
'''

#-Authorship information-########################################################################################################################
__author__ = 'Wilco Terink'
__copyright__ = 'Wilco Terink'
__version__ = '1.0'
__email__ = 'wilco.terink@ecan.govt.nz'
__date__ ='June 2019'
#################################################################################################################################################

import pandas as pd
from pdsql import mssql
import numpy as np

database = 'hydro'
server = 'edwprod01'

waps = ['K37/2710', 'K37/2752', 'K37/2607', 'K37/0879', 'K37/0727', 'K37/1012', 'K37/0793', 'K37/0593', 'L37/1770']

from_date = '2009-01-01'
to_date = '2018-12-31'

df_ts = mssql.rd_sql(server, database, 'TSDataNumericDaily', col_names = ['ExtSiteID', 'DatasetTypeID', 'QualityCode', 'DateTime', 'Value'], where_in = {'ExtSiteID': waps, 'DatasetTypeID': [9, 12]})
df_ts['DateTime'] = pd.to_datetime(df_ts['DateTime'])
df_ts = df_ts.loc[(df_ts['DateTime']>=pd.Timestamp(from_date)) & (df_ts['DateTime']<=pd.Timestamp(to_date))]
df_ts.rename(columns={'DateTime': 'Date'}, inplace=True)

#-empty dataframe to be filled with metered ts
wap_ts_metered_df = pd.DataFrame(index=pd.date_range(from_date, to_date, freq='D'), columns=waps)
wap_ts_metered_df.rename_axis('Date', inplace=True)

#-fill self.wap_ts_metered_df with abstraction data from df_ts
lMessageList=[]
for w in waps:
    #print w
    df_sel = df_ts.loc[df_ts['ExtSiteID']==w,['Date', 'Value']]
    df_sel.set_index('Date', inplace=True)
    #-Set non-reliable valuues (<0) to NaN
    df_sel.loc[df_sel['Value']<0] = np.nan
    df_sel.rename(columns={'Value': w}, inplace=True)
    try:
        wap_ts_metered_df[[w]] = df_sel
    except:
        lMessage = w + ' has multiple DatasetTypeIDs assigned to it. This is not possible and needs to be checked; i.e. a WAP can not be a divert and surface water take at the same time!!'
        lMessageList.append(lMessage) 
        print(lMessage)
        #-use the first occuring entry two prevent double datasettypeids
        df_sel.reset_index(inplace=True)
        df_sel = df_sel.groupby('Date').first()
        wap_ts_metered_df[[w]] = df_sel

wap_ts_metered_df = wap_ts_metered_df / (24*3600)        
        
wap_ts_metered_df.to_csv(r'C:\Active\Projects\Ashburton\naturalisation\results4\selected_wap_usage.csv')        
        
        