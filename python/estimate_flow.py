# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd
from pdsql import mssql
import geopandas as gpd
from gistools import vector
import statsmodels.api as sm

import matplotlib.pyplot as plt


pd.options.display.max_columns = 10

#-Authorship information-########################################################################################################################
__author__ = 'Wilco Terink'
__copyright__ = 'Wilco Terink'
__version__ = '1.0'
__email__ = 'wilco.terink@ecan.govt.nz'
__date__ ='May 2019'
#################################################################################################################################################

server = 'edwprod01'
database = 'hydro'
dataset_type_table = 'DatasetType'
ts_summ_table = 'TSDataNumericDailySumm'
ts_table = 'TSDataNumericDaily'
site_table = 'ExternalSite'

min_obs = 2
buf_dist = 10000


#-Get lowflow sites
flow_sites_gdf = gpd.read_file(r'C:\Active\Projects\Ashburton\naturalisation\results\lowflow_sites.shp')
flow_sites = flow_sites_gdf.flow_site.tolist()
print(flow_sites_gdf)

## Read in datasettypes
datasets = mssql.rd_sql(server, database, dataset_type_table, ['DatasetTypeID', 'CTypeID'], where_in={'FeatureID': [1], 'MTypeID': [2], 'CTypeID': [1, 2], 'DataCodeID': [1]})

#-Get datasetTypes for recorded data and manual data
rec_datasetTypes = datasets[datasets.CTypeID == 1].DatasetTypeID.tolist()
man_datasetTypes = datasets[datasets.CTypeID == 2].DatasetTypeID.tolist()
all_datasetTypes = rec_datasetTypes
all_datasetTypes.extend(man_datasetTypes)

#-Get summary table for the lowflow sites
site_summ = mssql.rd_sql(server, database, ts_summ_table, where_in={'DatasetTypeID': all_datasetTypes, 'ExtSiteID': flow_sites})
site_summ.FromDate = pd.to_datetime(site_summ.FromDate)
site_summ.ToDate = pd.to_datetime(site_summ.ToDate)
site_summ.drop('ModDate', axis=1, inplace=True)

#-Throw out sites from summary table that have less records than 'min_obs'
too_short = site_summ.loc[site_summ.Count<min_obs, ['ExtSiteID','Count']]
if len(too_short)>0:
    for j in too_short.iterrows():
        print('ExtSiteID %s is not used because it only has %s records, which is less than %s.' %(j[1]['ExtSiteID'], int(j[1]['Count']), min_obs))
        site_summ = site_summ.loc[site_summ['ExtSiteID']!=j[1]['ExtSiteID']]
#-Lowflow sites to keep
keep_sites = pd.unique(site_summ['ExtSiteID']).tolist()

#-Get site ids for recorder and manual sites of the lowflow sites and create geodataframe for manually recorded lowflow sites
rec_sites = site_summ.loc[(site_summ.DatasetTypeID.isin(rec_datasetTypes)) & (site_summ.ExtSiteID.isin(keep_sites)),'ExtSiteID'].tolist()
man_sites = site_summ.loc[(site_summ.DatasetTypeID.isin(man_datasetTypes)) & (site_summ.ExtSiteID.isin(keep_sites)),'ExtSiteID'].tolist()
man_sites_gdf = flow_sites_gdf[flow_sites_gdf.flow_site.isin(man_sites)].copy()

#-get all recorder flow sites within a buffer distance from the manually recorded lowflow sites
man_sites_buffer_gdf = man_sites_gdf.copy()
man_sites_buffer_gdf['geometry'] = man_sites_gdf.buffer(buf_dist)
all_recorder_flowsites = mssql.rd_sql(server, database, ts_summ_table, ['ExtSiteID'], where_in={'DatasetTypeID': rec_datasetTypes})
all_recorder_flowsites = pd.unique(all_recorder_flowsites.ExtSiteID).tolist()
all_recorder_flowsites =  mssql.rd_sql(server, database, site_table, ['ExtSiteID', 'NZTMX', 'NZTMY'], where_in={'ExtSiteID': all_recorder_flowsites})
all_recorder_flowsites_gdf = vector.xy_to_gpd('ExtSiteID', 'NZTMX', 'NZTMY', all_recorder_flowsites)
all_recorder_flowsites_buffer_gdf = vector.sel_sites_poly(all_recorder_flowsites_gdf, man_sites_buffer_gdf)
all_recorder_flowsites_buffer_gdf.to_file(r'C:\Active\Projects\Ashburton\naturalisation\results\test.shp')
all_recorder_flowsites = None; all_recorder_flowsites_gdf = None; del all_recorder_flowsites, all_recorder_flowsites_gdf  

#-merge list of lowflow sites with list of recorder sites in buffer to create list of all sites for which to extract data
all_flow_sites = flow_sites
all_flow_sites.extend(pd.unique(all_recorder_flowsites_buffer_gdf.ExtSiteID).tolist())
all_flow_sites = pd.DataFrame(columns=all_flow_sites)
all_flow_sites = pd.unique(all_flow_sites.columns).tolist()


#-Get time-series of all_flow_sites
ts_df = mssql.rd_sql(server, database, ts_table, ['ExtSiteID', 'DatasetTypeID', 'DateTime', 'Value', 'QualityCode'], where_in={'DatasetTypeID': all_datasetTypes, 'ExtSiteID': all_flow_sites})
ts_df = ts_df.loc[ts_df.QualityCode>=100] #-everything otherwise than missing data
ts_df.DateTime = pd.to_datetime(ts_df.DateTime)
ts_df_sites = pd.unique(ts_df.ExtSiteID).tolist()
all_flow_sites = None; del all_flow_sites

#-Check if there is data for all lowflow sites
for j in flow_sites:
    if j not in ts_df_sites:
        print('There is zero flow data for site %s, or record of observations for this site was too short.' %j) 
#-Minimum and maximum date of all data
min_date = ts_df.DateTime.min()
max_date = ts_df.DateTime.max()

#-Fill dataframe with data for min_date through max_date
df_final = pd.DataFrame(index=pd.date_range(min_date, max_date, freq='D'), columns=ts_df_sites)
df_final.rename_axis('DateTime', inplace=True)
for j in ts_df_sites:
    df_short = ts_df.loc[ts_df.ExtSiteID==j, ['DateTime', 'Value','QualityCode']]
    #-keep only the records with the highest qualitycode
    df_short_group = df_short.groupby(['DateTime', 'QualityCode']).mean()
    df_short_group.reset_index(inplace=True)
    df_short_group.sort_values(by=['DateTime', 'QualityCode'], inplace=True)
    df_short_group.drop_duplicates(subset='DateTime', keep='last', inplace=True)
    df_short_group.drop('QualityCode', inplace=True, axis=1)
    df_short_group.set_index('DateTime', inplace=True)
    df_short_group.rename(columns={'Value':j}, inplace=True)
    df_final[[j]] = df_short_group
ts_df = None; df_short_group = None; del ts_df, df_short_group

# print(df_final.head())
# df_final.to_csv(r'C:\Active\Projects\Ashburton\naturalisation\results\df.csv')

#-loop over the lowflow sites and calculate regressions
df_regression = pd.DataFrame(columns=['y', 'x', 'nobs', 'rmse', 'Adj. R2-squared', 'F-statistic', 'p-value'])
for j in flow_sites:
    for s in df_final.columns:
        print(j,s)
        sel_df = df_final[[j, s]]
        sel_df.replace(0., np.nan, inplace=True)
        sel_df.dropna(inplace=True)
        
        if len(sel_df)>min_obs:
            #-set x and y
            x = sel_df[[s]]
            x = sm.add_constant(x)  #-needed for intercept
            y = sel_df[[j]]
            #-linear fit
            model = sm.OLS(y, x).fit()
            
            #-some stats
            print(model.params)
            print(model.nobs)
            print(model.rsquared_adj)
            print(model.f_pvalue)
            
            #-predict using the model
            predictions = model.predict(x) # make the predictions by the model
            rmse_val = np.sqrt(  np.mean(   (predictions.to_numpy() - y.to_numpy())**2   )  )
            
            print('RMSE: %s' %rmse_val)
            
            #-figure for linear fit
            fig, ax = plt.subplots()
            plt.scatter(x[[s]], y)
            plt.plot(x[[s]], predictions)
            plt.show()
            
            #-TEST FOR POWER LAW FITTING
            print('POWER LAW FITTING....\n\n')
#             y = ax^b
#             log(y) = log(a) + b*log(x)
            logY = np.log(y)
            logX = np.log(x[[s]])
            logX = sm.add_constant(logX)
            print(logX, logY)
            model = sm.OLS(logY, logX).fit()
            
            print(model.params)
            #print(model.rsquared)
            print(model.nobs)
            print(model.rsquared_adj)
            #print(model.pvalues)
            #print(model.mse_model)
            print(model.f_pvalue)
            
            
            
            #----   y = ax^b
            x = x[s].to_numpy()
            #-create a range of x values for smooth curve
            x1 = np.arange(np.min(x), np.max(x), (np.max(x) - np.min(x))/100)
            #-smooth curve to display
            predictions = np.exp(model.params[0]) * x1**model.params[1]
            
            fig, ax = plt.subplots()
            plt.scatter(x, y)
            plt.plot(x1, predictions)
            plt.show()

            #-predictions of the exact x values for calculating rmse            
            predictions = np.exp(model.params[0]) * x**model.params[1]
            rmse_val = np.sqrt(  np.mean(   (predictions - y.to_numpy())**2   )  )

            
            print('RMSE: %s' %rmse_val)
            

