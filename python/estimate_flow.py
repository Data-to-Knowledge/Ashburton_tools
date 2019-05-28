# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd
from pdsql import mssql
import geopandas as gpd
from gistools import vector
import statsmodels.api as sm

import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams.update({'font.size': 8})


pd.options.display.max_columns = 100

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
flow_sites_copy = flow_sites.copy()

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
flow_sites = flow_sites_copy; flow_sites_copy = None

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

# #-loop over the lowflow sites and calculate regressions
# df_regression = pd.DataFrame(columns=['y', 'x', 'mean_y', 'mean_x', 'nobs', 'rmse', 'Adj. R2-squared', 'p-value', 'slope', 'intercept', 'power', 'fittype'])
# i=0
# for j in flow_sites:
# 
#     for s in df_final.columns:
#          
#         sel_df = df_final[[j, s]]
#         sel_df.replace(0., np.nan, inplace=True)
#         sel_df.dropna(inplace=True)
#         #-only fit for sites that have the minimum number of observations
#         if len(sel_df)>=min_obs:
#             try:
#                 #-set x and y
#                 x = sel_df[[s]]
#                 x = sm.add_constant(x)  #-needed for intercept
#                 y = sel_df[[j]]
#                  
#                 #-linear fit
#                 model = sm.OLS(y, x).fit()
#                  
#                 #-predict using the model
#                 predictions = model.predict(x) # make the predictions by the model
#                 rmse_val = np.sqrt(  np.mean(   (predictions.to_numpy() - y.to_numpy())**2   )  )
#                  
#                 #-fill dataframe with stats for linear fit
#                 df_regression.loc[i, 'y'] = j
#                 df_regression.loc[i, 'x'] = s
#                 df_regression.loc[i, 'mean_y'] = np.nanmean(y)
#                 df_regression.loc[i, 'mean_x'] = np.nanmean(x[s].to_numpy())
#                 df_regression.loc[i, 'nobs'] = model.nobs
#                 df_regression.loc[i, 'rmse'] = rmse_val
#                 df_regression.loc[i, 'Adj. R2-squared'] = model.rsquared_adj
#                 df_regression.loc[i, 'p-value'] = model.f_pvalue
#                 df_regression.loc[i, 'slope'] = model.params[1]
#                 df_regression.loc[i, 'intercept'] = model.params[0]
#                 df_regression.loc[i, 'fittype'] = 'linear'
#                  
#                 i+=1
#  
# #                 #-figure for linear fit
# #                 fig, ax = plt.subplots()
# #                 plt.scatter(x[[s]], y)
# #                 plt.plot(x[[s]], predictions)
# #                 plt.show()
#             except:
#                 print('Could not establish linear fit for %s and %s...' %(j,s))
# 
#             try:
#                 #-set x and y
#                 x = sel_df[[s]]
#                 x = sm.add_constant(x)  #-needed for intercept
#                 y = sel_df[[j]]                 
# 
#                 #-***********************************************************
#                 #-                           y = ax^b
#                 #-                           log(y) = log(a) + b*log(x)
#                 #-***********************************************************
#                 logY = np.log(y)
#                 logX = np.log(x[[s]])
#                 logX = sm.add_constant(logX)
#                 model = sm.OLS(logY, logX).fit()
#  
#                 x = x[s].to_numpy()
#                 #-predictions of the exact x values for calculating rmse            
#                 predictions = np.exp(model.params[0]) * x**model.params[1]
#                 rmse_val = np.sqrt(  np.mean(   (predictions - y.to_numpy())**2   )  )
#                  
#                 #-fill dataframe with stats for power fit
#                 df_regression.loc[i, 'y'] = j
#                 df_regression.loc[i, 'x'] = s
#                 df_regression.loc[i, 'mean_y'] = np.nanmean(y)
#                 df_regression.loc[i, 'mean_x'] = np.nanmean(x)
#                 df_regression.loc[i, 'nobs'] = model.nobs
#                 df_regression.loc[i, 'rmse'] = rmse_val
#                 df_regression.loc[i, 'Adj. R2-squared'] = model.rsquared_adj
#                 df_regression.loc[i, 'p-value'] = model.f_pvalue
#                 df_regression.loc[i, 'slope'] = np.exp(model.params[0])
#                 df_regression.loc[i, 'power'] = model.params[1]
#                 df_regression.loc[i, 'fittype'] = 'power'
#                  
# #                 #-create a range of x values for smooth curve
# #                 x1 = np.arange(np.min(x), np.max(x), (np.max(x) - np.min(x))/100)
# #                 #-smooth curve to display
# #                 predictions = np.exp(model.params[0]) * x1**model.params[1]
# #                 fig, ax = plt.subplots()
# #                 plt.scatter(x, y)
# #                 plt.plot(x1, predictions)
# #                 plt.show()
#  
#                 i+=1
#                  
#             except:
#                 print('Could not establish power fit for %s and %s...' %(j,s))
# sel_df = None; del sel_df
# #-calculate absolute R2 for sorting
# df_regression['Adj. R2-squared absolute'] = df_regression['Adj. R2-squared'].abs()
# #-remove negative slopes
# df_regression.loc[df_regression.slope<0,:] = np.nan
# df_regression.dropna(how='all', inplace=True)
# 
# df_regression.to_csv(r'C:\Active\Projects\Ashburton\naturalisation\results\test.csv', index=False)
# df_regression = pd.read_csv(r'C:\Active\Projects\Ashburton\naturalisation\results\test.csv')             
# print(df_regression)
# 
# 
# #-loop over lowflow sites, and select best four fits by sorting on R2, rmse, nobs
# df_regression_best = pd.DataFrame(columns=df_regression.columns)
# for j in flow_sites:
#     sel_df = df_regression.loc[df_regression.y == j]
#     sel_df.sort_values(by=['Adj. R2-squared absolute', 'rmse', 'nobs'], ascending=[False, True, False], inplace=True)
#     sel_df = sel_df.iloc[0:6,:]
#     df_regression_best = pd.concat([df_regression_best, sel_df])
# sel_df = None; del sel_df
# print(df_regression_best)
#    
# df_regression_best.to_csv(r'C:\Active\Projects\Ashburton\naturalisation\results\test_best.csv', index=False)
df_regression_best = pd.read_csv(r'C:\Active\Projects\Ashburton\naturalisation\results\test_best.csv')

#-make 6 plots for each site to visually pick the best
unique_y = pd.unique(df_regression_best['y'])
unique_y = list(unique_y)
for j in flow_sites:
    if int(j) in unique_y:

        #-get regression results for site j
        sel_df_regression = df_regression_best.loc[df_regression_best.y == int(j)]
        #-get the y values
        y = df_final[j].to_numpy()
        #-plot the best 6 fits
        fig = plt.figure(figsize=(11,8))
        kk=1
        for k in sel_df_regression.iterrows():
            ax1 = plt.subplot(3,2,kk)
            xSite = str(k[1]['x'])
            x = df_final[xSite].to_numpy()
            plt.scatter(x,y)
            ax1.grid(True)
            ax1.set_xlabel(xSite)
            ax1.set_ylabel(j)
            kk+=1

        fig.tight_layout()
        plt.show()
        break


            
                       
        
        
    #     plt.sc
    

    
