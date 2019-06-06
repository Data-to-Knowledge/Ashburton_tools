# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd
from pdsql import mssql
from gistools import vector
import statsmodels.api as sm
import geopandas as gpd

import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams.update({'font.size': 8})


pd.options.display.max_columns = 100

#-Authorship information-########################################################################################################################
__author__ = 'Wilco Terink'
__copyright__ = 'Wilco Terink'
__version__ = '1.0'
__email__ = 'wilco.terink@ecan.govt.nz'
__date__ ='June 2019'
#################################################################################################################################################


server = 'edwprod01' 
database = 'hydro'

from_date = '2009-01-01'
from_date = pd.Timestamp(from_date)#.date()
to_date = '2018-12-31'
to_date = pd.Timestamp(to_date)#.date()

dataset_type_table = 'DatasetType'
ts_summ_table = 'TSDataNumericDailySumm'
ts_table = 'TSDataNumericDaily'
site_table = 'ExternalSite'

min_flow_obs = 4

#-Get lowflow sites
flow_sites_gdf = gpd.read_file(r'C:\Active\Projects\Ashburton\naturalisation\results4\lowflow_sites.shp')
flow_sites = flow_sites_gdf.flow_site.tolist()
print(flow_sites)

sel_sites = ['168841', '1688218']

#-Read time-series of flow
flow_ts_df = pd.read_csv(r'C:\Active\Projects\Ashburton\naturalisation\results4\recorder_gauged_flow.csv', parse_dates=True, index_col=0, dayfirst=True)
flow_ts_df = flow_ts_df[sel_sites] 

## Read in datasettypes for recorded groundwater (gwl) levels
datasets = mssql.rd_sql(server, database, dataset_type_table, ['DatasetTypeID'], where_in={'FeatureID': [2], 'MTypeID': [1], 'CTypeID': [1], 'DataCodeID': [1]})

sites_buffer_gdf = flow_sites_gdf.copy()
sites_buffer_gdf['geometry'] = sites_buffer_gdf.buffer(60000)

all_gwl_sites = mssql.rd_sql(server, database, ts_summ_table, ['ExtSiteID'], where_in={'DatasetTypeID': datasets.DatasetTypeID.tolist()})
all_gwl_sites = pd.unique(all_gwl_sites.ExtSiteID).tolist()
all_gwl_sites =  mssql.rd_sql(server, database, site_table, ['ExtSiteID', 'NZTMX', 'NZTMY'], where_in={'ExtSiteID': all_gwl_sites})
all_gwl_sites_gdf = vector.xy_to_gpd('ExtSiteID', 'NZTMX', 'NZTMY', all_gwl_sites)
sel_gwl_sites_gdf = vector.sel_sites_poly(all_gwl_sites_gdf, sites_buffer_gdf)
all_gwl_sites = None; all_gwl_sites_gdf = None

#-Get summary table for the gwl sites and filter period of interest
site_summ = mssql.rd_sql(server, database, ts_summ_table, where_in={'DatasetTypeID': datasets.DatasetTypeID.tolist(), 'ExtSiteID': sel_gwl_sites_gdf.ExtSiteID.tolist()})
site_summ.FromDate = pd.to_datetime(site_summ.FromDate)
site_summ.ToDate = pd.to_datetime(site_summ.ToDate)
site_summ.drop('ModDate', axis=1, inplace=True)
site_summ = site_summ.loc[(site_summ.FromDate<to_date) & (site_summ.ToDate>from_date)]

#-Throw out gwl sites from summary table that have less records than 'min_flow_obs'
too_short = site_summ.loc[site_summ.Count<min_flow_obs, ['ExtSiteID','Count']]
if len(too_short)>0:
    for j in too_short.iterrows():
        print('ExtSiteID %s is not used because it only has %s records, which is less than %s.' %(j[1]['ExtSiteID'], int(j[1]['Count']), min_flow_obs))
        site_summ = site_summ.loc[site_summ['ExtSiteID']!=j[1]['ExtSiteID']]

#-Get time-series of selected gwl sites
ts_df = mssql.rd_sql(server, database, ts_table, ['ExtSiteID', 'DatasetTypeID', 'DateTime', 'Value', 'QualityCode'], where_in={'DatasetTypeID': datasets.DatasetTypeID.tolist(), 'ExtSiteID': site_summ.ExtSiteID.tolist()})
ts_df = ts_df.loc[ts_df.QualityCode>100] #-everything otherwise than missing data
ts_df.DateTime = pd.to_datetime(ts_df.DateTime)
ts_df_sites = pd.unique(ts_df.ExtSiteID).tolist()
all_flow_sites = None; del all_flow_sites
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

#-rolling mean
df_final = df_final.rolling(window=3, center=True).mean()

#-keep only winter flows for estimating correlations
df_final.reset_index(inplace=True)
df_final['Month'] = df_final['DateTime'].dt.strftime('%m').astype(np.int)
df_final.set_index('DateTime', inplace=True)
df_final = df_final.loc[(df_final.Month>4) & (df_final.Month<10)]
df_final.dropna(how='all', inplace=True)

#-concat the flow data to it (flow sites are identified under 'flow_sites' and gwl sites under ts_df_sites)
df_final = pd.concat([df_final, flow_ts_df], axis=1)
df_final.dropna(how='all', inplace=True)

#-loop over the lowflow sites and calculate regressions x=gwl, y=flow data
df_regression = pd.DataFrame(columns=['y', 'x', 'mean_y', 'mean_x', 'nobs', 'rmse', 'Adj. R2-squared', 'p-value', 'slope', 'intercept', 'power', 'fittype'])
i=0
for j in sel_sites:
    for s in ts_df_sites:
        sel_df = df_final[[j, s]]
        sel_df.replace(0., np.nan, inplace=True)
        sel_df.dropna(inplace=True)
        #-only fit for sites that have the minimum number of observations
        if len(sel_df)>=min_flow_obs:
            try: #-linear regression
                #-set x and y
                x = sel_df[[s]] *-1
                x = sm.add_constant(x)  #-needed for intercept
                y = sel_df[[j]]
                   
                #-linear fit
                model = sm.OLS(y, x).fit()
                   
                #-predict using the model
                predictions = model.predict(x) # make the predictions by the model
                rmse_val = np.sqrt(  np.mean(   (predictions.to_numpy() - y.to_numpy())**2   )  )
                   
                #-fill dataframe with stats for linear fit
                df_regression.loc[i, 'y'] = j
                df_regression.loc[i, 'x'] = s
                df_regression.loc[i, 'mean_y'] = np.nanmean(y)
                df_regression.loc[i, 'mean_x'] = np.nanmean(x[s].to_numpy())
                df_regression.loc[i, 'nobs'] = model.nobs
                df_regression.loc[i, 'rmse'] = rmse_val
                df_regression.loc[i, 'Adj. R2-squared'] = model.rsquared_adj
                df_regression.loc[i, 'p-value'] = model.f_pvalue
                df_regression.loc[i, 'slope'] = model.params[1]
                df_regression.loc[i, 'intercept'] = model.params[0]
                df_regression.loc[i, 'fittype'] = 'linear'
                   
                i+=1
            except:
                print('Could not establish linear fit for %s and %s...' %(j,s))
  
            try: #-power fit regression
                #-set x and y
                x = sel_df[[s]] *-1
                x = sm.add_constant(x)  #-needed for intercept
                y = sel_df[[j]]                 
  
                #-***********************************************************
                #-                           y = ax^b
                #-                           log(y) = log(a) + b*log(x)
                #-***********************************************************
                logY = np.log(y)
                logX = np.log(x[[s]])
                logX = sm.add_constant(logX)
                model = sm.OLS(logY, logX).fit()
   
                x = x[s].to_numpy()
                #-predictions of the exact x values for calculating rmse            
                predictions = np.exp(model.params[0]) * x**model.params[1]
                rmse_val = np.sqrt(  np.mean(   (predictions - y.to_numpy())**2   )  )
                   
                #-fill dataframe with stats for power fit
                df_regression.loc[i, 'y'] = j
                df_regression.loc[i, 'x'] = s
                df_regression.loc[i, 'mean_y'] = np.nanmean(y)
                df_regression.loc[i, 'mean_x'] = np.nanmean(x)
                df_regression.loc[i, 'nobs'] = model.nobs
                df_regression.loc[i, 'rmse'] = rmse_val
                df_regression.loc[i, 'Adj. R2-squared'] = model.rsquared_adj
                df_regression.loc[i, 'p-value'] = model.f_pvalue
                df_regression.loc[i, 'slope'] = np.exp(model.params[0])
                df_regression.loc[i, 'power'] = model.params[1]
                df_regression.loc[i, 'fittype'] = 'power'
                i+=1
            except:
                print('Could not establish power fit for %s and %s...' %(j,s))
sel_df = None; del sel_df

df_regression.to_csv(r'C:\Active\Projects\Ashburton\naturalisation\results4\test.csv')

#-loop over lowflow sites, and select best six fits by sorting on R2, rmse, nobs
best_regressions_df = pd.DataFrame(columns=df_regression.columns)
for j in sel_sites:
    sel_df = df_regression.loc[df_regression.y == j]
    sel_df.sort_values(by=['Adj. R2-squared', 'rmse', 'nobs'], ascending=[False, True, False], inplace=True)
    sel_df = sel_df.iloc[0:6,:]
    best_regressions_df = pd.concat([best_regressions_df, sel_df])
sel_df = None; del sel_df
best_regressions_df.to_csv(r'C:\Active\Projects\Ashburton\naturalisation\results4\test_best.csv')

#-make 6 plots for each site to visually pick the best
unique_y = pd.unique(best_regressions_df['y'])
unique_y = list(unique_y)
for j in sel_sites:
    if j in unique_y:
        #-get regression results for site j
        sel_df_regression = best_regressions_df.loc[best_regressions_df.y == j]
 
        #-plot the best 6 fits
        fig = plt.figure(figsize=(11,8))
        kk=1
        for k in sel_df_regression.iterrows():
            try:
                ax1 = plt.subplot(3,2,kk)
                xSite = k[1]['x']
                ySite = k[1]['y']
                 
                xy = df_final[[xSite, ySite]]
                xy.dropna(inplace=True)
                x = xy[xSite].to_numpy()*-1
                y = xy[ySite].to_numpy()
     
                xmin = np.nanmin(x)
                xmax = np.nanmax(x)
                dx = (xmax-xmin)/100
                xvals = np.arange(xmin, xmax, dx)
                ymin = np.nanmin(y)
                ymax = np.nanmax(y)
                 
                #-Get the stats from the table
                fit_type = k[1]['fittype']
                if fit_type == 'linear':
                    eq_str = 'y = %.3fx + %.3f' %(k[1]['slope'], k[1]['intercept'])
                    predictions = k[1]['slope'] * xvals + k[1]['intercept']
                else:
                    eq_str = 'y = %.3fx$^{%.3f}$' %(k[1]['slope'], k[1]['power'])
                    predictions = k[1]['slope'] * (xvals ** k[1]['power'])
                r2 = k[1]['Adj. R2-squared']
                rmse_val = k[1]['rmse']
                p_val = k[1]['p-value']
                nobs = k[1]['nobs']
                plt.plot(xvals, predictions, label='fit')
                plt.scatter(x,y, color='red', label='data')
                ax1.grid(True)
                ax1.set_xlabel('GWL at %s [m$^3$ s$^{-1}$]' %xSite)
                ax1.set_ylabel('Flow at %s [m$^3$ s$^{-1}$]' %ySite)
                ax1.legend(loc='lower right')
                 
                dy = (ymax-ymin)/10
                dx = (xmax-xmin)/10
                 
                #-plot the stats in the plot
                plt.text(xmin, ymax, eq_str)
                plt.text(xmin, ymax-dy, 'R$^2$ %.2f' %r2)
                plt.text(xmin, ymax-2*dy, 'RMSE %.2f' %rmse_val)
                plt.text(xmin, ymax-3*dy, 'p-value %.2f' %p_val)
                plt.text(xmin, ymax-4*dy, 'nobs %.0f' %nobs)
                ax1.set_xlim([xmin-dx, xmax+dx])
                ax1.set_ylim([ymin-dy, ymax+dy])
                kk+=1
            except:
                pass
        fig.tight_layout()

        plt.savefig(r'C:\Active\Projects\Ashburton\naturalisation\results4\%s_gwl_test.png' %ySite, dpi=300)
