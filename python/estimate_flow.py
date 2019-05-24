# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd
from pdsql import mssql
import geopandas as gpd

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

min_obs = 2


#-Get lowflow sites
flow_sites_gdf = gpd.read_file(r'C:\Active\Projects\Ashburton\naturalisation\results\lowflow_sites.shp')
flow_sites = flow_sites_gdf.flow_site.tolist()

## Read in datasettypes
datasets = mssql.rd_sql(server, database, dataset_type_table, ['DatasetTypeID', 'CTypeID'], where_in={'FeatureID': [1], 'MTypeID': [2], 'CTypeID': [1, 2], 'DataCodeID': [1]})

#-Get datasetTypes for recorded data and manual data
rec_datasetTypes = datasets[datasets.CTypeID == 1].DatasetTypeID.tolist()
man_datasetTypes = datasets[datasets.CTypeID == 2].DatasetTypeID.tolist()
all_datasetTypes = rec_datasetTypes
all_datasetTypes.extend(man_datasetTypes)
print(all_datasetTypes)

#-Get summary table
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
keep_sites = pd.unique(site_summ['ExtSiteID']).tolist()

#-Get site ids for recorder and manual sites
rec_sites = site_summ.loc[site_summ.DatasetTypeID.isin(rec_datasetTypes),'ExtSiteID'].tolist()
man_sites = site_summ.loc[site_summ.DatasetTypeID.isin(man_datasetTypes),'ExtSiteID'].tolist()

#-Get time-series of flow
ts_df = mssql.rd_sql(server, database, ts_table, ['ExtSiteID', 'DatasetTypeID', 'DateTime', 'Value', 'QualityCode'], where_in={'DatasetTypeID': all_datasetTypes, 'ExtSiteID': flow_sites})
ts_df = ts_df.loc[ts_df.QualityCode>=100] #-everything otherwise than missing data
ts_df = ts_df.loc[ts_df.ExtSiteID.isin(keep_sites)] #-throw away records with too little observations
ts_df.DateTime = pd.to_datetime(ts_df.DateTime)
ts_df_sites = pd.unique(ts_df.ExtSiteID).tolist()

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

print(df_final.head())


for j in ts_df_sites:
    fig, ax = plt.subplots()
    plt.scatter(df_final[j], df_final['168834'])
    ax.set_xlabel(j)
    ax.set_ylabel('168834')
    plt.show(block=True)