import pandas as pd
import numpy as np
from pdsql import mssql

import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams.update({'font.size': 8})

from python.stream_depletion import SD

server = 'edwprod01' 
database = 'hydro'
dataset_type_table = 'DatasetType'
ts_summ_table = 'TSDataNumericDailySumm'
ts_table = 'TSDataNumericDaily'

catch_area = 13319000  # m2


#######-Flow data for 1688218
#-DatasetTypeIDs for recorded flow
datasets = mssql.rd_sql(server, database, dataset_type_table, ['DatasetTypeID', 'CTypeID'], where_in={'FeatureID': [1], 'MTypeID': [2], 'CTypeID': [1, 2], 'DataCodeID': [1]})
#-Get summary for site 1688218
site_sum = mssql.rd_sql(server, database, ts_summ_table, where_in={'ExtSiteID': ['1688218'], 'DatasetTypeID': datasets.DatasetTypeID.tolist()})

#-Get recorded flow time-series for that site for winter only
ts = mssql.rd_sql(server, database, ts_table, ['DatasetTypeID', 'DateTime', 'Value', 'QualityCode'], where_in={'ExtSiteID': ['1688218'], 'DatasetTypeID': site_sum.DatasetTypeID.tolist()})
ts.DateTime = pd.to_datetime(ts.DateTime)
#-only recorder flow
ts = ts.loc[ts.DatasetTypeID==5]

# ts['Month'] = ts['DateTime'].dt.strftime('%m').astype(np.int)
ts.set_index('DateTime', inplace=True)
# ts.loc[(ts.Month>4) & (ts.Month<10)] = np.nan
ts.drop(['DatasetTypeID', 'QualityCode'], axis=1, inplace=True)
ts.rename(columns={'Value': 'Flow [m3/s]'}, inplace=True)
minDate = ts.index.min()
maxDate = ts.index.max()


# ############-Precipitation
# p_sites = mssql.rd_sql(server, database, ts_summ_table, col_names=['ExtSiteID'], where_in={'DatasetTypeID': [15, 38, 3068, 4554]})
# p_sites = pd.unique(p_sites.ExtSiteID).tolist()
# p_sites = mssql.rd_sql(server, database, 'ExternalSite', col_names=['ExtSiteID', 'NZTMX', 'NZTMY'], where_in={'ExtSiteID': p_sites})
# p_sites.to_csv(r'C:\Active\Projects\Ashburton\naturalisation\results4\all_precipitation_sites.csv', index=False)


sel_P_sites = [39005, 39845, 4763, 4762, 4761, 4760, 4759, 4758, 4757, 4756, 38974, 319602, 26170, 4787, 4785, 4784, 4783, 4780, 4779, 4778, 4777, 4776, 4774, 5047, 5049, 5051]
P_ts = mssql.rd_sql(server, database, ts_table, col_names=['ExtSiteID', 'DateTime', 'Value'], where_in={'DatasetTypeID': [15, 38, 3068, 4554], 'ExtSiteID': sel_P_sites})
P_ts.DateTime = pd.to_datetime(P_ts.DateTime)

###############-ETr
# et_sites = mssql.rd_sql(server, database, ts_summ_table, col_names=['ExtSiteID'], where_in={'DatasetTypeID': [103, 24, 3082]})
# et_sites = pd.unique(et_sites.ExtSiteID).tolist()
# et_sites = mssql.rd_sql(server, database, 'ExternalSite', col_names=['ExtSiteID', 'NZTMX', 'NZTMY'], where_in={'ExtSiteID': et_sites})
# et_sites.to_csv(r'C:\Active\Projects\Ashburton\naturalisation\results4\all_ETr_sites.csv', index=False)


sel_et_sites = [26170, 38974, 39661, 39845, 41200, 4764]
et_ts = mssql.rd_sql(server, database, ts_table, col_names=['ExtSiteID', 'DateTime', 'Value'], where_in={'DatasetTypeID': [103, 24, 3082], 'ExtSiteID': sel_et_sites})
et_ts.DateTime = pd.to_datetime(et_ts.DateTime)

 

##################-add precipitation and et to dataframe

df_final = ts.copy()
for p in sel_P_sites:
    sel_P = P_ts.loc[P_ts.ExtSiteID==str(p), ['DateTime', 'Value']]
    sel_P.set_index('DateTime', inplace=True)
    sel_P.rename(columns={'Value': p}, inplace=True)
    df_final[[p]] = sel_P
df_final['Pavg [mm/d]'] = df_final.iloc[:,1::].mean(axis=1)
#-keep only average Precip
df_final = df_final[['Flow [m3/s]', 'Pavg [mm/d]']]
for e in sel_et_sites:
    sel_et = et_ts.loc[et_ts.ExtSiteID==str(e), ['DateTime', 'Value']]
    sel_et.set_index('DateTime', inplace=True)
    sel_et.rename(columns={'Value': e}, inplace=True)
    df_final[[e]] = sel_et
df_final['ETavg [mm/d]'] = df_final.iloc[:,2::].mean(axis=1)
#-keep only average ET
df_final = df_final[['Flow [m3/s]', 'Pavg [mm/d]','ETavg [mm/d]']]
df_final.reset_index(inplace=True)
df_final.drop_duplicates(inplace=True)
df_final.set_index('DateTime', inplace=True)

##################-add upstream takes to flow data

#-waps upsteam of 1688218
upstream_waps = ['K37/0455', 'K37/1427', 'K37/1442', 'K37/0556', 'K37/0819', 'K37/2633']

crc_wap_details = pd.read_csv(r'C:\Active\Projects\Ashburton\naturalisation\results4\Ashburton_crc_wap.csv')
crc_wap_details = crc_wap_details.loc[crc_wap_details.wap.isin(upstream_waps),['wap', 'Activity', 'Distance', 'T_Estimate', 'S']]
crc_wap_details.drop_duplicates(inplace=True)

date_allo_usage = pd.read_csv(r'C:\Active\Projects\Ashburton\naturalisation\results4\Ashburton_date_allo_usage.csv', parse_dates=True, index_col=0, dayfirst=True)
date_allo_usage = date_allo_usage[['wap', 'crc_wap_metered_abstraction_filled [m3]']]
date_allo_usage = date_allo_usage.loc[date_allo_usage.wap.isin(upstream_waps)]

#-add the stream depletion based on usage
gw_waps = pd.unique(crc_wap_details.loc[crc_wap_details.Activity == 'Take Groundwater', 'wap']).tolist()
sd_df = pd.DataFrame(index= df_final.index)
sd_df.reset_index(inplace=True)
for wap in gw_waps:
    print(wap)
    df_sel = date_allo_usage.loc[date_allo_usage.wap == wap, ['crc_wap_metered_abstraction_filled [m3]']]
    df_sel.rename(columns={'crc_wap_metered_abstraction_filled [m3]': wap}, inplace=True)
    df_sel.index.names = ['DateTime']
    df_sel.reset_index(inplace=True)

    sd_df = pd.merge(sd_df, df_sel, how='left', on='DateTime')
 
    qpump = sd_df[wap].to_numpy()
    T = crc_wap_details.loc[crc_wap_details.wap == wap, 'T_Estimate'].to_numpy()[0]
    S = crc_wap_details.loc[crc_wap_details.wap == wap, 'S'].to_numpy()[0]
    D = crc_wap_details.loc[crc_wap_details.wap == wap, 'Distance'].to_numpy()[0]
    sd = SD(D, S, T, qpump)
    sd_df[wap] = sd
sd_df.set_index('DateTime', inplace=True)
#-rename the columns
for wap in gw_waps:
    sd_df.rename(columns={wap: wap + ' SD [m3]'}, inplace=True)
df_final = pd.concat([df_final, sd_df], axis=1)


#-add sw use
sw_waps = pd.unique(crc_wap_details.loc[crc_wap_details.Activity == 'Take Surface Water', 'wap']).tolist()
sw_df = pd.DataFrame(index= df_final.index)
sw_df.reset_index(inplace=True)
for wap in sw_waps:
    print(wap)
    df_sel = date_allo_usage.loc[date_allo_usage.wap == wap, ['crc_wap_metered_abstraction_filled [m3]']]
    df_sel.rename(columns={'crc_wap_metered_abstraction_filled [m3]': wap + '_SW [m3]'}, inplace=True)
    df_sel.index.names = ['DateTime']
    df_sel.reset_index(inplace=True)
    sw_df = pd.merge(sw_df, df_sel, how='left', on='DateTime')
sw_df.set_index('DateTime', inplace=True)
df_final = pd.concat([df_final, sw_df], axis=1)

#-calculate sum of all takes (sw + sd)
take_cols = list(sd_df.columns)
take_cols.extend(list(sw_df.columns))
df_final['Total take [m3]'] = df_final[take_cols].sum(axis=1)

#-convert everything to mm/d
df_final['Total take [mm/d]'] = (df_final['Total take [m3]'] / catch_area) * 1000
df_final['Flow [mm/d]'] = ((df_final['Flow [m3/s]'] * 3600 * 24) / catch_area) * 1000
df_final['Natural flow [mm/d]'] = df_final['Total take [mm/d]'] + df_final['Flow [mm/d]']



df_final.to_csv(r'C:\Active\Projects\Ashburton\naturalisation\results4\test.csv')






# #-make a figure of the scenario with average pumping for the entire period, and on/off pumping at different rates
# plt.figure(facecolor='#FFFFFF')
# for wap in gw_waps:
#     lines = plt.plot(sd_df.index,sd_df.iloc[:,gw_waps.index(wap)].to_numpy(), label=wap)
# plt.xlabel('Time')
# plt.ylabel('Stream depletion [m3/day]')
# plt.grid(True)
# plt.legend()
# plt.show(block=True)




