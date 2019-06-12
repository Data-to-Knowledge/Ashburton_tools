# -*- coding: utf-8 -*-

import os, math
import numpy as np
import pandas as pd
from pdsql import mssql
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


def getCorrelations(self):
    '''
    Calculate correlations (fits) for flow sites that do not have a recorder.
    Returns:
        - dataframe with the 6 best correlations for each flow site
        - dataframe with flow time-series for all sites (recorder and manual gaugings)
        - geopandas dataframe with locations of recorder sites within a buffer distance from the manual sites
    Writes:
        - csv-file with the 6 best correlations for each flow site
        - csv-file with flow time-series for all sites (recorder and manual gaugings)
        - shapefile with locations of recorder sites within a buffer distance from the manual sites
    '''
    
    dataset_type_table = 'DatasetType'
    ts_summ_table = 'TSDataNumericDailySumm'
    ts_table = 'TSDataNumericDaily'
    site_table = 'ExternalSite'
    
    self.min_flow_obs = self.config.getint('FLOW_CORRELATIONS', 'min_flow_obs')
    self.buf_dist = self.config.getint('FLOW_CORRELATIONS', 'buf_dist')
    self.filter_winter_flow = self.config.getint('FLOW_CORRELATIONS', 'filter_winter_flow')
    
    #-lists of stations and corresponding dates for which flow records should be removed before correlation is calculated
    remove_stat_dates = self.config.get('FLOW_CORRELATIONS', 'remove_stat_dates').split(',')
    if len(remove_stat_dates)>1:
        remove_stats = remove_stat_dates[0:-1:2]
        remove_dates = remove_stat_dates[1::2]
    else:
        remove_stats = False
        remove_dates = False
    remove_stat_dates = None

    #-Get lowflow sites
    flow_sites = self.flow_sites_gdf.flow_site.tolist()
 
    ## Read in datasettypes
    datasets = mssql.rd_sql(self.server, self.database, dataset_type_table, ['DatasetTypeID', 'CTypeID'], where_in={'FeatureID': [1], 'MTypeID': [2], 'CTypeID': [1, 2], 'DataCodeID': [1]})
     
    #-Get datasetTypes for recorded data and manual data
    rec_datasetTypes = datasets[datasets.CTypeID == 1].DatasetTypeID.tolist()
    man_datasetTypes = datasets[datasets.CTypeID == 2].DatasetTypeID.tolist()
    all_datasetTypes = rec_datasetTypes.copy()
    all_datasetTypes.extend(man_datasetTypes)
     
    #-Get summary table for the lowflow sites
    site_summ = mssql.rd_sql(self.server, self.database, ts_summ_table, where_in={'DatasetTypeID': all_datasetTypes, 'ExtSiteID': flow_sites})
    site_summ.FromDate = pd.to_datetime(site_summ.FromDate)
    site_summ.ToDate = pd.to_datetime(site_summ.ToDate)
    site_summ.drop('ModDate', axis=1, inplace=True)
     
    #-Throw out sites from summary table and lowflow sites that have less records than 'min_flow_obs'
    too_short = site_summ.loc[site_summ.Count<self.min_flow_obs, ['ExtSiteID','Count']]
    if len(too_short)>0:
        for j in too_short.iterrows():
            print('ExtSiteID %s is not used because it only has %s records, which is less than %s.' %(j[1]['ExtSiteID'], int(j[1]['Count']), self.min_flow_obs))
            site_summ = site_summ.loc[site_summ['ExtSiteID']!=j[1]['ExtSiteID']]
            flow_sites.remove(j[1]['ExtSiteID'])
     
    #-Get site ids for recorder and manual sites of the lowflow sites and create geodataframe for manually recorded lowflow sites
    rec_sites = site_summ.loc[site_summ.DatasetTypeID.isin(rec_datasetTypes),'ExtSiteID'].tolist()
    man_sites = site_summ.loc[site_summ.DatasetTypeID.isin(man_datasetTypes),'ExtSiteID'].tolist()
    man_sites_gdf = self.flow_sites_gdf[self.flow_sites_gdf.flow_site.isin(man_sites)].copy()
     
    #-get all recorder flow sites within a buffer distance from the manually recorded lowflow sites
    man_sites_buffer_gdf = man_sites_gdf.copy()
    man_sites_buffer_gdf['geometry'] = man_sites_gdf.buffer(self.buf_dist)
    all_recorder_flowsites = mssql.rd_sql(self.server, self.database, ts_summ_table, ['ExtSiteID'], where_in={'DatasetTypeID': rec_datasetTypes})
    all_recorder_flowsites = pd.unique(all_recorder_flowsites.ExtSiteID).tolist()
    all_recorder_flowsites =  mssql.rd_sql(self.server, self.database, site_table, ['ExtSiteID', 'NZTMX', 'NZTMY'], where_in={'ExtSiteID': all_recorder_flowsites})
    all_recorder_flowsites_gdf = vector.xy_to_gpd('ExtSiteID', 'NZTMX', 'NZTMY', all_recorder_flowsites)
    self.rec_sites_buffer_gdf = vector.sel_sites_poly(all_recorder_flowsites_gdf, man_sites_buffer_gdf)
    #-write recorder sites within buffer to a shapefile
    self.rec_sites_buffer_gdf.to_file(os.path.join(self.results_path, self.config.get('FLOW_CORRELATIONS', 'rec_sites_shp')))
    all_recorder_flowsites = None; all_recorder_flowsites_gdf = None; del all_recorder_flowsites, all_recorder_flowsites_gdf  
     
     
    #-merge list of lowflow sites with list of recorder sites in buffer to create list of all sites for which to extract data
    all_flow_sites = flow_sites.copy()
    all_flow_sites.extend(pd.unique(self.rec_sites_buffer_gdf.ExtSiteID).tolist())
    all_flow_sites = pd.DataFrame(columns=all_flow_sites)
    all_flow_sites = pd.unique(all_flow_sites.columns).tolist()
     
    #-Get time-series of all_flow_sites
    ts_df = mssql.rd_sql(self.server, self.database, ts_table, ['ExtSiteID', 'DatasetTypeID', 'DateTime', 'Value', 'QualityCode'], where_in={'DatasetTypeID': all_datasetTypes, 'ExtSiteID': all_flow_sites})
    ts_df = ts_df.loc[ts_df.QualityCode>100] #-everything otherwise than missing data
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
    df_final.dropna(how='all', inplace=True)
     
    #-create a copy of all gauged and recorded flow (before filtering) to make it accessible in the main class
    self.flow_ts_df = df_final.copy()
    self.flow_ts_df.to_csv(os.path.join(self.results_path, self.config.get('FLOW_CORRELATIONS', 'flow_ts_csv')))
     
    #-remove zero records
    df_final[df_final==0] = np.nan
    
    #-remove unreliable values using lists of stations and dates
    if remove_stats and remove_dates:
        df_final.reset_index(inplace=True)
        i = 0
        for s in remove_stats:
            stat_date = remove_dates[i]
            df_final.loc[(df_final['DateTime']==stat_date), s] = np.nan
            i+=1
        df_final.set_index('DateTime', inplace=True)
     
    #-keep only winter flows for estimating correlations
    if self.filter_winter_flow:
        df_final.reset_index(inplace=True)
        df_final['Month'] = df_final['DateTime'].dt.strftime('%m').astype(np.int)
        df_final.set_index('DateTime', inplace=True)
        df_final = df_final.loc[(df_final.Month>4) & (df_final.Month<10)]
    df_final.dropna(how='all', inplace=True)
     
    #-loop over the lowflow sites and calculate regressions
    df_regression = pd.DataFrame(columns=['y', 'x', 'mean_y', 'mean_x', 'nobs', 'rmse', 'Adj. R2-squared', 'p-value', 'slope', 'intercept', 'power', 'fittype'])
    i=0
    for j in flow_sites:
        for s in self.rec_sites_buffer_gdf.ExtSiteID.tolist():
            sel_df = df_final[[j, s]]
            sel_df.replace(0., np.nan, inplace=True)
            sel_df.dropna(inplace=True)
            #-only fit for sites that have the minimum number of observations
            if len(sel_df)>=self.min_flow_obs:
                try: #-linear regression
                    #-set x and y
                    x = sel_df[[s]]
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
                    x = sel_df[[s]]
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
    #-remove negative correlations
    df_regression.loc[df_regression['Adj. R2-squared']<0.,:] = np.nan
    df_regression.dropna(how='all', inplace=True)
      
    #-loop over lowflow sites, and select best six fits by sorting on R2, rmse, nobs
    self.best_regressions_df = pd.DataFrame(columns=df_regression.columns)
    for j in flow_sites:
        sel_df = df_regression.loc[df_regression.y == j]
        sel_df.sort_values(by=['Adj. R2-squared', 'rmse', 'nobs'], ascending=[False, True, False], inplace=True)
        sel_df = sel_df.iloc[0:6,:]
        self.best_regressions_df = pd.concat([self.best_regressions_df, sel_df])
    sel_df = None; del sel_df
    self.best_regressions_df.to_csv(os.path.join(self.results_path, self.config.get('FLOW_CORRELATIONS', 'correlations_csv')), index=False)
     
    #-make 6 plots for each site to visually pick the best
    unique_y = pd.unique(self.best_regressions_df['y'])
    unique_y = list(unique_y)
    for j in flow_sites:
        if j in unique_y:
            #-get regression results for site j
            sel_df_regression = self.best_regressions_df.loc[self.best_regressions_df.y == j]
     
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
                    x = xy[xSite].to_numpy()
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
                    ax1.set_xlabel('Flow at %s [m$^3$ s$^{-1}$]' %xSite)
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
            if self.filter_winter_flow:
                plt.savefig(os.path.join(self.results_path, '%s_correlations_winteronly.png' %ySite), dpi=300)
            else:
                plt.savefig(os.path.join(self.results_path, '%s_correlations.png' %ySite), dpi=300)
                
                
                
def estimateFlow(self):
    '''
    - Fill gaps in the recorder time-series using the most recent value until a recorded value appears again.
    - Estimate flow for non-recorder sites using a csv-file with correlations.
    - Writes results to csv-files for:
        - Recorder flow time-series with gaps filled
        - Estimated flow for non-recorder sites based on correlations between recorder sites and non-recorder sites
        - Same as above, but then the extreme high estimates are filled with NaNs, and then filled with the most recent reliable estimate. This is based on the
          90-percentile of flow of the main downstream flow site; i.e. when flow exceeds the 90-percentile for that site on a particular date, then the estimates
          for all non-recorder sites are set to NaN for that same date.
    - Creates hydrographs with corresponding statistics for the main downstream recorder site, and similarly for the non-recorder sites.
    '''

    #-Get the time-series for the recorder sites to be filled with the most recent recorded value    
    flow_ts_df = self.flow_ts_df[self.rec_sites_buffer_gdf.ExtSiteID.tolist()]
    
    #-select one site to calculate statistics for
    statistics_site = self.config.get('ESTIMATE_FLOW', 'statistics_site')
    print('Calculating some statistics for the main site %s...' %statistics_site)
    sel_df = flow_ts_df[[statistics_site]]
    sel_df.reset_index(inplace=True)
    sel_df = sel_df.loc[(sel_df.DateTime>=pd.Timestamp(self.from_date)) & (sel_df.DateTime<=pd.Timestamp(self.to_date))]
    sel_df.set_index('DateTime', inplace=True)
    sel_df.to_numpy()
    p10 = np.nanpercentile(sel_df, 10)
    p50 = np.nanpercentile(sel_df, 50)
    p90 = np.nanpercentile(sel_df, 90)
    #statistics_site_p90 = p90
    avg = np.nanmean(sel_df)
    print('Flow statistics for %s are:' %statistics_site)
    print('\tP10 = %.2f cumecs' %p10)
    print('\tP50 = %.2f cumecs' %p50)
    print('\tP90 = %.2f cumecs' %p90)
    print('\tMean = %.2f cumecs' %avg)
    #-Plot streamflow for one site with statistics (percentiles and mean) for the period of interest
    print('Creating hydrograph for the main site %s...' %statistics_site)
    fig = plt.figure(figsize=(10, 8))
    xdates = pd.date_range(self.from_date, self.to_date, freq='D')
    plt.plot(xdates, sel_df, label='Recorded flow', linewidth=0.75)
    plt.hlines(p10, xdates[0], xdates[-1], label='P10', color='red')
    plt.hlines(p50, xdates[0], xdates[-1], label='P50', color='orange')
    plt.hlines(p90, xdates[0], xdates[-1], label='P90', color='black')
    plt.hlines(avg, xdates[0], xdates[-1], label='Mean', color='green', linestyles='dotted')
    ymax = np.nanmax(sel_df)
    plt.ylim([0, ymax])
    plt.xlim([xdates[0], xdates[-1]])
    plt.grid(True)
    plt.xlabel('Date')
    plt.ylabel('Flow [m$^3$ s$^{-1}$]')
    plt.title(statistics_site)
    plt.legend(loc='upper right')
    xtext_pos = xdates[math.floor(len(xdates)/20)]
    dy = ymax/15
    plt.text(xtext_pos, ymax-dy, 'P10: %.2f' %p10)
    plt.text(xtext_pos, ymax-1.5*dy, 'P50: %.2f' %p50)
    plt.text(xtext_pos, ymax-2*dy, 'P90: %.2f' %p90)
    plt.text(xtext_pos, ymax-2.5*dy, 'Mean: %.2f' %avg)
    fig.tight_layout()
    plt.savefig(os.path.join(self.results_path, '%s_hydrograph_stats.png' %statistics_site), dpi=300)
    
    #-FILL GAPS IN ALL RECORDER SITES   
    print('Filling gaps in recorded flow series...')
    flow_ts_df = flow_ts_df.fillna(method='pad')
    flow_ts_df.reset_index(inplace=True)
    flow_ts_df = flow_ts_df.loc[(flow_ts_df.DateTime>=pd.Timestamp(self.from_date)) & (flow_ts_df.DateTime<=pd.Timestamp(self.to_date))]
    flow_ts_df.set_index('DateTime', inplace=True)
    #-write filled time-series to csv file
    csvF = os.path.join(self.results_path, self.config.get('ESTIMATE_FLOW', 'flow_filled_ts_csv'))
    flow_ts_df.to_csv(csvF)
    
    #-ESTIMATE FLOW FOR THE NON-RECORDER SITES
    #-best correlations to use
    correlations_df = pd.read_csv(os.path.join(self.inputs_path, self.config.get('ESTIMATE_FLOW', 'correlations_csv')), dtype = {'x': object, 'y': object})
    #-empty dataframe to be filled with estimated flow
    flow_ts_estimated_df = pd.DataFrame(index=flow_ts_df.index)
    flow_ts_estimated_df.index.names = ['DateTime']
    for i in correlations_df.iterrows():
        y = i[1]['y']
        x = i[1]['x']
        print('Estimating flow for %s...' %y)
        fittype = i[1]['fittype']
        xdata = flow_ts_df[x].to_numpy()
        if fittype == 'power':
            # y = ax^b
            ydata = np.maximum(0, i[1]['slope'] * np.power(xdata, i[1]['power']))
        else:
            ydata = np.maximum(0, (i[1]['slope'] * xdata) + i[1]['intercept'])
        flow_ts_estimated_df[y] = ydata
    #-write to estimated flow to csv file    
    csvF = os.path.join(self.results_path, self.config.get('ESTIMATE_FLOW', 'estimated_ts_csv'))
    flow_ts_estimated_df.to_csv(csvF)
    
    #-MAKE FIGURES OF ESTIMATED FLOW
    for y in flow_ts_estimated_df.columns:
        print('Creating hydrograph for %s...' %y)
        sel_df = flow_ts_estimated_df[[y]]
        xdates = sel_df.index
        sel_df.to_numpy()
        p10 = np.nanpercentile(sel_df, 10)
        p50 = np.nanpercentile(sel_df, 50)
        p90 = np.nanpercentile(sel_df, 90)
        avg = np.nanmean(sel_df)
        print('Flow statistics for %s are:' %y)
        print('\tP10 = %.2f cumecs' %p10)
        print('\tP50 = %.2f cumecs' %p50)
        print('\tP90 = %.2f cumecs' %p90)
        print('\tMean = %.2f cumecs' %avg)
        #-Plot streamflow for one site with statistics (percentiles and mean) for the period of interest
        fig = plt.figure(figsize=(10, 8))
        plt.plot(xdates, sel_df, label='Recorded flow', linewidth=0.75)
        plt.hlines(p10, xdates[0], xdates[-1], label='P10', color='red')
        plt.hlines(p50, xdates[0], xdates[-1], label='P50', color='orange')
        plt.hlines(p90, xdates[0], xdates[-1], label='P90', color='black')
        plt.hlines(avg, xdates[0], xdates[-1], label='Mean', color='green', linestyles='dotted')
        ymax = np.nanmax(sel_df)
        plt.ylim([0, ymax])
        plt.xlim([xdates[0], xdates[-1]])
        plt.grid(True)
        plt.xlabel('Date')
        plt.ylabel('Flow [m$^3$ s$^{-1}$]')
        plt.title(y)
        plt.legend(loc='upper right')
        xtext_pos = xdates[math.floor(len(xdates)/20)]
        dy = ymax/15
        plt.text(xtext_pos, ymax-dy, 'P10: %.2f' %p10)
        plt.text(xtext_pos, ymax-1.5*dy, 'P50: %.2f' %p50)
        plt.text(xtext_pos, ymax-2*dy, 'P90: %.2f' %p90)
        plt.text(xtext_pos, ymax-2.5*dy, 'Mean: %.2f' %avg)
        fig.tight_layout()
        plt.savefig(os.path.join(self.results_path, '%s_hydrograph_estimated.png' %y), dpi=300)
    
    print('Filling gaps and estimating flow has been comepleted.')  
    ###-SECTION BELOW MIGHT NEED SOME IMPROVEMENT. CURRENTLY NOT USED BECAUSE DUE TO TRAVEL TIMES WE CANNOT CANCEL OUT VALUES IF FLOW ON THE MAIN SITE IS ABOVE THE THRESHOLD. THEN LIKELY
    ###-FOR THE SITE OF INTEREST THE FLOW ON THE DAY BEFORE SHOULD BE CANCELLED. PROBABLY BETTER TO DETERMINE THE P90 THRESHOLD FOR EACH OF THE SITES INDIVIDUALLY, OR ....
#     #################-REPEAT STEPS ABOVE, BUT THEN FILTER OUT FLOW ABOVE THE SELECT SITE P90 THRESHOLD
#     mask = flow_ts_df[statistics_site]
#     mask = mask>statistics_site_p90
#     for y in flow_ts_estimated_df.columns:
#         flow_ts_estimated_df.loc[mask, y] = np.nan
#     flow_ts_estimated_df.fillna(method='pad', inplace=True)
#     #-write to estimated flow to csv file with the high values replaced by NaN, and then filled by the most recent value    
#     csvF = os.path.join(self.results_path, self.config.get('ESTIMATE_FLOW', 'estimated_ts_P90_cutoff_csv'))
#     flow_ts_estimated_df.to_csv(csvF)
    
    
    
    
    
        
        
