# -*- coding: utf-8 -*-

import os
from configparser import ConfigParser
import pandas as pd
from pdsql import mssql
from gistools import vector
import geopandas as gpd
from osgeo import gdal
import pcraster as pcr
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.ticker import FormatStrFormatter

#-Authorship information-########################################################################################################################
__author__ = 'Wilco Terink'
__copyright__ = 'Wilco Terink'
__version__ = '1.0'
__email__ = 'wilco.terink@ecan.govt.nz'
__date__ ='May 2019'
#################################################################################################################################################


rcParams.update({'font.size': 8})

pd.options.display.max_columns = 100

def tif_to_pcr(tifF, pcrF):
    '''
    Convert GTiff to pcraster map:
    ------------------------------
    
    tifF: Full path name to GeoTIFF raster
    pcrF: Full path name to PCRaster map
    '''
    
    format_out = 'PCRaster'
    
    #-open source tif
    src_ds = gdal.Open(tifF)
    
    driver = gdal.GetDriverByName(format_out)
    
    #-Output to pcraster
    dst_ds = driver.CreateCopy(pcrF, src_ds, 0 )
     
    #-Properly close the datasets to flush to disk
    dst_ds = None
    src_ds = None
    
    


class myHydroTool():
    def __init__(self):
        #-Set reading of configuration file
        self.config = ConfigParser()
        self.py_path = os.path.realpath(os.path.dirname(__file__))
        self.config.read(os.path.join(self.py_path, 'parameters.cfg'))
        #-Set the server
        self.server = self.config.get('SERVER', 'server')
        self.database = 'hydro'
        
        #-Read the paths
        self.project_path = self.config.get('PATH', 'project_path')
        self.results_path = os.path.join(self.project_path, self.config.get('PATH', 'results_path'))
        self.inputs_path = os.path.join(self.project_path, self.config.get('PATH', 'inputs_path'))
        if not os.path.exists(self.results_path):
            os.makedirs(self.results_path)
        if not os.path.exists(self.inputs_path):
            os.makedirs(self.inputs_path)
            
        #-Date settings
        self.from_date = self.config.get('DATE', 'from_date')
        self.from_date = pd.Timestamp(self.from_date).date()
        self.to_date = self.config.get('DATE', 'to_date')
        self.to_date = pd.Timestamp(self.to_date).date()
            
            
        #-Create shapefile of lowflow sites? Otherwise read the shapefile from results path
        create_lf_shp = self.config.getint('SITES', 'create_lf_shp')
        if create_lf_shp:
            print('Creating shapefile for lowflow sites')
            self.create_lowflow_sites_shp()
        else:
            print('Reading shapefile for lowflow sites')
            self.read_lowflow_sites_shp()
        
        #####-PCRASTER LDD AND CATCHMENT CREATION-###################
        #-Check if flow direction map should be created
        make_ldd = self.config.getint('PCRASTER_LDD', 'make_ldd')
        if make_ldd:
            print('Creating flow direction netword (ldd), accuflux, and river pcraster maps')
            self.pcr_flowdir()
        #-Check if catchment map should be created
        make_catchment = self.config.getint('PCRASTER_LDD', 'make_catchment')
        if make_catchment:
            print('Delineating catchment')
            self.delineate_catchment()
        #-Check if catchment maps need to be cleaned (clipped)    
        clean_up_maps = self.config.getint('PCRASTER_LDD', 'clean_up_maps')
        if clean_up_maps:
            print('Clipping pcraster maps')
            self.cleanup_catchment_maps()
        #############################################################
        
        ################-GET CONSENT AND WAP DETAILS FROM DATABASE AND SPATIALLY JOIN WITH LOWFLOW SITE CATCHMENTS-##########################
        get_crc_wap_details = self.config.getint('CRC_WAP', 'get_crc_wap_details')
        if get_crc_wap_details:
            self.get_CrcWap_details()
        else:
            self.crc_wap_df = pd.read_csv(os.path.join(self.results_path, self.config.get('CRC_WAP', 'crc_wap_csv')), parse_dates=[1, 2, 3],dayfirst=True)
            self.waps_gdf = gpd.read_file(os.path.join(self.results_path, self.config.get('CRC_WAP', 'wap_shp')))
        #-Get time-series for the waps
        create_wap_ts = self.config.getint('CRC_WAP', 'create_wap_ts')
        if create_wap_ts:
            self.wapTS()
        else:
            self.wap_ts_metered_df = pd.read_csv(os.path.join(self.results_path, self.config.get('CRC_WAP', 'wap_ts_csv')), index_col=0, parse_dates=True, dayfirst=True)
        ######################################################################################################################################
        
        #################-CALCULATE ABSTRACTION RATIOS USING THE RATIO OF METERED ABSTRACTION OVER MAXIMUM ALLOCATED RATE PER CRC/WAP-########
        crc_wap_ratio = self.config.getint('CALC_RATIOS','crc_wap_ratio')
        if crc_wap_ratio:
            self.abstraction_ratios()
        else:
            self.date_crc_wap_ratio = pd.read_csv(os.path.join(self.results_path, self.config.get('CALC_RATIOS', 'crc_wap_ratio_csv')), parse_dates=True, dayfirst=True)
        plot_ratios = self.config.getint('CALC_RATIOS','plot_ratios')
        #-plot all daily ratios to check for extreme/unrealistic ratios
        if plot_ratios:
            self.plot_ratios()
        #-filter and groupby month, swaz, use_type
        group_ratios = self.config.getint('CALC_RATIOS','group_ratios')
        if group_ratios:
            self.group_ratios()
        else:
            self.crc_wap_ratio_grouped = pd.read_csv(os.path.join(self.results_path, self.config.get('CALC_RATIOS', 'crc_wap_ratio_grouped_csv')))
        ######################################################################################################################################            
        
        ##################-ESTIMATE DEMAND FOR DATES WHERE NO METERED VALUE/STRANGE VALUE IS AVAILABLE USING RATIOS AND MAXIMUM ALLOCATED VOLUME-####
        estimate_usage = self.config.getint('ESTIMATE_USAGE','estimate_usage')
        if estimate_usage:
            self.estimate_usage()
        else:
            self.date_allo_usage = pd.read_csv(os.path.join(self.results_path, self.config.get('ESTIMATE_USAGE', 'estimated_usage_csv')), parse_dates=True, dayfirst=True)
#        print(self.date_allo_usage)

        print(self.flow_sites_gdf)
        
        
            
                



    def create_lowflow_sites_shp(self):
        '''
        Create a shapefile of the lowflow sites specified in the csv-file under "rec_rivers_250m.map"
        '''
 
        sites_table = 'ExternalSite'
        
        #-Read the Ashburton site IDs from csv
        sites = list(pd.read_csv(os.path.join(self.inputs_path, self.config.get('SITES', 'lowflow_sites'))).site.astype(str))
        
        #-Get selected sites from ExternalSite
        self.LF_sites = mssql.rd_sql(self.server, self.database, sites_table, col_names=['ExtSiteID', 'NZTMX', 'NZTMY', 'SwazGroupName', 'SwazName'], where_in={'ExtSiteID': sites})
        self.LF_sites.rename(columns={'ExtSiteID': 'flow_site'}, inplace=True)
        
        #-Shapefile name
        flow_sites_shp = os.path.join(self.results_path, self.config.get('SITES','lowflow_shp'))
        
        #-Convert the selected sites to a GeoDataFrame and save it to a shape file
        self.flow_sites_gdf = vector.xy_to_gpd('flow_site', 'NZTMX', 'NZTMY', self.LF_sites)
        self.flow_sites_gdf.to_file(flow_sites_shp)
        
        
    def read_lowflow_sites_shp(self):
        '''
        Read the shapefile of the lowflow sites specified under "lowflow_shp"
        '''

        flow_sites_shp = os.path.join(self.results_path, self.config.get('SITES','lowflow_shp'))
        self.flow_sites_gdf = gpd.read_file(flow_sites_shp)
        
    def pcr_flowdir(self):
        '''
        Make a PCRaster ldd (local drain direction) map based on DEM. If "burn_rivers==1", then river raster will be used to burn in for better delineation.
        '''
        
        demF = os.path.join(self.inputs_path, self.config.get('PCRASTER_LDD', 'dem'))
        dem = pcr.readmap(demF)
        
        #-burn rivers?
        burn_rivers = self.config.getint('PCRASTER_LDD', 'burn_rivers')
        if burn_rivers:
            print('Burning rivers')
            #-Read the rivers to burn map
            riversF = os.path.join(self.inputs_path, self.config.get('PCRASTER_LDD', 'raster_rivers'))
            riverBurn = pcr.readmap(riversF)
            riverBurn = pcr.cover(riverBurn, 0.)
            
            #-Max and min of Dem
            demMax = pcr.mapmaximum(dem)
            demMin = pcr.mapminimum(dem)
            #-Relative dem
            relDem = (dem - demMin) / (demMax - demMin)
            #-burn rivers in the relative dem
            burnDem = relDem - riverBurn
            #-assign burnDem to dem
            dem = burnDem
        
        #-Calculate the flow direction
        lddF = os.path.join(self.results_path, self.config.get('PCRASTER_LDD', 'ldd'))
        ldd = pcr.lddcreate(dem, 1e31, 1e31, 1e31, 1e31)
        #-Write ldd to results folder
        pcr.report(ldd, lddF)
        
        #-Make accuflux map
        accufluxF = os.path.join(self.results_path, self.config.get('PCRASTER_LDD', 'accuflux'))
        accuflux = pcr.accuflux(ldd, 1)
        pcr.report(accuflux, accufluxF)
        
        #-Make a river map (can be used to check with REC)
        riversF =  os.path.join(self.results_path, self.config.get('PCRASTER_LDD', 'rivers'))
        river_cells =  self.config.getint('PCRASTER_LDD', 'river_cells')
        rivers = accuflux>river_cells
        pcr.report(rivers, riversF)
    
    def delineate_catchment(self):
        '''
        Delineate catchment using ldd and sites map
        '''
        
        lddF = os.path.join(self.results_path, self.config.get('PCRASTER_LDD', 'ldd'))
        ldd = pcr.readmap(lddF)
        
        sitesF = os.path.join(self.inputs_path, self.config.get('PCRASTER_LDD', 'stations'))
        stations = pcr.readmap(sitesF)
        
        catchmentF = os.path.join(self.results_path, self.config.get('PCRASTER_LDD', 'catchment'))
        catchment = pcr.catchment(ldd, stations)
        catchment2 = pcr.boolean(catchment)
        catchment3 = pcr.ifthen(catchment2, catchment) 
        pcr.report(catchment3, catchmentF)
        
        subcatchmentF = os.path.join(self.results_path, self.config.get('PCRASTER_LDD', 'subcatchment'))
        subcatchment = pcr.subcatchment(ldd, stations)
        subcatchment2 = pcr.boolean(subcatchment)
        subcatchment3 = pcr.ifthen(subcatchment2, subcatchment) 
        pcr.report(subcatchment3, subcatchmentF)
        
    def cleanup_catchment_maps(self):
        '''
        Clip maps to catchment border
        '''
        
        #-Read the original ldd
        lddF = os.path.join(self.results_path, self.config.get('PCRASTER_LDD', 'ldd'))
        ldd = pcr.readmap(lddF)
        
        #-Read the catchment map (clip mask)
        catchmentF = os.path.join(self.results_path, self.config.get('PCRASTER_LDD', 'catchment'))
        catchment = pcr.readmap(catchmentF)
        
        #-Clip the ldd to the catchment mask
        ldd_clean = pcr.lddrepair(pcr.ifthen(pcr.boolean(catchment), ldd))
        lddF = os.path.join(self.results_path, self.config.get('PCRASTER_LDD', 'ldd_clean'))
        pcr.report(ldd_clean, lddF)
        
        #-New accuflux
        accuflux = pcr.accuflux(ldd_clean, 1)
        accufluxF = os.path.join(self.results_path, self.config.get('PCRASTER_LDD', 'accuflux_clean'))
        pcr.report(accuflux, accufluxF)
        
        #-Read the stations
        sitesF = os.path.join(self.inputs_path, self.config.get('PCRASTER_LDD', 'stations'))
        stations = pcr.readmap(sitesF)
        
        #-New catchment map
        catchmentF = os.path.join(self.results_path, self.config.get('PCRASTER_LDD', 'catchment_clean'))
        catchment = pcr.catchment(ldd_clean, stations)
        pcr.report(catchment, catchmentF)
        
        #-New sub-catchment map
        subcatchmentF = os.path.join(self.results_path, self.config.get('PCRASTER_LDD', 'subcatchment_clean'))
        subcatchment = pcr.subcatchment(ldd_clean, stations)
        pcr.report(subcatchment, subcatchmentF)
        
    def get_CrcWap_details(self):
        '''
        Get detailed information on crc/wap level and write results to a csv-file.
        '''
        
        #-Dictionary to convert string from month and to month to numbers
        month_dict = {'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'Jun': 6,'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12}
        #-Dictionary to replace Yes/No values with 1,0
        yes_no_dict = {'Yes': 1, 'No': 0, 'yes': 1, 'no': 0, 'YES': 1, 'NO': 0}
          
        #-Get list of SWAZ names for which to extract waps
        swaz_csv = os.path.join(self.inputs_path, self.config.get('CRC_WAP', 'swaz_csv'))
        SWAZ = list(pd.read_csv(swaz_csv).SWAZ.astype(str))
          
        #-Get bore depth cutoff
        well_depth = self.config.getint('CRC_WAP', 'well_depth')
  
        #-Get all the WAPs from the database that are within those SWAZs
        print('Getting all WAPs within the selected SWAZs...')                        
        SWAZ_waps = mssql.rd_sql('sql2012prod03', 'DataWarehouse', 'D_SW_WellsDetails', col_names = ['WellNo', 'SWAllocationZone', 'Depth'], where_in={'SWAllocationZone': SWAZ})
        SWAZ_waps.rename(columns={'SWAllocationZone':'SWAZ', 'WellNo': 'wap'}, inplace=True)
        SWAZ_waps_copy = SWAZ_waps.copy()
        SWAZ_waps_copy.drop('Depth', axis=1, inplace=True)
          
        ########-Filter out WAPs that OR have a screen depth <=well_depth or a bore depth of <=well_depth (the or condition is needed because not all Wells have screens).
        print('Filtering wells with a depth <= %s meter...' %well_depth)
        SWAZ_waps_xm = SWAZ_waps.loc[SWAZ_waps['Depth']<=well_depth, ['wap']]
        #-Get Wells with top_screen <=well_depth
        print('Filtering wells with a screen depth <= %s meter from Wells...' %well_depth)
        WAP_screens = mssql.rd_sql('sql2012prod05', 'Wells', 'SCREEN_DETAILS', col_names = ['WELL_NO', 'TOP_SCREEN'], where_in={'WELL_NO': list(SWAZ_waps['wap'])})
        WAP_screens.rename(columns={'WELL_NO': 'wap'}, inplace=True)
        WAP_screens = WAP_screens.groupby('wap')['TOP_SCREEN'].min().reset_index()
        WAP_screens = WAP_screens.loc[WAP_screens['TOP_SCREEN']<=well_depth]
        WAP_screens.drop('TOP_SCREEN', axis=1, inplace=True)
        #-Concat the two and only keep unique WAP numbers
        gw_waps = pd.concat([SWAZ_waps_xm, WAP_screens])
        gw_waps.drop_duplicates(inplace=True)
        gw_waps['take_type'] = 'Take Groundwater'
        SWAZ_waps_xm = None; WAP_screens = None; del SWAZ_waps_xm, WAP_screens
           
        print('Extracting the Surface Water Take waps from the database...')
        sw_waps = mssql.rd_sql(self.server, self.database, 'CrcWapAllo', ['wap', 'take_type'], where_in={'take_type': ['Take Surface Water'], 'wap': list(SWAZ_waps['wap'])})
        sw_waps.drop_duplicates(inplace=True)
        print('Merging Surface Water Take and Groundwater Take waps...')
        SWAZ_waps = pd.concat([gw_waps, sw_waps])
        sw_waps = None; gw_waps = None; del sw_waps, gw_waps 
          
          
        #-Get all the consents related to the WAPs within the selected SWAZs
        print('Getting all consents belonging to the waps...')
        SWAZ_WAP_consents = mssql.rd_sql('sql2012prod03', 'DataWarehouse', 'D_ACC_Act_Water_TakeWaterWAPAlloc', col_names = ['RecordNo', 'WAP'], where_in={'WAP': list(SWAZ_waps['wap'])})
        SWAZ_WAP_consents.rename(columns={'RecordNo': 'crc', 'WAP': 'wap'}, inplace=True)
        crc = pd.unique(SWAZ_WAP_consents['crc'])
      
        #-Get all the consents from the F_ACC_Permit table from the DataWarehouse that are part of the SWAZ_WAP_consents selection
        df = mssql.rd_sql('sql2012prod03', 'DataWarehouse', 'F_ACC_Permit', col_names = ['B1_ALT_ID','fmDate','toDate','toDateText','Given Effect To','Expires','OriginalRecord','ParentAuthorisations','ChildAuthorisations','HolderAddressFullName'], where_in={'B1_ALT_ID': list(crc)})
        df['toDate'] = pd.to_datetime(df['toDate'], errors='coerce')
        df['fmDate'] = pd.to_datetime(df['fmDate'], errors='coerce')
        df['Given Effect To'] = pd.to_datetime(df['Given Effect To'], errors='coerce')
        df['Expires'] = pd.to_datetime(df['Expires'], errors='coerce')
          
          
        #-Select consents that were active between sdate and edate
        print('Filter consents that were active between %s and %s...' %(self.from_date.strftime('%d-%m-%Y'), self.to_date.strftime('%d-%m-%Y')))
        df1 = df.loc[(df['toDate']>pd.Timestamp(self.from_date)) & (df['fmDate']<=pd.Timestamp(self.to_date))]
        #-If 'Given Effect To' date is later than 'toDate', then consent was never active in between the fmDate-toDate period, and is therefore removed from the dataframe
        df1.loc[(df1['Given Effect To'] > df1['toDate']),:]=np.nan
        df2 = df1.dropna(how='all')
        #-If 'Given Effect To' date is later than 'fmDate', then the 'fmDate' field is set to 'Given Effect To'
        df2.loc[(df2['fmDate'] < df2['Given Effect To']),['fmDate']]=  df2['Given Effect To']
          
        #-Unique consent numbers of 'OriginalRecord'
        ori_records = pd.unique(df2['OriginalRecord'])
        df2_columns = list(df2.columns)
        fmDate_index = df2_columns.index('fmDate')
        toDate_index = df2_columns.index('toDate')
        #-Make sure toDate is always 1 day before the fmDate of the child consent. Required to make sure that a consent isn't active twice on one day
        for c in ori_records:
            #-select the consents that belong to the same group (have same parent so to speak)
            df_short = df2.loc[df2['OriginalRecord']==c]
            for i in range(0,len(df_short)-1):
                toDate = df_short.iloc[i,toDate_index] #-toDate of current record
                fmDate = df_short.iloc[i+1,fmDate_index] #-fromDate of child record
                if toDate == fmDate: #-cannot be equal. If so, then decrease the todate of the current record with one day
                    df_short.iloc[i, toDate_index] = toDate - dt.timedelta(days=1)
            df2.loc[df2['OriginalRecord']==c] = df_short
        #-get rid of old dataframes
        df = df2.copy()
        df1 = None; df2 = None; del df1, df2
        #-For consents that are active for one day, the toDate may now (because of extracting one day from toDate) be smaller than fmDate. Those records are removed
        df = df.loc[df['toDate']>=df['fmDate']]
        df = df[['B1_ALT_ID','fmDate','toDate','Given Effect To','HolderAddressFullName']] #-This dataframe contains all take consents for the specified period within the selected SWAZs
        df.rename(columns={'B1_ALT_ID': 'crc'}, inplace=True)
        #-it may be the case that given effect is empty because consent was never activated. This results in empty cells for 'Given Effect To'. These are dropped.
        df.dropna(inplace=True)
          
        #-Get dataframe of all water takes and diverts on consent level
        print('Retrieve take info on consent level...')
        crcAllo = mssql.rd_sql('sql2012prod03', 'DataWarehouse', 'D_ACC_Act_Water_TakeWaterConsent', 
                                     col_names = ['RecordNo', 'Activity', 'B1_PER_ID3', 'Consented Annual Volume (m3/year)','Combined Annual Volume', 'Complex Allocation',
                                                  'Has a low flow restriction condition?'],
                                     where_in={'RecordNo': list(df['crc'])})
        crcAllo.rename(columns={'RecordNo': 'crc', 'Consented Annual Volume (m3/year)': 'crc_ann_vol [m3]', 'Combined Annual Volume': 'crc_ann_vol_combined [m3]',
                                'Complex Allocation': 'complex_allo', 'Has a low flow restriction condition?': 'lowflow_restriction'}, inplace=True)
        #-Consents with consented annual volume of 0 are incorrect. An annual volume of 0 is not possible, and should be interpreted as no annual volume is present in the consent conditions.
        #-These values are therefore replaced with NaNs. Same is true for consented annual combined volumes.
        crcAllo.loc[crcAllo['crc_ann_vol [m3]']==0,'crc_ann_vol [m3]']=np.nan
        crcAllo.loc[crcAllo['crc_ann_vol_combined [m3]']==0,'crc_ann_vol_combined [m3]']=np.nan
        #-change yes/no, to 1/0
        crcAllo.replace({'complex_allo': yes_no_dict, 'lowflow_restriction': yes_no_dict},inplace=True)
        #-consents for which complex_allo and lowflow_restriction have no value specified, it is assumed that these conditions are false and therefore set to 0.
        crcAllo.loc[pd.isna(crcAllo['complex_allo']),'complex_allo'] = 0
        crcAllo.loc[pd.isna(crcAllo['lowflow_restriction']),'lowflow_restriction'] = 0
          
        #-Get dataframe of all water takes and diverts on WAP level
        print('Retrieve take info on WAP level...')
        crcWapAllo = mssql.rd_sql('sql2012prod03', 'DataWarehouse', 'D_ACC_Act_Water_TakeWaterWAPAlloc',
                                        col_names = ['RecordNo', 'Activity', 'From Month', 'To Month', 'Allocation Block', 'WAP', 'Max Rate for WAP (l/s)', 
                                                     'Max Rate Pro Rata (l/s)', 'Max Vol Pro Rata (m3)', 'Consecutive Day Period', 'Include in SW Allocation?', 'First Stream Depletion Rate'],
                                        where_in={'RecordNo': list(df['crc']), 'WAP': list(SWAZ_waps['wap']), 'Activity': ['Take Surface Water', 'Take Groundwater']})
        crcWapAllo.rename(columns={'RecordNo': 'crc', 'Allocation Block': 'allo_block', 'WAP': 'wap', 'Max Rate for WAP (l/s)': 'wap_max_rate [l/s]', 'From Month': 'from_month',
                                   'To Month': 'to_month', 'Max Rate Pro Rata (l/s)': 'wap_max_rate_pro_rata [l/s]', 'Max Vol Pro Rata (m3)': 'wap_max_vol_pro_rata [m3]',
                                   'Consecutive Day Period': 'wap_return_period [d]', 'Include in SW Allocation?': 'in_sw_allo', 'First Stream Depletion Rate': 'first_sd_rate [l/s]'}, inplace=True)
        #-A few waps were not in capitals, which results in errors in joins later on. Therefore all waps were capitalized
        crcWapAllo['wap'] = crcWapAllo['wap'].str.upper()
        crcWapAllo.replace({'from_month': month_dict, 'to_month': month_dict},inplace=True)
        #-if wap max pro rata volume is specified, but the return period itself not, then assume return period equals 1
        crcWapAllo.loc[(crcWapAllo['wap_max_vol_pro_rata [m3]']>0) & pd.isna(crcWapAllo['wap_return_period [d]']), 'wap_return_period [d]'] = 1
        #-WAPs with wap "wap_max_rate [l/s]" and "wap_max_rate_pro_rata [l/s]" both being zero do not have water take/divert related consent conditions and are therefore dropped
        crcWapAllo.loc[(crcWapAllo['wap_max_rate [l/s]']==0) & (crcWapAllo['wap_max_rate_pro_rata [l/s]']==0),:] = np.nan
        #-WAPs where wap_max_vol_pro_rata and wap_return_period are 0 are set to NaN
        crcWapAllo.loc[(crcWapAllo['wap_max_vol_pro_rata [m3]']==0) & (crcWapAllo['wap_return_period [d]']==0),['wap_max_vol_pro_rata [m3]', 'wap_return_period [d]']] = np.nan
        crcWapAllo.dropna(how='all', inplace=True)
        #-Replace yes/no in in_sw_allo with 1/0
        crcWapAllo.replace({'in_sw_allo': yes_no_dict},inplace=True)
          
        #-merge selected consents with WAPs (takes and diverts)
        print('Merge details on WAP level...')
        df1 = pd.merge(df, crcWapAllo, how='left', on='crc')
        df1.drop_duplicates()
        crcWapAllo = None; del crcWapAllo
          
        #-merge SWAZ names
        SWAZ_waps_copy.loc[SWAZ_waps_copy.wap.isin(pd.unique(df1['wap']))]
        df1 = pd.merge(df1, SWAZ_waps_copy, how='left', on='wap')
        df1 = df1.loc[~pd.isna(df1['Activity'])]
          
        #-add the WAP NZTMX and NZTMY and SwazGroupName
        print('Adding NZTM and NZTMY coordinates...')
        extsite_df = mssql.rd_sql(self.server, self.database, 'ExternalSite', col_names = ['ExtSiteID', 'NZTMX', 'NZTMY', 'SwazGroupName'], where_in = {'ExtSiteID': list(df1['wap'])})
        df1 = pd.merge(df1, extsite_df, how='left', left_on='wap', right_on='ExtSiteID')
        df1.drop('ExtSiteID', axis=1, inplace=True)
        extsite_df = None; del extsite_df
  
        #-get stream depletion info and merge with df1
        print('Replace NZTMX and NZTMY for Groundwater Take WAPs with SD point coordinates...')
        gw_waps = pd.unique(df1.loc[df1['Activity']=='Take Groundwater', 'wap'])
        sd_df = mssql.rd_sql('sql2012prod05', 'Wells', 'Well_StreamDepletion_Locations', col_names = ['Well_No', 'NZTMX', 'NZTMY'], where_in = {'Well_No': list(gw_waps)})
        sd_df.rename(columns={'Well_No': 'wap'}, inplace=True)
        gw_waps = None; del gw_waps
          
        df2 = df1.loc[df1['Activity']=='Take Groundwater']
        df2.drop(['NZTMX', 'NZTMY'], axis=1, inplace=True)
        df2 = pd.merge(df2, sd_df, how='left', on='wap')
        df3 = df1.loc[df1['Activity']=='Take Surface Water']
        df1 = pd.concat([df2, df3])
        df2 = None; df3 = None; del df2, df3
          
        #-merge consent level dataframe
        df1 = pd.merge(df1, crcAllo, how='left', on=['crc','Activity'])
        crcAllo = None; del crcAllo
        df1.drop_duplicates(inplace=True)
          
        #-Get use type and irrigated area info
        print('Adding water use type and irrigated area...')
        hydro_crc_allo_df = mssql.rd_sql(self.server, self.database, 'CrcAllo', col_names = ['crc', 'take_type', 'allo_block', 'irr_area', 'use_type'], where_in = {'crc': list(df1['crc'])})
        hydro_crc_allo_df.rename(columns={'irr_area': 'irr_area [ha]'}, inplace=True)
        df1 = pd.merge(df1, hydro_crc_allo_df, how='left', left_on=['crc', 'Activity', 'allo_block'], right_on=['crc', 'take_type', 'allo_block'])
        df1.drop_duplicates(inplace=True)
        df1.drop('take_type', axis=1, inplace=True)
        hydro_crc_allo_df = None; del hydro_crc_allo_df
           
        #-Get SD connection info and merge
        print('Adding SD connection info...')
        hydro_wap_allo_df = mssql.rd_sql(self.server, self.database, 'CrcWapAllo', col_names = ['crc', 'take_type', 'allo_block', 'wap', 'sd1_7', 'sd1_30', 'sd1_150'], 
                                         where_in = {'crc': list(df1['crc']), 'wap': list(df1['wap'])})
        df1 = pd.merge(df1, hydro_wap_allo_df, how='left', left_on=['crc', 'wap', 'Activity', 'allo_block'], right_on=['crc', 'wap', 'take_type', 'allo_block'])
        df1.drop(['take_type', 'B1_PER_ID3'], axis=1, inplace=True)
        #-Fill NaNs for sd1_7, etc with zero (these are non-depleting)
        df1.loc[(pd.isna(df1['sd1_7'])) & (df1['Activity'] == 'Take Groundwater'), 'sd1_7'] = 0
        df1.loc[(pd.isna(df1['sd1_30'])) & (df1['Activity'] == 'Take Groundwater'), 'sd1_30'] = 0
        df1.loc[(pd.isna(df1['sd1_150'])) & (df1['Activity'] == 'Take Groundwater'), 'sd1_150'] = 0
        df1.loc[df1['Activity'] == 'Take Surface Water', ['sd1_7', 'sd1_30', 'sd1_150']] = np.nan
        df1.drop_duplicates(inplace=True)
            
        ###-re-organize order of columns
        df_final = df1[['crc', 'fmDate', 'toDate', 'Given Effect To', 'HolderAddressFullName', 'Activity', 'use_type', 'irr_area [ha]', 'from_month', 'to_month', 'SWAZ', 'SwazGroupName',
                    'in_sw_allo', 'allo_block', 'lowflow_restriction', 'complex_allo', 'crc_ann_vol [m3]', 'crc_ann_vol_combined [m3]', 'wap', 'wap_max_rate [l/s]', 'wap_max_rate_pro_rata [l/s]', 'wap_max_vol_pro_rata [m3]',
                    'wap_return_period [d]', 'first_sd_rate [l/s]', 'sd1_7', 'sd1_30', 'sd1_150','NZTMX', 'NZTMY']]
        df_final.loc[pd.isna(df_final['complex_allo']),'complex_allo'] = 0
        df_final.loc[pd.isna(df_final['lowflow_restriction']),'lowflow_restriction'] = 0
        df_final.loc[pd.isna(df_final['in_sw_allo']),'in_sw_allo'] = 0
        df1 = None; del df1
        
        #-keep only unique waps and coordinates for converting to geopandas dataframe
        print('Converting waps to GeoPandas DataFrame...')
        waps = df_final[['wap', 'SWAZ', 'SwazGroupName','NZTMX', 'NZTMY']].drop_duplicates()
        wap_sites = vector.xy_to_gpd(['wap', 'SWAZ', 'SwazGroupName','NZTMX', 'NZTMY'], 'NZTMX', 'NZTMY', waps,  crs=2193)
        wap_sites.drop(['NZTMX', 'NZTMY'], axis=1, inplace=True)
        waps = None; del waps
        
        #-Get the list of lowflow sites
        self.LF_sites = self.flow_sites_gdf['flow_site']
        #-Loop over the lowflow sites and join the waps that are located within the upstream catchment of that lowflow site
        catch_path =  os.path.join(self.inputs_path, self.config.get('CRC_WAP', 'catch_path'))
        gpd_list = []
        for j in self.LF_sites:
            #-read the catchment shapefile into a GeoPandas Dataframe
            catch = gpd.read_file(os.path.join(catch_path, j + '.shp'))
            #-join the waps located within that catchment
            waps_gdf, poly1 = vector.pts_poly_join(wap_sites, catch, 'flow_site')
            #-add the GeoPandas Dataframe to a list
            gpd_list.append(waps_gdf)
        #-Concat the individual GeoPandas Dataframes into one and write to a shapefile
        self.waps_gdf = gpd.GeoDataFrame(pd.concat(gpd_list, ignore_index=True))
        shpF = os.path.join(self.results_path, self.config.get('CRC_WAP','wap_shp'))
        #-Writing waps with lowflow sites to shapefile
        print('Writing waps with lowflow sites to %s...' %shpF)   
        self.waps_gdf.to_file(shpF)

        #-remove waps from df_final that do not have a lowflow site
        waps = pd.unique(self.waps_gdf['wap'])
        self.crc_wap_df = df_final.loc[df_final.wap.isin(waps)]
        csvF = os.path.join(self.results_path, self.config.get('CRC_WAP', 'crc_wap_csv'))
        self.crc_wap_df.to_csv(csvF, index=False)
        
        print('Consent and wap extracting completed successfully.')
        
    def wapTS(self):
        '''
        Extract metered abstraction data for selected waps and time period from database and write to a csv-file
        '''        

        #-waps to extract
        waps = list(pd.unique(self.crc_wap_df['wap']))
        
        #-get the wap abstraction data for rivers (9) and aquifer (12) for the selected waps and only keep period of interest
        print('Extracting metered abstraction time-series for %s waps from database...' %len(waps))
        df_ts = mssql.rd_sql(self.server, self.database, 'TSDataNumericDaily', col_names = ['ExtSiteID', 'DatasetTypeID', 'QualityCode', 'DateTime', 'Value'], where_in = {'ExtSiteID': waps, 'DatasetTypeID': [9, 12]})
        df_ts['DateTime'] = pd.to_datetime(df_ts['DateTime'])
        df_ts = df_ts.loc[(df_ts['DateTime']>=pd.Timestamp(self.from_date)) & (df_ts['DateTime']<=pd.Timestamp(self.to_date))]
        df_ts.rename(columns={'DateTime': 'Date'}, inplace=True)
        
        #-empty dataframe to be filled with metered ts
        self.wap_ts_metered_df = pd.DataFrame(index=pd.date_range(self.from_date, self.to_date, freq='D'), columns=waps)
        self.wap_ts_metered_df.rename_axis('Date', inplace=True)
        
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
                self.wap_ts_metered_df[[w]] = df_sel
            except:
                lMessage = w + ' has multiple DatasetTypeIDs assigned to it. This is not possible and needs to be checked; i.e. a WAP can not be a divert and surface water take at the same time!!'
                lMessageList.append(lMessage) 
                print(lMessage)
                #-use the first occuring entry two prevent double datasettypeids
                df_sel.reset_index(inplace=True)
                df_sel = df_sel.groupby('Date').first()
                self.wap_ts_metered_df[[w]] = df_sel
        
        #-csv-file for wap time-series (raw metered data)
        wap_ts_csv = os.path.join(self.results_path, self.config.get('CRC_WAP', 'wap_ts_csv'))
        self.wap_ts_metered_df.to_csv(wap_ts_csv)
        print('Metered abstraction successfully written to %s' %wap_ts_csv)
        
        
    def abstraction_ratios(self):
        '''
        Create a dataframe with for each date, crc, and wap:
            - The maximum allocated volume (crc_wap_max_vol [m3]) for that crc/wap/date
            - The metered abstraction (crc_wap_metered_abstraction [m3]) for that crc/wap/date
            - If the crc/wap/date has a metered value (metered [yes/no]) yes or no (1/0)
            - The ratio (only for records were a metered value is availble) of metered abstraction over the maximum allocated volume
        Columns in dataframe:
            - Date
            - crc
            - wap
            - Activity
            - use_type
            - SWAZ
            - in_sw_allo
            - allo_block
            - lowflow_restriction
            - crc_wap_max_vol [m3]
            - crc_wap_metered_abstraction [m3]
            - metered [yes/no]
            - ratio
        Writes the resulting dataframe to a csv-file
        '''
        
        print('Creating dataframe with maximum allocated volume, metered abstraction, and ratio for each date, consent, and wap...')
        
        #-dictionary to reclassify use_types
        use_type_dict = {'Aquaculture': 'irrigation', 'Dairy Shed (Washdown/Cooling)': 'stockwater', 'Intensive Farming - Dairy': 'irrigation', 'Intensive Farming - Other (Washdown/Stockwater/Cooling)': 'stockwater', 'Intensive Farming - Poultry': 'irrigation', 'Irrigation - Arable (Cropping)': 'irrigation', 'Irrigation - Industrial': 'irrigation', 'Irrigation - Mixed': 'irrigation', 'Irrigation - Pasture': 'irrigation', 'Irrigation Scheme': 'irrigation' , 'Viticulture': 'irrigation', 'Community Water Supply': 'water_supply', 'Domestic Use': 'water_supply', 'Construction': 'industrial', 'Construction - Dewatering': 'industrial', 'Cooling Water (non HVAC)': 'industrial', 'Dewatering': 'industrial', 'Gravel Extraction/Processing': 'industrial', 'HVAC': 'industrial', 'Industrial Use - Concrete Plant': 'industrial', 'Industrial Use - Food Products': 'industrial', 'Industrial Use - Other': 'industrial', 'Industrial Use - Water Bottling': 'industrial', 'Mining': 'industrial', 'Firefighting ': 'municipal', 'Firefighting': 'municipal', 'Flood Control': 'municipal', 'Landfills': 'municipal', 'Stormwater': 'municipal', 'Waste Water': 'municipal', 'Stockwater': 'stockwater', 'Snow Making': 'industrial', 'Augment Flow/Wetland': 'other', 'Fisheries/Wildlife Management': 'other', 'Other': 'other', 'Recreation/Sport': 'other', 'Research (incl testing)': 'other', 'Power Generation': 'hydroelectric', 'Drainage': 'municipal', 'Frost Protection': 'municipal'}
        #-reclassify use_type
        self.crc_wap_df.replace({'use_type': use_type_dict},inplace=True)
        #-do another classification on top (from Mike script)
        use_type_dict2 = {'industrial': 'other', 'municipal': 'other'}
        self.crc_wap_df.replace({'use_type': use_type_dict2},inplace=True)
        #-some records have NaN for use type: these are filled with the use type of the consent to which the wap belongs (if that consent has a use type)
        crc = pd.unique(self.crc_wap_df.loc[pd.isna(self.crc_wap_df['use_type']),'crc'])
        print('%s consents have one or more waps with a NaN for uses_type. These are filled with the use_type of the consent if this is known...' %len(crc))
        check_crc_list = []
        for c in crc:
            df_short = self.crc_wap_df.loc[self.crc_wap_df['crc']==c]
            df_short = df_short.loc[~pd.isna(df_short['use_type'])]
            df_short_use_type = pd.unique(df_short['use_type'])
            if len(df_short_use_type)==1:
                self.crc_wap_df.loc[(self.crc_wap_df['crc']==c) & pd.isna(self.crc_wap_df['use_type']),'use_type'] = df_short_use_type[0]
            else:
                check_crc_list.append(c)
        if len(check_crc_list)>0:
            print('For %s consent the use_type could not be filled because all waps of this consent do not have a use_type specified...' %len(check_crc_list))
            print('Check:')
            for j in check_crc_list:
                print('\t%s' %j)
        #-write updated crc_wap_df to csv-file         
        csvF = os.path.join(self.results_path, self.config.get('CRC_WAP', 'crc_wap_csv'))
        self.crc_wap_df.to_csv(csvF, index=False)

        #-Shorter column version of dataframe
        crc_wap_df = self.crc_wap_df[['crc', 'fmDate', 'toDate', 'Activity', 'use_type', 'from_month', 'to_month', 'SWAZ', 'in_sw_allo', 'allo_block', 'lowflow_restriction', 'wap', 'wap_max_rate [l/s]']]        
                  
        #-calculate maximum daily allowed volumes for crc/wap in m3
        crc_wap_df[['wap_max_vol [m3]']] = crc_wap_df[['wap_max_rate [l/s]']] * 86.4
        crc_wap_df.drop('wap_max_rate [l/s]', axis=1, inplace=True)
        crc_wap_df.loc[crc_wap_df['wap_max_vol [m3]']==0] = np.nan #-remove records with 0 wap_max_vol (cannot be trusted for demand estimation)
        crc_wap_df.dropna(how='all', inplace=True)
          
        #-Define columns of dataframe to be created 
        cols = list(crc_wap_df.columns.drop(['fmDate', 'toDate']))
        cols.insert(0, 'Date')
        cols.insert(1, 'Month')
        cols.insert(2,'active')
          
        #-Initialize empty dataframe
        df = pd.DataFrame(columns=cols)
        dates_range = pd.date_range(self.from_date, self.to_date, freq='D')
        #-Copy of empty dataframe to be used later for concatenation
        df_full = df.copy()
        #-Loop over all the consents/waps in the table, make a dataframe for the date range in which they were active, and add it to the final dataframe
        for c in crc_wap_df.iterrows():
            fmDate = c[1]['fmDate']
            toDate = c[1]['toDate']
            if (fmDate<= pd.Timestamp(self.to_date)) & (toDate>= pd.Timestamp(self.from_date)):
                print(c[1]['wap'])
                #-Temporaray dataframe for current consent/wap
                df_temp = df.copy()
                df_temp['Date'] = pd.date_range(c[1]['fmDate'], c[1]['toDate'], freq='D')
                df_temp['Month'] = df_temp['Date'].dt.strftime('%m')
                df_temp['Month'] = df_temp['Month'].astype(np.int) 
                df_temp['crc'] = c[1]['crc']
                df_temp['wap'] = c[1]['wap']
                df_temp['Activity'] = c[1]['Activity']
                df_temp['use_type'] = c[1]['use_type']
                df_temp['SWAZ'] = c[1]['SWAZ']
                df_temp['in_sw_allo'] = c[1]['in_sw_allo']
                df_temp['allo_block'] = c[1]['allo_block']
                df_temp['lowflow_restriction'] = c[1]['lowflow_restriction']
                df_temp['from_month'] = c[1]['from_month']
                df_temp['to_month'] = c[1]['to_month']
                df_temp['wap_max_vol [m3]'] =   c[1]['wap_max_vol [m3]']
                df_temp['active'] = 0
                mlist = []
                if c[1]['from_month'] < c[1]['to_month']:
                    for m in range(int(c[1]['from_month']), int(c[1]['to_month'])+1):
                        mlist.append(m)
                elif c[1]['from_month'] > c[1]['to_month']:
                    for m in range(int(c[1]['from_month']), 12+1):
                        mlist.append(m)
                    for m in range(1, int(c[1]['to_month']) + 1):
                        mlist.append(m)
                else:
                    mlist.append(int(c[1]['from_month']))
                df_temp.loc[df_temp['Month'].isin(mlist), 'active'] = 1
                   
                #-Concat temporary dataframe to final dataframe
                df_full = pd.concat([df_full, df_temp])
                df_temp = None; del df_temp
        df = None; del df
          
        #-Merge for the period of interest only
        crc_wap_df = pd.DataFrame(columns=['Date'])
        crc_wap_df['Date'] = dates_range
        crc_wap_df = pd.merge(crc_wap_df, df_full, how='left', on='Date')
        df_full = None; del df_full
          
        #-Drop records where consent is not executed
        crc_wap_df.loc[crc_wap_df['active']!=1] = np.nan
        crc_wap_df.dropna(how='all', inplace=True)
        crc_wap_df.drop(['Month', 'from_month', 'to_month', 'active'], axis=1, inplace=True)
        crc_wap_df.drop_duplicates(inplace=True)
          
        #-Group by date and wap to calculate the maximum volume that may be extracted from a wap on a specific date
        df_wap_max = crc_wap_df.groupby(['Date', 'wap'])['wap_max_vol [m3]'].sum().reset_index()
        df_wap_max.set_index('Date', inplace=True)
          
        #-Merge the metered time-series
        df_wap_max['wap_metered_abstraction [m3]'] = np.nan
        for i in list(self.wap_ts_metered_df.columns):
            #-Get the WAP ts
            df = self.wap_ts_metered_df[[i]]
            df.columns = ['wap_metered_abstraction [m3]']
            df_wap_max.loc[df_wap_max['wap']==i,['wap_metered_abstraction [m3]']] = df
        df_wap_max.reset_index(inplace=True)

        #-Merge df_wap_max to crc_wap_df
        crc_wap_df = pd.merge(crc_wap_df, df_wap_max, how='left', on=['Date', 'wap'])
        crc_wap_df.rename(columns={'wap_max_vol [m3]_x': 'crc_wap_max_vol [m3]', 'wap_max_vol [m3]_y': 'wap_max_vol [m3]'}, inplace=True)
        df_wap_max = None; del df_wap_max
        
        #-Calculate pro-rata distribution of metered abstraction to consent/wap level
        crc_wap_df['crc_wap_max_vol_ratio [-]'] = crc_wap_df['crc_wap_max_vol [m3]'] / crc_wap_df['wap_max_vol [m3]']
        crc_wap_df['crc_wap_metered_abstraction [m3]'] = crc_wap_df['wap_metered_abstraction [m3]'] * crc_wap_df['crc_wap_max_vol_ratio [-]']
        crc_wap_df.drop(['wap_max_vol [m3]','wap_metered_abstraction [m3]','crc_wap_max_vol_ratio [-]',], axis=1, inplace=True)
        
        #-Add column with ones if record is measured for that crc/wap/date yes (1) or no (0)
        crc_wap_df['metered [yes/no]'] = np.nan
        crc_wap_df.loc[pd.isna(crc_wap_df['crc_wap_metered_abstraction [m3]']), 'metered [yes/no]'] = 0
        crc_wap_df.loc[pd.notna(crc_wap_df['crc_wap_metered_abstraction [m3]']), 'metered [yes/no]'] = 1
        
        #-Calculate abstraction ratio
        crc_wap_df['ratio'] = crc_wap_df['crc_wap_metered_abstraction [m3]'] / crc_wap_df['crc_wap_max_vol [m3]']
        #-Move 'wap' column
        cols = crc_wap_df.columns.tolist()
        cols.insert(2, cols.pop(cols.index('wap')))
        q = crc_wap_df.reindex(columns=cols)
        self.date_crc_wap_ratio = q
        self.date_crc_wap_ratio.drop_duplicates(inplace=True)
        self.date_crc_wap_ratio.replace({'use_type': {'Irrigation': 'irrigation'}},inplace=True)
        #-Write to csv-file
        csvF = os.path.join(self.results_path, self.config.get('CALC_RATIOS','crc_wap_ratio_csv'))
        self.date_crc_wap_ratio.to_csv(csvF, index=False)
        print('Abstraction ratios successfully written to %s' %csvF)
       

    def plot_ratios(self, swazs='all', use_type='all', max_ratio=20):
        '''
        Make a histogram of the ratios to check for unrealistic ratios
        '''
        #-color maps for plotting
        colors = [(31, 119, 180), (174, 199, 232), (255, 127, 14), (255, 187, 120),    
                 (44, 160, 44), (152, 223, 138), (214, 39, 40), (255, 152, 150),    
                 (148, 103, 189), (197, 176, 213), (140, 86, 75), (196, 156, 148),    
                 (227, 119, 194), (247, 182, 210), (127, 127, 127), (199, 199, 199),    
                 (188, 189, 34), (219, 219, 141), (23, 190, 207), (158, 218, 229)]
        for i in range(len(colors)):
            r, g, b = colors[i]
            colors[i] = (r / 255., g / 255., b / 255.)
        
        ratio_df = self.date_crc_wap_ratio.copy()
        
        fig, ax = plt.subplots()
        #b = np.arange(1, max_ratio+1, 0.5)
        b = np.arange(0, max_ratio+1, 0.5)
        if (swazs=='all') and (use_type=='all'):
            plt.hist(ratio_df['ratio'], density=True, bins=b, cumulative=True, color=colors[0], edgecolor='black')
            plt.title('Ratios for all SWAZs and all use_types')
        else:
            ratio_df = ratio_df.loc[(ratio_df['SWAZ']==swazs) & (ratio_df['use_type']==use_type)]
            plt.hist(ratio_df['ratio'], density=True, bins=b, cumulative=True, color=colors[0], edgecolor='black')
            plt.title('Ratio for %s and use_type=%s' %(swazs, use_type))
        ax.set_xlabel('Ratio [-]')
        ax.set_ylabel('Cumulative density [-]')
        ax.set_xticks(np.arange(0, max_ratio+1, 1))
        ax.set_xlim([0, max_ratio])
        fig.tight_layout()
        plt.show(block=True)
        
    def group_ratios(self):
        '''
        Group the ratios by use_type, SWAZ, and month. A filter may be applied to remove ratios above a threshold before grouping is done.
        '''
        
        print('Grouping ratios by use_type, SWAZ, and month...')
        csvF = os.path.join(self.results_path, self.config.get('CALC_RATIOS', 'crc_wap_ratio_grouped_csv'))
        
        #-only keep records where a ratio was calculated (i.e. records without a metered abstraction are removed)
        date_crc_wap_ratio = self.date_crc_wap_ratio.loc[~pd.isna(self.date_crc_wap_ratio['ratio'])]
        try:
            self.thresh = self.config.getfloat('CALC_RATIOS', 'ratio_threshold')
        except:
            self.thresh = False
        #-filter ratios above threshold
        if self.thresh:
            print('Removing ratios > %.2f' %self.thresh)
            date_crc_wap_ratio = date_crc_wap_ratio.loc[date_crc_wap_ratio['ratio']<=self.thresh]
        date_crc_wap_ratio['month'] = pd.DatetimeIndex(date_crc_wap_ratio['Date']).month
        #-group by use_type, SWAZ, month, and calculate average
        self.crc_wap_ratio_grouped = date_crc_wap_ratio.groupby(['use_type', 'SWAZ', 'month']).mean()
        self.crc_wap_ratio_grouped.drop(['in_sw_allo', 'lowflow_restriction', 'crc_wap_max_vol [m3]', 'crc_wap_metered_abstraction [m3]', 'metered [yes/no]'], axis=1, inplace=True)
        self.crc_wap_ratio_grouped.reset_index(inplace=True)
        self.crc_wap_ratio_grouped.to_csv(csvF, index=False)
        
        #-Check if average ratio could be calculated for all combinations of SWAZs and use_types. If not, then use the mean of all SWAZs for that use_type
        fill_missing_ratios = self.config.getint('CALC_RATIOS', 'fill_missing_ratios')
        if fill_missing_ratios:
            crc_wap_ratio_grouped_avg = self.crc_wap_ratio_grouped.groupby(['use_type','month']).mean()
            crc_wap_ratio_grouped_avg.reset_index(inplace=True)
            use_types = pd.unique(self.date_crc_wap_ratio['use_type']).tolist()
            swazs = pd.unique(self.date_crc_wap_ratio['SWAZ']).tolist()
            for u in use_types:
                for s in swazs:
                    t = self.crc_wap_ratio_grouped.loc[(self.crc_wap_ratio_grouped['use_type']==u) & (self.crc_wap_ratio_grouped['SWAZ']==s)]
                    if len(t)==0:
                        tt = self.date_crc_wap_ratio.loc[(self.date_crc_wap_ratio['use_type']==u) & (self.date_crc_wap_ratio['SWAZ']==s)]
                        if len(tt)!=0:
                            print('Average ratio was not calculated for use_type=%s and SWAZ=%s. Ratio is calculated using the average ratio of %s of the other SWAZs.' %(u,s,u))
                            df = crc_wap_ratio_grouped_avg.loc[crc_wap_ratio_grouped_avg['use_type']==u].copy()
                            df['SWAZ'] = s
                            df1 = df[['use_type', 'SWAZ', 'month', 'ratio']]
                            self.crc_wap_ratio_grouped = pd.concat([self.crc_wap_ratio_grouped, df1], axis=0)
            self.crc_wap_ratio_grouped.to_csv(csvF, index=False)
        
        #-Plot ratios for each use_type and SWAZ?
        plot_grouped_ratios = self.config.getint('CALC_RATIOS', 'plot_grouped_ratios')
        if plot_grouped_ratios:
            #use_types = pd.unique(self.crc_wap_ratio_grouped['use_type'])
            use_types = pd.unique(self.crc_wap_ratio_grouped['use_type']).tolist()
            swazs = pd.unique(self.crc_wap_ratio_grouped['SWAZ']).tolist()
            for u in use_types:
                for s in swazs:
                    df = self.crc_wap_ratio_grouped.loc[(self.crc_wap_ratio_grouped['use_type']==u) & (self.crc_wap_ratio_grouped['SWAZ']==s),['month', 'ratio']]
                    if len(df)>0:
                        fig, ax1 = plt.subplots()
                        ax1.set_xlabel('Month')
                        ax1.set_ylabel('Ratio [-]')
                        ax1.plot(df['month'], df['ratio'], color='red')
                        ax1.grid(True)
                        ax1.set_xlim([1, 12])
                        ax1.set_xticks([i for i in range(1,13)])
                        ax1.set_ylim([0, np.max(df['ratio'])])
                        plt.title(u + ' - ' + s)
                        ax1.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))  
                        fig.tight_layout()
                        plt.show()
    
    def estimate_usage(self):
        '''
        Estimate usage for dates without a metered abstraction value or if the ratio is > threshold.
        '''
        
        #-dataframe to fill with estimated abstraction        
        self.date_allo_usage = self.date_crc_wap_ratio.copy()
        self.date_allo_usage['month'] = pd.DatetimeIndex(self.date_allo_usage['Date']).month
        self.date_allo_usage['month_ratio'] = np.nan
        
        #-filter ratios above threshold
        try:
            self.thresh = self.config.getfloat('CALC_RATIOS', 'ratio_threshold')
        except:
            self.thresh = False
        if self.thresh:
            print('Replacing metered values having ratios > %.2f with NaN' %self.thresh)
            self.date_allo_usage.loc[self.date_allo_usage['ratio']>self.thresh, 'crc_wap_metered_abstraction [m3]'] = np.nan
        #-Create column for estimated abstraction
        self.date_allo_usage['crc_wap_estimated_abstraction [m3]'] = np.nan
        
        #-use_types and swazs
        use_types = pd.unique(self.date_allo_usage['use_type'].tolist())
        swazs = pd.unique(self.date_allo_usage['SWAZ'].tolist())
        #-fill the 'month_ratio' field
        print('Applying monthly ratios...')
        for m in range(1, 12+1):
            for u in use_types:
                for s in swazs:
                    print('Month %s, %s, %s' %(m,u,s))
                    try:
                        v = self.crc_wap_ratio_grouped.loc[(self.crc_wap_ratio_grouped['month']==m) & (self.crc_wap_ratio_grouped['use_type']==u) & (self.crc_wap_ratio_grouped['SWAZ']==s), ['ratio']].to_numpy()[0][0]
                        self.date_allo_usage.loc[(self.date_allo_usage['month']==m) & (self.date_allo_usage['use_type']==u) & (self.date_allo_usage['SWAZ']==s), 'month_ratio'] = v
                    except:
                        pass
        print('Estimating abstraction using monthly ratio and maximum daily volume...')
        #-estimate the demand by multiplying 'crc_wap_max_vol [m3]' with 'month_ratio'
        self.date_allo_usage['crc_wap_estimated_abstraction [m3]'] =   self.date_allo_usage['crc_wap_max_vol [m3]'] * self.date_allo_usage['month_ratio']
        #-use estimated field where metered is missing
        self.date_allo_usage['crc_wap_metered_abstraction_filled [m3]'] = np.nan
        self.date_allo_usage.loc[pd.isna(self.date_allo_usage['crc_wap_metered_abstraction [m3]']), 'crc_wap_metered_abstraction_filled [m3]'] = self.date_allo_usage.loc[pd.isna(self.date_allo_usage['crc_wap_metered_abstraction [m3]']), 'crc_wap_estimated_abstraction [m3]']
        self.date_allo_usage.loc[pd.notna(self.date_allo_usage['crc_wap_metered_abstraction [m3]']), 'crc_wap_metered_abstraction_filled [m3]'] = self.date_allo_usage.loc[pd.notna(self.date_allo_usage['crc_wap_metered_abstraction [m3]']), 'crc_wap_metered_abstraction [m3]']
        self.date_allo_usage.drop(['month', 'ratio'], axis=1, inplace=True)
        #-csv-file to write to
        csvF = os.path.join(self.results_path, self.config.get('ESTIMATE_USAGE','estimated_usage_csv'))
        self.date_allo_usage.to_csv(csvF, index=False)    
        print('Estimating usage completed successfully.')     


        
                



                
        

        
        


