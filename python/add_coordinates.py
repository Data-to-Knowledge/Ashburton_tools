# -*- coding: utf-8 -*-

'''
Adds coordinates to csv-files with lowflow sites for the current situation and 2013 situation
'''

#-Authorship information-########################################################################################################################
__author__ = 'Wilco Terink'
__copyright__ = 'Wilco Terink'
__version__ = '1.0'
__email__ = 'wilco.terink@ecan.govt.nz'
__date__ ='July 2019'
#################################################################################################################################################

import pandas as pd
from pdsql import mssql
import numpy as np

database = 'hydro'
server = 'edwprod01'


curr_sites = pd.read_csv(r'C:\Active\Projects\Ashburton\GIS\current_sites.csv')
future_sites = pd.read_csv(r'C:\Active\Projects\Ashburton\GIS\2023_sites.csv')

curr_sites_xy = mssql.rd_sql(server, database, 'ExternalSite', col_names=['ExtSiteID', 'NZTMX', 'NZTMY'], where_in={'ExtSiteID': list(curr_sites.ID.astype(str))})
curr_sites.ID = curr_sites.ID.astype(str)
curr_sites = pd.merge(curr_sites, curr_sites_xy, how='left', left_on='ID', right_on='ExtSiteID')
curr_sites.to_csv(r'C:\Active\Projects\Ashburton\GIS\current_sites_xy.csv', index=False)

future_sites_xy = mssql.rd_sql(server, database, 'ExternalSite', col_names=['ExtSiteID', 'NZTMX', 'NZTMY'], where_in={'ExtSiteID': list(future_sites.ID.astype(str))})
future_sites.ID = future_sites.ID.astype(str)
future_sites = pd.merge(future_sites, future_sites_xy, how='left', left_on='ID', right_on='ExtSiteID')
future_sites.to_csv(r'C:\Active\Projects\Ashburton\GIS\future_sites_xy.csv', index=False)