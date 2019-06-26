# -*- coding: utf-8 -*-

'''
Just a simple script to summarize the irrigated area active on one day
'''

#-Authorship information-########################################################################################################################
__author__ = 'Wilco Terink'
__copyright__ = 'Wilco Terink'
__version__ = '1.0'
__email__ = 'wilco.terink@ecan.govt.nz'
__date__ ='June 2019'
#################################################################################################################################################


import pandas as pd

df = pd.read_csv(r'C:\Active\Projects\Ashburton\naturalisation\results4\Ashburton_crc_wap.csv', parse_dates=[1,2], dayfirst=True)
edate = pd.Timestamp(2018,12,31)
df = df.loc[df['toDate']>=edate, ['crc', 'fmDate', 'toDate', 'irr_area [ha]']]
df.drop_duplicates(inplace=True)

irr_area = df['irr_area [ha]'].sum()
print(irr_area)
