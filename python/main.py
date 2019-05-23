# -*- coding: utf-8 -*-

from python.tools import tif_to_pcr, myHydroTool


#-Authorship information-########################################################################################################################
__author__ = 'Wilco Terink'
__copyright__ = 'Wilco Terink'
__version__ = '1.0'
__email__ = 'wilco.terink@ecan.govt.nz'
__date__ ='May 2019'
#################################################################################################################################################

 
# #-Converting GeoTiff to PCraster        
# fin = r'C:\Active\Projects\Ashburton\GIS\rec_rivers_250m_very_downstream_only.tif'        
# fout = r'C:\Active\Projects\Ashburton\GIS\rec_rivers_250m_very_downstream_only.map'
# tif_to_pcr(fin, fout)       

ashburton = myHydroTool()

#ashburton.plot_ratios(swazs='North Branch', use_type='stockwater')
