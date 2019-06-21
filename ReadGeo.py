#!/usr/bin/env python
# coding: utf-8

# # Read Geography Data

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcol
from mpl_toolkits.basemap import Basemap
from netCDF4 import Dataset
import pandas as pd
import os


# ## - Read Topography

# In[2]:


def Etopo(lon_area, lat_area, resolution):
    
    ###############################################################
    # Read the topography model
    # The data downloaded from:
    # https://www.ngdc.noaa.gov/mgg/global/global.html
    #
    # R. Kiuchi (Jun. 2018)
    ###############################################################
    
    ### Input 
    # resolution: resolution of topography for both of longitude and latitude [deg]
    # (Original resolution is 0.0167 deg)
    # lon_area and lat_area: the region of the map which you want like [100, 130], [20, 25]
    ###
    
    ### Output
    # Mesh type longitude, latitude, and topography data
    ### Output
    
    ###############----For input file----###############
    # Setting the directory
    homedir = os.getcwd()
    path = homedir + '/data'
    os.chdir(path)
    
    # Read NetCDF data
    # The data from https://www.ngdc.noaa.gov/mgg/global/global.html
    data = Dataset("ETOPO1_Ice_g_gdal.grd", "r")
    
    # Get data
    lon_range = data.variables['x_range'][:]
    lat_range = data.variables['y_range'][:]
    topo_range = data.variables['z_range'][:]
    spacing = data.variables['spacing'][:]
    dimension = data.variables['dimension'][:]
    z = data.variables['z'][:]

    lon_num = dimension[0]
    lat_num = dimension[1]
    
    
    # Prepare array
    # For longitude
    lon_input = np.zeros(lon_num); lat_input = np.zeros(lat_num)
    for i in range(lon_num):
        lon_input[i] = lon_range[0] + i * spacing[0]

    for i in range(lat_num):
        lat_input[i] = lat_range[0] + i * spacing[1]

    # Create 2D array
    lon, lat = np.meshgrid(lon_input, lat_input)

    # Convert 2D array from 1D array for z value
    topo = np.reshape(z, (lat_num, lon_num))

    
    # Skip the data for resolution
    if ((resolution < spacing[0]) | (resolution < spacing[1])):
        print('Set the highest resolution')
    else:
        skip = int(resolution/spacing[0])
        lon = lon[::skip,::skip]
        lat = lat[::skip,::skip]
        topo = topo[::skip,::skip]

        
    topo = topo[::-1]
    
    
    # Select the range of map
    range1 = np.where((lon>=lon_area[0]) & (lon<=lon_area[1]))
    lon = lon[range1]; lat = lat[range1]; topo = topo[range1]
    range2 = np.where((lat>=lat_area[0]) & (lat<=lat_area[1]))
    lon = lon[range2]; lat = lat[range2]; topo = topo[range2]

    
    # Convert 2D again
    lon_num = len(np.unique(lon))
    lat_num = len(np.unique(lat))
    lon = np.reshape(lon, (lat_num, lon_num))
    lat = np.reshape(lat, (lat_num, lon_num))
    topo = np.reshape(topo, (lat_num, lon_num))
    

    return lon, lat, topo


    


# In[ ]:





# In[ ]:




