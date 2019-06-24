#!/usr/bin/env python
# coding: utf-8

# # 3D seismicity Plot using Plotly

# ##  Ryota Kiuchi
# https://sites.google.com/view/rkiuchi/home
# 
# https://www.linkedin.com/in/ryota-kiuchi-b121819b/

# #### Examples
# #### - 2018 Osaka-Hokubu earthquake (M 6.1) aftershock activity
# #### - 2019 Yamagata-oki earthquake (M 6.7) aftershock activity
# #### made by R.Kiuchi June, 2018

# ### Purpose: 
# 
# Visualise aftershock distribution to capture clusters, fault planes, and spatio-temporal evolution

# In[2]:


from plotly.offline import iplot, init_notebook_mode, plot
import plotly.plotly as py
import plotly.graph_objs as go
import cufflinks as cf
import pandas as pd
import numpy as np 
import glob
import os

import seaborn as sns

import Plotly_code as Pcode
from importlib import reload

init_notebook_mode(connected=True) 
cf.set_config_file(offline=True, theme='pearl')

homedir = os.getcwd()
outdir = homedir + '/results' 


# # 2019 Yamagata-oki EQ (M 6.7)

# ## - Import the data file for earthquakes

# In[4]:


# Setting the directory
datadir = homedir + '/data/2019Yamagata-oki'

file_list = glob.glob(datadir + "/HinetHypo.csv")
for i in range(len(file_list)):
    eachdata = pd.read_csv(file_list[i])
    if (i == 0):
        data = eachdata
    else:
        data = pd.concat([data, eachdata])

    
### If you download JMA Unified Catalog
# Otherwise skip these part
# Data fill NaN for empty cells
data.replace('', np.nan, inplace=True)
data['Mag'] = data['Mag'].str.strip('v')
data['Mag'] = data['Mag'].str.strip('V')
data['Mag'] = data['Mag'].str.strip('D')
data['Mag'] = data['Mag'].astype(float)
### If you download JMA Unified Catalog 


# Data selection
#data = data[data.Depth > 9] 

'''
data = data[data.Depth <= 30]        
data = data[data.Evla >= 34.7]
data = data[data.Evla < 35.0]
data = data[data.Evlo >= 135.45]
data = data[data.Evlo < 135.75]
'''
data['Mag'] = data['Mag'].str.strip('v')
data['Mag'] = data['Mag'].str.strip('V')
data['Mag'] = data['Mag'].str.strip('D')
data['Mag'] = data['Mag'].astype(float)
data = data[data.Mag >= 1.0]

# Change format to datetime for event date
# from Hi-net data source
data.DateTime=pd.to_datetime(data.DateTime, format='%Y-%m-%d %H:%M:%S')

date = np.array(data['DateTime'])
evlon = np.array(data['Evlo'])
evlat = np.array(data['Evla'])
evDepth = np.array(data['Depth'])
evMag = np.array(data['Mag'])


# Calculate time difference from Foreshock and Mainshock
MshockTime = date[np.where(evMag == 6.7)]
Timefrom_Mshock = (date - MshockTime)/np.timedelta64(1,'h')

### Data filtering
# For time from mainshock
# Hours before and after mainshock
Before = 0
After = 48
date = date[np.where(Timefrom_Mshock >= Before)]
evlon = evlon[np.where(Timefrom_Mshock >= Before)]
evlat = evlat[np.where(Timefrom_Mshock >= Before)]
evMag = evMag[np.where(Timefrom_Mshock >= Before)]
evDepth = evDepth[np.where(Timefrom_Mshock >= Before)]
Timefrom_Mshock = Timefrom_Mshock[np.where(Timefrom_Mshock >= Before)]

date = date[np.where(Timefrom_Mshock <= After)]
evlon = evlon[np.where(Timefrom_Mshock <= After)]
evlat = evlat[np.where(Timefrom_Mshock <= After)]
evMag = evMag[np.where(Timefrom_Mshock <= After)]
evDepth = evDepth[np.where(Timefrom_Mshock <= After)]
Timefrom_Mshock = Timefrom_Mshock[np.where(Timefrom_Mshock <= After)]



# ## Prepare 3D scatter map

# In[5]:


reload(Pcode)
init_notebook_mode(connected=True) 
cf.set_config_file(offline=True, theme='pearl')

#Convert color bar in Matplotlib
cbar = 'jet_r'
Cscale = Pcode.MlibCscale_to_Plotly(cbar)

# Set the region
lonmin = 139.
lonmax = 140.
lonbin = 0.25
latmin = 38.
latmax = 39.
latbin = 0.25
depmax = 30.
depmin = 0.


# Plot 3D
# Plot events in 3D
seis_3D = go.Scatter3d(x = evlon,
                      y = evlat,
                      z = evDepth,
                      mode='markers',
                      name='measured',
                      marker = dict(
                          size = 10.*evMag,
                          cmax = 50.,
                          cmin = 0.,
                          colorbar = dict(
                              title = 'Source Depth',
                              titleside = 'right',
                              tickmode = 'array',
                              ticks = 'outside',
                              ticktext=list(np.arange(0,60,10)),
                              tickvals=list(np.arange(0,60,10))
                          ),
                          ### choose color option
                          color = evDepth,
                          ### choose color option
                          colorscale = Cscale,
                          showscale = True,
                          opacity=1.0)
                      )


seis_3D = go.Scatter3d(x = evlon,
                      y = evlat,
                      z = evDepth,
                      mode = 'markers',
                      marker = dict(
                          size = 20*evMag,
                          cmax = 0.0,
                          cmin = -20,
                          colorbar = dict(
                              title = 'Source Depth',
                              titleside = 'right',
                              tickmode = 'array',
                              ticks = 'outside',
                              ticktext=list(np.arange(20,0,-5)),
                              tickvals=list(np.arange(-20,0,5))
                          ),
                          ### choose color option
                          color = evDepth,
                          ### choose color option
                          colorscale = Cscale,
                          showscale = True,
                          opacity=1.0))



# Plot 3D
seis_3D_time = go.Scatter3d(x = evlon,
                      y = evlat,
                      z = evDepth,
                      mode = 'markers',
                      marker = dict(
                          size = 10*evMag,
                          ### choose color option
                          color = Timefrom_Mshock,
                          ### choose color option
                          colorscale = Cscale,
                          showscale = True,
                          opacity=1.0))




# Plot 2D
z_offset=-100*np.ones(evDepth.shape)  # Plot at the bottom
seis_2D = go.Scatter3d(x = evlon,
                      y = evlat,
                      z = z_offset,
                      mode = 'markers',
                      marker = dict(
                          size = 2,
                          cmax = 0.0,
                          cmin = -290,
                          colorbar = dict(
                              title = 'Source Depth',
                              titleside = 'right',
                              tickmode = 'array',
                              ticks = 'outside',
                              ticktext=list(np.arange(20,0,-5)),
                              tickvals=list(np.arange(-20,0,5))
                          ),
                          ### choose color option
                          color = evDepth,
                          ### choose color option
                          colorscale = Cscale,
                          showscale = False,
                          opacity=1.0))
                    



# ## - Import Topography Data

# In[6]:


import ReadGeo
from importlib import reload
reload(ReadGeo)

# Import topography data
# Select the area you want
resolution = 0.001
lon_area = [lonmin-resolution, lonmax+resolution]
lat_area = [latmin-resolution, latmax+resolution]

# Get mesh-shape topography data
lon_topo, lat_topo, topo = ReadGeo.Etopo(lon_area, lat_area, resolution)



# In[7]:


# Input value
x = lon_topo
y = lat_topo
z = topo

# Import color scale
reload(Pcode)
name = "topo"
Ctopo = Pcode.Colorscale_Plotly(name)
cmin = -8000
cmax = 8000

topo3D = go.Surface(x=x,y=y,z=z,
                   colorscale=Ctopo, cmin=cmin, cmax=cmax)


# The position of z-axis
z_offset=depmax*np.ones(z.shape)  # Plot at the bottom
#z_offset=0*np.ones(z.shape)  # Plot at 0 level

topo_surf = go.Surface(z=z_offset, x=x, y=y,
                colorscale=Ctopo, cmin=cmin, cmax=cmax,
                showlegend=False,
                showscale=False,
                surfacecolor=topo,
                hoverinfo='text')


# ## - Draw 3D Map

# In[8]:


layout = go.Layout(
    autosize=False, width=1200, height=1200, 
    title = '2019 Yamagata-oki EQ Distribution',
    showlegend = False,
    scene = dict(
        xaxis = dict(title = 'Longitude', range=[lonmin, lonmax]),
        yaxis = dict(title = 'Latitude', range=[latmin, latmax]),
        zaxis = dict(title = 'Depth', range=[depmax, depmin]),
    aspectmode='manual',
    aspectratio=go.layout.scene.Aspectratio(
        x=3, y=3, z=1)),
    yaxis = dict(autorange='reversed'))

plot_data=[seis_3D_time, topo_surf]




fig = go.Figure(data=plot_data, layout=layout)
plot(fig, validate = False, filename=outdir+'/2019Yamagata_3DEQ.html', auto_open=True)


# # 2018 Northern Osaka EQ (M 6.1)

# ## - Import the data file for earthquakes

# In[9]:


###############----Read the Hi-net event catalog----###############
os.chdir(homedir)
file_list = glob.glob(homedir + "/data/2018Osaka/HinetHypo.csv")
for i in range(len(file_list)):
    eachdata = pd.read_csv(file_list[i])
    if (i == 0):
        data = eachdata
    else:
        data = pd.concat([data, eachdata])

        
# Data selection
#data = data[data.Depth > 9] 


data = data[data.Depth <= 30]        
data = data[data.Evla >= 34.7]
data = data[data.Evla < 35.0]
data = data[data.Evlo >= 135.45]
data = data[data.Evlo < 135.75]
data = data[data.Mag >= 1.0]

# Change format to datetime for event date
# from Hi-net data source
data.DateTime=pd.to_datetime(data.DateTime, format='%Y-%m-%d %H:%M:%S')

date = np.array(data['DateTime'])
evlon = np.array(data['Evlo'])
evlat = np.array(data['Evla'])
evDepth = np.array(data['Depth'])
evMag = np.array(data['Mag'])


# Calculate time difference from Foreshock and Mainshock
MshockTime = date[np.where(evMag == 6.2)]
Timefrom_Mshock = (date - MshockTime)/np.timedelta64(1,'h')

### Data filtering
# For time from mainshock
Before = 0
After = 48
date = date[np.where(Timefrom_Mshock >= Before)]
evlon = evlon[np.where(Timefrom_Mshock >= Before)]
evlat = evlat[np.where(Timefrom_Mshock >= Before)]
evMag = evMag[np.where(Timefrom_Mshock >= Before)]
evDepth = evDepth[np.where(Timefrom_Mshock >= Before)]
Timefrom_Mshock = Timefrom_Mshock[np.where(Timefrom_Mshock >= Before)]

date = date[np.where(Timefrom_Mshock <= After)]
evlon = evlon[np.where(Timefrom_Mshock <= After)]
evlat = evlat[np.where(Timefrom_Mshock <= After)]
evMag = evMag[np.where(Timefrom_Mshock <= After)]
evDepth = evDepth[np.where(Timefrom_Mshock <= After)]
Timefrom_Mshock = Timefrom_Mshock[np.where(Timefrom_Mshock <= After)]



# ## Prepare 3D scatter map

# In[10]:


reload(Pcode)
init_notebook_mode(connected=True) 
cf.set_config_file(offline=True, theme='pearl')

#Convert color bar in Matplotlib
cbar = 'jet_r'
Cscale = Pcode.MlibCscale_to_Plotly(cbar)

# Set the region
lonmin = 139.
lonmax = 140.
lonbin = 0.25
latmin = 38.
latmax = 39.
latbin = 0.25
depmax = 30.
depmin = 0.


# Plot 3D
# Plot events in 3D
seis_3D = go.Scatter3d(x = evlon,
                      y = evlat,
                      z = evDepth,
                      mode='markers',
                      name='measured',
                      marker = dict(
                          size = 10.*evMag,
                          cmax = 50.,
                          cmin = 0.,
                          colorbar = dict(
                              title = 'Source Depth',
                              titleside = 'right',
                              tickmode = 'array',
                              ticks = 'outside',
                              ticktext=list(np.arange(0,60,10)),
                              tickvals=list(np.arange(0,60,10))
                          ),
                          ### choose color option
                          color = evDepth,
                          ### choose color option
                          colorscale = Cscale,
                          showscale = True,
                          opacity=1.0)
                      )


seis_3D = go.Scatter3d(x = evlon,
                      y = evlat,
                      z = evDepth,
                      mode = 'markers',
                      marker = dict(
                          size = 20*evMag,
                          cmax = 0.0,
                          cmin = -20,
                          colorbar = dict(
                              title = 'Source Depth',
                              titleside = 'right',
                              tickmode = 'array',
                              ticks = 'outside',
                              ticktext=list(np.arange(20,0,-5)),
                              tickvals=list(np.arange(-20,0,5))
                          ),
                          ### choose color option
                          color = evDepth,
                          ### choose color option
                          colorscale = Cscale,
                          showscale = True,
                          opacity=1.0))



# Plot 3D
seis_3D_time = go.Scatter3d(x = evlon,
                      y = evlat,
                      z = evDepth,
                      mode = 'markers',
                      marker = dict(
                          size = 10*evMag,
                          ### choose color option
                          color = Timefrom_Mshock,
                          ### choose color option
                          colorscale = Cscale,
                          showscale = True,
                          opacity=1.0))




# Plot 2D
z_offset=-100*np.ones(evDepth.shape)  # Plot at the bottom
seis_2D = go.Scatter3d(x = evlon,
                      y = evlat,
                      z = z_offset,
                      mode = 'markers',
                      marker = dict(
                          size = 2,
                          cmax = 0.0,
                          cmin = -290,
                          colorbar = dict(
                              title = 'Source Depth',
                              titleside = 'right',
                              tickmode = 'array',
                              ticks = 'outside',
                              ticktext=list(np.arange(20,0,-5)),
                              tickvals=list(np.arange(-20,0,5))
                          ),
                          ### choose color option
                          color = evDepth,
                          ### choose color option
                          colorscale = Cscale,
                          showscale = False,
                          opacity=1.0))
                    




# ## - Draw 3-D seismicity map

# In[34]:


layout = go.Layout(
    autosize=False, width=1200, height=1200, 
    title = '2018 Osaka EQ Distribution',
    showlegend = False,
    scene = dict(
        xaxis = dict(title = 'Longitude'),
        yaxis = dict(title = 'Latitude'),
        zaxis = dict(title = 'Depth', )),
    yaxis = dict(autorange='reversed'))

plot_data=[seis_3D_time]


fig = go.Figure(data=plot_data, layout=layout)
plot(fig, validate = False, filename='results/3DEQDistribution.html', auto_open=True)


# In[ ]:




