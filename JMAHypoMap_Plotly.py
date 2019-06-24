#!/usr/bin/env python
# coding: utf-8

# # 3D seismicity Plot using Plotly

# ##  Ryota Kiuchi
# https://sites.google.com/view/rkiuchi/home
# 
# https://www.linkedin.com/in/ryota-kiuchi-b121819b/

# #### Example - 2018 Osaka-Hokubu earthquake (M 6.1) aftershock activity
# #### made by R.Kiuchi June, 2018

# ### Purpose: 
# 
# Visualise aftershock distribution to capture clusters, fault planes, and spatio-temporal evolution

# In[1]:


from plotly.offline import iplot, init_notebook_mode, plot
import plotly.plotly as py
import plotly.graph_objs as go
import cufflinks as cf
import pandas as pd
import numpy as np 
import glob
import os

import Plotly_code as Pcode

init_notebook_mode(connected=True) 
cf.set_config_file(offline=True, theme='pearl')


# ## - Import the data file for earthquakes

# In[4]:


###############----Read the Hi-net event catalog----###############
file_list = glob.glob("data/2018Osaka/HinetHypo.csv")
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
# For time from foreshock
date = date[np.where(Timefrom_Mshock >= 0)]
evlon = evlon[np.where(Timefrom_Mshock >= 0)]
evlat = evlat[np.where(Timefrom_Mshock >= 0)]
evDepth = evDepth[np.where(Timefrom_Mshock >= 0)]
Timefrom_Mshock = Timefrom_Mshock[np.where(Timefrom_Mshock >= 0)]


evDepth = -1*evDepth


# ## Prepare 3D scatter mapcatter map

# In[5]:


#Convert color bar in Matplotlib
cbar = 'jet'
Cscale = Pcode.MlibCscale_to_Plotly(cbar)

# Plot 3D
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
                          size = 20*evMag,
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

# In[5]:


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




