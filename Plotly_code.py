#!/usr/bin/env python
# coding: utf-8

# # The code list used in Plotly

# ## - Make Matplotlib color bar applicable in Plotly

# In[2]:


import matplotlib
from matplotlib import cm
import numpy as np

def matplotlib_to_plotly(cmap, pl_entries):
    h = 1.0/(pl_entries-1)
    pl_colorscale = []
    
    for k in range(pl_entries):
        C = list(map(np.uint8, np.array(cmap(k*h)[:3])*255))
        pl_colorscale.append([k*h, 'rgb'+str((C[0], C[1], C[2]))])
        
    return pl_colorscale

def MlibCscale_to_Plotly(cbar):
    cmap = matplotlib.cm.get_cmap(cbar)
    rgb = []
    norm = matplotlib.colors.Normalize(vmin=0, vmax=255)

    for i in range(0, 255):
        k = matplotlib.colors.colorConverter.to_rgb(cmap(norm(i)))
        rgb.append(k)

    Cscale = matplotlib_to_plotly(cmap, 255)
    
    return Cscale


# ## - Make my own color scale for Plotly

# In[3]:


def Colorscale_Plotly(name):
    
    if (name == "topo"):
        Cscale = [[0, 'rgb(0,0,70)'], [0.2, 'rgb(0,90,150)'], [0.4, 'rgb(150,180,230)'], [0.5, 'rgb(210,230,250)'], [0.50001, 'rgb(0,120,0)'], [0.57, 'rgb(220,180,130)'], [0.65, 'rgb(120,100,0)'], [0.75, 'rgb(80,70,0)'], [0.9, 'rgb(200,200,200)'], [1.0, 'rgb(255,255,255)']]  
        
    elif (name == "gray"):
        Cscale = [[0, 'rgb(60,60,60)'], [1.0, 'rgb(200,200,200)']]


    return Cscale
        


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




