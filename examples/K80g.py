'''
This example shows how to use exoMMR to make a pretty heatmap.

The heatmap function requires three arrays: x data, y data, and an array
that will colour the heatmap. exoMMR.myplot.get_hm() will create bins
along each axis (lengths defined as the two numerical inputs) and will color
the box with the mean of the color data.

For this example, we use a file which contains the mass and eccentricity of
a planet from numerous simulations and an array titled "existence", where ex=0
means that the planet could not exist (simulation becomes unstable or the known
resonances are disrupted) and ex=1 means that the planet is not forbidden from
existing

'''

from astropy.table import Table
import matplotlib
matplotlib.use('Agg',force=True)
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
from exoMMR.myplot import set_pp, get_hm
import numpy as np

# change some defaults to be nicer
set_pp()
mys = 16

data = Table.read('mass_info.dat',format='ascii')
mass = data['mass']
ecc = data['eccentricity']
existence_array = data['existence']
hm_name = 'K80g-hm.png'

# create a fig and ax to send to get_hm
fig, ax = plt.subplots(figsize=(10,8))
xdat, ydat, zdat = mass[(ecc<0.1)], ecc[(ecc<0.1)], existence_array[(ecc<0.1)]
# returns handle for hm for nice colorbar manipulation
hm = get_hm(xdat,ydat,zdat,13,13,fig,ax,'hot')

# adjust other plot parts
ax.set_xlabel('Mass [$M_{\oplus}$]',fontsize=mys,weight='bold')
ax.set_ylabel('Eccentricity',fontsize=mys,weight='bold')
ax.set(xticks=np.linspace(0,1,6),yticks=np.linspace(0,0.1,6))
ax.tick_params(labelsize=mys)

# add a colorbar
cbar_ax = fig.add_axes([0.87,0.22,0.02,0.6]) # defines the position (bottom left corner xpos, ypos) then size (width, height) of the colorbar
cb = fig.colorbar(hm,cax = cbar_ax)
cb.ax.set_ylabel('% Allowed',fontsize=mys, weight='bold')
cb.ax.tick_params(labelsize=mys)
fig.subplots_adjust(wspace=0, hspace=0,left=0.14,bottom=0.09,top=0.97)

plt.savefig(hm_name)
