'''
This example shows how to analyze the results from exoMMR simulations

Analysis such as this requires that the folder {sysName/res_cen} exists
and that it is populated with *txt files produced by exoMMR.make_rescen()

After char_res(), we graph the resonances; this requires that the folder
{sysName/bin_files} exists and that it is populated with *bin files
from rebound (created in exoMMR.sim(), but can be any rebound simulation)

'''

from exoMMR import combine_res, calc_res, char_res
from exoMMR.plotRes import plotRes
import rebound
import os

# define name of the system (as defined in first line of *in file)
sysName = 'KOI-500'

# analyze potential resonances in the system
combine_res(sysName)
calc_res(sysName)
char_res(sysName,incb=False)

# make some helpful graphs
if os.path.isdir(f'./{sysName}/graphs') != True:
	os.mkdir(f'./{sysName}/graphs')
for ii in range(15):
	try:
		sim = rebound.Simulation(f'./{sysName}/bin_files/end{ii}.bin')
	except:
		continue
	plotRes(sim,savename=f'./{sysName}/graphs/res{ii}.pdf')
