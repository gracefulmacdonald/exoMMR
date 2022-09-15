'''
This example shows how to create and run simulations with exoMMR

This example includes a target list of system numbers; such a set-up is only
required for use of exoMMR.make_input(), but the .in file for each system could
be made by hand or by a unique script.

'''

import numpy as np
from exoMMR import make_input, read_data, make_sim_file, run_sim
import os

# load in all targets and choose one
targets = np.loadtxt('TOI.targets')
t = 0

sysName = f'TOI-{int(targets[t])}'

# make_input requires the path to system data, the star/system's name, and the
# name of the column of star/system names in the system data file
datfile = './data/toi_err.csv'
make_input(datfile,int(targets[t]),col_name='TOI')

input_file = f'{sysName}.in'

### start here if .in file already exists
# read input file and returns data for simulations
sim_par_str, sim_par, pl_data = read_data(input_file)

# make directory for system with expected sub-directories
dir = os.getcwd()
sysDir = dir+'/'+ sim_par_str[0]
if os.path.isdir(sysDir) != True:
	os.mkdir(sysDir)
if os.path.isdir(sysDir+'/job_files') != True:
	os.mkdir(sysDir+'/job_files')
if os.path.isdir(sysDir+'/res_cen') != True:
	os.mkdir(sysDir+'/res_cen')
if os.path.isdir(sysDir+'/bin_files') != True:
	os.mkdir(sysDir+'/bin_files')

# create large txt file of simulation parameters, each row is each sim
make_sim_file(sysDir, sim_par_str, sim_par, pl_data)

# run all simulations as SLURM jobs
for ii in range(int(sim_par[3])): #numSim
	run_sim(ii,sim_par_str[0],dir)
