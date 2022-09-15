# called by run_sim with 'sim.py %i %s \n' %(num,sysName)
# must be proceeded by exoMMR.read_data() and exoMMR.make_sim_file()
# establishes and integrates the simulation
# saves the end of simulation in './{sysName}/bin_files/end{ind}.bin'
# saves the simulationArchive, if requested in input file, as './bin_files/{sysName}-{ind}}.bin'
# saves centers and amplitudes of res angles as './{sysName}/res_cen/{sysName}_res_{ind}.txt'



def sim(ind, sysName, track_res = True, Nout = 2000):
	import numpy as np
	from astropy.table import Table
	import rebound
	import sys
	import pandas as pd
	from .tools import txtrow_to_sim
	from .analysis import make_rescen
	M_earth = 3.e-6
	toearth = 3.33e5

	pldatfile = './'+sysName+'/'+sysName+'-allsims.in'
	simparamfile = './'+sysName+'/'+sysName+'-rb.param'

	pl_data = Table.read(pldatfile,format='ascii',fast_reader=False)
	sim_param = pd.read_csv(simparamfile,sep='\t',header=None)
	sim_param = sim_param.iloc[0]
	units_arr = sim_param[2] #('days', 'AU', 'Msun')
	sim_integrator = sim_param[1] #'whfast'
	T1 = float(sim_param[7])*365.25
	dt = float(sim_param[8])
	myint = float(sim_param[9]) #interval for rebound to save points; ignored if savebin==0
	M_star = float(sim_param[5])
	numpl = int(sim_param[3])
	archive = int(sim_param[4])

	sim = txtrow_to_sim(pl_data[ind],numpl,M_star,dt,sim_integrator)
	binname = './bin_files/%s-%i.bin' %(sysName,ind)
	if archive: sim.automateSimulationArchive(binname,interval=myint,deletefile=True)

	try:
		sim.integrate(T1)

	except rebound.Encounter as error:
		print(error)
		ps = sim.particles
		print(f'time={sim.t}, dt = {sim.dt}')
		print(f'P=[{ps[1].P},{ps[2].P},{ps[3].P},{ps[4].P}]')
		print(f'e=[{ps[1].e},{ps[2].e},{ps[3].e},{ps[4].e}]')
		print(f'mass=[{ps[1].m*toearth},{ps[2].m*toearth},{ps[3].m*toearth},{ps[4].m*toearth}]')
		exit()

	endsim = f'./{sysName}/bin_files/end{ind}.bin'
	sim.save(endsim)

	make_rescen(sysName,endsim,incb=False)
	print("integration complete")
