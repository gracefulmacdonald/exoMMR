import numpy as np

def read_data(filename):
    # Function: Sorts info from input file into different arrays
    # Input: name of input file (created with exoMMR.make_input())
    # Output: returns three arrays to be passed to make_sim_file()
    import pandas as pd

    c = 0
    for line in open(filename):
        li = line.strip()
        if not li.startswith('#'):
            if c==1:
                numpl = int(line.rstrip())
                break
            elif c==0: c=1

    col_heads=[*(f'pl{ii}' for ii in range(1, (numpl+1)))]
    data = pd.read_csv(filename,sep='\t',header=None,comment = '#',names = col_heads)
    sys_name = data.iloc[0,0] #string
    numpl = int(data.iloc[1,0]) #int
    m_s = float(data.iloc[2,0]) #float
    units = str(data.iloc[3,0:3]) #this keeps the keys for some reason
    units = ['days','Msun','AU']
    pl_data = np.array(data.iloc[4:14,:], dtype = float) #2D array, float
    numSim = int(data.iloc[14,0]) #int
    int_time = float(data.iloc[15,0]) #float
    dt = float(data.iloc[16,0]) #float
    savebin = int(data.iloc[17,0]) #bool
    integrator = data.iloc[18,0] #string
    archive_step = float(data.iloc[19,0]) #float

    sim_par_str = [sys_name,integrator,units]
    sim_par = np.array([numpl,savebin,m_s,numSim,int_time,dt,archive_step])

    return sim_par_str, sim_par, pl_data

def make_sim_file(dir, sim_par_str, sim_par, pl_data):
    # Function: creates two files, one with all system parameters for all simulations
    # and another with all rebound parameters
    # Input: path of system directory; three arrays from read_data
    # Output: None returned, but two files saved as '{sysName}/{sysName}-allsims.in'
    # and '{sysName}/{sysName}-rb.param'
    from astropy.table import Table
    from numpy.random import seed, uniform
    import rebound
    from .tools import drawnormal
    from os.path import exists

    filename = dir+'/'+sim_par_str[0]+'-allsims.in'
    filename2 = dir+'/'+sim_par_str[0]+'-rb.param'

    if exists(dir+'/../slurm.in') != True:
        fs = open(dir+'/../slurm.in','w')
        fs.write('#SBATCH --ntasks=1\n')
        fs.write('#SBATCH --nodes=1\n')
        fs.write('#SBATCH --constraint="skylake|broadwell"\n')
        fs.write('#SBATCH --partition=normal\n')
        fs.write('#SBATCH --time=4:00:00\n')
        fs.close()

    #7/29: do I add units to row 1?
    numpl = int(sim_par[0])
    nCol = 1+numpl*6
    nRow = int(sim_par[3]) #numSim
    data = np.zeros([nRow,nCol])

    for ii in range(nRow):
    	seed(ii)
    	m = drawnormal(pl_data[0,:],pl_data[1,:])
    	p = drawnormal(pl_data[2,:],pl_data[3,:])
    	inc = drawnormal(pl_data[4,:],pl_data[5,:])
    	ecc = drawnormal(pl_data[6,:],pl_data[7,:])
    	ecc = [0.0 if x < 0 else x for x in ecc]
    	Om = list(uniform(0, 2*np.pi,numpl))
    	mLon = list(np.mod(-2*np.pi+pl_data[8,:]/p,2*np.pi))
    	data[ii,:] = np.array([ii]+m+p+inc+ecc+Om+mLon)

    col_heads=['ind',*(f'm{ii}' for ii in range(1, (numpl+1))),*(f'p{ii}' for ii in range(1, (numpl+1))),*(f'inc{ii}' for ii in range(1, (numpl+1))),*(f'ecc{ii}' for ii in range(1, (numpl+1))),*(f'Om{ii}' for ii in range(1, (numpl+1))),*(f'mLon{ii}' for ii in range(1, (numpl+1)))]
    wdata = Table(data,names=col_heads)
    wdata.write(filename, format='ascii', overwrite=True)

    f = open(filename2,'w')
    for t in sim_par_str: f.write('{}\t'.format(t))
    for t in sim_par: f.write('{}\t'.format(str(t)))
    f.close()

def sim(ind, sysName, track_res = True, Nout = 2000):
	import numpy as np
	from astropy.table import Table
	import rebound
	import sys
	import pandas as pd
	from .tools import txtrow_to_sim
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

def run_sim(num,sysName,wd):
	# Function: submits SLURM job to run rebound simulation
	# Input: simulation number/ID, system name, working directory
	# Output: None returned, but creates and executes SLURM file
	import os

	sysDir = wd+'/'+sysName
	jobDir = '/job_files'
	fn = sysDir+'/'+sysName+'-allsims.in' #wd/sysName/sysName-allsims.in
	fn2 = sysDir+'/'+sysName+'-rb.param'
	fn3 = wd+'/'+'slurm.in'

	f3 = open(fn3,"r")
	lines = f3.readlines()
	f3.close()

	os.system('echo %i' % num)
	f=open('%s%s/submit%i.sh' %(sysDir,jobDir,num),'w')
	f.write('#!/bin/bash \n')
	f.write('\n')
	f.write('#SBATCH --chdir=%s/ \n' %wd)
	#f.write('#SBATCH --chdir=../../ \n' )
	f.write('#SBATCH --job-name=res%i \n' %num)
	f.write('#SBATCH --output=%s%s/job.%i.out  \n' %(sysDir,jobDir,num))
	for l in lines:
		f.write(l)
	#f.write('#SBATCH --ntasks=1   \n')
	#f.write('#SBATCH --nodes=1  \n')
	#f.write('#SBATCH --constraint="skylake|broadwell"  \n')
	#f.write('#SBATCH --partition=normal  \n')
	#f.write('#SBATCH --time=4:00:00  \n')
	f.write('\n')
	f.write('module load python \n')
	f.write('\n')
	f.write('echo \"Starting on \"`date` \n')
	#f.write('python sim.py %i %s \n' %(num,sysName))
	f.write('python -c \"import exoMMR; exoMMR.sim(%i,\'%s\')\" \n' %(num,sysName))

	f.write('echo \"Finished on \"`date` \n')
	f.close()

	cmdstr = 'sbatch %s%s/submit%i.sh' %(sysDir,jobDir,num)
	os.system(cmdstr)

def combine_res(sysName):
	# Function: combines amp/cen res files from all simulations into one file;
	# must be proceeded by either track_res = True in exoMMR.sim() or by exoMMR.make_rescen()
	# Input: system name
	# Output: None returned, but creates file './{sysName}/{sysName}_res_cen.dat'
	import os

	comb_file = f'./{sysName}/{sysName}_res_cen.dat'
	f = open(comb_file,'w')
	ii=0

	for file in os.listdir(f'./{sysName}/res_cen/'):
		if file.endswith(".txt"):
			#read file and put stuff into other doc
			ii=ii+1
			if ii%25==0: print(ii)
			newfile = file
			f1 = open(f'./{sysName}/res_cen/'+newfile,'r')
			if ii==1:
				temp_string = f1.read()
				f.write(temp_string)
				f.write('\n')
			else:
				temp_string = f1.readlines()
				f.write(temp_string[2])
				f.write('\n')
			f1.close()
	f.close()

def calc_res(sysName):
    # Function: performs statistics on number of simulations with librating angles
    # must be proceeded by combine_res()
    # Input: system name
    # Output: None returned, but creates two files.
    # './{sysName}/{sysName}_librating.dat' contains flags for which angle librated
    # in which simulation; './{sysName}/{sysName}_stats.dat' contains centers and
    # amplitudes for librating angles
    from astropy.table import Table

    comb_file = f'./{sysName}/{sysName}_res_cen.dat'
    lib_file = f'./{sysName}/{sysName}_librating.dat'
    stats_file = f'./{sysName}/{sysName}_stats.dat'

    data = Table.read(comb_file,format='ascii')
    names = data.colnames
    a =  len(data)#a is rows, b is columns
    b = len(names)
    num_angles = int((b-1)/2)
    names = np.array([x[:-2] for x in names])
    name = names[np.array([((ii*2)+1) for ii in range(num_angles)])]

    amp_max = 150.0

    lib = np.zeros([a,(num_angles+1)])
    lib[:,0] = np.array(data[0][:])

    f1 = open(stats_file,'w')
    for ii in range(num_angles):
        this_cen = data.columns[((ii*2)+1)]
        this_amp = data.columns[((ii*2)+2)]
        this_amp = np.array([200 if x==0 else x for x in this_amp])
        lib[:,(ii+1)] = np.array([1 if x<amp_max else 0 for x in this_amp])
        try:
            c1,c2,c3 = np.percentile(this_cen[this_amp<amp_max],[16,50,84])
            a1,a2,a3 = np.percentile(this_amp[this_amp<amp_max],[16,50,84])
            f1.write(f'{name[ii]}_c\t{c2}\t{c3-c2}\t{c2-c1}\t{(sum(lib[:,(ii+1)])/a)*100.0:.2f}%\n')
            f1.write(f'{name[ii]}_a\t{a2}\t{a3-a2}\t{a2-a1}\t{(sum(lib[:,(ii+1)])/a)*100.0:.2f}%\n')
        except:
            f1.write(f'{name[ii]}_c\tNone\tNone\tNone\t0.00%\n')
            f1.write(f'{name[ii]}_a\tNone\tNone\tNone\t0.00%\n')
    f1.close()

    wdata = Table(lib,names=np.append(['index'],name))
    wdata.write(lib_file, format='ascii', overwrite=True)

def char_res(sysName,incb=False):
	# Function: characterizes the resonance chain(s) in system. Must be proceeded
	# by calc_res(); a chain requires 3+ planets
	# Input: system name; bool for if both two-body angle are included in *librating.dat
	# Output: None returned, but creates './{sysName}/{sysName}_char-res.dat' with
	# characterization information
	from itertools import groupby
	from astropy.table import Table
	from .tools import getCol

	lib_file = f'./{sysName}/{sysName}_librating.dat'
	char_file = f'./{sysName}/{sysName}_char-res.dat'

	data = Table.read(lib_file, format = 'ascii')
	aDat = np.lib.recfunctions.structured_to_unstructured(data.as_array())
	N,b = np.shape(aDat)
	index = aDat[0,:]
	numpl = int((b+2)/2)
	if incb: numpl = int((b+3)/3)
	if numpl<3:
		print("This function requires at least three planets")
		exit()

	part_chain = np.ones([N,numpl])*300 #is the planet part of the chain in this simulation
	col = getCol(numpl)

	if incb:
		for ii in range(numpl-1):
			aDat[:,1+ii] = [max(aDat[q,1+ii], aDat[q,2*numpl-2+ii] ) for q in range(N)]

	for pl in range(numpl):
		part_chain[:,pl] = [1 if(sum(aDat[x,col[pl]])>0) else 0 for x in range(N)]

	max_chain = np.zeros(N)

	for x in range(N):
		if sum(part_chain[x,:])>1:
			max_chain[x] = max(len(list(g)) for k,g in groupby(part_chain[x,:]))

	for ii,m in enumerate(max_chain):
		if m == numpl:
			if (sum(aDat[ii,1:numpl]) < (numpl-1)) & (sum(aDat[ii,numpl:])==0): #if not all two-body angles librate AND no three-body angles librate
				max_chain[ii] = 2

	f = open(char_file,'w')
	for pl in range(numpl):
		f.write(f'Percentage of stable simulations where planet {pl+1} is in res chain: {100*sum(part_chain[:,pl])/N}%\n')
	for c in range(1,numpl):
		f.write(f'Percentage of stable simulations with at least a {c+1}-body resonance: {100*len(max_chain[(max_chain>(c))])/N }%\n')

	f.close()

	all_data = np.zeros([N,(b+1+numpl)])
	header = data.colnames
	header += [f'pl{pl+1}' for pl in range(numpl)]
	header += ['len_chain']

	all_data[:,:b] = aDat
	all_data[:,b:(b+numpl)] = part_chain
	all_data[:,-1] = max_chain

	wdata = Table(all_data,names=header)
	allfile = f'./{sysName}/{sysName}_all.dat'
	wdata.write(allfile, format='ascii', overwrite=True)

def make_input(datfile,KOI,col_name='KOI'):
	# Function: creates the input file for read_data() from a csv; user may also
	# update an example input file by hand
	# Input: system number, either KOI or TOI
	# Output: None returned, but creates '{sysName}.in' for read_data()
	import pandas as pd
	from .tools import mr

	#datfile = './data/koi_err.csv'
	#datfile = './data/k2_err.csv'
	#datfile = './data/toi_err.csv'
	dat = pd.read_csv(datfile,sep=',')
	nrows = dat.shape[0]
	rows = np.array([int(x) for x in np.linspace(0,nrows,nrows)])
	all_KOI = np.array(dat[col_name].values.tolist())
	myrows = rows[(all_KOI==KOI)]

	#sysName = f'KOI-{KOI}'
	#sysName = f'K2-{KOI}'
	sysName = f'{col_name}-{KOI}'
	numpl = len(myrows)

	mass = np.array(dat['mass'][myrows])
	m_err = np.array(dat['m_err'][myrows])
	period = np.array(dat['period'][myrows])
	p_err = np.array(dat['p_err'][myrows])
	inc = np.array(dat['inc'][myrows])
	i_err = np.array(dat['i_err'][myrows])
	ecc = np.array(dat['ecc'][myrows])
	e_err = np.array(dat['e_err'][myrows])
	t0 = np.array(dat['t0'][myrows])
	t0_err = np.array(dat['t0_err'][myrows])
	radius = np.array(dat['radius'][myrows])
	r_err = np.array(dat['r_err'][myrows])
	Mstar = np.array(dat['st_mass'][myrows])[0]

	sort = np.argsort(period)
	mass = mass[sort]
	m_err = m_err[sort]
	period = period[sort]
	p_err = p_err[sort]
	inc = inc[sort]
	i_err = i_err[sort]
	ecc = ecc[sort]
	e_err = e_err[sort]
	t0 = t0[sort]
	t0_err = t0_err[sort]
	radius = radius[sort]
	r_err = r_err[sort]

	for m in range(numpl):
		if np.isnan(mass[m]):
			mass[m]=mr(radius[m],radius=True)
			m_err[m]=mass[m]*0.3

	'''
	# KOI host mass in separate file
	dat2 = pd.read_csv('./data/mstar_koi.csv',sep=',') #KOI,Mstar
	mstars = np.array(dat2['Mstar'])
	mKOI = np.array(dat2['KOI'])
	Mstar = float(mstars[(mKOI==KOI)])
	'''

	num = 500
	T1 = 1e6
	tstep = 0.05
	savebin = 0
	integrator = 'whfast'
	arch_step = 1e4

	filename = f'{sysName}.in'
	f = open(filename,'w')
	f.write( '# System name\n{}\n# Number of planets\n{}\n# Stellar mass\n{}\n# Units\ndays\tMsun\tAU\n# Planet masses\n'.format(sysName,numpl,Mstar) )
	for x in mass: f.write('{}\t'.format(x))
	f.write('\n# Mass error\n')
	for x in m_err: f.write('{}\t'.format(x))
	f.write('\n# Periods\n')
	for x in period: f.write('{}\t'.format(x))
	f.write('\n# Period error\n')
	for x in p_err: f.write('{}\t'.format(x))
	f.write('\n# Inc\n')
	for x in inc: f.write('{}\t'.format(x))
	f.write('\n# Inc error\n')
	for x in i_err: f.write('{}\t'.format(x))
	f.write('\n# Ecc\n')
	for x in ecc: f.write('{}\t'.format(x))
	f.write('\n# Ecc error\n')
	for x in e_err: f.write('{}\t'.format(x))
	f.write('\n# t0\n')
	for x in t0: f.write('{}\t'.format(x))
	f.write('\n# t0 error #unused\n')
	for x in t0_err: f.write('{}\t'.format(x))
	f.write('\n# Number of simulations\n{}\n# Integration time\n{}\n# Time step percentage\n{}\n'.format(num,T1,tstep))
	f.write('# Flag for saving bin file\n{}\n'.format(savebin))
	f.write('# Rarely changed parameters:\n# Integrator\n{}\n# Sim-archive step\n{}\n'.format(integrator,arch_step))
	f.close()

def make_rescen(sysName,binname,incb=False):
	# Function: creates res file with centers and amplitudes for a simulation
	# Input: system name; string for the simulationArchive bin file; bool to include
	# additional two-body MMR (other pomega)
	# Output: None returned, but creates './{sysName}/res_cen/{sysName}_res_{ind}.txt'
	from re import findall
	import rebound
	from .tools import maxe, nearestRes, wrap, wrap180

	ind = int(findall(r'[+]?[\d]+(?:,\d\d\d)*(?:[eE][-+]?\d+)?',binname)[-1])
	try:
		sm = rebound.Simulation(binname)
	except:
		print(f'file {binname} does not exist')
		return
	sim = sm.copy()
	sim_time=sim.t #from kwargs, 0 if not given

	Nout=2000
	ps = sim.particles
	numpl = len(ps)-1
	maxe(ps,numpl)
	l = np.zeros([numpl,Nout]); po = np.zeros([numpl, Nout]); p = np.zeros([numpl, Nout]); e = np.zeros([numpl, Nout])
	int_time = np.linspace(0,6000,Nout)

	for ii,t in enumerate(int_time):
		sim.integrate(sim_time+t)
		for pl in range(numpl):
			l[pl,ii] = sim.particles[pl+1].l
			po[pl,ii] = sim.particles[pl+1].pomega
			p[pl,ii] = sim.particles[pl+1].P
			e[pl,ii] = sim.particles[pl+1].e
	ps = sim.particles
	maxe(ps,numpl)

	allper = np.array([ps[pl+1].P for pl in range(numpl)])
	res_ang = np.zeros([(numpl*2-3),Nout])
	if incb: phi_bs = np.zeros([(numpl-1),Nout])
	thbr_names = ['lap1','lap2','lap3','lap4','lap5']
	twbr_names = ['phi12','phi23','phi34','phi45','phi56','phi67']
	ang_names = []

	res = nearestRes(allper)

	for kk in range(numpl-1):
		phi = res[kk,0]*l[(kk+1),:] + res[kk,1]*l[kk,:] + res[kk,2]*po[kk,:]
		phib = res[kk,0]*l[(kk+1),:] + res[kk,1]*l[kk,:] + res[kk,2]*po[(kk+1),:]
		res_ang[kk,:] = wrap(phi)
		if incb: phi_bs[kk,:] = wrap(phib)
		c_name = twbr_names[kk]+'_c'
		a_name = twbr_names[kk]+'_a'
		ang_names = ang_names + [c_name,a_name]
	for kk in range(numpl-2):
		lap = res[kk+numpl-1,0]*l[(kk+2),:] + res[kk+numpl-1,1]*l[(kk+1),:] + res[kk+numpl-1,2]*l[kk,:]
		res_ang[(kk+numpl-1),:] = wrap180(lap)
		c_name = thbr_names[kk]+'_c'
		a_name = thbr_names[kk]+'_a'
		ang_names = ang_names + [c_name,a_name]

	txtfile = f'./{sysName}/res_cen/{sysName}_res_{ind}.txt'

	f = open(txtfile,'w')
	f.write("# index\t")
	for ii in range(len(ang_names)):
		f.write(ang_names[ii]+'\t')
	if incb: f.write('phi12b_c\tphi12b_a\tphi23b_c\tphi23b_a\n')
	else: f.write('\n')
	f.write("# angles = ")
	for ii in range(len(res_ang)):
		f.write(f'[{res[ii,0]},{res[ii,1]},{res[ii,2]}]\t')

	f.write('\n'+str(ind))

	for ii in range(len(res_ang)):
		this_ang = res_ang[ii,:]
		center = np.median(this_ang)
		amp = 2*np.std(this_ang)
		f.write('\t'+str(center)+'\t'+str(amp))

	if incb:
		for ii in range(numpl-1):
			this_ang = phi_bs[ii,:]
			center = np.median(this_ang)
			amp = 2*np.std(this_ang)
			f.write('\t'+str(center)+'\t'+str(amp))
	f.close()

def constrain_prop(sysName):
	# Function: calls exoMMR.compare() to constrain mass of each planet
	# based on librating angles
	# Input: system name; must be proceeded by calc_res()
	# Output: None returned
	from .tools import getCol, compare

	filename1 = f'./{sysName}/{sysName}-allsims.in'
	filename2 = f'./{sysName}/{sysName}_librating.dat'
	with open(filename2) as f:
		ang_names = np.array(f.readline().rstrip().split(' '))

	dat1 = np.loadtxt(filename1,skiprows=1)
	dat2 = np.loadtxt(filename2,skiprows=1)
	numpl = int((np.shape(dat1)[1]-1)/6)

	index = dat2[:,0]
	index = np.array([int(x) for x in index])
	col = getCol(numpl)
	for pl in range(numpl):
		mass = dat1[index,(pl+1)]
		ang = dat2[:,col[pl]]
		anames = ang_names[col[pl]]
		for ii in range(len(col[pl])):
			print(f'planet {pl+1}: {anames[ii]}')
			compare(mass,ang[:,ii])
