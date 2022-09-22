def txtrow_to_sim(row,npl,Mstar,dt,integrator):
	# Funtion: sets up rebound Simulation from row of '{sysName}/{sysName}-allsims.in'
	# Input: system parameters from '{sysName}/{sysName}-allsims.in' for sim ID;
	# number of planets; mass of star; integration step size; integrator
	# Output: returns rebound Simulation to integrate
	import rebound
	import numpy as np

	sim=rebound.Simulation()
	sim.units=('Msun','days','AU') #hard coded for now
	#sim.units=(m_un,t_un,d_un)
	#Mstar = row['Mstar']
	sim.add(m=Mstar)
	for ii in range(1,npl+1):
	   m = row['m{}'.format(ii)]*3.0e-6 #convert from M_Earth to M_Sun
	   per = row['p{}'.format(ii)]
	   inc = (90-row['inc{}'.format(ii)])*np.pi/180.0
	   ecc = row['ecc{}'.format(ii)]
	   Omega = row['Om{}'.format(ii)]
	   mean_long = row['mLon{}'.format(ii)]
	   #pomega = row['pomega{}'.format(ii)]
	   sim.add(m=m,P=per,inc=inc,e=ecc,Omega=Omega,l=mean_long,hash=ii)
	sim.move_to_com()
	sim.integrator=integrator
	minP = min([sim.particles[x+1].P for x in range(npl)])
	sim.dt = minP*dt
	set_min_distance(sim,2)
	return sim

def set_min_distance(sim,rHillFactor):
    # Function: sets minimum distance allowed for rebound
    # Input: rebound simulation, number of hill radii as min
    # Output: None
    import numpy as np

    ps=sim.particles[1:]
    rHill = np.min([p.rhill for p in ps if p.m > 0])
    minDist = rHillFactor*rHill
    sim.exit_min_distance=minDist

def drawmass(value, mass=False, **kwargs):
    # Function: draws mass from normal distribution
    # Input: either mass or radius; bool for if passing mass or radius, default is
    # radius
    # Optional input: err =  uncertainty on mass distribution, default is 30%
    # Output: returns mass
    if mass: #value is the mass
        if "err" not in kwargs:
            kwargs["err"] = 0.3*value
        m_val = drawnormal(value,kwargs["err"])

    else: #value is the radius
        draw_mass = mr(value) #returns a radius
        if "err" not in kwargs:
            kwargs["err"] = 0.3*draw_mass
        m_val = drawnormal(draw_mass,kwargs["err"])
    return m_val

def mr(value, mass=False, radius=True):
    # Function: returns mass or radius from M-R relation (Bashi et al 2017)
    # Input: mass or radius; bool for passing mass, default is False; bool
    # Output: returns the radius or mass

    if mass: #return a radius for some reason
        if value < 124.0: return value**0.55
        else: return value**0.01

    if radius: #return a mass
        if value < 12.1: return value**(1.0/0.55)
        else: return value**100.0

def resProx(p1,p2,MMR):
    # Function: returns proximity to first-order resonance
    # Input: orbital period 1, orbital period 2, order
    # Output: P2/P1 - (MMR+1)/MMR  where MMR = 1 is for 2:1, 2 is 3:2, 3 is 4:3 etc

    if p1>p2:
        dummy = p1
        p1 = p2
        p2 = dummy
    prox = (p2/p1) - ((MMR+1)/MMR)
    return prox

def nearestRes(p):
    # Function: returns coefficients for two- and three-body resonant angles, based
    # on period ratios
    # Input: array of orbital periods
    # Output: returns coefficients of two- and three-body resonant angles; returns
    # array of zeros if period ratios are not near any strong resonances
    import numpy as np

    npl = len(p)
    n2br = npl-1
    n3br = npl-2
    res = np.zeros([(n2br+n3br),3])
    for i in range(n2br):
        this_ratio = p[i+1]/p[i]
        if this_ratio < 3.2 and this_ratio > 2.98 :
            res[i,:] = np.array([3,-1,-2])
        elif this_ratio < 2.7 and this_ratio > 2.48 :
            res[i,:] = np.array([5,-2,-3])
        elif this_ratio < 2.2 and this_ratio > 1.88 :
            res[i,:] = np.array([2,-1,-1])
        elif this_ratio < 1.73 and this_ratio > 1.66 :
            res[i,:] = np.array([5,-3,-2])
        elif this_ratio < 1.6 and this_ratio > 1.48 :
            res[i,:] = np.array([3,-2,-1])
        elif this_ratio < 1.48 and this_ratio > 1.3 :
            res[i,:] = np.array([4,-3,-1])
        elif this_ratio < 1.3 and this_ratio > 1.2 :
            res[i,:] = np.array([5,-4,-1])
        else: res[i,:] = np.array([0,0,0])

    for i in range(n3br):
        order= -1.0*np.array([res[i,2],res[i+1,2]])
        c1 = res[i+1,0]*order[0]
        c2 = res[i+1,1]*order[0]-res[i,0]*order[1]
        c3 = -1.0*res[i,1]*order[1]
        if np.mod(abs(c1),2) == 0 and np.mod(abs(c2),2) == 0 and np.mod(abs(c3),2) == 0:
            c1 = c1/2; c2 = c2/2; c3 = c3/2
        if np.mod(abs(c1),3) == 0 and np.mod(abs(c2),3) == 0 and np.mod(abs(c3),3) == 0:
            c1 = c1/3; c2 = c2/3; c3 = c3/3

        res[i+n2br,:] = [c1,c2,c3]
    if sum(sum(res)): print('Warning: no resonances likely')
    return res

def mad(array):
    # Function: calculates and returns the MAD of an array
    # Input: array of numerical values
    # Output: returns MAD
    import numpy as np

    this_med = np.median(array)
    return np.median([abs(a-this_med) for a in array])

def wrap(vals):
    # Function: wraps an angle between 0 and 360 degress
    # Input: array of values, in radians
    # Output: array of values, wrapped, in degrees
    import numpy as np

    for ii in range(len(vals)):
      while vals[ii] < 0.0:
      	vals[ii] += 2.0*np.pi
      while vals[ii] > 2.0*np.pi:
      	vals[ii] -=2.0*np.pi
    return vals*180.0/np.pi

def wrap180(vals):
    # Function: wraps an angles between -180 and 180 degrees
    # Input: array of values, in radians
    # Output: array of values, wrapped, in degrees
    import numpy as np

    for ii in range(len(vals)):
      while vals[ii] < -1.0*np.pi:
      	vals[ii] += 2.0*np.pi
      while vals[ii] > np.pi:
      	vals[ii] -=2.0*np.pi
    return vals*180.0/np.pi

def drawnormal(mean, sigma):
	# Function: draws value from a normal distribution
	# Input: mean and standard deviation; can be arrays
	# Output: returns draw from normal distribution, value must be positive
	from numpy.random import normal
	import numpy as np

	vals = np.zeros(len(mean))
	for ii in range(len(mean)):
		val = -1
		while val < 0:
			val = normal(mean[ii], sigma[ii])
		vals[ii] = val
	return list(vals)

def getCol(numpl):
    # Function: returns the column numbers required for characterizing resonances
    # called in calc_res() and constrainProp()
    # Input: number of planets
    # Output: returns array of column indices
    col={}
    if numpl == 3:
        col[0] = [1,3]
        col[1] = [1, 2, 3]
        col[2] = [2, 3]
    else:
        for pl in range(numpl):
            if pl==0: col[pl] = [1,numpl]
            elif pl==1: col[pl] = [1,2,numpl,numpl+1]
            elif pl==(numpl-2): col[pl] = [pl,pl+1, numpl-2+pl,numpl-1+pl]
            elif pl==(numpl-1): col[pl] = [pl,numpl-2+pl]
            else: col[pl] = [pl,pl+1,numpl-2+pl,numpl-1+pl,numpl+pl]
    return col

def clean_up(sysName, bin=0, res=1, job=1,submit=1):
	# Function: cleans up directories
	# Input: system name; bool for removing simulationArchive bin files, default is
	# False; bool for removing res amp/cen files, default is True so make sure to
	# run combine_res() first; bool for removing SLURM job files, default is
	# True; bool for removing submit files, default is True
	# Output: None returned, files deleted
	import os
	import os.path

	if bin:
		os.system(f'rm ./{sysName}/bin_files/*.bin')

	if res:
		if os.path.exists(f'./{sysName}/{sysName}_res_cen.dat'):
			os.system(f'rm ./{sysName}/res_cen/*.txt')

	if job:
		os.system(f'rm ./{sysName}/job_files/job*.out')

	if submit:
		os.system(f'rm ./{sysName}/job_files/*.sh')

def maxe(ps,numpl):
	# Function: exits python if any planet's ecc is greater than 1 (unbound)
	# Input: sim.particles of rebound Simulation; number of planets
	# Output: None returned
	import numpy as np

	maxecc = max(np.array([ps[pl+1].e for pl in range(numpl)]))
	if maxecc > 1.0 :
		print('Warning: Planet ejected in integration')
		return 1

def stats(x):
	# Function: calculates and returns statistics of an array
	# Input: array of values
	# Output: returns the median, upper uncertainty, and lower uncertainty of array
	import numpy as np
	m = np.median(x)
	u = np.percentile(x,84)-m
	d = m-np.percentile(x,16)
	return m,u,d

def compare(a,res):
	# Function: compares a parameter, split by a binary array
	# Input: distribution of one parameter; array of bool to separate populations
	# Output: None returned, but prints to screen the median +\- uncertainties of
	# each population as well as the p-val from a two-sample K-S test
	from scipy.stats import ks_2samp

	if((sum(res)==len(a)) | (sum(res)==0)):
		return
	st, pval = ks_2samp(a[(res==1)],a[(res==0)])
	if pval < 0.05:
		a1,a2,a3 = stats(a[(res==1)])
		b1,b2,b3 = stats(a[(res==0)])
		print(f'res: {a1:.2f}+{a2:.2f}-{a3:.2f}')
		print(f'non-res: {b1:.2f}+{b2:.2f}-{b3:.2f}')
		print(f'k-s p-vale = {pval}')
		print('**********************************')
