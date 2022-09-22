import numpy as np
import rebound
from datetime import datetime
import matplotlib
matplotlib.use('Agg',force=True)
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from .tools import nearestRes,wrap180,wrap
from .myplot import set_pp

def plotRes(sm, addlines=True,incb=0,**kwargs):
    # Function: plots the resonant angles of simulation
    # Input: rebound simulation; bool for adding lines at center and amp of angles; bool for including
	# alternative two-body angles (with other planet's pomega)
    # Optional inputs: savename = resulting image file name, default is time of creation + .pdf;
    # time = how much to integrate the simulation before plotting, default is 0;
    # ntick = how many tick marks on x- and y-axes, default is 5
    # Output: None
	sim = sm.copy()
	mylw=2
	mycol = set_pp()

	if "savename" not in kwargs:
		now = datetime.now()
		mysave = now.strftime("%b-%d-%y-%H:%M:%S")
		kwargs["savename"] = mysave+'.pdf'
	savename = kwargs["savename"]

	if "time" not in kwargs:
		kwargs["time"] = 0.0 #do not move the simulation forward before plot
	int_time = kwargs["time"]

	if "ntick" not in kwargs:
		kwargs["ntick"] = 5
	ntick = kwargs["ntick"]

	sim_time=sim.t+int_time #from kwargs, 0 if not given
	sim.integrate(sim_time)
	Nout=2000
	ps = sim.particles
	numpl = len(ps)-1
	if numpl < 3: print('Warning: simulation contains fewer than 3 planets')

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

	ps=sim.particles

	maxe(ps,numpl)

	allper=np.array([ps[pl+1].P for pl in range(numpl)])
	res_ang = np.zeros([(numpl*2-3),Nout])
	phi_bs=np.array([])
	if incb: phi_bs = np.zeros([(numpl-1),Nout])
	ang_names = []
	b_names=[]
	thbr_names = ['lap1','lap2','lap3','lap4','lap5']
	twbr_names = ['phi12','phi23','phi34','phi45','phi56','phi67']

	res = nearestRes(allper)

	for kk in range(numpl-1):
		phi = res[kk,0]*l[(kk+1),:] + res[kk,1]*l[kk,:] + res[kk,2]*po[kk,:]
		phib = res[kk,0]*l[(kk+1),:] + res[kk,1]*l[kk,:] + res[kk,2]*po[(kk+1),:]
		res_ang[kk,:] = wrap180(phi)
		if incb: phi_bs[kk,:] = wrap180(phib)
		ang_names = ang_names + [twbr_names[kk]]
		b_names+=[twbr_names[kk]+'b']
	for kk in range(numpl-2):
		lap = res[kk+numpl-1,0]*l[(kk+2),:] + res[kk+numpl-1,1]*l[(kk+1),:] + res[kk+numpl-1,2]*l[kk,:]
		res_ang[(kk+numpl-1),:] = wrap(lap)
		ang_names = ang_names + [thbr_names[kk]]

	time = (sim_time+int_time)/365.25

	numres = np.shape(res_ang)[0]+len(phi_bs)
	fig,axs = plt.subplots(numres,sharex=True,figsize=(8,12))
	fig.subplots_adjust(wspace=0, hspace=0,left=0.2,bottom=0.11,top=0.95)
	i = 0

	if incb:
		res_ang = np.concatenate(res_ang,phi_bs)
		ang_names +=b_names

	for ax in axs:
		#ax.plot(time,res_ang[i,:], mycol[i+1],marker='.',ls='')
		ax.plot(time,res_ang[i,:], 'k',marker='.',ls='')
		ax.set_ylabel(ang_names[i],weight='bold')
		if addlines:
			ax.plot(time,np.ones(len(time))*np.median(res_ang[i,:]),'b-',lw=2)
			ax.plot(time,np.ones(len(time))*(np.median(res_ang[i,:])+2*np.std(res_ang[i,:])),'b--',lw=1)
			ax.plot(time,np.ones(len(time))*(np.median(res_ang[i,:])-2*np.std(res_ang[i,:])),'b--',lw=1)
		i=i+1
	axs[-1].set_xlabel('Time (yr)',weight='bold')

	for ax in axs:
		ax.set_yticks([-180,0,180])
		ax.set_yticks(np.linspace(-180,180,9),minor=True)
		ax.set_yticklabels(['']*9,minor=True)
		ax.set(ylim=[-180,180])
		ax.label_outer()

	for ii in range(numpl-2):
		axs[numpl-1+ii].set_yticks([0,180,360])
		axs[numpl-1+ii].set_yticks(np.linspace(0,360,9),minor=True)
		axs[numpl-1+ii].set_yticklabels(['']*9,minor=True)
		axs[numpl-1+ii].set(ylim=[0,360])

	if incb:
		for ii in range(numpl-1):
			axs[2*numpl-3+ii].set_yticks([0,180,360])
			axs[2*numpl-3+ii].set_yticks(np.linspace(0,360,9),minor=True)
			axs[2*numpl-3+ii].set_yticklabels(['']*9,minor=True)
			axs[2*numpl-3+ii].set(ylim=[0,360])

	pp = PdfPages(savename)
	pp.savefig(fig)
	pp.close()
	plt.close('all')
	plt.cla()
	plt.clf()
	print("plot saved as %s" %savename)
