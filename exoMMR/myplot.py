import numpy as np
import matplotlib
matplotlib.use('Agg',force=True)
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

def set_pp(type='paper'):
    # Function:
    # Input: type of plot, default is paper, else is poster; user can implement
    # more types as necessary
    # Output: returns array of colorblind friendly colors
    if type=='paper':
        S_SIZE = 12
        L_SIZE = 14
        mylw=2

        plt.rc('font', size=L_SIZE,weight='bold')          # controls default text sizes
        #plt.rc('axes', titlesize=S_SIZE)     # fontsize of the axes title
        plt.rc('axes', labelsize=L_SIZE,linewidth=mylw)    # fontsize of the x and y labels
        plt.rc('xtick', labelsize=L_SIZE)    # fontsize of the tick labels
        plt.rc('ytick', labelsize=L_SIZE)    # fontsize of the tick labels
        #plt.rc('legend', fontsize=S_SIZE)    # legend fontsize
        #plt.rc('text',usetex=True)
        plt.rcParams['xtick.major.size']=6
        plt.rcParams['xtick.minor.size']=5
        plt.rcParams['ytick.major.size']=6
        plt.rcParams['ytick.minor.size']=5
        plt.rcParams['xtick.major.width']=1.7
        plt.rcParams['xtick.minor.width']=1.7
        plt.rcParams['ytick.major.width']=1.7
        plt.rcParams['ytick.minor.width']=1.7

    else:
        S_SIZE = 28
        L_SIZE = 30
        mylw=4

        plt.rc('font', size=L_SIZE,weight='bold')          # controls default text sizes
        #plt.rc('axes', titlesize=S_SIZE)     # fontsize of the axes title
        plt.rc('axes', labelsize=L_SIZE,linewidth=mylw)    # fontsize of the x and y labels
        plt.rc('xtick', labelsize=L_SIZE)    # fontsize of the tick labels
        plt.rc('ytick', labelsize=L_SIZE)    # fontsize of the tick labels
        #plt.rc('legend', fontsize=S_SIZE)    # legend fontsize
        #plt.rc('text',usetex=True)

        plt.rcParams['xtick.major.size']=12
        plt.rcParams['xtick.minor.size']=10
        plt.rcParams['ytick.major.size']=12
        plt.rcParams['ytick.minor.size']=10
        plt.rcParams['xtick.major.width']=3
        plt.rcParams['xtick.minor.width']=3
        plt.rcParams['ytick.major.width']=3
        plt.rcParams['ytick.minor.width']=3

    safe_col=['#BBBBBB','#0077BB','#33BBEE','#009988','#EE7733','#CC3311','#EE3377']
    other_col = ['k','mediumblue','green','darkorange','purple','cornflowerblue']
    return other_col

def prettyplot(x,y,type='paper',ls='None',marker='o',markersize=5, **kwargs):
    # Function: Create a plot with nice defaults; allows arrays of sizes and colors
    # to be passed, along with multi-dimensional data. Hopefully smart enough to
    # not break when asked something weird
    # Input: x array; y array; purpose of plot (paper or poster, determines sizes
    # and widths), default is paper; linestyle, default is none; marker, default is
    # circles; markersize, default is 5
    # Optional input: s = array of marker sizes, default is all 25 for papertype and
    # all 35 for postertype; color = string or array of colors, if data are multi-
    # dimensional will select smallest dimension to assign different colors to,
    # default is black or different CBF colors defined in set_pp; xlabel = string
    # for xlabel of plot, default is none; ylabel = string for ylabel of plot,
    # default is none; filename = string for name of plot, default is none and no
    # plot will be saved but fig, ax will be returned
    # Output: if no filename, fig, ax will be returned.
    x = np.array(x)
    y = np.array(y)
    if type=='paper':
        myc = set_pp()
        fig,ax = plt.subplots(figsize=(8,6))
        fig.subplots_adjust(left=0.13,bottom=0.11,top=0.97,right=0.97)
        myt = 14
        mys=25
    if type=='poster':
        myc = set_pp(type='poster')
        fig,ax = plt.subplots(figsize=(26,14))
        myt = 30
        mys = 35

    if 's' in kwargs:
        mys = kwargs["s"]
        kwargs = {key: value for key, value in kwargs.items() if key != 's'}

    if 'color' in kwargs:
        mycol = kwargs['color']
        if not isinstance(mycol,list): mycol = [mycol]
        kwargs = {key: value for key, value in kwargs.items() if key != 'color'}
    else:
        if (x.ndim==2): mycol = myc[:min(np.shape(x))]
        else: mycol = myc[0]
        if not isinstance(mycol,list): mycol = [mycol]

    if (len(mycol)>1)&(len(mycol)!=len(x)):
        if (x.ndim==2):
            if (np.shape(x)[0]!=len(mycol))&(np.shape(x)[1]!=len(mycol)):
                print('Warning: length of color array does not match any dimension of data')
                mycol = mycol[:min(np.shape(x))]
        else:
            print('Warning: length of color array does not match any dimension of data')
            mycol = [mycol[0]]

    print(f'x len is {len(x)}, x dim is {x.ndim}, x shape is {np.shape(x)}, and mycol is {mycol}')

    if 'xlabel' in kwargs:
        xlab = kwargs['xlabel']
        kwargs = {key: value for key, value in kwargs.items() if key != 'xlabel'}
    else: xlab = ''
    if 'ylabel' in kwargs:
        ylab = kwargs['ylabel']
        kwargs = {key: value for key, value in kwargs.items() if key != 'ylabel'}
    else: ylab = ''

    if 'filename' in kwargs:
        filename = kwargs['filename']
        kwargs = {key: value for key, value in kwargs.items() if key != 'filename'}
    else: filename=''

    if len(mycol)>1:
        if (len(mycol)==len(x))&(x.ndim==1):
            plt.scatter(x,y,c=mycol,s=mys,marker=marker,**kwargs)
        elif (x.ndim==2):
            if (np.shape(x)[0]==len(mycol)):
                for ii in range(np.shape(x)[0]):
                    plt.scatter(x[ii,:],y[ii,:],color=mycol[ii],s=mys,marker=marker,**kwargs)
            elif (np.shape(x)[1]==len(mycol)):
                for ii in range(np.shape(x)[1]):
                    plt.scatter(x[:,ii],y[:,ii],color=mycol[ii],s=mys,marker=marker,**kwargs)
    else:
        plt.plot(x,y,color = str(mycol[0]),ls=ls,marker=marker,markersize=markersize,**kwargs)

    if len(xlab)>0:
        ax.set_xlabel(xlab,fontsize=myt,weight='bold')

    if len(ylab)>0:
        ax.set_ylabel(ylab,fontsize=myt,weight='bold')

    if len(filename)>0:
        fig.savefig(filename)
        plt.close('all')
        plt.cla()
        plt.clf()
    else:
        return fig,ax

def plotRebound(sm, default=True, **kwargs):
	# Function: plot orbits of rebound simulation
	# Input: rebound simulation; bool for changing default settings (rebound's plotOrbit)
	# Optional inputs: savename = resulting image file name, default is time of creation + .pdf;
	# n_orbits = how many orbits to plot, default is 1; time = how much to integrate the simulation
	# before plotting, default is 0; ntick = how many tick marks on x- and y-axes, default is 5
	from datetime import datetime

	sim = sm.copy()
	mylw=2
	mycol=set_pp()

	if "savename" not in kwargs:
		now = datetime.now()
		mysave = now.strftime("%b-%d-%y-%H:%M:%S")
		kwargs["savename"] = mysave+'.pdf'
	savename = kwargs["savename"]

	if "n_orbits" not in kwargs:
		kwargs["n_orbits"] = 1
	else:
		default = False
	n_orbits = kwargs["n_orbits"]

	if "time" not in kwargs:
		kwargs["time"] = 0 #do not move the simulation forward before plot
	else:
		default = False #time is included, so we need to not use default
	int_time = kwargs["time"]

	if "ntick" not in kwargs:
		kwargs["ntick"] = 5
	ntick = kwargs["ntick"]

	fig,ax = plt.subplots(figsize=(7,7))
	ax.set_aspect("equal")

	ps = sim.particles
	lim = 0
	for pl in ps:
		this_max = max(pl.x,pl.y)
		if this_max > lim: lim = this_max

	fig.subplots_adjust(wspace=0, hspace=0,left=0.25,bottom=0.11,top=0.95)

	if default:
		sim_time=sim.t+int_time
		ax.scatter(ps[0].x, ps[0].y, s=55, marker='*', facecolor='k',zorder=3) #s=35
		ii=0
		for planet in ps[1:]:
			ax.scatter(planet.x, planet.y, s=30, facecolor='k',zorder=3) #s=10
			o = np.array(planet.sample_orbit())
			lc = fading_line(o[:,0],o[:,1],color=mycol[ii],fading=False,lw=2)
			ax.add_collection(lc)
			ii+=1

	else:
		sim_time=sim.t+int_time #from kwargs, 0 if not given
		sim.integrate(sim_time)
		npts = 300*n_orbits
		ps = sim.particles
		numpl = len(ps)-1
		out_P = ps[numpl].P
		maxe = 0
		for ii in range(numpl):
			if ps[ii+1].e > maxe:
				maxe = ps[ii+1].e
		if maxe > 1.0 :
			print('Warning: Planet ejected in integration')

		'''
		if bin_P*n_orbits < 0.1*ps[2].P:
			print("Warning: plotting less than 0.01 of planet orbit")
			print("n_orbits spans %f days" %n_orbits*bin_P)
			print('binary period: %f ; planet period: %f' %(bin_P,ps[2].P))
			print("Scaling n_orbits")
		'''

		time_array = np.linspace(0,out_P*n_orbits,npts)
		xs = np.zeros([npts,numpl])
		ys = np.zeros([npts,numpl])

		for tt in range(len(time_array)):
			sim.integrate(sim_time+time_array[tt])
			for pl in range(numpl):
				xs[tt,pl] = sim.particles[pl+1].x
				ys[tt,pl] = sim.particles[pl+1].y

		ps = sim.particles
		ax.scatter(ps[0].x, ps[0].y, s=55, marker='*', facecolor='k',zorder=3) #s=35

		for pl in range(numpl):
			ax.scatter(ps[pl+1].x, ps[pl+1].y, s=30, facecolor='k',zorder=3) #s=10
			lc = fading_line(xs[:,pl],ys[:,pl],color=mycol[pl],fading=False,lw=2)
			ax.add_collection(lc)
			this_max = max(xs[:,pl])
			if this_max > lim: lim = this_max
			this_max = max(ys[:,pl])
			if this_max > lim: lim = this_max

	lim = 1.25*lim
	ax.set_xlim([-lim, lim])
	ax.set_ylim([-lim, lim])
	ax.set_yticks(np.linspace(-lim,lim,ntick))
	ax.set_xticks(np.linspace(-lim,lim,ntick))
	ax.set_xlabel('AU',weight='bold') #ax.set(ylabel='AU',xlabel='AU')
	ax.set_ylabel('AU',weight='bold')

	pp = PdfPages(savename)
	pp.savefig(fig)
	pp.close()
	plt.close('all')
	plt.cla()
	plt.clf()

def plotBinary(sm, default=True, **kwargs):
	# Function: plot orbits of circumbinary planets
	# Input: rebound simulation; bool for changing default settings (rebound's plotOrbit)
	# Optional inputs: savename = resulting image file name, default is time of creation + .pdf;
	# n_orbits = how many orbits to plot, default is 1; time = how much to integrate the simulation
	# before plotting, default is 0; ntick = how many tick marks on x- and y-axes, default is 7
	from datetime import datetime

	sim = sm.copy()
	S_SIZE = 12
	L_SIZE = 18
	mylw=2

	mycol=set_pp()

	if "savename" not in kwargs:
			now = datetime.now()
			mysave = now.strftime("%b-%d-%y-%H:%M:%S")
			kwargs["savename"] = mysave+'.pdf'
	savename = kwargs["savename"]
	if "n_orbits" not in kwargs:
		kwargs["n_orbits"] = 1
	else:
		default = False
	n_orbits = kwargs["n_orbits"]

	if "time" not in kwargs:
		kwargs["time"] = 0 #do not move the simulation forward before plot
	else:
		default = False #time is included, so we need to not use default
	int_time = kwargs["time"]

	if "ntick" not in kwargs:
		kwargs["ntick"] = 7
	ntick = kwargs["ntick"]

	fig,ax = plt.subplots(figsize=(7,7))
	ax.set_aspect("equal")

	ps = sim.particles
	lim = 0
	for pl in ps:
		this_max = max(pl.x,pl.y)
		if this_max > lim: lim = this_max

	fig.subplots_adjust(wspace=0, hspace=0,left=0.25,bottom=0.11,top=0.95)

	if default:
		sim_time=sim.t+int_time

		for star in ps[:2]:
			ax.scatter(star.x, star.y, s=55, marker='*', facecolor='k',zorder=3) #s=35
		o = np.array(ps[1].sample_orbit())
		lc = fading_line(o[:,0],o[:,1],color=mycol[2],fading=False,lw=2)
		ax.add_collection(lc)


		for planet in ps[2:]:
			ax.scatter(planet.x, planet.y, s=30, facecolor='k',zorder=3) #s=10
			o = np.array(planet.sample_orbit())
			lc = fading_line(o[:,0],o[:,1],color=mycol[4],fading=False,lw=2)
			ax.add_collection(lc)

	else:
		sim_time=sim.t+int_time #from kwargs, 0 if not given
		sim.integrate(int_time)
		npts = 800
		ps = sim.particles
		bin_P = ps[1].P
		if ps[2].e > 1.0:
			print('Planet ejected in integration')
		if bin_P*n_orbits < 0.1*ps[2].P:
			print("Warning: plotting less than 0.01 of planet orbit")
			print("n_orbits spans %f days" %n_orbits*bin_P)
			print('binary period: %f ; planet period: %f' %(bin_P,ps[2].P))
			print("Scaling n_orbits")

		time_array = np.linspace(0,bin_P*n_orbits,npts)
		s1 = np.zeros([npts,2])
		s2 = np.zeros([npts,2])
		pl1 = np.zeros([npts,2])

		for tt in range(len(time_array)):
			sim.integrate(sim_time+time_array[tt])
			s1[tt,0] = sim.particles[0].x
			s1[tt,1] = sim.particles[0].y
			s2[tt,0] = sim.particles[1].x
			s2[tt,1] = sim.particles[1].y
			pl1[tt,0] = sim.particles[2].x
			pl1[tt,1] = sim.particles[2].y

		ps = sim.particles
		for star in ps[:2]:
			ax.scatter(star.x, star.y, s=55, marker='*', facecolor='k',zorder=3) #s=35
		lc1 = fading_line(s1[:,0],s1[:,1],color=mycol[0],fading=False,lw=2)
		ax.add_collection(lc1)
		lc2 = fading_line(s2[:,0],s2[:,1],color=mycol[2],fading=False,lw=2)
		ax.add_collection(lc2)

		for planet in ps[2:]:
			ax.scatter(planet.x, planet.y, s=30, facecolor='k',zorder=3) #s=10
		lc = fading_line(pl1[:,0],pl1[:,1],color=mycol[4],fading=False,lw=2)
		ax.add_collection(lc)

		this_max = max(pl1[:,0])
		if this_max > lim: lim = this_max
		this_max = max(pl1[:,1])
		if this_max > lim: lim = this_max

	lim = 1.25*lim
	ax.set_xlim([-lim, lim])
	ax.set_ylim([-lim, lim])
	ax.set_yticks(np.linspace(-lim,lim,ntick))
	ax.set_xticks(np.linspace(-lim,lim,ntick))
	ax.set_xlabel('AU',weight='bold') #ax.set(ylabel='AU',xlabel='AU')
	ax.set_ylabel('AU',weight='bold')
	#plt.show()

	pp = PdfPages(savename)
	pp.savefig(fig)
	pp.close()
	plt.close('all')
	plt.cla()
	plt.clf()

def planetGif(sm,name, del_dir=True,inverse=False,num=100):
    # Function: creates gif of system through one full orbit of farthest planet
    # Input: rebound simulation of the full system; name of gif; bool to delete
    # the directory and frames for gif, default is True; bool to invert colors
    # so background is black, default is white background; number of frames
    # Output: function does not return anything, but saves gif to wd
    import progressbar
    import os

    if inverse: matplotlib.rcParams['axes.edgecolor'] = 'white'
    print('This might take a while')
    sim = sm.copy()
    sim.move_to_com()
    ps = sim.particles
    numpl = len(ps)-1
    maxP = ps[numpl].P
    mycol = set_prettyplot(type='poster')
    x,y = np.zeros([num,numpl]), np.zeros([num,numpl])
    time = sim.t + np.linspace(0,maxP,num)
    if name[-4:] != '.gif': name +='.gif'

    # create a directory to store the frames; will be deleted after gif is made
    dirn = f'gif-{name[:-4]}'
    if os.path.isdir(dirn) != True:
        os.mkdir(dirn)

    # create the frames
    for ii in range(num):
        sim.integrate(time[ii])
        x[ii,:] = np.array([ps[pl+1].x for pl in range(numpl)])
        y[ii,:] = np.array([ps[pl+1].y for pl in range(numpl)])

    lim = 1.25*max(abs(np.min(x)),np.max(x),abs(np.min(y)),np.max(y))
    widgets = ['Constructing frames: ', progressbar.AnimatedMarker()]
    bar = progressbar.ProgressBar(max_value=num,widgets=widgets).start()
    #bar = progressbar.ProgressBar(max_value=num)
    #widgets=[' [', progressbar.Timer(), '] ', progressbar.Bar(), ' (', progressbar.ETA(), ') ',]

    for ii in range(num):
        if ii%3 == 0:
            fig, ax = plt.subplots(figsize=(14,12))
            fig.subplots_adjust(right=0.96,left=0.23,bottom=0.14,top=0.95)
            fign = f'./{dirn}/{name[:-4]}-{ii:03}.png'  #'{0:03}'.format(1)
            for pl in range(numpl):
                plt.plot(x[:,pl],y[:,pl],color=mycol[pl],lw=8)

            plt.scatter(x[ii,:],y[ii,:],color=mycol[0:numpl],s=400)
            ax.set_xlabel('x [AU]',fontsize=35,weight='bold')
            ax.set_ylabel('y [AU]',fontsize=35,weight='bold')
            ax.axis('equal')
            ax.set(xlim=[-lim, lim], ylim =[-lim, lim])
            ax.set_yticks(np.linspace(-lim,lim,5))
            ax.set_xticks(np.linspace(-lim,lim,3))

            if inverse:
                fig.set_facecolor('black')
                ax.set_facecolor('black')
                ax.tick_params(axis='x', colors='white')
                ax.tick_params(axis='y', colors='white')
                ax.xaxis.label.set_color('white')
                ax.yaxis.label.set_color('white')
            plt.savefig(fign,facecolor=fig.get_facecolor(), edgecolor='none')
            plt.close('all')
            plt.cla()
            plt.clf()
            bar.update(ii)

    # Create the gif
    print('\nCreating gif...')
    print('Sorry there are no fancy animations.....')
    os.system(f'convert -delay 20 -loop 0 ./{dirn}/*.png {name}')
    print(f'gif saved as {name}')

    if del_dir:
        # remove dirn and frames
        print('removing frames')
        os.system(f'rm -fr {dirn}')

def get_hm(xdat,ydat,zdat,x_size,y_size,fig,ax,cm):
    # Function: creates a pretty heatmap
    # Input: array defining the x-axis locations; array defining the y-axis locations;
    # array defining the values for the heatmap; size of x-dim for heatmap; size of
    # y-dim for heatmap; figure handle; axis handle; colormap;
    # Output: returns handle to heatmap
    hm_data = np.zeros([y_size, x_size])
    x_bins, y_bins = np.linspace(min(xdat),max(xdat),(x_size+1)), np.linspace(min(ydat),max(ydat), (y_size+1))
    index = np.array(range(len(xdat)))

    for ii in range(x_size):
    	for jj in range(y_size):
    		these_sims = index[(xdat>x_bins[ii])*(xdat<x_bins[ii+1])*(ydat>y_bins[jj])*(ydat<y_bins[jj+1])]
    		if len(these_sims)>0: hm_data[jj,ii] = np.mean(zdat[these_sims])

    hm = ax.imshow(hm_data,cmap=cm,interpolation='gaussian',aspect='auto',extent=[min(xdat),max(xdat),min(ydat),max(ydat)],origin='lower')

    fig.subplots_adjust(left=0.14,right = 0.85,bottom=0.09,top=0.98)

    return hm

def fading_line(x, y, color='black', alpha=1, fading=True, fancy=False, **kwargs):
    """
    Stolen with love from rebound (Rein & Tamayo 2015 )
    Returns a matplotlib LineCollection connecting the points in the x and y lists.
    Can pass any kwargs you can pass to LineCollection, like linewidgth.
    Parameters
    ----------
    x       : list or array of floats for the positions on the (plot's) x axis.
    y       : list or array of floats for the positions on the (plot's) y axis.
    color   : Color for the line. 3-tuple of RGB values, hex, or string. Default: 'black'.
    alpha   : float, alpha value of the line. Default 1.
    fading  : bool, determines if the line is fading along the orbit.
    fancy   : bool, same as fancy argument in OrbitPlot()
    """
    try:
        from matplotlib.collections import LineCollection
        import numpy as np
    except:
        raise ImportError("Error importing matplotlib and/or numpy. Plotting functions not available. If running from within a jupyter notebook, try calling '%matplotlib inline' beforehand.")


    if "lw" not in kwargs:
        kwargs["lw"] = 1
    lw = kwargs["lw"]

    if fancy:
        kwargs["lw"] = 1*lw
        fl1 = fading_line(x, y, color=color, alpha=alpha, fading=fading, fancy=False, **kwargs)
        kwargs["lw"] = 2*lw
        alpha *= 0.5
        fl2 = fading_line(x, y, color=color, alpha=alpha, fading=fading, fancy=False, **kwargs)
        kwargs["lw"] = 6*lw
        alpha *= 0.5
        fl3 = fading_line(x, y, color=color, alpha=alpha, fading=fading, fancy=False, **kwargs)
        return [fl3,fl2,fl1]

    Npts = len(x)
    if len(y) != Npts:
        raise AttributeError("x and y must have same dimension.")

    color = get_color(color)
    colors = np.zeros((Npts,4))
    colors[:,0:3] = color
    if fading:
        colors[:,3] = alpha*np.linspace(0,1,Npts)
    else:
        colors[:,3] = alpha

    segments = np.zeros((Npts-1,2,2))
    segments[:,0,0] = x[:-1]
    segments[:,0,1] = y[:-1]
    segments[:,1,0] = x[1:]
    segments[:,1,1] = y[1:]

    lc = LineCollection(segments, color=colors, **kwargs)
    return lc
