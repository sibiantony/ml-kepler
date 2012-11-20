"""
Support vector regression on kepler data
Requires scikits.learn

Author : Sibi Antony 
         sibi [dot] antony [at] gmail [dot] com
"""

from pyraf import iraf
import pylab, numpy
from pylab import *
from matplotlib import *
import kepio, kepmsg, kepkey
import re
from scikits.learn.svm import SVR

def keptest(infile,outfile,datacol,ploterr,errcol,quality,
	    lcolor,lwidth,fcolor,falpha,labelsize,ticksize,
	    xsize,ysize,fullrange,plotgrid,verbose,logfile,status): 

# log the call 

    hashline = '----------------------------------------------------------------------------'
    kepmsg.log(logfile,hashline,verbose)
    call = 'KEPTEST -- '
    call += 'infile='+infile+' '
    call += 'outfile='+outfile+' '
    call += 'datacol='+datacol+' '
    perr = 'n'
    if (ploterr): perr = 'y'
    call += 'ploterr='+perr+ ' '
    call += 'errcol='+errcol+' '
    qual = 'n'
    if (quality): qual = 'y'
    call += 'quality='+qual+ ' '
    call += 'lcolor='+str(lcolor)+' '
    call += 'lwidth='+str(lwidth)+' '
    call += 'fcolor='+str(fcolor)+' '
    call += 'falpha='+str(falpha)+' '
    call += 'labelsize='+str(labelsize)+' '
    call += 'ticksize='+str(ticksize)+' '
    call += 'xsize='+str(xsize)+' '
    call += 'ysize='+str(ysize)+' '
    frange = 'n'
    if (fullrange): frange = 'y'
    call += 'fullrange='+frange+ ' '
    pgrid = 'n'
    if (plotgrid): pgrid = 'y'
    call += 'plotgrid='+pgrid+ ' '
    chatter = 'n'
    if (verbose): chatter = 'y'
    call += 'verbose='+chatter+' '
    call += 'logfile='+logfile
    kepmsg.log(logfile,call+'\n',verbose)

# start time

    kepmsg.clock('KEPTEST started at',logfile,verbose)

# test log file

    logfile = kepmsg.test(logfile)

# open input file

    if status == 0:
        struct, status = kepio.openfits(infile,'readonly',logfile,verbose)
    if status == 0:
        tstart, tstop, bjdref, cadence, status = kepio.timekeys(struct,infile,logfile,verbose,status)

# read table structure

    if status == 0:
	table, status = kepio.readfitstab(infile,struct[1],logfile,verbose)

# read table columns

    if status == 0:
        intime, status = kepio.readtimecol(infile,table,logfile,verbose)
        #intime += bjdref
	indata, status = kepio.readfitscol(infile,table,datacol,logfile,verbose)
	if (ploterr):
            indataerr, status = kepio.readfitscol(infile,table,errcol,logfile,verbose)
    if status == 0:
        gaps = zeros(len(indata))

# read table quality column

    if status == 0 and quality:
        try:
            qualtest = table.field('SAP_QUALITY')
        except:
            message = 'ERROR -- KEPTEST: no SAP_QUALITY column found in file ' + infile
            message += '. Use keptest quality=n'
            status = kepmsg.err(logfile,message,verbose)
    if status == 0 and quality:
        gaps, status = kepio.readfitscol(infile,table,'SAP_QUALITY',logfile,verbose)       

# close infile

    if status == 0:
	status = kepio.closefits(struct,logfile,verbose)

# remove infinities and bad data

    if status == 0:
	barytime = []; data = []; dataerr = []
        if 'ap_raw' in datacol or 'ap_corr' in datacol:
            cadenom = cadence
        else:
            cadenom = 1.0
	for i in range(len(intime)):
            if numpy.isfinite(indata[i]) and indata[i] != 0.0 and gaps[i] == 0:
                barytime.append(intime[i])
                data.append(indata[i] / cadenom)
                if (ploterr):
                    dataerr.append(indataerr[i])
	barytime = numpy.array(barytime,dtype='float64')
	data = numpy.array(data,dtype='float64')
	if (ploterr):
            dataerr = numpy.array(dataerr,dtype='float64')

# clean up x-axis unit

    if status == 0:
	barytime0 = float(int(tstart / 100) * 100.0)
	barytime -= barytime0
        xlab = 'BJD $-$ %d' % barytime0

# clean up y-axis units

        try:
            nrm = len(str(int(data.max())))-1
        except:
            nrm = 0
	data = data / 10**nrm
	ylab1 = '10$^%d$ e$^-$ s$^{-1}$' % nrm

# data limits

	xmin = barytime.min()
	xmax = barytime.max()
	ymin = data.min()
	ymax = data.max()
	xr = xmax - xmin
	yr = ymax - ymin
	data[0] = ymin - yr * 2.0
	data[-1] = ymin - yr * 2.0
        if fullrange:
            data[0] = 0.0
            data[-1] = 0.0

# define plot formats

        try:
            rc('text', usetex=True)
            rc('font',**{'family':'sans-serif','sans-serif':['sans-serif']})
            params = {'backend': 'png',
                      'axes.linewidth': 2.5,
                      'axes.labelsize': labelsize,
                      'axes.font': 'sans-serif',
                      'axes.fontweight' : 'bold',
                      'text.fontsize': 12,
                      'legend.fontsize': 12,
                      'xtick.labelsize': ticksize,
                      'ytick.labelsize': ticksize}
            pylab.rcParams.update(params)
        except:
            pass

# define size of plot on monitor screen

	pylab.figure(figsize=[xsize,ysize])

# delete any fossil plots in the matplotlib window

        pylab.clf()

	# position axes inside the plotting window

	ax = pylab.axes([0.06,0.1,0.93,0.88])

# force tick labels to be absolute rather than relative

        pylab.gca().xaxis.set_major_formatter(pylab.ScalarFormatter(useOffset=False))
        pylab.gca().yaxis.set_major_formatter(pylab.ScalarFormatter(useOffset=False))

# rotate y labels by 90 deg

        labels = ax.get_yticklabels()
        setp(labels, 'rotation', 90, fontsize=12)
	
# plot data time series as an unbroken line, retaining data gaps

	ltime = []; ldata = []; ldataerr = []; ldatagaps = []
        dt = 0

	# SVR 
	svr_rbf = SVR(kernel='rbf', C=1, gamma=0.1)
	svr_lin = SVR(kernel='linear', C=1)
	svr_poly = SVR(kernel='poly', C=1, degree=2)
	svr_ltime = []; svr_ldata = []


	for i in range(len(indata)):
            if i > 0:
		if numpy.isfinite(indata[i]) and indata[i] != 0.0 : 
			# print intime[i], " ", indata[i]
			ltime.append(intime[i])
			ldata.append(indata[i])
			svr_ltime.append([intime[i]])

	ltime = array(ltime, dtype=float64)
	ldata = array(ldata, dtype=float64)

	if len(ldata) > 0 and len(ltime) > 0 :
		pylab.scatter (ltime, ldata, s=1, color=lcolor, label='Data:Input lightcurve')

	svr_ltime = array(svr_ltime, dtype='float64')
	svr_ldata = array(ldata, dtype='float64')

	svr_ldata_rbf = svr_rbf.fit(svr_ltime, svr_ldata).predict(svr_ltime)

	## Get the transits!
	# Identify the difference of data min. and the regression line
	# = An approximate initial dip value.
	
	ldata_min = min(ldata)
	ldata_min_i = ldata.tolist().index(ldata_min)
	fluxdip = svr_ldata_rbf[ldata_min_i] - ldata_min
	# fluxthresh = (svr_ldata_rbf[ldata_min_i] + ldata_min ) / 2.0
	print "ldata min = ", ldata_min, "fluxdip =", fluxdip
	thresh_x = []; thresh_y = [];

	# Sequentially scan the inputs, look for y-points below the 
	# initial mean. Group the points
	i = 0
	while i < len(ldata):
		# print intime[i], " ", indata[i]
		fluxmin = fluxthresh = svr_ldata_rbf[i] - fluxdip/2.0
		if ldata[i] < fluxthresh:
			thresh_y.append(fluxthresh); thresh_x.append(ltime[i])
		# Identify the local min, calculate difference with regression line.
			while i < len(ldata) and ldata[i] < fluxthresh :
				if ldata[i] < fluxmin:
					fluxmin = ldata[i]
					fluxmin_i = i
				i += 1
			
		# We got the local min, now plot the line,
		# converge the dip value with the newly calculated one.	
			pylab.plot([ ltime[fluxmin_i], ltime[fluxmin_i] ], 
				[ ldata[fluxmin_i], svr_ldata_rbf[fluxmin_i] ], 
				'r-', linewidth=1)
			fluxdip = (fluxdip + svr_ldata_rbf[fluxmin_i] - fluxmin)/2.0
		i += 1


	pylab.plot(thresh_x, thresh_y, c='c', label='Adapted transit threshold')
	pylab.scatter(thresh_x, thresh_y, c='k', s=1)
	pylab.plot(svr_ltime, svr_ldata_rbf, c='g', label='Cum. RBF model')


	if (ploterr):
            ldataerr = numpy.array(ldataerr,dtype='float32')


# plot labels

	pylab.xlabel(xlab, {'color' : 'k'})
        try:
            pylab.ylabel(ylab1, {'color' : 'k'})
        except:
            ylab1 = '10**%d e-/s' % nrm
            pylab.ylabel(ylab1, {'color' : 'k'})

# make grid on plot

	if plotgrid: pylab.grid()

# paint plot into window
	pylab.legend()

        pylab.draw()

# save plot to file

    if status == 0 and outfile.lower() != 'none':
	pylab.savefig(outfile)

# -----------------------------------------------------------
# main

parfile = iraf.osfn("kepler$keptest.par")
t = iraf.IrafTaskFactory(taskname="keptest", value=parfile, function=keptest)
