ml-kepler
=========
These are some simple/stupid machine learning hacks, to analyze the kepler mission data. The kepler mission intends to identify Earth-sized planets, by observing repeated transits of a planet in front of their stars, and measuring the resulting brightness reduction. The code here has some dip detection methods, and to run regression/classification on the formatted data.

The original Kepler mission data can be found here : http://archive.stsci.edu/kepler/publiclightcurves.html or here : http://kepler.nasa.gov/Science/ForScientists/dataarchive/

There are also some modules written for PyKE (http://keplergo.arc.nasa.gov/PyKE.shtml) distribution, a framework for kepler data analysis.


Directory contents 
==================

dipfinder/data				- The original data released is in the FITS format. The example data in this directory
						has been formatteed into plaintext column data.
dipfinder/dipfinder.py			- A dip detection program. Has the options to choose the algorithm/k-window/threshold multiplier etc.

classification/plot_oneclass.py		- A one class unsupervised classfication, uses scikits.learn

pyke_modules/kepregr.py 		- A PyKE module to perform a k-NN regression
pyke_modules/kepdip.py			- A PyKE module to perform dip detection
sv_regression/keptest.py		- SV Regression, initially written as a PyKE module. Requires PyRAF/PyKE to launch

