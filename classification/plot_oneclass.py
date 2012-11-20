"""
One-class SVM with a non-linear RBF kernel 

One-class SVM is an unsupervised algorithm that
estimates outliers in a dataset.

"""
print __doc__

import numpy as np
import pylab as pl
from scikits.learn import svm

import csv 

def import_text(filename, separator):
	for line in csv.reader(open(filename), delimiter=separator,
		skipinitialspace=True):
		if line:
			yield line

X = []; y = []
for data in import_text('test_1.fmt', ' '):
	X.append([data[0], data[1]])

X = np.array(X, dtype='float64')
print X

x1_max = X[:][:,0].max()
x1_min = X[:][:,0].min()
x2_max = X[:][:,1].max()
x2_min = X[:][:,1].min()
x_size = len(X)

xx, yy = np.meshgrid(np.linspace(x1_min, x1_max, x_size), np.linspace(x2_min, x2_max, x_size))

# fit the model
clf = svm.OneClassSVM(nu=0.1, kernel="linear", gamma=0.1)
clf.fit(X)

# plot the line, the points, and the nearest vectors to the plane
Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
y_pred = clf.predict(X)

pl.set_cmap(pl.cm.Paired)
pl.contourf(xx, yy, Z)
pl.scatter(X[y_pred>0,0], X[y_pred>0,1], c='white', label='inliers')
pl.scatter(X[y_pred<=0,0], X[y_pred<=0,1], c='black', label='outliers')
pl.axis('tight')
pl.legend()
pl.show()
