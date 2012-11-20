#!/usr/bin/python
# A simple dip finder for discrete temporal data
# Use one of the following methods for dip detection
# 	kneighb - mean of k-nearest neighbors \n"
#       entropy - entropy and kernel density estimate with a gaussian kernel
#       chebyshev - outlier detection with chebyshev inequality
#
# Author : Sibi Antony 
#	   sibi [dot] antony [at] gmail [dot] com


import numpy as np
import pylab as pl
import csv
import sys
import os
import argparse

def read_fmt(filename, separator):
    for line in csv.reader(open(filename), delimiter=separator, 
        skipinitialspace=True): 
        if line:
            yield line

# Mean of ( maximum of (Signed distances of (xi from k left))
#           maximum of (Signed distances of (xi from k right))
#   )
def dip_kneighb_mean(k, i, y, arg):
    y_i = y[i]
    max_left = max(y[i-k:i] - y_i)
    max_right = max(y[i+1:i+k+1] - y_i)
    min_left = min(y[i-k:i] - y_i)
    min_right = min(y[i+1:i+k+1] - y_i)

    # Compute the 'distance' maximum
    # The term 'maximum of the signed distances' is rather
    # confusing. Here I compute the maximum distance, then
    # return with their sign. 
    if (abs(max_left) < abs(min_left)):
        max_left = min_left
    if (abs(max_right) < abs(min_right)):
        max_right = min_right

    return (max_left + max_right)/2.0

def dip_kneighb_mean_2(k, i, y, arg):
    y_i = y[i]
    avg_left = np.mean(y[i-k:i] - y_i)
    avg_right = np.mean(y[i+1:i+k+1] - y_i)

    return (avg_left + avg_right)/2.0

def dip_kneighb_mean_3(k, i, y, arg):
    y_i = y[i]
    avg_left = np.mean(y[i-k:i])
    avg_right = np.mean(y[i+1:i+k+1])

    return ((avg_left - y_i) + (avg_right - y_i))/2.0


def entropy(A):
    H_w = []
    by_2pi = 1 / (np.sqrt(2.0 * 3.1415))

    A_std = np.std(A)
    w = 1.06 * A_std * pow(len(A), (-1/5))
    for a_i in A:
        k_x = (a_i - A)/w
        # Gaussian kernel
        kernel_k_x = by_2pi * np.exp(-0.5 * k_x * k_x)
        #kernel_k_x = by_2pi * np.sign(k_x) * np.exp(-0.5 * k_x * k_x)

        # Polynomial kernel
        #kernel_k_x = 3.0/4 * np.array([1.0 - x * x for x in k_x if abs(x) < 1])

        # probability density for this a_i
        p_div = 1.0 / (len(A) * w)
        pdf_w = p_div * sum(kernel_k_x)
        #pdf_w = abs(p_div * sum(kernel_k_x))
        #pdf_w = 1 if (pdf_w == 0) else pdf_w

        H_w.append(-1.0 * pdf_w * np.log(pdf_w))

    return sum(H_w)

def dip_entropy_kde(k, i, y, arg):
    
    N_wi_i = y[i-k:i+k+1]
    N_wo_i = np.concatenate((y[i-k:i], y[i+1:i+k+1]))
    #N_wi_i = np.concatenate((y[i+1:i+k+1], (y[i-k:i])[::-1]))
    #N_wo_i = np.concatenate((y[i+1:i+k+1], np.array([y[i]]), (y[i-k:i])[::-1]))

    e_N_wo_i = entropy(N_wo_i);
    e_N_wi_i = entropy(N_wi_i);
    return (e_N_wi_i - e_N_wo_i)
    #return (e_N_wi_i)
    
def dip_chebyshev(k, i, y, arg):

    y_i = y[i]
    N_wo_i = np.concatenate((y[i-k:i], y[i+1:i+k+1]))

    N_mean = np.mean(N_wo_i)
    N_std = np.std(N_wo_i)
    #print "y_i ", y_i, ",mean ", N_mean, ",std ", N_std, ", y_i - N_mean ", abs(y_i - N_mean), \
    #  ", h * std ", arg.h * N_std
    if (y_i <= N_mean) and (abs(y_i - N_mean) >= 0.025 * N_std):
        # We caught an outlier.
        # return abs(y_i - N_mean) 
        # A variant of the dip measure by adding std to the dip
        # This way, a distribution with low std will get punished.
        return abs(y_i - N_mean) + 2.8 * N_std
    else:
        return 0.0

def _read_lightcurves(infile):

    X = []; y = []; 
    for data in read_fmt(infile, ' '):
        X.append(data[0])
        y.append(data[1])

    X = np.array(X, dtype='float64')
    y = np.array(y, dtype='float64')

    return X, y

def _find_dips(X, y, method, arg):

    dip = []; y_out = []; x_out = []
    X_len = len(X)

    if method == "kneighb":
        _dip_func = dip_kneighb_mean
    elif method == "entropy":
        _dip_func = dip_entropy_kde
    elif method == "chebyshev":
        _dip_func = dip_chebyshev
    else:
        print "Error. Incorrect method."
        sys.exit()

    for i in range(X_len):
        # Dip detection is based on sliding windows. For consistency, 
        # we skip the first k and the last k.
        if arg.k > i or (arg.k + i) >= len(y):
            dip.append('0.0')
            continue
        dip.append(_dip_func(arg.k, i, y, arg))

    dip = np.array(dip, dtype='float64')

    # We need to calculate mean/SD only for the
    #   +ve elements in the list.
    dip_p = [i for i in dip if i > 0]
    dp_mean = np.mean(dip_p)
    dp_std = np.std(dip_p)
    # print "Mean", dp_mean, "Std", dp_std, "h * dp_std", arg.h * dp_std

    # We would need to invert the entropy based on the whole distribution
    #   for dips.
    if method == "entropy":
        dip_no_k = dip[arg.k:-arg.k]
        dnk_mean = np.mean(dip_no_k)
        dnk_std = np.std(dip_no_k)

        dip_inv = np.array([0.0] * X_len)
        dip_inv[arg.k:-arg.k] = 2 * dnk_mean - dip_no_k

        #print "Mean", dnk_mean, "Std", dnk_std, "h * dp_std", arg.h * dnk_std
        dip = dip_inv
        dip_mean = dnk_mean
        dip_std = dnk_std

    # Remove local dips, that are 'small' in global context
    for i in range(len(dip)):
        if(dip[i] > 0.0 and ((dip[i] - dp_mean) > arg.h * dp_std)):
            y_out.append( (y[i], i) )
            #print i, ",", y[i], ",", X[i]

    # Retain only one dip within the reach of k.
    # We introduce a 'u' to take care of the proper indexing
    # after we pop elements from the list.
    u = 0
    for t in range(len(y_out) - 1):
        dip_i, i = y_out[t-u]
        dip_j, j = y_out[t-u + 1]
        if abs(i - j) <= arg.k:
            if dip_i == max(dip_i, dip_j):
                y_out.pop(t-u)
            else:
                y_out.pop(t-u + 1)

            # Make sure, next read will include one of the current 
            # elements
            u += 1

    for i in range(len(y_out)):
        x_out.append(X[y_out[i][1]])

    return x_out, y_out

def _plot_input(X, y):
    pl.figure(figsize=[18,10])
    pl.scatter(X, y, s=3, c='b', label='lightcurve', linewidths=0)

def _plot_dips(x_out, y_out, color):

    x_out = np.array(x_out)
    y_out = np.array(y_out)
    if len(y_out) > 0:
        pl.scatter(x_out, y_out[:][:,0], c=color, label='dips', 
            s=35, marker='o', linewidths=1, alpha=0.3)

def _save_plot(fileout):

    pl.legend()
    pl.title('Dip detector with configurable functions')
    pl.draw()
    pl.savefig(fileout)

# Parse the params, pass on to find_dip()
parser = argparse.ArgumentParser(description='Simple dip finder', \
            formatter_class=argparse.RawTextHelpFormatter)

parser.add_argument('-k', action='store', dest='k', default=10, type=int,
                    help='Window size. Default = 10')
parser.add_argument('-s', action='store', dest='h', default=1.5, type=float, 
                    help='Threshold multiplier for SD. Default=1.5')
parser.add_argument('-w', action='store', dest='w', default=5, type=int, 
                    help='Parzen window size. Only for entropy option. Default=5')
parser.add_argument('-f', action='store', dest='method', default="kneighb", type=str, 
                    choices = ['kneighb', 'entropy', 'chebyshev'],
                    help="Method of dip detection,\n" 
                    "kneighb - mean of k-nearest neighbors \n"
                    "entropy - entropy and kernel density estimate with a gaussian kernel \n"
                    "chebyshev - outlier detection with chebyshev inequality \n" )
parser.add_argument('file', metavar='file', type=str,
                    help='Input formatted file')

arg = parser.parse_args()
filename = arg.file
filebase = os.path.splitext(filename)[0]
fileout = filebase + "_dip_h"+str(arg.h)+"_k"+str(arg.k)+"_w"+str(arg.w)+".png"

X, y = _read_lightcurves(filename)
x_out, y_out = _find_dips(X, y, arg.method, arg)
_plot_input(X, y)
_plot_dips(x_out, y_out, 'g')
_save_plot(fileout)

