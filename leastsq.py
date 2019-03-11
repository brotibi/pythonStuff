# Fitting U.S. population numbers from Wikipedia, 1610-2010

import csv
import numpy as np
import matplotlib.pyplot as plt
import math
#import uspop.txt
#import uspop1700_1900.txt


# Return the n x 2 matrix where each row is year, population
# (return as a numpy array so we can easily extract columns)
def read_csv(filename):
	matrix_out = []
	with open(filename,newline='') as csvfile:
		myreader = csv.reader(csvfile, delimiter = ',')
		for row in myreader:
			# Convert from list of strings to list of ints
			matrix_out.append(list(map(int, row)))
	return np.array(matrix_out)
	
# Return the best straight-line fit params m,b (y = mx + b) fitting with least-squares
# Assumes input is 2d array with independent variable (years) in column 0
# and dependent variable (population) in column 1
def linear_fit(data):
	years = data[:,0]   # Grab 0th column
	pops = data[:,1]
	# modeled after the example at
	# https://docs.scipy.org/doc/numpy-1.15.0/reference/generated/numpy.linalg.lstsq.html
	A = np.vstack([years, np.ones(len(years))]).T
	m, b = np.linalg.lstsq(A, pops, rcond=None)[0]
	return m, b

# Plot the least-squares fit line under the model y = mx + b
def plotline(data, m, b):
	# again following
	# https://docs.scipy.org/doc/numpy-1.15.0/reference/generated/numpy.linalg.lstsq.html
	years = data[:,0]
	pops = data[:,1]
	plt.plot(years, pops, 'o', markersize=10)
	plt.plot(years, m*years + b, 'r')
	plt.show()

# Solve an exponential-fit problem y = Ce^(ax) by taking the log of both sides,
# ln y = ln C + ax, and solving the linear fit problem for input a, output ln y,
# to find parameters ln C and a.  Return the real C,a.
def exponential_fit(data):
	years = data[:,0]   # Grab 0th column
	pops = data[:,1]
	v = np.log(pops)
	A = np.vstack([years, np.ones(len(years))]).T
	a, C = np.linalg.lstsq(A, v, rcond=None)[0]
	return a, np.exp(C)
	
# Plot an exponential fit, following the model of plotline.  Notice that we just
# need to feed calculated fit datapoints to plot; it's not very different from plotline
# except in that equation.
# Plot the least-squares fit line under the model y = Ce^(ax)
def plotcurve(data, m, C):
	years = data[:,0]   # Grab 0th column
	pops = data[:,1]
	plt.plot(years, pops, 'o', markersize=10)
	plt.plot(years, C*np.exp(m*years), 'r')
	plt.show()

uspop1 = read_csv("uspop.txt")
uspop1700 = read_csv("uspop1700_1900.txt")
uspopln = np.c_[uspop1[:,0],np.log(uspop1[:,1])]
plotcurve(uspop1700, *(exponential_fit(uspop1700)))
plotline(uspopln, *(linear_fit(uspopln)))
print (exponential_fit(uspop1700))