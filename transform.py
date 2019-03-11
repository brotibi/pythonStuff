import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt
import math as math

# Based on a matplotlib example at:
# https://matplotlib.org/mpl_toolkits/mplot3d/tutorial.html
def plotspiral(x_rot, y_rot, z_rot, scale):
	mpl.rcParams['legend.fontsize'] = 10

	fig = plt.figure()
	ax = fig.gca(projection='3d')
	theta = np.linspace(-4 * np.pi, 4 * np.pi, 100)
	z = np.linspace(-2, 2, 100)
	r = z**2 + 1
	x = r * np.sin(theta)
	y = r * np.cos(theta)
	x, y, z = transform(x, y, z, x_rot, y_rot, z_rot, scale)
	ax.plot(x, y, z, label='parametric curve')
	ax.legend()

	plt.show()

# Rotate around the x-axis, then the y-axis, then the z-axis, then scale
def transform(x, y, z, x_rot, y_rot, z_rot, scale):
	# TODO
	Matrix = np.array([x,y,z])
	#Rx = [[1,0,0],[0,math.cos(x_rot),math.sin(x_rot)],[0,-math.sin(x_rot), math.cos(x_rot)]]#X Rotation Matrix
	#Ry = [[math.cos(y_rot),0,-math.sin(y_rot)],[0,1,0],[math.sin(y_rot), 0,math.cos(y_rot)]]#Y Rotation Matrix
	#Rz = [[math.cos(z_rot),math.sin(x_rot),0],[-math.sin(z_rot), math.cos(z_rot),0],[0,0,1]]#Z Rotation Matrix
	Rx = [[1,0,0],
		  [0,math.cos(x_rot),-math.sin(x_rot)],
		  [0,math.sin(x_rot),math.cos(x_rot)]]
	Ry = [[math.cos(y_rot),0,math.sin(y_rot)],
		  [0,1,0],
		  [-math.sin(y_rot),0,math.cos(y_rot)]]

	Rz = [[math.cos(z_rot),-math.sin(z_rot),0],
		  [math.sin(z_rot),math.cos(z_rot),0],
		  [0, 0, 1]]

	#MAT = np.matmul(Rx,Ry)
	#MAT = np.matmul(MAT, Rz)
	#Matrix = np.matmul(MAT, Matrix)

	PT =  np.matmul(np.matmul(Rz,np.matmul(Ry,Rx)),Matrix)*scale

	return PT[0],PT[1],PT[2]

plotspiral(math.pi/2,math.pi/2,0,1)