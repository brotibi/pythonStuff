# Perceptron experiment.
# Our space will be the RGB color space, and we'll try to learn
# the concept of ``gold'' (like yellow but a little broader)   Our 
# points come from:
# http://www.tayloredmktg.com/rgb/
# The names for the lists are somewhat broad:  orange_points includes red, for example.
#
# Be sure to use Python3 (python3 at the terminal) since this code uses list copy(),
# which is too new for Python 2.

import random
import matplotlib.pyplot as plt
import numpy as np
import mpl_toolkits.mplot3d as A

# Python can return two values, comma separated.  You can use this like:
# examples, true_classes = perceptron.create_data()
def create_data():
	gold_points = [[238, 232, 170], [250, 250, 210], [255,255,224],[255,255,0],[255,215,0], [238,221,130],[218,165,32],[184,134,11]]

	white_points = [[255,250,250], [240,240,230],[238,213,183],[238,232,205]]

	blue_points = [[25,25,112], [0,0,128],[30,144,255],[0,206,209]]

	green_points = [[102,205,170], [152, 251, 152], [127, 255, 0], [107,142,35]]

	brown_points = [[139, 69, 19], [222,184,135], [244,164,96], [178,34,34]]

	orange_points = [[233,150,122], [255,165,0], [255,99,71], [255,0,0]]

	non_gold_points = white_points.copy()
	non_gold_points.extend(blue_points)
	non_gold_points.extend(green_points)
	non_gold_points.extend(brown_points)
	non_gold_points.extend(orange_points)

	all_examples = gold_points.copy()
	all_examples.extend(non_gold_points)
	
	all_examples = scaleTo0To1(all_examples)

	# [1]*n creates a list of n 1's
	true_classes = [1]*len(gold_points)
	true_classes.extend([0]*len(non_gold_points))
	
	# Shuffle both the examples and their classifications in exactly the same way
	random.seed(0)
	random.shuffle(all_examples)
	random.seed(0)
	random.shuffle(true_classes)
	
	return all_examples, true_classes

# Scaling the data to the range 0 to 1 instead of 0 to 255.
# This is a common machine learning practice since it reduces the guesswork for other
# parameters, and it also lets matplotlib recognize the values as colors.
def scaleTo0To1(points):
	scaled_points = []
	# There are slicker Python-y ways of doing the following, but we'll keep it simple
	for point in points:
		scaled_points.append([point[0]/255.0, point[1]/255.0, point[2]/255.0])
	return scaled_points

# We'll treat a perceptron as simply a vector of weights.  For convenience,
# the last weight will be the bias (so that we don't have an awkward offset
# from the input indices)
def new_perceptron():
	# Red weight, green weight, blue weight, bias
	return [0.25, 0.25, 0.25, 0.25]

# Returns 1 if we predict it's in the target class, 0 if we predict it's not
def perceptron_predict(ptron, example):
	if ptron[0]*example[0] + ptron[1]*example[1] + ptron[2]*example[2] + ptron[3] >= 0:
		return 1
	else:
		return 0


# Returns the new perceptron after training on an example once.
def perceptron_learn(ptron, example, true_class, learn_rate):
	x = ptron[0] + learn_rate*example[0]
	y = ptron[1] + learn_rate*example[1]
	z = ptron[2] + learn_rate*example[2]
	d = ptron[3] + learn_rate*perceptron_predict(ptron, example)
	if true_class - perceptron_predict(ptron, example) == -1 :
		return [-x*ptron[0],-y*ptron[1],-z*ptron[2], -d]
	elif true_class - perceptron_predict(ptron, example) == 1 :
		return [x*ptron[0],y*ptron[1],z*ptron[2], d]
	else:
		return ptron

# Calculate accuracy on the training data, which should go up over time if the learning works.
# Note that we could calculate this during learning, but this is just a cleaner function
# design and not significantly more time to calculate.
# Btw:  Accuracy isn't a great measure of learning when one class is more common
# than another, since we could get reasonable-looking accuracy just by predicting the more common class.
def perceptron_accuracy(ptron, examples, true_classes):
	correct = 0.0
	incorrect = 0.0
	for i in range(len(examples)):
		out_class = perceptron_predict(ptron, examples[i])
		if out_class == true_classes[i]:
			correct += 1.0
		else:
			incorrect += 1.0
	return correct/(correct+incorrect)
		
def perceptron_train(ptron, examples, true_classes, learn_rate, iterations):
	print("Starting accuracy: " + str(perceptron_accuracy(ptron,examples, true_classes)))
	for iteration in range(iterations):
		for i in range(len(examples)):
			ptron = perceptron_learn(ptron, examples[i], true_classes[i], learn_rate)
		accuracy = perceptron_accuracy(ptron, examples, true_classes)
	print("Final accuracy: " + str(accuracy))
	return ptron

def plot_everything(ptron, examples):
	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')
	for example in examples:
		ax.scatter(example[0], example[1], example[2], 'z', s=20, c=[(example[0], example[1], example[2])])
	xx, yy = np.meshgrid(np.arange(0,1.1,0.1),np.arange(0,1.1,0.1))
	# ax + by + cz + d == 0  becomes z == (-d - ax - by)/c
	z = (-ptron[3] - ptron[0]*xx - ptron[1]*yy)/ptron[2]
	ax.plot_surface(xx, yy, z, alpha=0.2)
	plt.show()

import perceptron
ptron = perceptron.new_perceptron()
examples, classes = perceptron.create_data()
perceptron.plot_everything(ptron, examples)
perceptron.perceptron_train(ptron, examples, classes, 0.01, 100)
perceptron.plot_everything(ptron, examples)

