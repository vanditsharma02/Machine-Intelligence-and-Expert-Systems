# Import modules
from random import seed
from random import randrange
from random import random
from math import exp
import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV file and prepare dataset
def load_csv(filename):
    dataset = list()
    df = pd.read_csv(filename)
    df.drop(['Id'], axis=1)
    for index, row in df.iterrows():
        dataset.append([str(row['SepalLengthCm']), str(row['SepalWidthCm']), str(row['PetalLengthCm']), str(row['PetalWidthCm']), row['Species']])
    return dataset

# Convert string column to float
def str_column_to_float(dataset, column):
	for row in dataset:
		row[column] = float(row[column].strip())

# Convert string column to integer
def str_column_to_int(dataset, column):
    class_values = [row[column] for row in dataset]
    unique = set(class_values)
    temp = list(unique)
    print('\nOutput-class mapping:')
    for i in range(len(temp)):
        print(str(i) + '=>' + temp[i])
    print()
    lookup = dict()
    for i, value in enumerate(unique):
        lookup[value] = i
    for row in dataset:
        row[column] = lookup[row[column]]
    return lookup

# Find the min and max values for each column
def dataset_minmax(dataset):
	minmax = list()
	stats = [[min(column), max(column)] for column in zip(*dataset)]
	return stats

# Normalize dataset columns to the range 0-1
def normalize_dataset(dataset, minmax):
	for row in dataset:
		for i in range(len(row)-1):
			row[i] = (row[i] - minmax[i][0]) / (minmax[i][1] - minmax[i][0])

# Split the dataset into 5 folds
def cross_validation_split(dataset, n_folds):
	dataset_split = list()
	dataset_copy = list(dataset)
	fold_size = int(len(dataset) / n_folds)
	for i in range(n_folds):
		fold = list()
		while len(fold) < fold_size:
			index = randrange(len(dataset_copy))
			fold.append(dataset_copy.pop(index))
		dataset_split.append(fold)
	return dataset_split

# Calculate accuracy percentage
def accuracy_metric(actual, predicted):
	correct = 0
	for i in range(len(actual)):
		if actual[i] == predicted[i]:
			correct += 1
	return correct / float(len(actual)) * 100.0

# Evaluate the network using a cross validation split (80-20)
def evaluate_nn(dataset, algorithm, n_folds, *args):
	folds = cross_validation_split(dataset, n_folds)
	scores = list()
	num = 1
	original_folds = folds
	for fold in folds:
		print('Fold number: '+ str(num))
		num = num + 1
		train_set = list(folds)
		train_set.remove(fold)
		train_set = sum(train_set, [])
		#Uncomment for testing accuracy
		test_set = list()
		for row in fold:
			row_copy = list(row)
			test_set.append(row_copy)
			row_copy[-1] = None
		predicted = algorithm(train_set, test_set, *args)
		actual = [row[-1] for row in fold]

		#Uncomment for training accuracy
		# test_set = list()
		# original_folds.remove(fold)
		# fold_new = []
		# for f in original_folds:
		# 	fold_new = fold_new + f
		# for row in fold_new:
		# 	row_copy = list(row)
		# 	test_set.append(row_copy)
		# 	row_copy[-1] = None
		# predicted = algorithm(train_set, test_set, *args)
		# actual = [row[-1] for row in fold_new]

		accuracy = accuracy_metric(actual, predicted)
		scores.append(accuracy)
	return scores

# Calculate neuron activation for an input
def activate(weights, inputs):
	activation = weights[-1]
	for i in range(len(weights)-1):
		activation += weights[i] * inputs[i]
	return activation

# Transfer neuron activation
def transfer(activation):
	return 1.0 / (1.0 + exp(-activation))

# Forward propagate input to a network output
def forward_propagate(network, row):
	inputs = row
	for layer in network:
		new_inputs = []
		for neuron in layer:
			activation = activate(neuron['weights'], inputs)
			neuron['output'] = transfer(activation)
			new_inputs.append(neuron['output'])
		inputs = new_inputs
	return inputs

# Calculate the derivative of an neuron output
def transfer_derivative(output):
	return output * (1.0 - output)

# Backpropagate error and store in neurons
def backward_propagate_error(network, expected):
	for i in reversed(range(len(network))):
		layer = network[i]
		errors = list()
		if i != len(network)-1:
			for j in range(len(layer)):
				error = 0.0
				for neuron in network[i + 1]:
					error += (neuron['weights'][j] * neuron['delta'])
				errors.append(error)
		else:
			for j in range(len(layer)):
				neuron = layer[j]
				errors.append(expected[j] - neuron['output'])
		for j in range(len(layer)):
			neuron = layer[j]
			neuron['delta'] = errors[j] * transfer_derivative(neuron['output'])

# Update network weights with error
def update_weights(network, row, l_rate):
	for i in range(len(network)):
		inputs = row[:-1]
		if i != 0:
			inputs = [neuron['output'] for neuron in network[i - 1]]
		for neuron in network[i]:
			for j in range(len(inputs)):
				neuron['weights'][j] += l_rate * neuron['delta'] * inputs[j]
			neuron['weights'][-1] += l_rate * neuron['delta']

# Train a network for a fixed number of epochs
def train_network(network, train, l_rate, n_epoch, n_outputs):
    x = []
    y = []
    for epoch in range(n_epoch):
        x.append(epoch+1)
        cost = 0
        for row in train:
            outputs = forward_propagate(network, row)
            expected = [0 for i in range(n_outputs)]
            expected[row[-1]] = 1
            cost += (1/2)*sum([(expected[i] - outputs[i]) ** 2 for i in range(len(expected))])
            backward_propagate_error(network, expected)
            update_weights(network, row, l_rate)
        #print('epoch=%d, cost=%.3f' % (epoch, cost))
        y.append(cost)
    plt.plot(x, y, linestyle='solid', color='blue')
    plt.title("Cost vs Epoch for training data")
    plt.xlabel("Epoch")
    plt.ylabel("Cost")
    plt.show()

# Initialize a network
def initialize_network(n_inputs, n_hidden, n_outputs):
    network = list()
    hidden_layer = [{'weights':[random() for i in range(n_inputs + 1)]} for i in range(n_hidden)]
    network.append(hidden_layer)
    output_layer = [{'weights':[random() for i in range(n_hidden + 1)]} for i in range(n_outputs)]
    network.append(output_layer)
    return network

# Make a prediction with a network
def predict(network, row):
	outputs = forward_propagate(network, row)
	return outputs.index(max(outputs))

# Backpropagation Algorithm With Gradient Descent
def back_propagation(train, test, l_rate, n_epoch, n_hidden, to_classify):
    n_inputs = len(train[0]) - 1
    n_outputs = len(set([row[-1] for row in train]))
    network = initialize_network(n_inputs, n_hidden, n_outputs)
    train_network(network, train, l_rate, n_epoch, n_outputs)
    predictions = list()
    for i in range(len(to_classify)):
        row = to_classify[i]
        prediction = predict(network, row)
        print('Classified given sample ' + str(to_classify_labels[i]) + ' as: ' + str(prediction))
    print()
    # Uncomment for testing accuracy
    for i in range(len(test)):
        row = test[i]
        prediction = predict(network, row)
        print('Classified (normalized) test sample ' + str(row) + ' as: ' + str(prediction))
        predictions.append(prediction)
    print()

    # Uncomment for training accuracy
    # for i in range(len(train)):
    #     row = train[i]
    #     prediction = predict(network, row)
    #     print('Classified test sample (normalized) ' + str(row) + ' as: ' + str(prediction))
    #     predictions.append(prediction)

    return(predictions)

# Set seed for random function
seed(1)
# Load and prepare dataset
filename = 'Iris.csv'
dataset = load_csv(filename)
for i in range(len(dataset[0])-1):
	str_column_to_float(dataset, i)
# convert class column to integers
str_column_to_int(dataset, len(dataset[0])-1)

# Get values of data before normalization
data = {}
for i in range(len(dataset[0])-1):
    data[i] = []
    for row in dataset:
        data[i].append(row[i])

data = [data[0], data[1], data[2], data[3]]
fig = plt.figure(figsize=(14, 14))
# Creating axes instance
ax = fig.add_subplot(211)
# Creating subplot
bp = ax.boxplot(data, labels = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm'])
plt.title('Before normalization')

# Store original dataset
original_dataset = dataset
original_dataset_split = list()

# Normalize input variables
minmax = dataset_minmax(dataset)
normalize_dataset(dataset, minmax)

# Get values of data after normalization
data = {}
for i in range(len(dataset[0])-1):
    data[i] = []
    for row in dataset:
        data[i].append(row[i])

data = [data[0], data[1], data[2], data[3]]
# Creating axes instance
ax = fig.add_subplot(212)
# Creating plot
bp = ax.boxplot(data, labels = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm'])
plt.title('After normalization')
# Show plot comparing data before and after normalization
plt.show()

# Load data to classify
to_classify = [[4.6, 3.5, 1.8, 0.2, None],
              [5.9, 2.5, 1.6, 1.6, None],
              [5, 4.2, 3.7, 0.3, None],
              [5.7, 4, 4.2, 1.2, None]]

to_classify_labels = [[4.6, 3.5, 1.8, 0.2],
					  [5.9, 2.5, 1.6, 1.6],
					  [5, 4.2, 3.7, 0.3],
					  [5.7, 4, 4.2, 1.2]]
# Normalize data before making classification
normalize_dataset(to_classify, minmax)

# Train and evaluate neural network
n_folds = 5
l_rate = 0.15
n_hidden = 3

# Get accuracy for different values of epoch
epochs = [2, 5, 8, 10, 20, 50, 80, 100, 200, 500, 800, 1000]
#epochs = [2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
accuracy = []
for epoch in epochs:
    print('Number of epochs: ' + str(epoch) + '\n')
    scores = evaluate_nn(dataset, back_propagation, n_folds, l_rate, epoch, n_hidden, to_classify)
    accuracy.append(sum(scores) / float(len(scores)))
    print('Mean accuracy with ' + str(epoch) + ' epochs = %.3f%%\n' % (sum(scores) / float(len(scores))))

plt.plot(epochs, accuracy, linestyle='solid', color='blue')
plt.title("Accuracy vs Number of Epochs")
plt.xlabel("Number of Epochs")
plt.ylabel("Accuracy")
plt.show()