import pandas as pd
import matplotlib.pyplot as plt
import random
import math

def calculate_accuracy(actual_labels, predicted_labels):
    # keep a count of predictions that match the provided labels
    correct = 0
    for i in range(len(actual_labels)):
        if actual_labels[i] == predicted_labels[i]:
            correct += 1
    # define accuracy as the percentage of correctly predicted labels
    accuracy = correct / float(len(actual_labels)) * 100.0
    return accuracy

# Calculate euclidean distance
def euclidean(train, test):
    ss = 0
    for i in range(len(train)):
        ss = ss + (train[i] - test[i])**2
    distance = math.sqrt(ss)
    return distance

# Calculate normalized euclidean distance
def normalized_euclidean(train, test):
    normalizing_factor = math.sqrt(sum([number ** 2 for number in train]))
    train = [float(number)/float(normalizing_factor) for number in train]
    normalizing_factor = math.sqrt(sum([number ** 2 for number in test]))
    test = [float(number)/float(normalizing_factor) for number in test]
    ss = 0
    for i in range(len(train)):
        ss = ss + (train[i] - test[i])**2
    distance = math.sqrt(ss)
    return distance

# Calculate cosine similarity
def cosine_similarity(train, test):
    normalizing_factor_train = math.sqrt(sum([number ** 2 for number in train]))
    normalizing_factor_test = math.sqrt(sum([number ** 2 for number in test]))
    similarity = 0
    for i in range(len(train)):
        similarity = similarity + (train[i]*test[i])
    similarity = float(similarity)/float(normalizing_factor_train*normalizing_factor_test)
    return similarity

# Locate the most similar neighbors
def get_neighbors(train, test_row, num_neighbors, distance):
	distances = list()
	for train_row in train:
		dist = distance(test_row, train_row)
		distances.append((train_row, dist))
	distances.sort(key=lambda tup: tup[1])
	neighbors = list()
	for i in range(num_neighbors):
		neighbors.append(distances[i][0])
	return neighbors

# Make a classification prediction with neighbors
def predict_classification(train, test_row, num_neighbors, distance):
	neighbors = get_neighbors(train, test_row, num_neighbors, distance)
	output_values = [row[-1] for row in neighbors]
	prediction = max(set(output_values), key=output_values.count)
	return prediction

df = pd.read_csv('cancer_dataset.csv')
df.drop('Index', inplace = True, axis = 1)
dataset = []
for i in range(len(df)):
    dataset.append(df.iloc[[i]].values[0])
random.shuffle(dataset)
train = dataset[:math.floor(0.8*len(dataset))]
test = dataset[math.floor(0.8*len(dataset)):]

for distance in [euclidean, normalized_euclidean, cosine_similarity]:
    scores = {}
    scores_list = []
    #k_range = range(1, 26)
    k_range = [1, 3, 5, 7]
    for k in k_range:
        prediction = []
        actual = []
        for i in range(len(test)):
            prediction.append(predict_classification(train, test[i], k, distance))
            actual.append(test[i][-1])
        scores[k] = calculate_accuracy(actual, prediction)
        scores_list.append(scores[k])

    plt.plot(k_range, scores_list, marker='o')
    plt.xlabel('Value of K for KNN')
    plt.ylabel('Testing Accuracy')
    if distance == euclidean:
        plt.title('Accuracy vs K with Euclidean Distance as distance metric')
        print('Accuracy values for different values of K with Euclidean Distance as distance metric:')
        print(scores)
        print()
    elif distance == normalized_euclidean:
        plt.title('Accuracy vs K with Normalized Euclidean Distance as distance metric')
        print('Accuracy values for different values of K with Normalized Euclidean Distance as distance metric:')
        print(scores)
        print()
    else:
        plt.title('Accuracy vs K with Cosine Similarity as distance metric')
        print('Accuracy values for different values of K with Cosine Similarity as distance metric:')
        print(scores)
        print()
    plt.show()
