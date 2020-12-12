import pandas as pd
from sklearn.tree import DecisionTreeClassifier

# function to calculate classification accuracy
def calculate_accuracy(data, labels, classifier):
    #print(classifier.tree_.max_depth)
    # predict labels using the trained classifier
    predicted_labels = classifier.predict(data)
    # keep a count of predictions that match the provided labels
    correct = 0
    for i in range(len(labels)):
        if labels.loc[i, 'class'] == predicted_labels[i]:
            correct += 1
    # define accuracy as the percentage of correctly predicted labels
    accuracy = correct / float(len(labels)) * 100.0
    return accuracy

# read training and test data while segregating into data and labels categories
data_training = pd.read_csv('iris_train_data.csv', usecols = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width'])
labels_training = pd.read_csv('iris_train_data.csv', usecols = ['class'])
data_test = pd.read_csv('iris_test_data.csv', usecols = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width'])
labels_test = pd.read_csv('iris_test_data.csv', usecols = ['class'])

# train classifier with criterion as entropy
classifier = DecisionTreeClassifier(criterion = 'entropy')
classifier.fit(data_training, labels_training)
# print accuracy on training and test data
print('Case I: entropy')
print('    Accuracy on training data: ' + str(calculate_accuracy(data_training, labels_training, classifier)))
print('    Accuracy on test data: ' + str(calculate_accuracy(data_test, labels_test, classifier)))
print()

# train classifier with criterion as entropy along with parameter tuning
classifier = DecisionTreeClassifier(criterion = 'entropy', max_depth = 3, min_samples_leaf = 10)
classifier.fit(data_training, labels_training)
# print accuracy on training and test data
print('Case II: entropy with parameter tuning')
print('    Accuracy on training data: ' + str(calculate_accuracy(data_training, labels_training, classifier)))
print('    Accuracy on test data: ' + str(calculate_accuracy(data_test, labels_test, classifier)))
print()

# train classifier with criterion as gini
classifier = DecisionTreeClassifier()
classifier.fit(data_training, labels_training)
# print accuracy on training and test data
print('Case III: gili')
print('    Accuracy on training data: ' + str(calculate_accuracy(data_training, labels_training, classifier)))
print('    Accuracy on test data: ' + str(calculate_accuracy(data_test, labels_test, classifier)))
print()

# train classifier with criterion as gini along with parameter tuning
classifier = DecisionTreeClassifier(max_depth = 2 , min_samples_leaf = 10)
classifier.fit(data_training, labels_training)
# print accuracy on training and test data
print('Case IV: gili with parameter tuning')
print('    Accuracy on training data: ' + str(calculate_accuracy(data_training, labels_training, classifier)))
print('    Accuracy on test data: ' + str(calculate_accuracy(data_test, labels_test, classifier)))
print()
