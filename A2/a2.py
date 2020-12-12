import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

dataset = pd.read_csv('Mall_Customers.csv')

# Uncomment to select the age and the annual income columns
#datapoints = dataset.iloc[:, [2,3]].values

# Uncomment to select the annual income and the spending score columns
datapoints = dataset.iloc[:,[3,4]].values

# Uncomment to select the age and the spending score columns
#datapoints = dataset.iloc[:, [2,4]].values

# Calculate the value of objective function for a range of number of clusters
objective_function_values = []
for i in range(1, 11):
    km = KMeans(
        n_clusters=i, init='random',
        n_init=10, max_iter=300,
        tol=1e-04, random_state=0
    )
    km.fit(datapoints)
    objective_function_values.append(km.inertia_)

# plot
plt.plot(range(1, 11), objective_function_values, marker='o')
# Choose appropriate axis labels and title
plt.title('Objective function vs Number of clusters')
plt.xlabel('Number of clusters')
plt.ylabel('Objective Function')
# Show plot
plt.show(block = False)
plt.figure()

# Infer the number of clusters by elbow finding
# In our case 5 seems to be a good number of clusters for all 3 combinations

km = KMeans(
    n_clusters=5, init='random',
    n_init=10, max_iter=300,
    tol=1e-04, random_state=0
)
y_km = km.fit_predict(datapoints)

# Plot the 5 clusters
plt.scatter(
    datapoints[y_km == 0, 0], datapoints[y_km == 0, 1],
    s=50, c='lightgreen',
    marker='s', edgecolor='black',
    label='Cluster 1'
)

plt.scatter(
    datapoints[y_km == 1, 0], datapoints[y_km == 1, 1],
    s=50, c='orange',
    marker='o', edgecolor='black',
    label='Cluster 2'
)

plt.scatter(
    datapoints[y_km == 2, 0], datapoints[y_km == 2, 1],
    s=50, c='cyan',
    marker='v', edgecolor='black',
    label='Cluster 3'
)

plt.scatter(
    datapoints[y_km == 3, 0], datapoints[y_km == 3, 1],
    s=50, c='yellow',
    marker='h', edgecolor='black',
    label='Cluster 4'
)

plt.scatter(
    datapoints[y_km == 4, 0], datapoints[y_km == 4, 1],
    s=50, c='violet',
    marker='d', edgecolor='black',
    label='Cluster 5'
)

# Plot the centroids of 5 clusters
plt.scatter(
    km.cluster_centers_[:, 0], km.cluster_centers_[:, 1],
    s=250, marker='*',
    c='red', edgecolor='black',
    label='Centroids'
)
plt.legend(scatterpoints=1)
# Choose appropriate axis labels and title
plt.title('Clusters of Customers')
plt.xlabel('Annual Income(k$)')
plt.ylabel('Spending Score(1-100)')
plt.grid()
# Show plot
plt.show()


