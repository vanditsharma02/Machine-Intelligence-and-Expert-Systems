1) Place the Mall_Customers.csv dataset in the same directory where the python program (a2.py) is placed.

Follow these steps to run the code file (a2.py)-

1) Install virtualenv in python using "pip install virtualenv".
2) Create a virtual environment using virtualenv in python. I used PyCharm for this purpose, but you can also use command line interface in Linux.
3) Install pandas, scikit-learn and matplotlib in the virtual environment. You can use PyCharm directly, or otherwise use pip to install these in case of command line interface.
4) Run the program file using PyCharm or python in command line ("python3 a2.py").
5) By default, you should see the plots corresponding to the combination {annual income, spending score}. The first plot is the objective function vs number of clusters and the second plot is the clustering output.
6) In order to run the code for other combinations, you need to load the correct datapoints corresponding to that combination. To do this, uncomment one desired code line among lines 8, 11, 14 and keep the other lines commented out.
7) You can also vary the number of clusters by changing the value of n_clusters in the code (line 41). Make sure to add all clusters to the scatter plot in case you decide to change the number of clusters.