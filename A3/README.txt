1) Place the Iris.csv dataset in the same directory where the python program (a3.py) is placed.

Follow these steps to run the code file (a3.py)-

1) Install virtualenv in python using "pip install virtualenv".
2) Create a virtual environment using virtualenv in python. I used PyCharm for this purpose, but you can also use command line interface in Linux.
3) Install pandas, and matplotlib in the virtual environment. You can use PyCharm directly, or otherwise use pip to install these in case of command line interface.
4) Run the program file using PyCharm or python in command line ("python3 a3.py").
5) By default, you should see the plots corresponding to the cost vs epochs and accuracy vs number of epochs. You should also see the different predictions being printed.
NOTE: I have implemented a 5-fold evaluation approach, meaning for each value of number of epochs, the training and testing takes place 5 times on 5 possible combinations of training and testing data (80:20). I report the accuracy which is the average of these 5 folds.
6) To switch between training and testing accuracies, you need to uncomment few lines of code, which has been written in the comments.