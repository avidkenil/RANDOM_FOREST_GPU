import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split

iris = datasets.load_iris()
iris_x = iris.data
iris_y = iris.target
iris_train_x, iris_test_x, iris_train_y, iris_test_y = train_test_split(
    iris_x, iris_y, test_size=0.33, random_state=42)

np.save("iris_train_x.npy", iris_train_x)
np.save("iris_train_y.npy", iris_train_y)
np.save("iris_test_x.npy", iris_test_x)
np.save("iris_test_y.npy", iris_test_y)
