import numpy as np
import os
import sys

from sklearn import datasets
from sklearn.model_selection import train_test_split
import sklearn.ensemble

import extra_trees


def get_accuracy(pred, true):
    return (pred == true).mean()


def display_stats(model_name, model, train_x, train_y, test_x, test_y):
    print(f"=> {model_name}")
    print("TRAIN: ACCURACY: {}".format(
        get_accuracy(model.predict(train_x), train_y))
    )
    print("TEST: ACCURACY: {}".format(
        get_accuracy(model.predict(test_x), test_y))
    )
    print()


def main(data_path):
    mnist_train_x = np.load(f"{data_path}/train_x.npy")
    mnist_train_y = np.load(f"{data_path}/train_y.npy")
    mnist_test_x = np.load(f"{data_path}/test_x.npy")
    mnist_test_y = np.load(f"{data_path}/test_y.npy")

    iris = datasets.load_iris()
    iris_x = iris.data
    iris_y = iris.target
    iris_train_x, iris_test_x, iris_train_y, iris_test_y = train_test_split(
        iris_x, iris_y, test_size=0.33, random_state=42)

    skl_et_iris = sklearn.ensemble.ExtraTreesClassifier(
        n_estimators=1,
        criterion="gini",
        n_jobs=-1,
    )
    skl_et_iris.fit(iris_train_x, iris_train_y)

    skl_et_mnist = sklearn.ensemble.ExtraTreesClassifier(
        n_estimators=1,
        criterion="gini",
        n_jobs=-1,
    )
    skl_et_mnist.fit(mnist_train_x, mnist_train_y)

    cpu_et_iris = extra_trees.Tree()
    cpu_et_iris.fit(iris_train_x, iris_train_y)

    cpu_et_mnist = extra_trees.Tree()
    cpu_et_mnist.fit(mnist_train_x, mnist_train_y)

    display_stats(
        model_name="SKLEARN IRIS", model=skl_et_iris,
        train_x=iris_train_x, train_y=iris_train_y,
        test_x=iris_test_x, test_y=iris_test_y,
    )

    display_stats(
        model_name="CPU IRIS", model=cpu_et_iris,
        train_x=iris_train_x, train_y=iris_train_y,
        test_x=iris_test_x, test_y=iris_test_y,
    )

    display_stats(
        model_name="SKLEARN MNIST", model=skl_et_mnist,
        train_x=mnist_train_x, train_y=mnist_train_y,
        test_x=mnist_test_x, test_y=mnist_test_y,
    )

    display_stats(
        model_name="CPU MNIST", model=cpu_et_mnist,
        train_x=mnist_train_x, train_y=mnist_train_y,
        test_x=mnist_test_x, test_y=mnist_test_y,
    )



if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise RuntimeError("Provide data path")
    main(sys.argv[1])
