import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.pyplot import axis
from numpy import *
import unittest


def get_data():
    np.random.seed(10)
    data = np.random.binomial(100, 0.5, size=100)
    X = data.reshape((100, 1))

    return X


def get_normal_distributed_features(x1, x2, y1, y2, size, k=1):
    sigma = 1
    x_length = x2 - x1
    y_length = y2 - y1

    data_x = (np.random.normal(0, sigma, size=size) + sigma * 3) / (k * 6 * sigma) * x_length + x1
    data_y = (np.random.normal(0, sigma, size=size) + sigma * 3) / (k * 6 * sigma) * y_length + y1

    return np.hstack([data_x.reshape([size, 1]),
                      data_y.reshape([size, 1])])


def euclidian_distance_metric(feature_line: np.ndarray, x, y):
    f_x = feature_line[:, 0, np.newaxis]
    f_y = feature_line[:, 1, np.newaxis]
    return np.sqrt((f_x - x) ** 2 + (f_y - y) ** 2)


#  ======================================
#  =================MODELS===============
#  --------- NN -----------

class NNmodel:

    features = None
    classes = None

    def getclassof(self, x, y):
        return self.classes[np.argmin(euclidian_distance_metric(self.features, x, y), axis=0)]

    def learn(self, features: np.ndarray, classes: np.array):
        self.features = features
        self.classes = classes
        return self

    def __init__(self):
        super().__init__()


class NNModelTest(unittest.TestCase):
    def test_learning(self):
        dataset = np.array([
            [1, 2, 0],
            [3, 4, 1],
            [5, 6, 3]
        ])

        self.assertEqual(NNmodel().learn(dataset[:, :2], dataset[:, 2]).getclassof(1, 1), 0)
        self.assertEqual(NNmodel().learn(dataset[:, :2], dataset[:, 2]).getclassof(3.1, 4.1), 1)
        self.assertEqual(NNmodel().learn(dataset[:, :2], dataset[:, 2]).getclassof(10, 10), 3)

    def test_matrics(self):
        dataset = np.array([
            [1, 2, 0],
            [3, 4, 1],
            [5, 6, 3]
        ])

        x = np.linspace(0, 10, 4)
        y = np.linspace(0, 10, 4)

        # Формируем координатную сетку
        X, Y = np.meshgrid(x, y)

        Xi = X.reshape(4 * 4)
        Yi = Y.reshape(4 * 4)

        Zi = NNmodel().learn(dataset[:, :2], dataset[:, 2]).getclassof(Xi, Yi)
        Z = Zi.reshape((4, 4))

        self.assertEqual(Z[0, 0], 0)
        self.assertEqual(Z[3, 3], 3)


# ----------- kNN -----------

class KNNmodel:
    def getclassof(self, x, y):
        weights = euclidian_distance_metric(self.features, x, y)

        sorted = argsort(weights[:], axis=0)
        # neibour = where(sorted<self.k, sorted)

        sortedClasses = self.classes[sorted]

        classes = delete(sortedClasses, s_[self.k:], 0)

        def get_max_count(a):
            elem, counts = unique(a, return_counts=True)
            return elem[argmax(counts)]

        resultClass = apply_along_axis(get_max_count, 0, classes)

        return resultClass

    def learn(self, features: np.ndarray, classes: np.array):
        self.features = features
        self.classes = classes
        return self

    def __init__(self, k):
        super().__init__()
        self.k = k


class KNNmodelTest(unittest.TestCase):
    def test_learn(self):
        dataset = np.array([
            [0, 0, 2],
            [1, 0, 2],
            [2, 0, 2],
            [1, 1.1, 1],
            # [1, 0.9, 1],
        ])

        self.assertEqual(KNNmodel(3).learn(dataset[:, :2], dataset[:, 2]).getclassof(1, 1), 2)

    def test_matrix_learn(self):
        dataset = np.array([
            [0, 0, 2],
            [1, 0, 2],
            [2, 0, 2],
            [1, 1.1, 1],
            [1, 0.9, 2],
            [2, 1, 1],
        ])

        x = np.linspace(0, 3, 4)
        y = np.linspace(0, 3, 4)

        # Формируем координатную сетку
        X, Y = np.meshgrid(x, y)

        Xi = X.reshape(4 * 4)
        Yi = Y.reshape(4 * 4)

        Zi = KNNmodel(3).learn(dataset[:, :2], dataset[:, 2]).getclassof(Xi, Yi)
        Z = Zi.reshape((4, 4))

        self.assertEqual(Z[0, 0], 2)
        self.assertEqual(Z[3, 3], 1)


if __name__ == '__main__':
    unittest.main()