import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.pyplot import axis
from numpy import *
import unittest
import sklearn.utils

from data_preparing import get_normal_distributed_features, NNmodel


class CrossValidation:
    def __init__(self, model, n, learning_k=0.8, printlog=False):
        super().__init__()
        if learning_k >= 1 or learning_k < 0:
            raise Exception("learning_percent must be between 0 and 100")
        self.printlog = printlog
        self.model = model
        self.n = n
        self.learning_k = learning_k

    def calculate(self, features, classes):
        full_error = 0
        if self.printlog:
            print("==== start calculate ====")
        for n in range(0, self.n):

            if self.printlog:
                print("[", n/self.n*100, "%]", end='\r')

            rand_mask = random.choice([True, False], features.shape[0], p=[self.learning_k, 1 - self.learning_k])

            learning_sample, learning_classes = features[rand_mask], classes[rand_mask]
            check_sample, check_classes = features[~rand_mask], classes[~rand_mask]

            self.model.learn(learning_sample, learning_classes)

            error = sum(self.model.getclassof(check_sample[:, 0], check_sample[:, 1]) - check_classes)**2

            if len(check_sample)==0:
                print("len(check_sample) == 0")
                continue

            error /= len(check_sample)
            full_error += error

        full_error /= self.n

        if self.printlog:
            print("==== finish calculate ====")

        return full_error


class CrossValidationTest(unittest.TestCase):
    def test_get_validation_value(self):
        part_size = 450

        dataset = array([
            [1, 8, 1],
            [1, 9, 1],
            [2, 9, 1],
            [8, 1, 2],
            [9, 1, 2],
            [9, 2, 2]
        ])

        cv = CrossValidation(NNmodel(), 10)
        result = cv.calculate(dataset[:, :2], dataset[:, 2])

        self.assertTrue(result >= 0 and result <= 1)


if __name__ == '__main__':
    unittest.main()
