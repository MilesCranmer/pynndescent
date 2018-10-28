import unittest
import numpy as np
from pynndescent import NNDescent

class TestAll(unittest.TestCase):
    def test_basic_load(self):
        data = np.random.normal((5, 4), (6, 0.3), size=(15, 2))
        index = NNDescent(data)

        print(index.query(np.array([6.00, 4.00]), k=15))

