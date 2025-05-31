import unittest
import joblib
import numpy as np

class TestIrisModel(unittest.TestCase):
    def setUp(self):
        self.model = joblib.load('iris_model.pkl')

    def test_prediction_setosa(self):
        input_data = np.array([[5.1, 3.5, 1.4, 0.2]])
        pred = self.model.predict(input_data)
        self.assertEqual(pred[0], 0)  # Assuming 'setosa' is encoded as 0

if __name__ == '__main__':
    unittest.main()
