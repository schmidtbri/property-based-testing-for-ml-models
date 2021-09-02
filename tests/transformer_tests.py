import unittest
import numpy as np

from mobile_handset_price_model.prediction.transformers import BooleanTransformer


class TransformersTests(unittest.TestCase):

    def test_boolean_transformer(self):
        # arrange
        boolean_transformer = BooleanTransformer(true_value=1, false_value=0)
        X = [[1], [0], [1]]

        # act
        boolean_transformer.fit(X)
        result = boolean_transformer.transform(X)

        # assert
        self.assertTrue((result == np.array([[True], [False], [True]])).all())

    def test_boolean_transformer_with_boolean_values(self):
        # arrange
        boolean_transformer = BooleanTransformer(true_value=1, false_value=0)
        X = [[True], [False], [False]]

        # act
        boolean_transformer.fit(X)
        result = boolean_transformer.transform(X)

        # assert
        self.assertTrue((result == np.array([[True], [False], [False]])).all())

    def test_boolean_transformer_with_bad_values(self):
        # arrange
        boolean_transformer = BooleanTransformer(true_value=1, false_value=0)
        X = [[1, 5], [0, 'asd'], [1, 0.5]]

        # act, assert
        boolean_transformer.fit(X)

        with self.assertRaises(ValueError):
            result = boolean_transformer.transform([[5, 'asd'], [0, 'asd'], [1, 'asd']])
            pass


if __name__ == '__main__':
    unittest.main()