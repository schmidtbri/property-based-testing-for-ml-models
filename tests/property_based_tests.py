import unittest
from unittest import TestCase
from hypothesis import given, strategies, settings

from mobile_handset_price_model.prediction.model import MobileHandsetPriceModel
from mobile_handset_price_model.prediction.schemas import MobileHandsetPriceModelInput, MobileHandsetPriceModelOutput, \
    PriceEnum


class ModelPropertyBasedTests(TestCase):

    def setUp(self) -> None:
        self.counter = 0
        self.model = MobileHandsetPriceModel()

    def tearDown(self) -> None:
        print("Generated and tested {} examples.".format(self.counter))

    @settings(deadline=None, max_examples=1000)
    @given(strategies.builds(MobileHandsetPriceModelInput))
    def test_model_input(self, data):
        # act
        result = self.model.predict(data=data)

        # assert
        self.assertTrue(type(result) is MobileHandsetPriceModelOutput)
        self.assertTrue(type(result.price_range) is PriceEnum)

        self.counter += 1


if __name__ == '__main__':
    unittest.main()
