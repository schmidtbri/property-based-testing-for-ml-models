import unittest
from pydantic import ValidationError

from mobile_handset_price_model.prediction.model import MobileHandsetPriceModel
from mobile_handset_price_model.prediction.schemas import MobileHandsetPriceModelInput, \
    MobileHandsetPriceModelOutput, PriceEnum


class ModelTests(unittest.TestCase):

    def test_model(self):
        # arrange
        model = MobileHandsetPriceModel()
        inpt = dict(battery_power=842,
                    has_bluetooth=True,
                    clock_speed=2.2,
                    has_dual_sim=False,
                    front_camera_megapixels=1,
                    has_four_g=False,
                    internal_memory=7,
                    depth=0.6,
                    weight=188,
                    number_of_cores=2,
                    primary_camera_megapixels=2,
                    pixel_resolution_height=20,
                    pixel_resolution_width=756,
                    ram=2549,
                    screen_height=9,
                    screen_width=7,
                    talk_time=19,
                    has_three_g=False,
                    has_touch_screen=False,
                    has_wifi=True)

        # act
        inpt = MobileHandsetPriceModelInput(**inpt)
        prediction = model.predict(inpt)

        # assert
        self.assertTrue(type(prediction) is MobileHandsetPriceModelOutput)
        self.assertTrue(type(prediction.price_range) is PriceEnum)

    def test_model_with_missing_optional_fields(self):
        # arrange
        model = MobileHandsetPriceModel()
        inpt = dict(has_bluetooth=False,
                    has_dual_sim=False,
                    has_four_g=False,
                    has_three_g=False,
                    has_touch_screen=False,
                    has_wifi=True)

        # act
        inpt = MobileHandsetPriceModelInput(**inpt)
        prediction = model.predict(inpt)

        # assert
        self.assertTrue(type(prediction) is MobileHandsetPriceModelOutput)
        self.assertTrue(type(prediction.price_range) is PriceEnum)

    def test_model_with_wrong_input_type(self):
        # arrange
        model = MobileHandsetPriceModel()
        inpt = dict(battery_power=842,
                    clock_speed=2.2,
                    has_dual_sim=False,
                    front_camera_megapixels=1,
                    has_four_g=False,
                    internal_memory=7,
                    weight=188,
                    number_of_cores=2,
                    primary_camera_megapixels=2,
                    pixel_resolution_height=20,
                    pixel_resolution_width=756,
                    ram=2549,
                    screen_height=9,
                    screen_width=7,
                    talk_time=19,
                    has_three_g=False,
                    has_touch_screen=False,
                    has_wifi=True)

        # act, assert
        with self.assertRaises(ValidationError):
            inpt = MobileHandsetPriceModelInput(**inpt)
            prediction = model.predict(inpt)


if __name__ == '__main__':
    unittest.main()
