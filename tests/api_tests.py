import unittest
from unittest import TestCase
from hypothesis import given, settings
import schemathesis
from schemathesis import DataGenerationMethod

from rest_model_service.main import app

schema = schemathesis.from_asgi("/openapi.json", app,
                                data_generation_methods=[DataGenerationMethod.negative])
model_metadata_strategy = schema["/api/models"]["GET"].as_strategy()
model_prediction_strategy = schema["/api/models/mobile_handset_price_model/prediction"]["POST"].as_strategy()


class APITests(TestCase):

    def setUp(self) -> None:
        self.counter = 0

    def tearDown(self) -> None:
        print("Generated and tested {} examples.".format(self.counter))

    @given(case=model_metadata_strategy)
    @settings(deadline=None)
    def test_model_metadata_endpoint(self, case):
        response = case.call_asgi()
        case.validate_response(response)
        self.counter += 1

    @given(case=model_prediction_strategy)
    @settings(deadline=None, max_examples=1000)
    def test_model_prediction_endpoint(self, case):
        response = case.call_asgi()
        case.validate_response(response)
        self.counter += 1


if __name__ == '__main__':
    unittest.main()
