import os
import joblib
import pandas as pd
from ml_base import MLModel

from mobile_handset_price_model import __version__
from mobile_handset_price_model.prediction.schemas import MobileHandsetPriceModelInput, MobileHandsetPriceModelOutput,\
    PriceEnum

# map used to convert the output of the model to the output Enum
output_class_map = {
    "0": PriceEnum.zero,
    "1": PriceEnum.one,
    "2": PriceEnum.two,
    "3": PriceEnum.three
}


class MobileHandsetPriceModel(MLModel):
    """Prediction functionality of the Mobile Handset Price Model."""

    @property
    def display_name(self) -> str:
        """Return display name of model."""
        return "Mobile Handset Price Model"

    @property
    def qualified_name(self) -> str:
        """Return qualified name of model."""
        return "mobile_handset_price_model"

    @property
    def description(self) -> str:
        """Return description of model."""
        return "Model to predict the price of a mobile phone."

    @property
    def version(self) -> str:
        """Return version of model."""
        return __version__

    @property
    def input_schema(self):
        """Return input schema of model."""
        return MobileHandsetPriceModelInput

    @property
    def output_schema(self):
        """Return output schema of model."""
        return MobileHandsetPriceModelOutput

    def __init__(self):
        """Class constructor that loads and deserializes the model parameters.

        .. note::
            The trained model parameters are loaded from the "model_files" directory.

        """
        dir_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
        with open(os.path.join(dir_path, "model_files", "1", "model.joblib"), 'rb') as file:
            self._svm_model = joblib.load(file)

    def predict(self, data: MobileHandsetPriceModelInput) -> MobileHandsetPriceModelOutput:
        """Make a prediction with the model.

        :param data: Data for making a prediction with the model. Object must meet requirements of the input schema.
        :rtype: The result of the prediction, the output object will meet the requirements of the output schema.

        """
        # converting the incoming data into a pandas dataframe that can be accepted by the model
        X = pd.DataFrame([[data.battery_power, data.has_bluetooth, data.clock_speed, data.has_dual_sim,
                           data.front_camera_megapixels, data.has_four_g, data.internal_memory, data.depth, data.weight,
                           data.number_of_cores, data.primary_camera_megapixels, data.pixel_resolution_height,
                           data.pixel_resolution_width, data.ram, data.screen_height, data.screen_width, data.talk_time,
                           data.has_three_g, data.has_touch_screen, data.has_wifi]],
                         columns=["battery_power", "has_bluetooth", "clock_speed", "has_dual_sim",
                                  "front_camera_megapixels", "has_four_g", "internal_memory", "depth", "weight",
                                  "number_of_cores", "primary_camera_megapixels", "pixel_resolution_height",
                                  "pixel_resolution_width", "ram", "screen_height", "screen_width", "talk_time",
                                  "has_three_g", "has_touch_screen", "has_wifi"])

        # making the prediction and extracting the result from the array
        y_hat = output_class_map[str(self._svm_model.predict(X)[0])]

        return MobileHandsetPriceModelOutput(price_range=y_hat)
