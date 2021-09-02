from typing import Optional
from pydantic import BaseModel, Field
from enum import Enum


class MobileHandsetPriceModelInput(BaseModel):
    """Schema for input of the model's predict method."""

    battery_power: Optional[int] = Field(None, title="battery_power", ge=500, le=2000,
                                         description="Total energy a battery can store in one time measured in mAh.")
    has_bluetooth: bool = Field(..., title="has_bluetooth", description="Whether the phone has bluetooth.")
    clock_speed: Optional[float] = Field(None, title="clock_speed", ge=0.5, le=3.0,
                                         description="Speed of microprocessor in gHz.")
    has_dual_sim: bool = Field(..., title="has_dual_sim", description="Whether the phone has dual SIM slots.")
    front_camera_megapixels: Optional[int] = Field(None, title="front_camera_megapixels", ge=0, le=20,
                                                   description="Front camera mega pixels.")
    has_four_g: bool = Field(..., title="has_four_g", description="Whether the phone has 4G.")
    internal_memory: Optional[int] = Field(None, title="internal_memory", ge=2, le=664,
                                           description="Internal memory in gigabytes.")
    depth: float = Field(None, title="depth", ge=0.1, le=1.0, description="Depth of mobile phone in cm.")
    weight: Optional[int] = Field(None, title="weight", ge=80, le=200, description="Weight of mobile phone.")
    number_of_cores: Optional[int] = Field(None, title="number_of_cores", ge=1, le=8,
                                           description="Number of cores of processor.")
    primary_camera_megapixels: Optional[int] = Field(None, title="primary_camera_megapixels", ge=0, le=20,
                                                     description="Primary camera mega pixels.")
    pixel_resolution_height: Optional[int] = Field(None, title="pixel_resolution_height", ge=0, le=1960,
                                                   description="Pixel resolution height.")
    pixel_resolution_width: Optional[int] = Field(None, title="pixel_resolution_width", ge=500, le=1998,
                                                  description="Pixel resolution width.")
    ram: Optional[int] = Field(None, title="ram", ge=256, le=3998, description="Random access memory in megabytes.")
    screen_height: Optional[int] = Field(None, title="screen_height", ge=5, le=19,
                                         description="Screen height of mobile in cm.")
    screen_width: Optional[int] = Field(None, title="screen_width", ge=0, le=18,
                                        description="Screen width of mobile in cm.")
    talk_time: Optional[int] = Field(None, title="talk_time", ge=2, le=20,
                                     description="Longest time that a single battery charge will last when on phone "
                                                 "call.")
    has_three_g: bool = Field(..., title="has_three_g", description="Whether the phone has 3G touchscreen or not.")
    has_touch_screen: bool = Field(..., title="has_touch_screen", description="Whether the phone has a touchscreen or "
                                                                              "not.")
    has_wifi: bool = Field(..., title="has_wifi", description="Whether the phone has wifi or not.")


class PriceEnum(str, Enum):
    zero = "zero"
    one = "one"
    two = "two"
    three = "three"


class MobileHandsetPriceModelOutput(BaseModel):
    """Schema for output of the model's predict method."""
    price_range: PriceEnum = Field(..., title="Price Range", description="Price range class.")
