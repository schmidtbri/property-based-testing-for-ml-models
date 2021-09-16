# Property-Based Testing for ML models

Building and deploying a regression ML model.

This code is used in this [blog post](https://brianschmidt-78145.medium.com/property-based-testing-for-ml-models-83847d6a781a).

## Requirements

Python 3

## Installation 

The Makefile included with this project contains targets that help to automate several tasks.

To download the source code execute this command:

```bash
git clone https://github.com/schmidtbri/property-testing-for-ml-models
```

Then create a virtual environment and activate it:

```bash
# go into the project directory
cd regression-model

make venv

source venv/bin/activate
```

Install the dependencies:

```bash
make dependencies
```

The requirements.txt file only includes the dependencies needed to make predictions with the model. To train the model you'll need to install the dependencies from the train_requirements.txt file:

```bash
make train-dependencies
```

## Running the Unit Tests
To run the unit test suite execute these commands:

```bash
# first install the test dependencies
make test-dependencies

# run the test suite
make test

# clean up the unit tests
make clean-test
```

## Running the Service

To start the service locally, execute these commands:

```bash
uvicorn rest_model_service.main:app --reload
```

To send a request to the service execute this command:

```bash
curl -X 'POST' \
  'http://127.0.0.1:8000/api/models/mobile_handset_price_model/prediction' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "battery_power": 501,
  "has_bluetooth": true,
  "clock_speed": 1.0,
  "has_dual_sim": true,
  "front_camera_megapixels": 1,
  "has_four_g": true,
  "internal_memory": 300,
  "depth": 0.2,
  "weight": 100,
  "number_of_cores": 4,
  "primary_camera_megapixels": 1,
  "pixel_resolution_height": 600,
  "pixel_resolution_width": 600,
  "ram": 500,
  "screen_height": 6,
  "screen_width": 1,
  "talk_time": 3,
  "has_three_g": true,
  "has_touch_screen": true,
  "has_wifi": true
}'
```

## Generating an OpenAPI Specification

To generate the OpenAPI spec file for the REST service that hosts the model, execute these commands:

```bash
export PYTHONPATH=./
generate_openapi --output_file=service_contract.yaml
```

## Docker

To build a docker image for the service, run this command:

```bash
docker build -t mobile_handset_price_model:0.1.0 .
```

To run the image, execute this command:

```bash
docker run -d -p 80:80 mobile_handset_price_model:0.1.0
```

To watch the logs coming from the image, execute this command:

```bash
docker logs $(docker ps -lq)
```

To stop the docker image, execute this command:

```bash
docker kill $(docker ps -lq)
```