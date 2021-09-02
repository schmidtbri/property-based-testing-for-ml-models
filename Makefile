TEST_PATH=./tests

.DEFAULT_GOAL := help

.PHONY: help download-dataset train-model save-model clean-pyc build clean-build venv dependencies test-dependencies clean-venv test test-reports clean-test check-codestyle check-docstyle

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'

download-dataset: ## download dataset from Kaggle
	mkdir -p data
	kaggle datasets download -d iabhishekofficial/mobile-price-classification/tasks -p ./data --unzip

train-model:  ## run training notebooks in order
	jupyter nbconvert --execute --to html mobile_handset_price_model/training/data_exploration.ipynb
	jupyter nbconvert --execute --to html mobile_handset_price_model/training/data_preparation.ipynb
	jupyter nbconvert --execute --to html mobile_handset_price_model/training/model_training.ipynb
	jupyter nbconvert --execute --to html mobile_handset_price_model/training/model_validation.ipynb

save-model:  ## save currently trained model to package
	mkdir -p mobile_handset_price_model/model_files/$(PARAMETERS_VERSION)
	mv mobile_handset_price_model/training/*.html mobile_handset_price_model/model_files/$(PARAMETERS_VERSION)
	mv mobile_handset_price_model/training/model.joblib mobile_handset_price_model/model_files/$(PARAMETERS_VERSION)/model.joblib

clean-training:  ## delete results of training run from training folder
	rm -rf mobile_handset_price_model/training/*.html
	rm -rf mobile_handset_price_model/training/*.joblib

clean-pyc: ## Remove python artifacts.
	find . -name '*.pyc' -exec rm -f {} +
	find . -name '*.pyo' -exec rm -f {} +
	find . -name '*~' -exec rm -f {} +

build: ## build the package
	python setup.py sdist bdist_wheel

clean-build:  ## clean build artifacts
	rm -rf build
	rm -rf dist
	rm -rf vendors
	rm -rf mobile_handset_price_model.egg-info

venv: ## create virtual environment
	python3 -m venv venv

dependencies: ## install dependencies from requirements.txt
	python -m pip install --upgrade pip
	python -m pip install --upgrade setuptools
	python -m pip install --upgrade wheel
	python -m pip install pip-tools
	pip install -r requirements.txt

test-dependencies: ## install dependencies from test_requirements.txt
	pip install -r test_requirements.txt

clean-venv: ## remove all packages from virtual environment
	pip freeze | grep -v "^-e" | xargs pip uninstall -y

test: clean-pyc ## Run unit test suite.
	pytest --verbose --color=yes $(TEST_PATH)

test-reports: clean-pyc clean-test ## Run unit test suite with reporting
	mkdir -p reports
	mkdir ./reports/unit_tests
	mkdir ./reports/coverage
	mkdir ./reports/badge
	-python -m coverage run --source mobile_handset_price_model -m pytest --verbose --color=yes --html=./reports/unit_tests/report.html --junitxml=./reports/unit_tests/report.xml $(TEST_PATH)
	-coverage html -d ./reports/coverage
	-coverage-badge -o ./reports/badge/coverage.svg
	rm -rf .coverage

clean-test:	## Remove test artifacts
	rm -rf .pytest_cache
	rm -rf .coverage
	rm -rf .hypothesis
	rm -rf reports

generate-modules:  ## generates python modules from jupyter notebooks
	jupyter nbconvert --to script ./mobile_handset_price_model/training/*.ipynb --PythonExporter.exclude_markdown=True

clean-modules:  ## delete all modules created from jupyter notebooks
	rm ./mobile_handset_price_model/training/*.py

check-codestyle: generate-modules  ## checks the style of the code against PEP8
	pycodestyle mobile_handset_price_model --ignore=W391,E402  --max-line-length=120

check-docstyle: generate-modules  ## checks the style of the docstrings against PEP257
	pydocstyle mobile_handset_price_model --ignore=D100,D203,D212,D213

check-security:  ## checks for common security vulnerabilities
	bandit -r mobile_handset_price_model

check-dependencies:  ## checks for security vulnerabilities in dependencies
	safety check -r requirements.txt

check-codemetrics: generate-modules  ## calculate code metrics of the package
	radon cc mobile_handset_price_model
