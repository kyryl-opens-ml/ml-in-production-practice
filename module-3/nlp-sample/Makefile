build:
	docker build -f Dockerfile -t nlp-sample:latest .

run_dev: build
	docker run -it -v ${PWD}:/main nlp-sample:latest /bin/bash

run_dev_gpu: build
	docker run --net=host --gpus all -it -v ${PWD}:/main nlp-sample:latest /bin/bash

format:
	black --line-length 120 nlp_sample tests
	isort -rc nlp_sample tests

lint:
	flake8 --max-line-length 120 nlp_sample tests

test:
	pytest --disable-warnings ./tests/

test_data:
	pytest --disable-warnings -rc ./tests/test_data.py

test_code:
	pytest --disable-warnings -rc ./tests/test_code.py

test_model:
	pytest --disable-warnings -ss ./tests/test_model.py

test_all:
	pytest --cov=nlp_sample tests/

train_example:
	python nlp_sample/cli.py load-cola-data ./data
	python nlp_sample/cli.py train ./conf/example.json
	python nlp_sample/cli.py upload-to-registry example_model /tmp/results

train_fast_ci:
	python nlp_sample/cli.py load-cola-data ./data
	python nlp_sample/cli.py train ./conf/fast.json
	python nlp_sample/cli.py upload-to-registry fast-model /tmp/results
	