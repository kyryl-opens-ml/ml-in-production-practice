build:
	docker build -f Dockerfile -t classic-example:latest .

push:
	docker tag classic-example:latest ghcr.io/kyryl-opens-ml/classic-example:latest
	docker push ghcr.io/kyryl-opens-ml/classic-example:latest

run_dev: build
	docker run -it -v ${PWD}:/main classic-example:latest /bin/bash

run_dev_gpu: build
	docker run --net=host --gpus all -it -v ${PWD}:/main classic-example:latest /bin/bash

format:
	ruff format classic_example/ tests/

lint:
	ruff check classic_example/ tests/

test:
	pytest --disable-warnings ./tests/

test_data:
	pytest --disable-warnings -rc ./tests/test_data.py

test_code:
	pytest --disable-warnings -rc ./tests/test_code.py

test_model:
	pytest --disable-warnings -ss ./tests/test_model.py

test_all:
	pytest --cov=classic_example tests/

train_example:
	python classic_example/cli.py load-sst2-data ./data
	python classic_example/cli.py train ./conf/example.json
	python classic_example/cli.py upload-to-registry example_model /tmp/results

train_fast_ci:
	python classic_example/cli.py load-sst2-data ./data
	python classic_example/cli.py train ./conf/fast.json
	python classic_example/cli.py upload-to-registry fast-model /tmp/results
	