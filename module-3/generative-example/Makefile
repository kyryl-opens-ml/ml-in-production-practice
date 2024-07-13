build:
	docker build -f Dockerfile -t generative-example:latest .

run_dev: build
	docker run -it -v ${PWD}:/main generative-example:latest /bin/bash

run_dev_gpu: build
	docker run --net=host --gpus all -it -v ${PWD}:/main generative-example:latest /bin/bash

format:
	ruff format nlp_sample/ tests/

lint:
	ruff check nlp_sample/ tests/

test:
	pytest --disable-warnings ./tests/

test_data:
	pytest --disable-warnings -rc ./tests/test_data.py

test_code:
	pytest --disable-warnings -rc ./tests/test_code.py

test_model:
	pytest --disable-warnings -ss ./tests/test_model.py

test_all:
	pytest --cov=generative_example tests/

train_example:
	python generative_example/cli.py load-sql-data --subsample 0.1 ./data 
	python generative_example/cli.py train ./conf/example.json
	python generative_example/cli.py upload-to-registry example_model /tmp/results

train_fast_ci:
	python generative_example/cli.py load-sql-data --subsample 0.1 ./data 
	python generative_example/cli.py train ./conf/fast.json
	python generative_example/cli.py upload-to-registry fast-model /tmp/results
	