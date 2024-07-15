build:
	docker build -f Dockerfile -t generative-example:latest .

run_dev: build
	docker run -it -v ${PWD}:/main generative-example:latest /bin/bash

run_dev_gpu: build
	docker run --net=host --gpus all -it -v ${PWD}:/main generative-example:latest /bin/bash

format:
	ruff format nlp_sample/ 

lint:
	ruff check nlp_sample/



train_example:
	python generative_example/cli.py load-sql-data --subsample 0.1 ./data 
	python generative_example/cli.py train ./conf/example.json
	python generative_example/cli.py upload-to-registry generative_example_model /tmp/results

	