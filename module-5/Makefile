build_all:
	docker build -f Dockerfile -t all:latest --target app-seldon .    

build_app_streamlit:
	docker build -f Dockerfile -t app-streamlit:latest --target app-streamlit .  

run_app_streamlit: build_app_streamlit
	docker run -it -p 8081:8080 -e WANDB_API_KEY=${WANDB_API_KEY} app-streamlit:latest

build_fast_api:
	docker build -f Dockerfile -t app-fastapi:latest --target app-fastapi .  

run_fast_api: build_fast_api
	docker run -it -p 8081:8080 -e WANDB_API_KEY=${WANDB_API_KEY} app-fastapi:latest

build_app_seldon:
	docker build -f Dockerfile -t app-seldon:latest --target app-seldon .  

run_app_seldon: build_app_seldon
	docker run -it -p 8081:8080 -e WANDB_API_KEY=${WANDB_API_KEY} app-seldon:latest

run_dev: build_all
	docker run -it --net=host -v $PWD:/dev_data -e WANDB_API_KEY=${WANDB_API_KEY} all:latest /bin/bash

format:
	black --line-length 120 serving tests
	isort -rc serving tests

lint:
	flake8 --max-line-length 120 serving tests

build_kserve:
	docker build -f Dockerfile -t app-kserve:latest --target app-kserve .

run_kserve:
	docker run -e PORT=8080 -e WANDB_API_KEY=${WANDB_API_KEY} -p 8081:8080 app-kserve:latest 


