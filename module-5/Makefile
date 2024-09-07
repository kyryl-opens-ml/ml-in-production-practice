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

build_pytriton:
	docker build -f Dockerfile -t app-pytriton:latest --target app-pytriton .  

run_pytriton: build_pytriton
	docker run -it -p 8001:8001 -p 8000:8000 -p 8002:8002 -e WANDB_API_KEY=${WANDB_API_KEY} app-pytriton:latest





build_app_seldon:
	docker build -f Dockerfile -t app-seldon:latest --target app-seldon .  

run_app_seldon: build_app_seldon
	docker run -it -p 8081:8080 -e WANDB_API_KEY=${WANDB_API_KEY} app-seldon:latest

build_kserve:
	docker build -f Dockerfile -t app-kserve:latest --target app-kserve .

run_kserve:
	docker run -e PORT=8080 -e WANDB_API_KEY=${WANDB_API_KEY} -p 8081:8080 app-kserve:latest 


