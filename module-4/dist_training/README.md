```
docker run -it -v $PWD:/app --net=host --gpus all --shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864 huggingface/transformers-pytorch-gpu:4.28.1 /bin/bash


torchrun --nnodes=2 --nproc_per_node=1 --node-rank=0 --master-addr=172.31.3.162 --master-port=8888 torch_native_dist.py 1000 100
torchrun --nnodes=2 --nproc_per_node=1 --node-rank=1 --master-addr=172.31.3.162 --master-port=8888 torch_native_dist.py 1000 100


accelerate launch --multi_gpu --num_machines 2 --num_processes 2 --machine_rank 0 --main_process_ip 172.31.3.162 --main_process_port 8888 accelerate_run.py
accelerate launch --multi_gpu --num_machines 2 --num_processes 2 --machine_rank 1 --main_process_ip 172.31.3.162 --main_process_port 8888 accelerate_run.py

accelerate launch accelerate_run.py
```