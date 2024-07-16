FROM huggingface/transformers-pytorch-gpu:4.35.2

WORKDIR /app

ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt
RUN MAX_JOBS=4 pip install flash-attn==2.5.7 --no-build-isolation

RUN ln -s /usr/bin/python3 /usr/bin/python

ENV PYTHONPATH /app
COPY . . 

CMD [ "bash" ]