# Base layer with all dependencies
FROM python:3.12 as base

WORKDIR /app

RUN pip install --upgrade pip


# Take models and create deployment
FROM base AS app-web
CMD python -m http.server 8080


# Take models and create training
FROM base AS app-ml
CMD python -c "import time; time.sleep(5); print(f'AUC = {0.9999}')"
