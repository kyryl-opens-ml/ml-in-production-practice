# Module 1

![alt text](./../docs/into.jpg)

# Practice 

[Practice task](./PRACTICE.md)

***

# Reference implementation

***

# H1: Initial Design Draft


[Google doc](https://docs.google.com/document/d/1mUAUVMdA6O3rxvjS87mm-tAisQXDQggRKEBv0nWPuP4/edit)


# H2: Infrastructure


# Docker

## Run hello-world

```bash
docker pull hello-world
docker run hello-world
```

Reference: https://hub.docker.com/_/hello-world

## Build and run

Build ml sample docker image

```bash
docker build --tag app-ml:latest ./app-ml
```

Run ml sample docker container

```bash
docker run -it --rm --name app-ml-test-run app-ml:latest
docker run -it --rm --name app-ml-test-run app-ml:latest python -c "import time; time.sleep(5); print(f'AUC = {0.0001}')"
```

Build web sample docker image

```bash
docker build --tag app-web:latest ./app-web
```

Build web sample docker image

```bash
docker run -it --rm -p 8080:8080 --name app-web-test-run app-web:latest
```

In a separate terminal, run the curl command to check access.

```bash
curl http://0.0.0.0:8080/
```

Bulti-build docker file, if you don't want to keep a lot of docker files.

```bash
docker build --tag app-web:latest --target app-web ./app-multi-build
docker build --tag app-ml:latest --target app-ml ./app-multi-build
```

## Share

Login to docker registry

```bash
export GITHUB_TOKEN=token
export GITHUB_USER=user
echo $GITHUB_TOKEN | docker login ghcr.io -u $GITHUB_USER --password-stdin
```

Tag images

```bash
docker tag app-ml:latest ghcr.io/kyryl-opens-ml/app-ml:latest
docker tag app-web:latest ghcr.io/kyryl-opens-ml/app-web:latest
```

Push image

```bash
docker push ghcr.io/kyryl-opens-ml/app-ml:latest
docker push ghcr.io/kyryl-opens-ml/app-web:latest
```

## Registry

- [github](https://github.com/features/packages)
- [dockerhub](https://hub.docker.com/)
- [aws](https://aws.amazon.com/ecr/)
- [gcp](https://cloud.google.com/container-registry)

# CI/CD

Check code in this file

[CI example](./../.github/workflows/module-1.yaml)

## Provides

- [circleci](https://circleci.com/)
- [GitHub Actions](https://docs.github.com/en/actions)
- [Jenkins](https://www.jenkins.io/)
- [Travis CI](https://www.travis-ci.com/)
- [List of Continuous Integration services](https://github.com/ligurio/awesome-ci)

# Kubernetes

## Setup

Install [kind](https://kind.sigs.k8s.io/docs/user/quick-start/)

```bash
brew install kind
```

Create cluster

```bash
kind create cluster --name ml-in-production
```

Install kubectl

```bash
brew install kubectl
```

Check current context

```bash
kubectl config get-contexts
```

Install "htop" for k8s

```bash
brew install derailed/k9s/k9s
```

Run "htop" for k8s

```bash
k9s -A
```

## Use

Create pod for app-web

```bash
kubectl create -f k8s-resources/pod-app-web.yaml
```

Create pod for app-ml

```bash
kubectl create -f k8s-resources/pod-app-ml.yaml
```

Create job for app-ml

```bash
kubectl create -f k8s-resources/job-app-ml.yaml
```

Create deployment for app-web

```bash
kubectl create -f k8s-resources/deployment-app-web.yaml
```

To access use port-forwarding

```bash
kubectl port-forward svc/deployments-app-web 8080:8080
```

## Provides 

- [EKS](https://aws.amazon.com/eks/)
- [GKE](https://cloud.google.com/kubernetes-engine)
- [CloudRun](https://cloud.google.com/run)
- [AWS Fargate/ECS](https://aws.amazon.com/fargate/)


# Bonus

Sometimes Kubernetes might be overkill for your problem, for example, if you are a small team, it's a pet project, or you just don't want to deal with complex infrastructure setup. In this case, I would recommend checking out serverless offerings, some good examples of which I use all the time.

## Railway

- [Railway infrastructure platform](https://railway.app/) - nice too to deploy simple app.

```bash
open https://railway.app/
```

## Modal

- [The serverless platform for AI/ML/data teams](https://modal.com/) - nice too to deploy ML heavy app.

```bash
pip install modal --upgrade
modal token new
modal run modal_hello_world.py
```
