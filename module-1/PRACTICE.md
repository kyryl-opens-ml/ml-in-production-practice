# Practice

This module focuses on drafting your ML system design and containerizing a
simple application. You'll also create CI/CD pipelines and basic Kubernetes
manifests.

### Key tasks

- Draft a design document using the MLOps template.
- Build and push a Docker image.
- Set up GitHub Actions for CI/CD.
- Deploy your image to a local Kubernetes cluster.

***

# H1: Initial Design Draft

## Reading list:

- [Ml-design-docs](https://github.com/eugeneyan/ml-design-docs)
- [How to Write Design Docs for Machine Learning Systems](https://eugeneyan.com/writing/ml-design-docs/)
- [Design Docs at Google](https://www.industrialempathy.com/posts/design-docs-at-google/)
- [How Big Tech Runs Tech Projects and the Curious Absence of Scrum](https://newsletter.pragmaticengineer.com/p/project-management-in-tech)
- [Continuous Delivery for Machine Learning](https://martinfowler.com/articles/cd4ml.html)
- [Best practices for implementing machine learning on Google Cloud](https://cloud.google.com/architecture/ml-on-gcp-best-practices)
- [The ML Test Score: A Rubric for ML Production Readiness and Technical Debt Reduction](https://storage.googleapis.com/pub-tools-public-publication-data/pdf/aad9f93b86b7addfea4c419b9100c6cdd26cacea.pdf)
- [datascience-fails](https://github.com/xLaszlo/datascience-fails)
- [CS 329S Lecture 1. Machine Learning Systems in Production](https://docs.google.com/document/d/1C3dlLmFdYHJmACVkz99lSTUPF4XQbWb_Ah7mPE12Igo/edit#)
- [MLOps Infrastructure Stack](https://ml-ops.org/content/state-of-mlops)
- [You Don't Need a Bigger Boat](https://github.com/jacopotagliabue/you-dont-need-a-bigger-boat)
- [Why to Hire Machine Learning Engineers, Not Data Scientists](https://www.datarevenue.com/en-blog/hiring-machine-learning-engineers-instead-of-data-scientists)


## Task:

Write a design doc [example](https://docs.google.com/document/d/14YBYKgk-uSfjfwpKFlp_omgUq5hwMVazy_M965s_1KA/edit#heading=h.7nki9mck5t64) with the MLOps template from the [MLOps Infrastructure Stack doc](https://ml-ops.org/content/state-of-mlops) and for the next points (you can use an example from your work or make it up).
- Models in production.
- Pros/Cons of the architecture. 
- Scalability.
- Usability.
- Costs.
- Evolution.
- Next steps.
- ML Test score from this [article](https://storage.googleapis.com/pub-tools-public-publication-data/pdf/aad9f93b86b7addfea4c419b9100c6cdd26cacea.pdf).
- Run over this design doc over [doc](https://github.com/xLaszlo/datascience-fails) and highlight all matches.
- Try to add business metrics into your design doc, [example](https://c3.ai/customers/ai-for-aircraft-readiness/).

## Criteria: 

- Approve / No approval.
- Notes: Repeat the same task at the end of the course for coursework.


# H2: Infrastructure

## Reading list:

- [0 to production-ready: a best-practices process for Docker packaging](https://www.youtube.com/watch?v=EC0CSevbt9k)
- [Docker and Python: making them play nicely and securely for Data Science and ML](https://www.youtube.com/watch?v=Jq68axbKIbg)
- [Docker introduction](https://docker-curriculum.com/)
- [Overview of Docker Hub](https://docs.docker.com/docker-hub/)
- [Introduction to GitHub Actions](https://docs.docker.com/build/ci/github-actions/)
- [Learn Kubernetes Basics](https://kubernetes.io/docs/tutorials/kubernetes-basics/)
- [Hello Minikube](https://kubernetes.io/docs/tutorials/hello-minikube/)
- [Kind Quick Start](https://kind.sigs.k8s.io/docs/user/quick-start/)
- [Why data scientists shouldn’t need to know Kubernetes](https://huyenchip.com/2021/09/13/data-science-infrastructure.html)
- [Scaling Kubernetes to 7,500 nodes](https://openai.com/research/scaling-kubernetes-to-7500-nodes)
- [Course: CI/CD for Machine Learning (GitOps)](https://www.wandb.courses/courses/ci-cd-for-machine-learning)
- [Book: Kubernetes in Action](https://www.manning.com/books/kubernetes-in-action)

## Task:

- PR1: Write a dummy Dockerfile with a simple server and push it to your docker hub or github docker registry.
- PR2: Write CI/CD pipeline with github action that does this for each PR.
- PR3: Write YAML definition for Pod, Deployment, Service, and Job with your Docker image, Use minikube/kind for testing it.
- Install [k9s](https://k9scli.io/) tool.

## Criteria:

- 3 PRs are merged 
- CI/CD is green 
