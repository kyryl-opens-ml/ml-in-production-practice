# Practice 

*** 


# H13: Monitoring and observability

## Reading list: 

- [Underspecification Presents Challenges for Credibility in Modern Machine Learning](https://arxiv.org/abs/2011.03395)
- [Why is pattern recognition often defined as an ill-posed problem?](https://stats.stackexchange.com/questions/433692/why-is-pattern-recognition-often-defined-as-an-ill-posed-problem)
- [How ML Breaks: A Decade of Outages for One Large ML Pipeline](https://www.usenix.org/conference/opml20/presentation/papasian)
- [Elastic Stack](https://www.elastic.co/elastic-stack)
- [Real-time Interactive Dashboards DataDog](https://www.datadoghq.com/product/platform/dashboards/)
- [Python OpenTelemetry Instrumentation](https://signoz.io/docs/instrumentation/python/)
- [Opentelemetry Devstats](https://opentelemetry.devstats.cncf.io/d/4/company-statistics-by-repository-group?orgId=1)
- [Log Monitoring 101 Detailed Guide [Included 10 Tips]](https://signoz.io/blog/log-monitoring/)
- [Lecture 10: Data Distribution Shifts & Monitoring Presentation](https://docs.google.com/presentation/d/1tuCIbk9Pye-RK1xqiiZXPzT8lIgDUL6CqBkFSYZXkbY/edit#slide=id.p)
- [Lecture 10. Data Distribution Shifts and Monitoring Notes](https://docs.google.com/document/d/14uX2m9q7BUn_mgnM3h6if-s-r0MZrvDb-ZHNjgA1Uyo/edit#heading=h.sqk67ofnp3ir)
- [Data Distribution Shifts and Monitoring](https://huyenchip.com/2022/02/07/data-distribution-shifts-and-monitoring.html)
- [Monitoring Machine Learning Systems Made with ML](https://madewithml.com/courses/mlops/monitoring/)
- [Inferring Concept Drift Without Labeled Data](https://concept-drift.fastforwardlabs.com/)
- [Failing Loudly: An Empirical Study of Methods for Detecting Dataset Shift](https://arxiv.org/abs/1810.11953)
- [The Playbook to Monitor Your Model’s Performance in Production](https://towardsdatascience.com/the-playbook-to-monitor-your-models-performance-in-production-ec06c1cc3245)
- [Ludwig](https://ludwig.ai/latest/)
- [Declarative Machine Learning Systems](https://arxiv.org/pdf/2107.08148.pdf)
- [Deploy InferenceService with Alibi Outlier/Drift Detector](https://kserve.github.io/website/0.10/modelserving/detect/alibi_detect/alibi_detect/)

## Task:



- PR1: Write code for integrating SigNoz monitoring into your application.
- PR2: Write code for creating a Grafana dashboard for your application.
- PR3: Write code for detecting drift in your pipeline (Kubeflow, AirFlow, or Dagster) within your input and output features.
- Update the Google document with system monitoring and a plan for ML monitoring, covering aspects like ground truth availability, drift detection, etc.

## Criteria: 

- 3 PRs merged 
- Monitoring plan for system and ML in the google doc.

# H14: Tools, LLMs and Data moat.

## Reading list:


- [Alibi Detect](https://github.com/SeldonIO/alibi-detect)
- [Evidently](https://github.com/evidentlyai/evidently)
- [A Guide to Monitoring Machine Learning Models in Production](https://developer.nvidia.com/blog/a-guide-to-monitoring-machine-learning-models-in-production/)
- [Top 7 ML Model Monitoring Tools in 2024](https://www.qwak.com/post/top-ml-model-monitoring-tools)
- [Best Tools to Do ML Model Monitoring](https://neptune.ai/blog/ml-model-monitoring-best-tools)
- [DataDog Machine Learning](https://www.datadoghq.com/solutions/machine-learning/)
- [Another tool won’t fix your MLOps problems](https://dshersh.medium.com/too-many-mlops-tools-c590430ba81b)
- [LLMOps](https://fullstackdeeplearning.com/llm-bootcamp/spring-2023/llmops/)
- [LangSmith Docs](https://www.langchain.com/langsmith)
- [RLAIF: Scaling Reinforcement Learning from Human Feedback with AI Feedback](https://arxiv.org/abs/2309.00267)
- [OpenLLMetry](https://github.com/traceloop/openllmetry?tab=readme-ov-file)
- [Open-source ML observability course](https://github.com/evidentlyai/ml_observability_course)
- [RLHF: Reinforcement Learning from Human Feedback](https://huyenchip.com/2023/05/02/rlhf.html)


## Task:

- PR1: Write code for utilizing a managed model monitoring tool for your model (e.g., Arize).
- PR2: Write code for LLM monitoring with the help of Langsmith or similar tools.
- PR3: Write code to close the loop: create a dataset for labeling from your monitoring solution.
- Update the Google document with the data moat strategy, detailing how you would enrich data from production and reuse it for building future models.


## Criteria:


- 3 PRs are merged 
- Data moat strategy in the google doc.
