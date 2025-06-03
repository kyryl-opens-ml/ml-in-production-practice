# Practice

Build end-to-end training and inference pipelines using Kubeflow,
Airflow and Dagster.

### Key tasks

- Deploy Kubeflow pipelines and write training/inference DAGs.
- Deploy Airflow with KubernetesPodOperator.
- Implement the same logic in Dagster.

***


# H7: Kubeflow + AirFlow pipelines

## Reading list:

- [Kubeflow pipelines Standalone Deployment](https://www.kubeflow.org/docs/components/pipelines/v1/installation/standalone-deployment/)
- [Kubeflow Pipelines SDK API Reference](https://kubeflow-pipelines.readthedocs.io/en/)
- [How we Reduced our ML Training Costs by 78%!!](https://blog.gofynd.com/how-we-reduced-our-ml-training-costs-by-78-a33805cb00cf)
- [Leveraging the Pipeline Design Pattern to Modularize Recommendation Services](https://doordash.engineering/2021/07/07/pipeline-design-pattern-recommendation/)
- [Why data scientists shouldn’t need to know Kubernetes](https://huyenchip.com/2021/09/13/data-science-infrastructure.html)
- [Orchestration for Machine Learning](https://madewithml.com/courses/mlops/orchestration/)
- [KubernetesPodOperator](https://airflow.apache.org/docs/apache-airflow-providers-cncf-kubernetes/stable/operators.html)


## Task:

For this task, you will need both a training and an inference pipeline. The training pipeline should include at least the following steps: Load Training Data, Train Model, Save Trained Models. Additional steps may be added as desired. Similarly, the inference pipeline should include at least the following steps: Load Data for Inference, Load Trained Model, Run Inference, Save Inference Results. You may also add extra steps to this pipeline as needed.

- PR1: Write a README with instructions on deploying Kubeflow pipelines.
- PR2: Write a Kubeflow training pipeline.
- PR3: Write a Kubeflow inference pipeline.

- PR4: Write a README with instructions on how to deploy Airflow.
- PR5: Write an Airflow training pipeline.
- PR6: Write an Airflow inference pipeline.


## Criteria:

- 6 PRs merged.


# H8: Dagster

## Reading list:

- [Orchestrating Machine Learning Pipelines with Dagster](https://dagster.io/blog/dagster-ml-pipelines)
- [ML pipelines for fine-tuning LLMs](https://dagster.io/blog/finetuning-llms)
- [Awesome open source workflow engines](https://github.com/meirwah/awesome-workflow-engines)
- [A framework for real-life data science and ML](https://metaflow.org/)
- [New in Metaflow: Train at scale with AI/ML frameworks](https://outerbounds.com/blog/distributed-training-with-metaflow/)
- [House all your ML orchestration needs](https://flyte.org/machine-learning)


## Task:

For this task, you will need both a training and an inference pipeline. The training pipeline should include at least the following steps: Load Training Data, Train Model, Save Trained Models. Additional steps may be added as desired. Similarly, the inference pipeline should include at least the following steps: Load Data for Inference, Load Trained Model, Run Inference, Save Inference Results. You may also add extra steps to this pipeline as needed.

- Update the Google Doc with the pipeline section for your use case, and compare Kubeflow, Airflow, and Dagster.
- PR1: Write a Dagster training pipeline.
- PR2: Write a Dagster inference pipeline.

## Criteria:


- 2 PRs merged.
- Pipeline section in the google doc.
