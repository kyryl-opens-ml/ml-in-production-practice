# Practice

***

# H5: Training & Experiments

## Reading list:

- [The Data Science Lifecycle Process](https://github.com/dslp/dslp#the-data-science-lifecycle-process)
- [Structuring Your Project](https://docs.python-guide.org/writing/structure/)
- [How to Organize Deep Learning Projects – Examples of Best Practices](https://neptune.ai/blog/how-to-organize-deep-learning-projects-best-practices)
- [Readme Driven Development](https://tom.preston-werner.com/2010/08/23/readme-driven-development.html)
- [Reengineering Facebook AI’s deep learning platforms for interoperability](https://ai.meta.com/blog/reengineering-facebook-ais-deep-learning-platforms-for-interoperability/)
- [The Twelve Factors](https://12factor.net/)
- [Weights & Biases Documentation](https://docs.wandb.ai/)
- [The value of a shared understanding of AI models](https://modelcards.withgoogle.com/about)
- [Aim experiment tracker](https://github.com/aimhubio/aim)
- [MosaicBERT: Pretraining BERT from Scratch for $20](https://www.mosaicml.com/blog/mosaicbert)
- [AutoML NNI](https://github.com/microsoft/nni)
- [15 Best Tools for ML Experiment Tracking and Management](https://neptune.ai/blog/best-ml-experiment-tracking-tools)
- [Some Techniques To Make Your PyTorch Models Train (Much) Faster](https://sebastianraschka.com/blog/2023/pytorch-faster.html)
- [Pytorch Distributed Overview](https://pytorch.org/tutorials/beginner/dist_overview.html)
- [Distributed Training](https://www.run.ai/guides/gpu-deep-learning/distributed-training)
- [Training 175B Parameter Language Models at 1000 GPU scale with Alpa and Ray](https://www.anyscale.com/blog/training-175b-parameter-language-models-at-1000-gpu-scale-with-alpa-and-ray)
- [Accelerate](https://github.com/huggingface/accelerate)
- [Ray Train: Scalable Model Training](https://docs.ray.io/en/latest/train/train.html#train-docs)

## Task:

You need to have a training pipeline for your model for this homework. You can take it from your test task for this course, bring your own or use this [code](https://github.com/huggingface/transformers/tree/main/examples/pytorch/text-classification) as an example.


- Update the Google Doc with the experiment section: experiment management tool and model card for your project.
- PR1: Write code for training your model using the W&B experiment logger.
- PR2: Write code for conducting hyperparameter searches with W&B.
- PR3: Write code to create a model card for your model, which can be a simple markdown or utilize this [toolset](https://github.com/tensorflow/model-card-toolkit)
- PR4 (optional): Write to replicate the [tutorial](https://www.mosaicml.com/blog/mosaicbert)
- PR5 (optional): Write code for hyperparameter searches using [NNI](https://github.com/microsoft/nni)
- PR6 (optional): Write code for distributed training with PyTorch, Accelerate, and Ray.
- Public link to your W&B project with experiments.

## Criteria:

- 6 PRs are merged.
- W&B project created.
- Description of experiment section in the google doc.

# H6: Testing & CI

## Reading list:

- [TestPyramid](https://martinfowler.com/bliki/TestPyramid.html)
- [PyTesting the Limits of Machine Learning](https://www.youtube.com/watch?v=GycRK_K0x2s)
- [Testing Machine Learning Systems: Code, Data and Models](https://madewithml.com/courses/mlops/testing/)
- [Beyond Accuracy: Behavioral Testing of NLP models with CheckList](https://github.com/marcotcr/checklist)
- [Robustness Gym is an evaluation toolkit for machine learning.](https://github.com/robustness-gym/robustness-gym)
- [ML Testing  with Deepchecks](https://github.com/deepchecks/deepchecks?tab=readme-ov-file)
- [Promptfoo: test your LLM app](https://github.com/promptfoo/promptfoo)
- [The Evaluation Framework for LLMs](https://github.com/confident-ai/deepeval)
- [Continuous Machine Learning (CML)](https://github.com/iterative/cml)
- [Using GitHub Actions for MLOps & Data Science](https://github.blog/2020-06-17-using-github-actions-for-mlops-data-science/)
- [Benefits of Model Serialization in ML](https://appsilon.com/model-serialization-in-machine-learning/)
- [Model registry](https://docs.wandb.ai/guides/model_registry)
- [Privacy Testing for Deep Learning](https://github.com/trailofbits/PrivacyRaven)
- [Learning Interpretability Tool (LIT)](https://github.com/PAIR-code/lit)

## Task:

You need to have a training pipeline for your model for this homework. You can take it from your test task for this course, bring your own or use this [code](https://github.com/huggingface/transformers/tree/main/examples/pytorch/text-classification) as an example.

- Google doc update with a testing plan for your ML model. 
- PR1: Write tests for [code](https://madewithml.com/courses/mlops/testing/#pytest), tests should be runnable from CI.
- PR2: Write tests for [data](https://madewithml.com/courses/mlops/testing/#data), tests should be runnable from CI.
- PR3: Write tests for [model](https://madewithml.com/courses/mlops/testing/#models), tests should be runnable from CI.
- PR4: Write code to store your model in model management with W&B.
- PR5 (optional) : Write code to use [LIT](https://github.com/PAIR-code/lit) for your model, in the case of other domains (CV, audio, tabular) find and use a similar tool.
- PR6 (optional): Write code to test LLM API (select any LLM - OpenAI, VertexAI, etc).

## Criteria:

- 6 PRs merged.
- Testing plan in the google doc.
