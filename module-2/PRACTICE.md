# Practice

***

# H3: Data storage & processing

## Reading list:

- [Data engineer roadmap](https://github.com/datastacktv/data-engineer-roadmap)
- [Minio using Kubernetes](https://github.com/kubernetes/examples/tree/master/staging/storage/minio)
- [What Goes Around Comes Around... And Around...](https://db.cs.cmu.edu/papers/2024/whatgoesaround-sigmodrec2024.pdf)
- [Small Big Data: using NumPy and Pandas when your data doesn't fit ](https://www.youtube.com/watch?v=8pFnrr0NnwY)
- [How to Choose the Right Python Concurrency API](https://superfastpython.com/python-concurrency-choose-api/)
- [Articles: Speed up your data science and scientific computing code](https://pythonspeed.com/datascience/#memory)
- [Data formats with Pandas and Numpy](https://aaltoscicomp.github.io/python-for-scicomp/data-formats/)
- [An Empirical Evaluation of Columnar Storage Formats](https://arxiv.org/pdf/2304.05028.pdf)
- [ML⇄DB Seminar Series — Fall 2023](https://db.cs.cmu.edu/seminar2023/)
- [High Performance I/O For Large Scale Deep Learning](https://arxiv.org/pdf/2001.01858.pdf)
- [Announcing CPP-based S3 IO DataPipes](https://pytorch.org/blog/announcing-cpp/)
- [Efficient PyTorch I/O library for Large Datasets, Many Files, Many GPUs](https://pytorch.org/blog/efficient-pytorch-io-library-for-large-datasets-many-files-many-gpus/)
- [AIStore: scalable storage for AI applications](https://github.com/NVIDIA/aistore)
- [Book: Designing Data-Intensive Applications by Martin Kleppmann](https://www.oreilly.com/library/view/designing-data-intensive-applications/9781491903063/ch04.html)
- [Book: The Data Engineering Cookbook](https://github.com/andkret/Cookbook)
- [Course: CMU Database Systems](https://15445.courses.cs.cmu.edu/fall2023/)
- [Course: Advanced Database Systems](https://15721.courses.cs.cmu.edu/spring2024/)

## Task:

- PR1: Write README instructions detailing how to deploy MinIO with the following options: Local, Docker, Kubernetes (K8S)-based.
- PR2: Develop a CRUD Python client for MinIO and accompany it with comprehensive tests.
- PR3: Write code to benchmark various Pandas formats in terms of data saving/loading, focusing on load time and save time.
- PR4: Create code to benchmark inference performance using single and multiple processes, and report the differences in time.
- PR5: Develop code for converting your dataset into the StreamingDataset format.
- PR6: Write code for transforming your dataset into a vector format, and utilize VectorDB for ingestion and querying.
- Google Doc: Update your proposal by adding a section on data storage and processing.

## Criteria:

- 6 PRs are merged
- Description of data section, storage and processing, in the google doc.


# H4: Data labeling & validation

## Reading list:

- [How to Write Data Labeling/Annotation Guidelines](https://eugeneyan.com/writing/labeling-guidelines/)
- [How to Develop Annotation Guidelines](https://nilsreiter.de/blog/2017/howto-annotation)
- [Label Studio](https://github.com/HumanSignal/label-studio)
- [Argilla](https://github.com/argilla-io/argilla)
- [Open Source Data Annotation & Labeling Tools](https://github.com/zenml-io/awesome-open-data-annotation)
- [Cleanlab detect issues in a ML dataset.](https://github.com/cleanlab/cleanlab)
- [Deepchecks](https://github.com/deepchecks/deepchecks)
- [Data generation](https://github.com/tatsu-lab/stanford_alpaca?tab=readme-ov-file#data-generation-process)

## Task:

- Google doc containing dataset labeling section: Estimate costs and time based on your experience labeling ~50 samples, provide instructions for future data labeling, and add a flow for data enrichment in production.
- PR1: Commit your data with DVC into the GitHub repo.
- PR2: Write code to deploy a labeling tool (e.g., Label Studio, Argilla), including README instructions.
- PR3: Write code to generate a synthetic dataset with ChatGPT.
- PR4: Write code to test your data after labeling (can use Cleanlab or Deepchecks).

## Criteria:

- 4 PRs are merged. 
- Description of data section, labeling and versions, in the google doc.
