# ML in Production Practice

This repository contains practical exercises and reference implementations for the [ML in Production](https://edu.kyrylai.com/courses/ml-in-production) course.

![Course banner](./docs/into.jpg)

## Setup

1. Clone the repo and create a Python **3.10+** virtual environment.
2. Each module is a self-contained example with its own dependencies. Check the module's `README.md` or `PRACTICE.md` for installation instructions.
3. Format the code with `ruff format` and run `ruff check` to verify style.
4. Execute `pytest` from the repository root.

Example:

```bash
git clone https://github.com/<user>/ml-in-production-practice.git
cd ml-in-production-practice
python -m venv .venv
source .venv/bin/activate
# install dependencies for a module
uv pip install -r module-3/classic-example/requirements.txt
ruff format && ruff check
pytest
```

## Project structure

```
.
├── module-1/  # containerization and infrastructure basics
├── module-2/  # data management and labeling
├── module-3/  # model training workflows
├── module-4/  # pipeline orchestration
├── module-5/  # serving with FastAPI
├── module-6/  # large model optimisation and load testing
├── module-7/  # monitoring and observability
├── module-8/  # additional production topics
└── docs/      # images used in documentation
```

Each module is self‑contained with its own `README.md`, assignments and reference code. You can dive into any module independently or work through them sequentially.


## Versioning

A protected `2024-version` branch preserves the 2024 and early 2025 edition of this course. The main branch contains the most up‑to‑date materials.

## Support

- [Create an issue](../../issues) if you encounter problems or have feature requests.
- Join the [course Discord](https://discord.gg/5NF2NAsGEM) to ask questions.
- Visit the [blog](https://kyrylai.com/blog/) for additional articles.
- See the [course page](https://edu.kyrylai.com/courses/ml-in-production) for curriculum details.

---

Released under the [MIT License](LICENSE).
