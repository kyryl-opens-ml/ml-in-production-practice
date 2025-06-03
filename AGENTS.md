# Contributor Guide

This repository contains homework solutions for the [ML in Production](https://edu.kyrylai.com/courses/ml-in-production) course.  Each module under `module-*` holds an independent example or exercise.

## Development
- Use Python 3.10+.
- Format Python code using `ruff format` and check style with `ruff check`.
- Execute tests with `pytest` from the repository root. Some modules also provide Makefiles (e.g. `module-3/classic-example`) with helper commands such as `make test`.
- Ensure all tests pass before committing.

## Pull Requests
- Keep PR titles concise, e.g. `[module-x] <short description>`.
- Summarize the changes in the body of the PR.
