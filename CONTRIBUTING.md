# Contributing to mne-denoise

Thank you for your interest in contributing to `mne-denoise`! We generally follow the [MNE-Python contribution guidelines](https://mne.tools/stable/contributing.html), but here are the specifics for this repository.

## Getting Started

1.  **Fork** the repository on GitHub.
2.  **Clone** your fork:
    ```bash
    git clone https://github.com/<your-username>/mne-denoise.git
    cd mne-denoise
    ```
3.  **Install** in editable mode with development dependencies:
    ```bash
    pip install -e .[dev,docs]
    ```
4.  **Install pre-commit hooks**:
    ```bash
    pre-commit install
    ```

## Development Workflow

1.  Create a branch for your feature or fix: `git checkout -b my-feature-branch`.
2.  Make your changes.
3.  Ensure your code is linted and formatted. The pre-commit hooks will handle this automatically on commit, but you can run them manually:
    ```bash
    pre-commit run --all-files
    ```

## Pull Request Checklist

Before submitting a Pull Request (PR), please ensure the following:

### 1. Style & Linting

We use **Ruff** for linting and formatting.

- Check for errors:
  ```bash
  ruff check .
  ```
- Auto-fix issues:
  ```bash
  ruff check . --fix
  ```
- Format code:
  ```bash
  ruff format .
  ```

### 2. Testing

Run the test suite to ensure no regressions.

- Run all tests:
  ```bash
  pytest
  ```
- Check coverage (aim for 100% on new code):
  ```bash
  pytest --cov=mne_denoise --cov-report=html
  # Open htmlcov/index.html to view report
  ```

### 3. Documentation

If you added new features or changed APIs, update the documentation.

- We use **NumPy style docstrings**.
- Build the docs locally to verify:
  ```bash
  make -C docs html
  ```
  The output will be in `docs/_build/html/index.html`.

### 4. Dependencies

If you added new dependencies, update `pyproject.toml` under `dependencies` or `optional-dependencies`.

## Submitting changes

1.  Push your branch to GitHub.
2.  Open a Pull Request against the `main` branch.
3.  Describe your changes clearly in the PR description. Link to any relevant issues.

Thank you for contributing!
