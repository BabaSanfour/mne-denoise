# Release checklist

This project uses manual versioning in `pyproject.toml`.

## 1. Prepare Release

- [ ] **Update Dependencies**:
    - Check for dependency updates.
    - Run `pip install .[dev,test,docs]` to ensure environment is fresh.
- [ ] **Run Checks**:
    - [ ] `ruff check .`
    - [ ] `ruff format .`
    - [ ] `pytest` (ensure all pass)
    - [ ] `make -C docs html` (ensure no warnings)
- [ ] **Bump Version**:
    - Update `version` in `pyproject.toml` (e.g., `0.0.1`).
    - Update `CHANGELOG.md` with release date and summary.
    - Commit changes: `git commit -am "Bump version to X.Y.Z"`

## 2. Tag and Publish (GitHub)

- [ ] **Tag Release**:
    - `git tag vX.Y.Z`
    - `git push upstream main`
    - `git push upstream vX.Y.Z`

## 3. Verify Release (GitHub & PyPI)

- [ ] **Verify CI/CD**:
    - Watch functionality of `ci.yml`.
    - Watch `release.yml` (triggered by tag).
    - **PyPI Publishing**: Verify the new version appears on PyPI (automated via Trusted Publishing).

## 4. Post-Release Maintenance

- [ ] **Set Next Version**:
    - Update `pyproject.toml` to next dev version (e.g., `0.1.0.dev0`).
    - Commit and push: `git commit -am "Back to dev" && git push upstream main`
- [ ] **Conda-Forge**:
    - Wait for the regro-cf-autotick-bot to open a PR on the feedstock.
    - Merge the PR to update the conda package.
