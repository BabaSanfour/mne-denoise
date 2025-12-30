# Release checklist

This project uses manual versioning in `pyproject.toml`.

1.  **Bump Version**:
    *   Update `version` in `pyproject.toml`.
    *   Update `CHANGELOG.md`.

2.  **Tag Release**:
    *   Commit changes: `git commit -am "Release vX.Y.Z"`
    *   Tag: `git tag vX.Y.Z`
    *   Push: `git push && git push --tags`

3.  **Build and Publish**:
    *   Clean: `rm -rf dist/`
    *   Build: `python -m build`
    *   Check: `twine check dist/*`
    *   Upload: `twine upload dist/*`

4.  **Post-Release**:
    *   Update version to next dev cycle (e.g. `X.Y.Z+1.dev0`) in `pyproject.toml`.
