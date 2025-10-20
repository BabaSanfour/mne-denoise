# Release checklist

This project uses `setuptools_scm` to derive versions from Git tags. Create
annotated tags in the form ``vX.Y.Z``.

1. Ensure `main` is up to date and the changelog documents the release.
2. Run the full quality suite locally: `pre-commit run --all-files`, `pytest`,
   and `make -C docs html`.
3. Update `CHANGELOG.md` and commit with the release notes.
4. Tag the release:

   ```bash
   git tag -a vX.Y.Z -m "Release vX.Y.Z"
   git push origin vX.Y.Z
   ```

5. GitHub Actions will build and publish wheels and source distributions to
   PyPI via OpenID Connect.

If a release needs to be yanked, use the PyPI interface and create a follow-up
patch release (e.g. `vX.Y.(Z+1)`) with the fix.
