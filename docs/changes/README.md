# Changelog Guide

We use `towncrier` to manage our changelog. This ensures that changes are documented as they happen, preventing merge conflicts in the changelog file and ensuring high-quality release notes.

## Adding a Changelog Entry

When you make a change (feature, bugfix, documentation update), you should add a fragment file to the `docs/changes/devel/` directory.

The filename should be your **PR number** (or Issue number if no PR yet) followed by the type of change and the extension `.md`.

Format: `<ISSUE_OR_PR_NUMBER>.<TYPE>.md`

### Available types:

* `feature`: New feature.
* `bugfix`: Bug fix.
* `doc`: Documentation improvement.
* `removal`: Deprecation or removal of a feature.
* `misc`: Internal changes, tooling, etc.

## Example

If you fixed a bug in PR #42, create a file `docs/changes/devel/42.bugfix.md`:

```markdown
Fixed a bug where the ZapLine algorithm would crash on empty data.
```

## Building the Changelog

To preview the changelog (requires `towncrier`):

```bash
towncrier build --draft
```
