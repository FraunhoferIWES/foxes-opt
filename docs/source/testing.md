# Testing

Use the developer-owned environment described in the
[installation guide](installation.md#development-setup) and
[development guide](https://github.com/FraunhoferIWES/foxes-opt/blob/main/docs/development.md). Do
not install or synchronize dependencies as part of a test run. If the
environment is missing or stale, ask the developer to re-sync it.

Run the tests from the repository root with mandatory `--no-sync`:

```console
uv run --no-sync pytest tests
```
