# Testing

Use the developer-owned environment described in the
[installation guide](installation.md#installation-as-developer) and
[development guide](../development.md#developer-owned-environment-setup). Do
not install or synchronize dependencies as part of a test run. If the
environment is missing or stale, ask the developer to re-sync it.

Run the tests from the repository root with mandatory `--no-sync`:

```console
uv run --no-sync pytest tests
```
