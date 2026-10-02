# ADR-0001: Forward-Only Development And Complete Closure

- Status: Accepted
- Date: 2026-10-02
- Supersedes: None

## Context

FOXES Optimization is an established scientific package with public Python,
YAML, command-line, pipeline, callback, optimization-variable,
objective/constraint component, and result contracts. Preserving every
superseded contract would accumulate parallel implementations, compatibility
branches, duplicate tests, and documentation that describes multiple eras of
the package. That increases maintenance cost and makes the intended current
behavior harder to identify.

Development closure also needs one repository-wide meaning. Code without its
tests, quality gates, public docstrings, changelog entry, or current technical
records leaves future contributors and agents with an incomplete contract.
The local environment also contains developer-managed editable FOXES and iwopy
installs. Implicit uv synchronization can replace those intended sources and
invalidate development results.

## Decision

FOXES Optimization development is strictly forward-looking by default:

- Implement the target contract directly and remove superseded code, tests,
	documentation, examples, aliases, formats, and fallback behavior in the same
	change.
- Update all maintained in-repository callers instead of adding compatibility
	shims for them.
- Add a legacy or migration path only when the user explicitly requires a
	bounded exception. Record its scope and objective removal condition in the
	governing ADR.
- Run every routine development command through `uv run --no-sync`. The
	developer owns `uv sync`, package installation, and environment repair; ask
	the developer to re-sync instead of changing the environment during a task.

A development change is complete only when:

- when code files changed, required tests are present and both focused tests
	and `uv run --no-sync pytest tests` pass;
- documentation-only changes pass applicable documentation validation instead
	of the full runtime test suite;
- `uv run --no-sync pre-commit run --all-files` passes after the final edit;
- every affected public docstring is accurate;
- the final version section of `CHANGELOG.md` matches `project.version` from
	`pyproject.toml` and records the change; and
- architecture, naming, development, API, example, and other affected FOXES
	Optimization documentation is synchronized in the same change.

## Compatibility And Migration

This policy applies to new development from this decision onward. It does not
retroactively require unrelated cleanup. When a task changes an existing
contract, all maintained FOXES Optimization callers move to the new contract
immediately; the old path is not retained by default.

Downstream consumers may need to update when a public contract changes. Such an
impact must be identified and documented, but it does not itself justify legacy
code. An explicitly authorized exception states exactly what remains, for whom,
and when or under which condition it is removed.

## Consequences

- FOXES Optimization has one current implementation and one documented contract
	for changed behavior.
- Changes may require coordinated updates across code, tests, examples,
	documentation, FOXES/iwopy integration, and downstream consumers in one
	development effort.
- Public changes can be immediately breaking for consumers that have not moved
	to the current contract.
- Routine development cannot silently replace the developer's editable FOXES or
	iwopy installs; dependency changes require developer intervention.
- Completion takes longer than code-only delivery because validation and all
	records close together.

## Verification

- For code changes, the completion report names focused and full test results;
	for documentation-only changes, it names applicable documentation checks.
- Reported local uv commands include `--no-sync`; environment changes are either
	absent or explicitly performed by the developer.
- The final pre-commit run covers the entire repository.
- Review compares affected public docstrings and FOXES Optimization records with
	the final implementation.
- The final `CHANGELOG.md` version heading is compared with `project.version` in
	`pyproject.toml` and contains the change.

## Alternatives Considered

- Preserve compatibility shims and deprecate gradually by default. Rejected
	because it creates legacy paths as the normal case.
- Complete code first and update tests or documentation later. Rejected because
	it leaves the repository in a knowingly inconsistent state.
- Require closure checks only for releases. Rejected because stale information
	and missing coverage compound between releases.

## References

- [Repository instructions](../../AGENTS.md#development-direction)
- [Development guide](../development.md)
- [Architecture](../architecture.md#cross-cutting-decisions)
