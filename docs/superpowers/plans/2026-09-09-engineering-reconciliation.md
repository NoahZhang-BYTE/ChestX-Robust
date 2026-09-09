# Engineering reconciliation implementation plan

User-authorized scope: execute PROJECT_STATUS.md in order, without training or replaying test inference; delete only proven disposable files after a final scan.

- [ ] P0: read-only reconciliation of B0–B5 and ensemble; verify existing hashes, checkpoints, arrays and patient splits. Record missing provenance as unknown.
- [ ] P0: publish immutable ledger generation and atomically switch the canonical manifest; regenerate state/registry/log views, retaining original snapshots. A reader sees one committed generation; legacy copies are checked projections.
- [ ] P1: unify config-driven training and resume, retain compatibility entrypoints; add CPU synthetic regression coverage for RNG, scheduler and early stopping.
- [ ] P1: unify threshold APIs without changing historical grid/exact protocols; add explicit CLI commands with no auto-dispatch from workflow stages.
- [ ] P1: provide environment pins, packaging, relative path remapping with hash validation, and current image manifest.
- [ ] P2: use runs/<experiment>/<run_id> for new jobs, reports/ for derived analysis; preserve existing hashed artifact locations.
- [ ] P2: descriptive patient bootstrap for B5/ensemble using saved predictions, plus an external-validation handoff. No new threshold fitting on test or model selection.
- [ ] P2: review code and evidence, full CPU test suite, provenance and ledger checks, then delete only project-generated caches and record exact deletions.

Historical training commit/commands and historical image bytes cannot be reconstructed from current files. Distinguish current audit fingerprints from run-time evidence. Do not infer hardware clearance from completed later runs. Preserve uncertainty around whole-project test exposure.

Validation: hash corruption rejection, missing artifact rejection, non-mutating audit, atomic pointer failure behavior, projection drift detection, deterministic resume with shuffled synthetic data, relocated hash-verified frozen bundle, CLI help/dispatch, full pytest and diff checks.
