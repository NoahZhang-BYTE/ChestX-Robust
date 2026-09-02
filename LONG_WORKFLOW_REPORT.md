# Long Workflow Report

This report is populated only after B2/B3/B4 completion gates have produced
validated artifacts. It is intentionally not marked complete while B2 is still
running.

Current workflow state is recorded in `workflow_state.json`; experiment rows
are recorded in `outputs/experiment_registry.csv`; detailed experiment notes
are recorded in `EXPERIMENT_LOG.md`.

## Stop Condition

B5 resolution expansion is never started automatically. After B4 analysis,
the workflow stops and records whether B5 is recommended for human review.
