# TensorBoard Training Visualization Design

## Purpose

Add low-overhead, near-real-time TensorBoard telemetry to formal training while preserving the existing CSV, JSON, manifest, and checkpoint artifacts as the authoritative experiment record.

The integration begins with the next training process or a checkpoint resume. It does not alter or restart a process that is already running.

## Scope

The feature applies to the shared training path in `baseline/engine.py` used by `scripts/training/formal_train.py`. Each run writes TensorBoard events inside its own output directory so experiments cannot share or overwrite telemetry.

The initial configuration is:

```yaml
training:
  tensorboard:
    enabled: true
    log_interval_seconds: 30
```

When `training.tensorboard` is absent or `enabled` is false, training behaves as it did before this change.

## Architecture

TensorBoard support will be isolated behind a small training telemetry component. The component owns the `SummaryWriter`, time-based throttling, scalar names, global-step calculation, flushing, and shutdown. The epoch loop supplies observations; it does not contain TensorBoard-specific timing or path logic.

The writer directory is:

```text
outputs/<experiment>/tensorboard/
```

TensorBoard is an explicit optional runtime feature. If configuration enables it but the TensorBoard package cannot be imported or the event directory cannot be created, startup fails with a clear error before the first training batch. The system will not silently continue without requested telemetry.

## Data Flow

At training startup, the telemetry component reads the resolved configuration, creates the event writer, and derives the initial global batch step from persisted batch history when resuming.

During training, every completed batch updates in-memory running totals. When at least 30 monotonic-clock seconds have elapsed since the last emission, the latest observation is written and flushed. The emission contains:

- `train/loss_batch`: loss from the most recently completed batch.
- `train/loss_running`: sample-weighted mean training loss for the current epoch so far.
- `train/learning_rate`: current optimizer learning rate.
- `performance/samples_per_second`: samples processed since the preceding telemetry emission divided by its elapsed time.
- `memory/gpu_allocated_gib`: PyTorch allocated CUDA memory.
- `memory/gpu_reserved_gib`: PyTorch reserved CUDA memory.

No artificial wait is introduced to meet the interval. If one batch takes longer than 30 seconds, telemetry is written after that batch completes. CPU training records loss, learning rate, and throughput while omitting CUDA-only memory values.

At the end of each epoch, the component records and flushes:

- `epoch/train_loss` and `epoch/val_loss`.
- Macro and micro AUROC, AUPRC, and F1, plus sample F1 when available.
- Per-label AUROC, AUPRC, F1, precision, and recall using stable label names.
- Learning rate before and after the scheduler step.
- Epoch duration and the existing process/GPU resource snapshot values when available.

Epoch metrics use the epoch number as their step. Batch telemetry uses a monotonically increasing global batch step. On resume, the global batch step continues after the batch records stored in the checkpoint, preventing overlapping or reset curves.

## Persistence and Provenance

Existing calls that atomically write `history.csv`, `history_per_label.csv`, `history_batches.csv`, `history.json`, and `history_manifest.json` remain unchanged in authority and cadence. Existing checkpoint content and resume behavior remain intact.

TensorBoard event files are a presentation and monitoring layer. They do not replace source tables, histories, manifests, checkpoints, or frozen evaluation evidence.

The resolved `config.json` records whether TensorBoard was enabled and its 30-second interval. Each experiment continues to use a distinct output directory.

## Shutdown and Failure Handling

The writer is flushed and closed when training finishes normally, stops through early stopping, or raises an exception. Writer cleanup must not hide the original training exception.

Event-write failures are raised with context because continuing after requested monitoring fails would produce misleading gaps. Existing completed history and checkpoints remain recoverable according to their current epoch-boundary cadence.

## User Workflow

Install project dependencies, start or resume a TensorBoard-enabled training run, and launch the dashboard from the repository root:

```powershell
.venv\Scripts\python.exe -m tensorboard.main --logdir outputs --port 6006
```

Open `http://127.0.0.1:6006`. TensorBoard can compare runs because each experiment has a separate event directory under `outputs/`.

## Testing

Tests use a recording writer supplied to the telemetry component rather than starting a TensorBoard server. They verify:

1. No batch emission occurs before 30 seconds and one occurs after the interval.
2. A batch emission contains latest loss, sample-weighted running loss, learning rate, throughput, and applicable CUDA memory values.
3. Epoch emission includes aggregate validation metrics and correctly named per-label metrics.
4. Flush occurs after emissions and close occurs for normal completion and exceptions.
5. Resume derives a strictly increasing global batch step from persisted batch records.
6. Disabled configuration creates no writer and preserves existing training behavior.
7. Enabled configuration without the TensorBoard dependency fails before training with an actionable message.

Targeted telemetry and training-history tests run first. The full test suite and `git diff --check` run before completion is claimed.

## Acceptance Criteria

- A configured training run produces TensorBoard events in its experiment output directory.
- During an active epoch, the dashboard receives batch telemetry no more frequently than every 30 seconds and without deliberate training delays.
- Completed epochs expose aggregate and per-label validation metrics.
- Resume keeps batch steps monotonic.
- TensorBoard failures are explicit, and writer resources are closed reliably.
- Existing history, checkpoint, split, threshold, and evaluation behavior remains unchanged.
