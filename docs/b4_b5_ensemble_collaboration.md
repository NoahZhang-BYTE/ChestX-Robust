# B4/B5 Ensemble Collaboration

This branch is the minimal competition-submission collaboration baseline. It
starts from the frozen B4+B5 ensemble workflow and excludes unrelated
TensorBoard, Long-Tail, and project-organization changes.

## Scope

- Use frozen B4 and B5 probability bundles as model inputs.
- Select one B4 weight per label using Validation AUPRC only.
- Freeze validation-derived per-label thresholds after weight selection.
- Evaluate Test once after weights and thresholds are frozen.
- Keep checkpoints and large NumPy bundles outside Git; rely on protocol paths
  and SHA-256 manifests.

## Entry Point

```powershell
& '.\.venv\Scripts\python.exe' -m scripts.analysis.ensemble_b4_b5_per_class_auprc `
  --b4-dir artifacts/B4_densenet121_sqrt_posweight_scheduler_eval_20260904_025154 `
  --b5-dir artifacts/B5_densenet121_asl_eval_20260904_1200 `
  --output-dir artifacts/B4B5_ensemble_per_class_auprc_YYYYMMDD
```

The output directory must be new and must not replace historical evidence.
