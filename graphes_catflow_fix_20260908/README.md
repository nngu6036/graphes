# CatFlow correction patch — 8 September 2026

Run from the existing GraphER root after extracting this archive:

```bash
python /tmp/graphes_catflow_fix_20260908/apply_catflow_fix.py --repo "$PWD" --dry-run
python /tmp/graphes_catflow_fix_20260908/apply_catflow_fix.py --repo "$PWD"
```

The installer targets graphes_four_baseline_wrappers_20260906.zip, backs up replaced
files, and refuses local-file conflicts before changing anything. Existing runs
and external CatFlow sources are untouched. A corrected model requires retraining
under a new run ID; old checkpoints retain legacy sampling behavior.

Read `files/docs/CATFLOW_PATH_AUDIT_20260908.md` for diagnosis, commands, the exact
source inconsistency, validation evidence, and important limits. The original user
MMD failure has not been rerun or shown to recover in this environment.
