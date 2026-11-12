#!/usr/bin/env python3
"""Inventory recorded experiment evidence without loading models or graph pickles.

Missing commit/environment/attempt/population metadata remains explicitly absent.
The current source hash is not retroactively assigned to historical runs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha(path):
    h=hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1<<20), b""):
            h.update(chunk)
    return h.hexdigest()


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    args=p.parse_args(argv)
    if not args.root.is_dir():
        p.error("Experiment artifact root does not exist")
    records=[]
    for path in sorted(args.root.rglob("manifest.json")):
        row={"manifest_path":str(path.resolve()),"manifest_sha256":sha(path)}
        try:
            data=json.loads(path.read_text())
            row["recorded_metadata"]={k:data[k] for k in ("format","model_id","run_id","dataset","train_seed","generation_seed","num_graphs","num_generated","num_requested","requested_count","completed_count","attempted_count","duration_seconds","checkpoint_selection","checkpoint","environment","source","source_commit","provenance","shared_context","sample_ids","requested_sample_ids","failures") if k in data}
            artifacts=[]
            for key,value in data.items():
                if not isinstance(value,dict) or "path" not in value or "sha256" not in value:
                    continue
                candidate=path.parent/str(value["path"])
                actual=sha(candidate) if candidate.is_file() else None
                artifacts.append({"key":key,"path":str(candidate),"recorded_sha256":value["sha256"],"observed_sha256":actual,"matches":actual==value["sha256"]})
            row["artifacts"]=artifacts
            row["missing_recorded_fields"]=[key for key, present in {
                "source_commit": bool(data.get("source_commit") or data.get("source",{}).get("commit") or data.get("environment",{}).get("git_commit")),
                "environment": bool(data.get("environment")),
                "population_or_dataset_identity": bool(data.get("sample_ids") or data.get("dataset") or data.get("provenance")),
                "attempt_count": any(k in data for k in ("attempted_count","num_attempts","N_att")),
            }.items() if not present]
        except (OSError,ValueError,TypeError,AttributeError) as exc:
            row["error"]=f"{type(exc).__name__}: {exc}"
        records.append(row)
    repo=Path(__file__).resolve().parents[1]
    files={str(path.relative_to(repo)):sha(path) for folder in ("src","scripts","commands","configs") for path in sorted((repo/folder).rglob("*")) if path.is_file() and path.suffix in (".py",".sh",".yaml",".yml")}
    report={"format":"graphes_experiment_inventory_v1","artifact_root":str(args.root.resolve()),"manifest_count":len(records),"runs":records,
            "current_source_files":files,"current_source_role":"audit tool snapshot only; not evidence of historical source revision", "note":"An absent historical value cannot be recovered by this inventory. Preserve original manifests/logs or rerun under a new recorded protocol."}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
    print(f"Inventoried {len(records)} manifests -> {args.output}")
    return 1 if any(r.get("error") or any(not a["matches"] for a in r.get("artifacts",[])) for r in records) else 0


if __name__=="__main__":
    raise SystemExit(main())
