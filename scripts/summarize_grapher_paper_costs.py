#!/usr/bin/env python3
"""Aggregate measured full-pipeline profiles; keep failures in cost denominators."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from pathlib import Path
import statistics


def stats(values):
    values=[v for v in values if v is not None]
    return {"n":len(values),"mean":statistics.mean(values) if values else None,"sample_sd":statistics.stdev(values) if len(values)>1 else None}


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--profile-root",type=Path,required=True)
    p.add_argument("--output-dir",type=Path,required=True)
    args=p.parse_args(argv)
    profiles=[]
    for path in sorted(args.profile_root.rglob("*.json")):
        value=json.loads(path.read_text())
        if not isinstance(value,dict) or value.get("format")!="graphes_paper_runtime_v1":
            continue
        label=value["label"]
        group=label.rsplit("_seed_",1)[0]
        row={"profile":str(path.resolve()),"profile_sha256":hashlib.sha256(path.read_bytes()).hexdigest(),"group":group,"seed":label.rsplit("_seed_",1)[-1],"phase":value["phase"],"status":value["status"],"requested_count":value["requested_count"],"attempted_samples":value.get("attempted_samples"),"completed_endpoints_before_rewiring":value.get("completed_endpoints_before_rewiring"),"persisted_output_count":value.get("persisted_output_count"),"wall_seconds":value["wall_seconds"],"seconds_per_requested_sample":value.get("seconds_per_requested_sample"),"peak_host_rss_kib":value.get("peak_host_rss_kib"),"peak_gpu_allocated_bytes":value.get("peak_gpu_allocated_bytes"),"artifact_dir":value.get("artifact_dir"),"artifact_manifest_sha256":value.get("artifact_manifest_sha256"),"environment":value.get("environment"),"cache_before":value.get("preprocessing_cache_before")}
        for stage in ("basis_reverse","degree_projection","topology_decode","attributes","rewiring","generation_pipeline","spectral_preprocessing","spectral_normalizer_fit","summary_targets","summary_targets_and_cache","typed_vocabulary_fit","training_pipeline"):
            measured=value.get("stages",{}).get(stage)
            seconds=measured["exclusive_seconds"] if measured else None
            # Not called is N/A/unknown, never silently a measured zero.
            row[stage+"_seconds"]=seconds
            row[stage+"_seconds_per_request"]=seconds/value["requested_count"] if seconds is not None and value["phase"]=="generate" else None
        row["unclassified_seconds"]=value.get("unclassified_seconds")
        profiles.append(row)
    if not profiles:
        p.error("No graphes_paper_runtime_v1 profiles found")
    summaries=[]
    for group in sorted({r["group"] for r in profiles}):
        rows=[r for r in profiles if r["group"]==group]
        for key in ("wall_seconds","seconds_per_requested_sample","peak_host_rss_kib","peak_gpu_allocated_bytes",*(k for k in rows[0] if k.endswith("_seconds_per_request"))):
            summaries.append({"group":group,"metric":key,**stats([r.get(key) for r in rows]),"failed_runs":sum(r["status"]!="complete" for r in rows),"costs_include_failed_runs":True})
    args.output_dir.mkdir(parents=True,exist_ok=True)
    report={"format":"graphes_paper_cost_summary_v1","profiles":profiles,"summary":summaries,"notes":["Exclusive stage wall times are additive; include generation/training pipeline overhead and unclassified overhead when checking totals.","Absent calls are null, including inapplicable attributes/projection; do not treat absent measurements as zero.","A failed run contributes time and its originally requested denominator. Report its completion/attempt counts alongside cost.","New profile populations do not recover historical N_att or establish identical output populations.","Repeated seeds measure run variation, not an independent hardware benchmarking replicate design."]}
    (args.output_dir/"cost_summary.json").write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
    with (args.output_dir/"cost_summary.csv").open("w",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(summaries[0]));writer.writeheader();writer.writerows(summaries)
    print(f"Summarized {len(profiles)} measured profiles -> {args.output_dir}")
    return 0


if __name__=="__main__":
    raise SystemExit(main())
