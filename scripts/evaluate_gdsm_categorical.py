#!/usr/bin/env python
"""Extra attributed metrics on all raw graphs (not a replacement for molecular/ORCA evaluation)."""
import argparse
import json
from pathlib import Path
from grapher.models.gdsm_simple.categorical.evaluation import evaluate


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--generated-dir',type=Path,required=True)
    p.add_argument('--reference-graphs',type=Path,required=True)
    p.add_argument('--output-json',type=Path)
    p.add_argument('--sigma',type=float,default=1.)
    p.add_argument('--max-reference',type=int)
    p.add_argument('--seed',type=int,default=42)
    args=p.parse_args()
    result=evaluate(args.generated_dir,args.reference_graphs,sigma=args.sigma,max_reference=args.max_reference,seed=args.seed)
    path=args.output_json or args.generated_dir/'categorical_metrics.json'
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
