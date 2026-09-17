#!/usr/bin/env python
"""Validate actual categorical outputs, current-step swaps and dynamic eigenpairs."""
import argparse
import json
from pathlib import Path
from grapher.models.gdsm_simple.categorical.evaluation import audit


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--generated-dir',type=Path,required=True)
    p.add_argument('--output-json',type=Path)
    args=p.parse_args();result=audit(args.generated_dir)
    path=args.output_json or args.generated_dir/'categorical_audit.json'
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
