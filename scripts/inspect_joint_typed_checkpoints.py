#!/usr/bin/env python
"""Inspect multi-criterion joint typed-edge snapshots and their matching priors."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import torch
from grapher.rewiring_mlp.attributed.joint_typed_edge_model import load_checkpoint
from grapher.rewiring_mlp.generic.joint_checkpointing import file_sha256,state_dict_sha256
from grapher.models.dhvae_hh.typed_degree_vae import TYPED_CHECKPOINT_FORMAT


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--training-dir',required=True);p.add_argument('--verify',action='store_true')
    args=p.parse_args();root=Path(args.training_dir)
    registry=json.loads((root/'checkpoint_registry.json').read_text())
    if registry.get('format')!='joint_typed_edge_registry_v1':raise ValueError('Not a joint typed edge registry.')
    print(f"{'Selection':18s} {'Epoch':>5s} {'Joint':>10s} {'Edge CE':>10s} {'Hist W1':>10s} {'Orbit logRMSE':>14s}")
    for kind,row in registry['selections'].items():
        m=row['metrics'];folder=root/row['path']
        if args.verify:
            for file,key in [('checkpoint.pt','checkpoint_sha256'),('degree_checkpoint.pt','degree_checkpoint_sha256')]:
                if file_sha256(folder/file)!=row[key]:raise ValueError(f'Hash mismatch: {folder/file}')
            model,joint=load_checkpoint(folder/'checkpoint.pt','cpu')
            print(f"  diffusion mode={model.spectral_mode}; independent_spectral_diffusion={model.diffusion_metadata()['independent_spectral_diffusion']}")
            prior=torch.load(folder/'degree_checkpoint.pt',map_location='cpu',weights_only=True)
            if prior.get('format')!=TYPED_CHECKPOINT_FORMAT:raise ValueError('Wrong exported typed prior format.')
            expected=row['embedded_typed_state_sha256']
            if state_dict_sha256(model.degree_model.state_dict())!=expected or state_dict_sha256(prior['model_state_dict'])!=expected:
                raise ValueError(f'Embedded/exported typed state mismatch: {kind}')
            if joint['epoch']!=row['epoch'] or prior['metrics']['epoch']!=row['epoch']:
                raise ValueError(f'Epoch mismatch: {kind}')
        hist=m.get('val_clustering_histogram_w1');orbit=m.get('val_orbit_summary_log_rmse')
        print(f"{kind:18s} {row['epoch']:5d} {m['val_joint_loss']:10.6f} {m['val_edge_ce_loss']:10.6f} "
              f"{f'{hist:.6f}' if hist is not None else '-':>10s} {f'{orbit:.6f}' if orbit is not None else '-':>14s}")
        if 'val_induced_graphlet_histogram_tv' in m:
            print(f"  induced graphlet TV: {m['val_induced_graphlet_histogram_tv']:.6f}")
            if args.verify and model.induced_graphlet_metadata() is not None:
                info = model.induced_graphlet_metadata()
                print(f"  attributed={info['attributed']} k={info['k']} bins={info['width']} "
                      f"fingerprint={info['fingerprint']}")
    if args.verify:print('All selected joint/typed checkpoints verified.')

if __name__=='__main__':main()
