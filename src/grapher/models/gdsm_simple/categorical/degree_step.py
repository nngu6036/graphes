"""One topology-first, bond-only reverse step (opt-in variant)."""
from __future__ import annotations

import numpy as np
import torch

from .model import predictions_numpy
from .noise import draw_categories, draw_bonds, bond_reverse_probs
from .refiner import refine
from .topology import spectral_topology_step


@torch.no_grad()
def reverse_step(model, pred, x, e, z, tv, sv, anchor, mask, total,
                 current_pairs, node_noise, bond_noise, cfg, basis, bins,
                 noise_rng, topology_rng, refine_rng):
    """Only the topology operators own support; categorical noise cannot edit it.

    Structural refinement first changes the noisy typed graph using same-type
    swaps. A separate spectral pair-score decoder can then change its binary
    support. The bond head is queried on the final proposed support, before
    persistent/born-edge bond labels are sampled. Old labels on deleted pairs
    never survive and are never reinterpreted as valid bond observations.
    """
    t, s = int(tv[0]), int(sv[0])
    guide = cfg['guidance']; topo = cfg['topology']
    guide_active = bool(guide['enabled']) and s/total <= guide['start_fraction'] and (s == 0 or s % int(guide['every']) == 0)
    topo_active = s/total <= topo['start_fraction'] and (s == 0 or s % int(topo['every']) == 0)
    support = e > 0
    events = [{'guidance': None, 'topology': None} for _ in range(len(mask))]
    if guide_active or topo_active:
        support = support.clone()
        for i, n in enumerate(mask.sum(1).tolist()):
            ee = e[i, :n, :n].cpu().numpy()
            if guide_active:
                xx = x[i, :n].cpu().numpy()
                targets = predictions_numpy(pred, i, n)
                ee, rd = refine(xx, ee, targets, basis, bins, guide, refine_rng)
                events[i]['guidance'] = {'from_t': t, 'to_t': s,
                                         'phase': 'before_spectral_topology_and_bond_update', **rd}
            a = ee > 0
            if topo_active:
                scores = pred['spectral_scores'][i, :n, :n].cpu().numpy()
                a, rd = spectral_topology_step(a, scores, topo, topology_rng)
                events[i]['topology'] = {'from_t': t, 'to_t': s, **rd}
            support[i, :n, :n] = torch.as_tensor(a, device=e.device)
    # Features/clean spectrum come from the noisy graph. The mask only chooses
    # output pairs: it is never fed back as the clean graph during training.
    if (support & ~(e > 0)).any():
        pred = model.query_bonds(pred, support)
    px = node_noise.reverse_probs(pred['node_logits'].softmax(-1), x, tv, sv)
    new_x = draw_categories(px, noise_rng).masked_fill(~mask, 0)
    if bond_noise.marginal.numel() == 1:
        new_e = support.long()  # Generic graph: no edge classifier or resampling.
    else:
        pe = bond_reverse_probs(bond_noise, pred['edge_logits'].softmax(-1), e, tv, sv)
        new_e = draw_bonds(pe, support, mask, noise_rng)
    if not torch.equal((e > 0).sum(2), (new_e > 0).sum(2)):
        raise AssertionError('Reverse step violated indexed ordinary degrees')
    if not torch.equal(new_e > 0, support):
        raise AssertionError('Categorical labels changed the topology')
    return new_x, new_e, pred, events
