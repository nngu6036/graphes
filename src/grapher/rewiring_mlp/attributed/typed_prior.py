"""Shared empirical typed-prior dispatch for both attributed predictor families."""
from __future__ import annotations
from copy import deepcopy

from grapher.models.dhvae_hh.typed_degree_perturbation import PerturbedEmpiricalTypedDegreeSampler


def uses_typed_empirical_kernel(config):
    generation=config.get('generation',{})
    source=generation.get('invariant_source',generation.get('degree_source',
        'learned' if config.get('joint_typed_degree',{}).get('enabled',False) else 'empirical'))
    mode=generation.get('invariant_rng_mode','legacy')
    if mode not in ('legacy','independent'):
        raise ValueError('generation.invariant_rng_mode must be legacy or independent.')
    active=source=='train_empirical_perturbed' or (source in ('train_empirical','empirical') and mode=='independent')
    if generation.get('degree_perturbation') and not active:
        raise ValueError('Typed degree_perturbation settings require invariant_source=train_empirical_perturbed; '
                         'they cannot be silently ignored for a learned/legacy source.')
    return active


def build_typed_empirical_sampler(config,train_graphs,*,seed,edge_types,node_attribute='atomic_num',
                                  edge_attribute='bond_type',vectorizer=None,graph_validator=None):
    if not uses_typed_empirical_kernel(config): return None
    generation=config.get('generation',{});source=generation.get('invariant_source',generation.get('degree_source',
        'learned' if config.get('joint_typed_degree',{}).get('enabled',False) else 'empirical'))
    settings=deepcopy(generation.get('degree_perturbation',{}))
    if source in ('train_empirical','empirical'):
        settings={'method':'unit_transfer','probability':0.,'failure_policy':'error'}
    ctor=deepcopy(config.get('constructor',{}));sig=config.get('typed_signature',{})
    # Intersect all declared generation caps; never silently relax one.
    degree_caps=[v for v in (ctor.get('max_ordinary_degree'),sig.get('max_ordinary_degree'),
                  getattr(vectorizer,'max_ordinary_degree',None)) if v is not None]
    if degree_caps:ctor['max_ordinary_degree']=min(degree_caps)
    maps=[m for m in (ctor.get('max_weighted_valence'),sig.get('max_weighted_valence'),
                      getattr(vectorizer,'max_weighted_valence',None)) if m is not None]
    if maps:
        common=set(maps[0]).intersection(*(set(m) for m in maps[1:]))
        ctor['max_weighted_valence']={k:min(m[k] for m in maps) for k in common}
    return PerturbedEmpiricalTypedDegreeSampler.fit(train_graphs,settings,seed=seed,
        edge_types=edge_types,node_attribute=node_attribute,edge_attribute=edge_attribute,
        constructor_config=ctor,graph_validator=graph_validator,
        allowed_signatures=vectorizer.vocabulary.signatures if vectorizer is not None else None)


def validate_strict_typed_refinement(refiner, enrichment=None):
    """The legacy attributed family permits drift unless explicitly prohibited."""
    for name,cfg in (('attributed_refiner',refiner),('source_enrichment.rewiring',enrichment)):
        if cfg is None:continue
        if not cfg.require_same_edge_type_pair or not cfg.preserve_typed_degree or not cfg.preserve_weighted_valence:
            raise ValueError(f'{name} must enable molecular.require_same_edge_type_pair, '
                'molecular.preserve_typed_degree, and molecular.preserve_weighted_valence '
                'for typed degree perturbation. Cross-type reassignment is not compatible.')
