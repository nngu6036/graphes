"""Explicit new-variant options; legacy extension keys cannot silently constrain it."""
from __future__ import annotations
import copy
import math

DEFAULTS={
    'enabled':True,
    'categories':{'node_attribute':None,'edge_attribute':None},
    'noise':{'type':'marginal','schedule':'cosine_exact_terminal','pseudocount':.001},
    'graphlets':{'size':3,'connected_only':True,'clustering_bins':100},
    'initialization':{'mode':'degree_basis','ridge':.001,'diagonal_weight':1.,'basis_max_per_size':32,
                      'degree_generator':{'type':'empirical','fallback':'error','postprocess_policy':'reject_only'}},
    'spectral_conditioning':True,
    'spectrum_feedback':.05,
    'feedback_start_fraction':.2,
    'loss_weights':{'spectral':1.,'node':1.,'edge':1.,'graphlet':.5,'mass':.5,'clustering':.5,'orbit':.25},
    'guidance':{'enabled':True,'start_fraction':.2,'every':50,'max_steps_per_event':2,
                'proposal_budget':128,'valid_candidate_budget':64,'same_edge_type':True,
                'preserve_connectivity_if_connected':True,'min_improvement':1e-8,
                'require_structure_improvement':True,
                'weights':{'graphlet':1.,'mass':1.,'clustering':.5,'orbit':.25,'spectral':.25,'edge':.1}},
    'save_trajectory':False,
}


def merge(base,changes):
    out=copy.deepcopy(base)
    for k,v in changes.items():
        if isinstance(v,dict) and isinstance(out.get(k),dict): out[k]=merge(out[k],v)
        else: out[k]=copy.deepcopy(v)
    return out


def _unknown(value,template,path):
    extra=set(value)-set(template)
    if extra: raise ValueError(f"Unknown {path} settings: {sorted(extra)}")


def resolve(options):
    ext=options.get('extensions',{})
    raw=ext.get('attributed_categorical',{})
    _unknown(raw,DEFAULTS,'attributed_categorical')
    cfg=merge(DEFAULTS,raw)
    if not cfg['enabled']: raise ValueError('attributed_categorical must be enabled')
    if ext.get('degree_conditioning') or ext.get('hh_initialization') or ext.get('degree_preserving_rewiring'):
        raise ValueError('Disable legacy degree/HH/rewiring flags; categorical.guidance controls event-local swaps')
    if ext.get('structural_summary','none') != 'none':
        raise ValueError('Use attributed_categorical.graphlets, not legacy structural_summary')
    for section in ('noise','graphlets','guidance','loss_weights'):
        _unknown(cfg[section],DEFAULTS[section],section)
    _unknown(cfg['guidance']['weights'],DEFAULTS['guidance']['weights'],'guidance.weights')
    if cfg['noise']['type']!='marginal' or cfg['noise']['schedule']!='cosine_exact_terminal':
        raise ValueError('This variant supports marginal node/edge noise with cosine_exact_terminal only')
    if not math.isfinite(float(cfg['noise']['pseudocount'])) or cfg['noise']['pseudocount']<=0: raise ValueError('noise.pseudocount must be >0')
    if cfg['graphlets']['size']!=3 or not cfg['graphlets']['connected_only']:
        raise ValueError('Use connected induced typed graphlets of size 3 plus connected-triple mass')
    if int(cfg['graphlets']['clustering_bins'])<2: raise ValueError('clustering_bins must be >=2')
    init=cfg['initialization']
    _unknown(init,DEFAULTS['initialization'],'initialization')
    if init['mode'] not in ('degree_basis','gaussian'): raise ValueError('Unknown initialization mode')
    if init['degree_generator']['type'] not in ('dhvae','empirical'): raise ValueError('Use ordinary-degree dhvae or empirical prior')
    if init['degree_generator'].get('fallback','error')!='error' or init['degree_generator'].get('postprocess_policy','reject_only')!='reject_only':
        raise ValueError('Degree prior requires fallback=error and postprocess_policy=reject_only')
    if not math.isfinite(float(init['ridge'])) or not math.isfinite(float(init['diagonal_weight'])) or init['ridge']<=0 or init['diagonal_weight']<0 or int(init['basis_max_per_size'])<1: raise ValueError('Invalid anchor settings')
    for v in list(cfg['loss_weights'].values())+list(cfg['guidance']['weights'].values()):
        if not math.isfinite(float(v)) or float(v)<0: raise ValueError('Loss/energy weights must be finite and nonnegative')
    if not any(cfg['loss_weights'].values()): raise ValueError('At least one loss is required')
    for key in ('spectrum_feedback','feedback_start_fraction'):
        if not 0<=float(cfg[key])<=1: raise ValueError(f'{key} must be in [0,1]')
    guide=cfg['guidance']
    if not guide['same_edge_type']: raise ValueError('First implementation uses same-edge-type event-local swaps only')
    if not 0<=guide['start_fraction']<=1 or int(guide['every'])<1 or int(guide['max_steps_per_event'])<0:
        raise ValueError('Invalid guidance schedule')
    if int(guide['proposal_budget'])==0 or int(guide['valid_candidate_budget'])==0: raise ValueError('Use positive budgets or -1 for exhaustive candidates')
    if not math.isfinite(guide['min_improvement']) or guide['min_improvement']<0: raise ValueError('Invalid minimum improvement')
    for key in ('epochs','batch_size','validation_every','log_every'):
        if int(options['train'].get(key,1))<1: raise ValueError(f'train.{key} must be positive')
    if int(options['diffusion']['steps'])<2 or int(options['sample']['steps'])<1: raise ValueError('Invalid diffusion/sample steps')
    if int(options['sample']['steps'])>int(options['diffusion']['steps']): raise ValueError('sample.steps cannot exceed diffusion.steps')
    if int(options['generation_batch_size'])<1: raise ValueError('generation_batch_size must be positive')
    return cfg
