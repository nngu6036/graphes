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
    'guidance':{'enabled':True,'selection':'guided','start_fraction':.2,'every':50,'max_steps_per_event':2,
                'proposal_budget':128,'valid_candidate_budget':64,'same_edge_type':True,
                'preserve_connectivity_if_connected':True,'min_improvement':1e-8,
                'require_structure_improvement':True,
                'weights':{'graphlet':1.,'mass':1.,'clustering':.5,'orbit':.25,'spectral':.25,'edge':.1}},
    # Generation-only sampler acceptance.  This is intentionally separate from
    # local rewiring: categorical reverse transitions may create disconnected
    # states, so molecular profiles can reject a disconnected FINAL sample and
    # continue sampling until the requested number of connected molecules is
    # returned.  The raw acceptance/yield is reported in the generation manifest.
    'final_acceptance':{'require_connected':False,'max_attempt_multiplier':10.0},
    'save_trajectory':False,
}


MULTISCALE_DEFAULTS = {
    'sizes': None, 'size_weights': None, 'counting': 'exact_connected',
    'max_vocab_per_size': None, 'min_train_count': 1,
    'max_connected_subsets': 1000000,
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
    for section in ('noise','guidance','loss_weights','final_acceptance'):
        _unknown(cfg[section],DEFAULTS[section],section)
    _unknown(cfg['guidance']['weights'],DEFAULTS['guidance']['weights'],'guidance.weights')
    if cfg['noise']['type']!='marginal' or cfg['noise']['schedule']!='cosine_exact_terminal':
        raise ValueError('This variant supports marginal node/edge noise with cosine_exact_terminal only')
    if not math.isfinite(float(cfg['noise']['pseudocount'])) or cfg['noise']['pseudocount']<=0: raise ValueError('noise.pseudocount must be >0')
    gc=cfg['graphlets']
    _unknown(gc,{**DEFAULTS['graphlets'],**MULTISCALE_DEFAULTS},'graphlets')
    if gc.get('sizes') is not None:
        # Opt-in so legacy config/checkpoint dictionaries are unchanged.
        gc=merge(MULTISCALE_DEFAULTS,gc);cfg['graphlets']=gc
        orders=gc['sizes']
        if not isinstance(orders,list) or not orders or any(type(k) is not int or k not in (3,4,5) for k in orders) or orders!=sorted(set(orders)):
            raise ValueError('graphlets.sizes must be a sorted unique subset of [3,4,5]')
        if not gc['connected_only'] or gc['counting']!='exact_connected':
            raise ValueError('Multiscale graphlets use exact connected induced counts')
        weights=gc['size_weights']
        if weights is None: weights=[1.]*len(orders)
        if not isinstance(weights,list) or len(weights)!=len(orders) or any(not math.isfinite(float(w)) or float(w)<=0 for w in weights):
            raise ValueError('One finite positive graphlet size weight per order is required')
        gc['size_weights']=[float(w) for w in weights]
        for key in ('max_vocab_per_size','max_connected_subsets'):
            if gc[key] is not None and (type(gc[key]) is not int or gc[key]<1):
                raise ValueError(f'graphlets.{key} must be null or a positive integer')
        if type(gc['min_train_count']) is not int or gc['min_train_count']<1:
            raise ValueError('graphlets.min_train_count must be a positive integer')
    elif gc['size']!=3 or not gc['connected_only']:
        raise ValueError('Use legacy size: 3 or explicit graphlets.sizes: [3,4,5]')
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
    if str(guide.get('selection','guided')).lower() not in ('guided','uniform'):
        raise ValueError("guidance.selection must be 'guided' or 'uniform'")
    guide['selection']=str(guide.get('selection','guided')).lower()
    if not guide['same_edge_type']: raise ValueError('First implementation uses same-edge-type event-local swaps only')
    if not 0<=guide['start_fraction']<=1 or int(guide['every'])<1 or int(guide['max_steps_per_event'])<0:
        raise ValueError('Invalid guidance schedule')
    if int(guide['proposal_budget'])==0 or int(guide['valid_candidate_budget'])==0: raise ValueError('Use positive budgets or -1 for exhaustive candidates')
    if not math.isfinite(guide['min_improvement']) or guide['min_improvement']<0: raise ValueError('Invalid minimum improvement')
    acceptance=cfg['final_acceptance']
    if type(acceptance['require_connected']) is not bool:
        raise ValueError('final_acceptance.require_connected must be boolean')
    multiplier=float(acceptance['max_attempt_multiplier'])
    if not math.isfinite(multiplier) or multiplier<1.0:
        raise ValueError('final_acceptance.max_attempt_multiplier must be finite and >= 1')
    acceptance['max_attempt_multiplier']=multiplier
    for key in ('epochs','batch_size','validation_every','log_every'):
        if int(options['train'].get(key,1))<1: raise ValueError(f'train.{key} must be positive')
    if int(options['diffusion']['steps'])<2 or int(options['sample']['steps'])<1: raise ValueError('Invalid diffusion/sample steps')
    if int(options['sample']['steps'])>int(options['diffusion']['steps']): raise ValueError('sample.steps cannot exceed diffusion.steps')
    if int(options['generation_batch_size'])<1: raise ValueError('generation_batch_size must be positive')
    return cfg
