"""Training-only perturbations of JOINT node-category/typed-degree multisets.

A prior perturbation may change node degrees and weighted valences, subject to
configured bounds. It preserves indexed node categories and every edge-category
incidence total. After source construction, strict rewiring preserves the NEW
indexed typed invariant. These are deliberately different guarantees.

Per-channel graphicality is necessary, not sufficient. Every accepted proposal
has an exact simultaneous, simple connected realization, optionally validated by
a domain callback. Constructor budget exhaustion is not a proof of infeasibility.
No original training adjacency is retained. Temporary witnesses are discarded.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, replace
import hashlib
import json
from typing import Any, Callable, Mapping, Sequence

import networkx as nx
import numpy as np
from scipy.optimize import linear_sum_assignment

from grapher.models.dhvae_hh.degree_perturbation import (
    DegreePerturbationConfig, DegreePerturbationError, _available_blocks,
    _moment_blocks, degree_summary, sum_preserving_round,
)
from grapher.models.dhvae_hh.typed_constructor import (
    TypedConstructorConfig, TypedConstructionError, construct_typed_graph,
)
from grapher.rewiring_mlp.molecular.typed_invariants import (
    TypedDegreeSignature, TypedInvariant, extract_typed_invariant,
    typed_invariant_errors, typed_invariant_matches_graph,
)


def validate_typed_input(invariant: TypedInvariant) -> None:
    """Do not let older dataclasses silently coerce fractional degrees to ints."""
    if not isinstance(invariant, TypedInvariant) or not invariant.num_nodes or not invariant.edge_types:
        raise ValueError('Expected a nonempty TypedInvariant with an edge vocabulary.')
    for signature in invariant.signatures:
        hash(signature.node_type)
        for value in signature.edge_degrees:
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 0:
                raise ValueError('Typed degrees must be nonnegative integers, not floats or booleans.')


def typed_key(invariant: TypedInvariant) -> tuple[TypedDegreeSignature, ...]:
    """Canonicalize WHOLE records, never independently sort type columns."""
    return tuple(sorted(invariant.signatures, key=lambda s: (repr(s.node_type), s.edge_degrees)))


def canonical_invariant(invariant: TypedInvariant) -> TypedInvariant:
    validate_typed_input(invariant)
    return replace(invariant, signatures=typed_key(invariant))


def typed_totals(invariant: TypedInvariant) -> tuple[int, ...]:
    return tuple(int(sum(s.edge_degrees[r] for s in invariant.signatures)) for r in range(len(invariant.edge_types)))


def typed_moments(invariant: TypedInvariant) -> tuple[int, tuple[int, ...]]:
    return (sum(s.degree**2 for s in invariant.signatures),
            tuple(sum(s.edge_degrees[r]**2 for s in invariant.signatures) for r in range(len(invariant.edge_types))))


def align_typed_rows(left: TypedInvariant, right: TypedInvariant) -> np.ndarray:
    """Minimum-L1 matching of complete vectors WITHIN each node category."""
    if left.edge_types != right.edge_types or left.num_nodes != right.num_nodes:
        raise ValueError('Typed alignment requires matching size and edge vocabulary.')
    if Counter(s.node_type for s in left.signatures) != Counter(s.node_type for s in right.signatures):
        raise ValueError('Typed alignment requires matching node-category counts.')
    result = np.empty((left.num_nodes, len(left.edge_types)), dtype=np.int64)
    for atom in sorted({s.node_type for s in left.signatures}, key=repr):
        ii = [i for i,s in enumerate(left.signatures) if s.node_type == atom]
        jj = sorted((j for j,s in enumerate(right.signatures) if s.node_type == atom),
                    key=lambda j: right.signatures[j].edge_degrees)
        a = np.array([left.signatures[i].edge_degrees for i in ii], dtype=np.int64)
        b = np.array([right.signatures[j].edge_degrees for j in jj], dtype=np.int64)
        rows, cols = linear_sum_assignment(np.abs(a[:,None,:]-b[None,:,:]).sum(-1))
        result[np.array(ii)[rows]] = b[cols]
    return result


def typed_distance(left: TypedInvariant, right: TypedInvariant) -> float:
    return float(np.abs(np.array([s.edge_degrees for s in left.signatures]) - align_typed_rows(left,right)).sum()/2)


def typed_fingerprint(invariants: Sequence[TypedInvariant]) -> str:
    payload = [canonical_invariant(inv).to_dict() for inv in invariants]
    return hashlib.sha256(json.dumps(payload,sort_keys=True,separators=(',',':'),default=_json_scalar).encode()).hexdigest()


def _json_scalar(value):
    if isinstance(value, np.generic): return value.item()
    raise TypeError(f'Unsupported typed invariant JSON value: {type(value).__name__}')


class PerturbedEmpiricalTypedDegreeSampler:
    """Sample one eligible training parent, then perturb or explicitly keep it.

    Constructor/domain failures never cause a different parent draw. Incomplete
    multi-step changes roll back transactionally. Static local parent exclusions
    are identical across methods and are reported. Validity of an unchanged
    parent is verified on a NEW realization, not its original training graph.
    """
    def __init__(self, invariants: Sequence[TypedInvariant], config=None, *, seed: int = 42,
                 constructor_config=None, allowed_signatures: Sequence[TypedDegreeSignature] | None = None,
                 endpoint_compatible: Callable | None = None, graph_validator: Callable[[nx.Graph], bool] | None = None):
        self.config = config if isinstance(config,DegreePerturbationConfig) else DegreePerturbationConfig.from_dict(config)
        self.seed = int(seed)
        if self.seed < 0: raise ValueError('seed must be nonnegative.')
        cfg = constructor_config if isinstance(constructor_config,TypedConstructorConfig) else TypedConstructorConfig.from_dict(constructor_config)
        if not cfg.ensure_connected: raise ValueError('Typed perturbation requires connected construction.')
        if cfg.candidate_ranking != 'uniform':
            raise ValueError('Typed empirical perturbation requires uniform witness ranking (no adjacency-derived prior).')
        ceilings = [x for x in (self.config.max_degree,cfg.max_ordinary_degree) if x is not None]
        self.constructor_config = replace(cfg,randomize_assignment=False,max_ordinary_degree=min(ceilings) if ceilings else None)
        self.allowed_signatures = None if allowed_signatures is None else frozenset(allowed_signatures)
        self.endpoint_compatible, self.graph_validator = endpoint_compatible,graph_validator
        original = tuple(canonical_invariant(inv) for inv in invariants)
        if not original: raise ValueError('Typed empirical sampling requires training invariants.')
        schema = (original[0].edge_types,original[0].node_attribute,original[0].edge_attribute)
        if any((v.edge_types,v.node_attribute,v.edge_attribute) != schema for v in original):
            raise ValueError('Training invariants have inconsistent edge vocabulary or attribute names.')
        self.training_set = {typed_key(inv) for inv in original}
        self.original_training_count = len(original)
        self.training_fingerprint = typed_fingerprint(original)
        self.exclusions = []
        eligible,indices = [],[]
        for index,inv in enumerate(original):
            reasons=self._local_errors(inv)
            if reasons: self.exclusions.append({'train_index':index,'reasons':reasons})
            else: eligible.append(inv); indices.append(index)
        if not eligible:
            raise ValueError('No training typed invariant satisfies generation caps/vocabulary; no fallback is allowed.')
        self.invariants, self.train_indices = tuple(eligible),tuple(indices)
        self.parent_rng=np.random.default_rng(np.random.SeedSequence(self.seed,spawn_key=(17,)))
        self.records: list[dict[str,Any]]=[]
        self._groups=defaultdict(dict)
        for inv in self.invariants: self._groups[self._group(inv)][typed_key(inv)] = inv
        self._neighbor_cache={}

    @classmethod
    def fit(cls, graphs: Sequence[nx.Graph], config=None, *, edge_types, node_attribute='atomic_num', edge_attribute='bond_type', **kwargs):
        # Discard all training edges immediately after invariant extraction.
        invs=[extract_typed_invariant(g,edge_types=edge_types,node_attribute=node_attribute,edge_attribute=edge_attribute) for g in graphs]
        return cls(invs,config,**kwargs)

    @staticmethod
    def _group(inv):
        atoms=tuple(sorted(Counter(s.node_type for s in inv.signatures).items(),key=lambda x:repr(x[0])))
        return inv.num_nodes,atoms,typed_totals(inv)

    def _local_errors(self,inv):
        validate_typed_input(inv)
        cfg=self.constructor_config
        errors=typed_invariant_errors(inv,require_connected=True,max_ordinary_degree=cfg.max_ordinary_degree,
            max_weighted_valence=cfg.max_weighted_valence,endpoint_compatible=self.endpoint_compatible)
        for r in range(len(inv.edge_types)):
            if not nx.is_graphical([s.edge_degrees[r] for s in inv.signatures],method='eg'):
                errors.append(f'edge type {inv.edge_types[r]!r}: nongraphical')
        if self.allowed_signatures is not None and any(s not in self.allowed_signatures for s in inv.signatures):
            errors.append('signature_outside_checkpoint_vocabulary')
        return errors

    def _realize(self,inv,rng,rejections):
        errors=self._local_errors(inv)
        if errors:
            rejections['local_feasibility_or_vocabulary']+=1
            return None,{'failure_reason':'local_feasibility_or_vocabulary','errors':errors}
        try:
            witness,diag=construct_typed_graph(inv,self.constructor_config,rng,endpoint_compatible=self.endpoint_compatible)
        except TypedConstructionError as exc:
            reason='joint_constructor_'+str(exc.diagnostics.get('failure_reason','failed'))
            rejections[reason]+=1
            return None,{'failure_reason':reason,'constructor':exc.diagnostics}
        if not typed_invariant_matches_graph(witness,inv): raise AssertionError('Witness lost indexed typed degrees.')
        if self.graph_validator is not None and not self.graph_validator(witness):
            rejections['domain_validator_rejected_witness']+=1
            return None,{'failure_reason':'domain_validator_rejected_witness'}
        return witness,diag

    def _neighbors(self,current):
        key=typed_key(current)
        if key not in self._neighbor_cache:
            candidates=[]
            for other_key,inv in self._groups.get(self._group(current),{}).items():
                if other_key==key: continue
                distance=typed_distance(current,inv)
                limit=self.config.interpolation_max_parent_distance
                if limit is None or distance <= limit: candidates.append((distance,repr(other_key),inv))
            candidates.sort(key=lambda row:(row[0],row[1]))
            if len(candidates)>self.config.interpolation_neighbors:
                boundary=candidates[self.config.interpolation_neighbors-1][0]
                candidates=[row for row in candidates if row[0]<=boundary]
            self._neighbor_cache[key]=tuple(row[2] for row in candidates)
        return self._neighbor_cache[key]

    @staticmethod
    def _replace_matrix(inv,matrix):
        return replace(inv,signatures=tuple(TypedDegreeSignature(s.node_type,tuple(int(x) for x in row))
                        for s,row in zip(inv.signatures,matrix)))

    def _proposals(self,current,rng):
        cfg=self.config; a=np.array([s.edge_degrees for s in current.signatures],dtype=np.int64)
        n,width=a.shape
        if cfg.method=='unit_transfer':
            actions=[(i,j,r) for r in range(width) for i in range(n) if a[i,r]>0 for j in range(n) if i!=j]
            for pos in rng.permutation(len(actions)):
                i,j,r=actions[int(pos)]; b=a.copy(); b[i,r]-=1; b[j,r]+=1
                yield self._replace_matrix(current,b),None,{'donor':i,'recipient':j,'edge_type':current.edge_types[r]}
        elif cfg.method=='moment_preserving':
            # A block shares category and all OTHER typed coordinates. Preserving
            # the active column's first two moments then also preserves sum(d^2).
            groups=defaultdict(list)
            for r in range(width):
                for i,s in enumerate(current.signatures): groups[(r,s.node_type,tuple(np.delete(a[i],r)))].append(i)
            keys=list(groups)
            for pos in rng.permutation(len(keys)):
                key=keys[int(pos)]; r=key[0]; indices=groups[key]
                if len(indices)<cfg.block_size: continue
                vals=[int(a[i,r]) for i in indices]
                blocks=_available_blocks(vals,cfg.block_size,cfg.max_block_patterns)
                ceiling=min(n-1,self.constructor_config.max_ordinary_degree or n-1)-sum(key[2])
                for bp in rng.permutation(len(blocks)):
                    old=blocks[int(bp)]; k=len(old); total=sum(old); squares=sum(x*x for x in old)
                    # Shift by 1 to support zero typed degrees in the positive catalogue.
                    alts=[tuple(x-1 for x in b) for b in _moment_blocks(k,total+k,squares+2*total+k,ceiling+1,cfg.max_block_alternatives)]
                    for ap in rng.permutation(len(alts)):
                        new=alts[int(ap)]
                        if new==old: continue
                        remaining=list(indices); selected=[]
                        for value in old:
                            i=next(i for i in remaining if a[i,r]==value); remaining.remove(i); selected.append(i)
                        b=a.copy()
                        for i,value in zip(selected,new): b[i,r]=value
                        yield self._replace_matrix(current,b),None,{'edge_type':current.edge_types[r],
                            'indices':selected,'old_block':list(old),'new_block':list(new)}
        elif cfg.method=='edge_relocation':
            witness,diag=self._realize(current,rng,self._active_rejections)
            if witness is None:
                self._proposal_failure=diag['failure_reason']; return
            actions=[]
            for u,v,data in sorted(witness.edges(data=True)):
                r=current.edge_types.index(data[current.edge_attribute])
                for i,j in ((u,v),(v,u)):
                    if current.signatures[i].degree<=1: continue
                    for w in range(n):
                        if w not in (i,j) and not witness.has_edge(w,j): actions.append((i,j,w,r))
            for pos in rng.permutation(len(actions)):
                u,v,w,r=actions[int(pos)]; graph=witness.copy(); graph.remove_edge(u,v)
                graph.add_edge(w,v,**{current.edge_attribute:current.edge_types[r]})
                b=a.copy();b[u,r]-=1;b[w,r]+=1
                yield self._replace_matrix(current,b),graph,{'removed_edge':[u,v],'inserted_edge':[w,v],
                    'edge_type':current.edge_types[r]}
        else:
            neighbors=self._neighbors(current)
            if not neighbors:
                self._proposal_failure='no_distinct_training_partner_same_n_node_counts_edge_counts'; return
            for _ in range(cfg.max_attempts):
                partner=neighbors[int(rng.integers(len(neighbors)))]; b=align_typed_rows(current,partner)
                x=(1-cfg.interpolation_alpha)*a+cfg.interpolation_alpha*b
                rounded=np.empty_like(a)
                for r,total in enumerate(typed_totals(current)):
                    rounded[:,r]=sum_preserving_round(x[:,r],total,rng,sort_result=False)
                yield self._replace_matrix(current,rounded),None,{'partner_typed_invariant':partner.to_dict(),
                    'alpha':cfg.interpolation_alpha,'parent_partner_distance':typed_distance(current,partner)}

    def _one_step(self,current,parent,visited,rng,rejections):
        self._active_rejections=rejections;self._proposal_failure=None
        checks=0
        self._step_checks=0
        for proposal,witness,info in self._proposals(current,rng):
            if checks>=self.config.max_attempts: break
            checks+=1; self._step_checks=checks; key=typed_key(proposal)
            if key in visited:
                rejections['unchanged_or_visited_multiset']+=1;continue
            if tuple(s.node_type for s in proposal.signatures)!=tuple(s.node_type for s in parent.signatures):
                raise AssertionError('Perturbation changed indexed node categories.')
            if typed_totals(proposal)!=typed_totals(parent): raise AssertionError('Perturbation changed per-type totals.')
            if self.config.require_novel and key in self.training_set:
                rejections['not_novel_vs_training']+=1;continue
            if self.config.max_distance is not None and typed_distance(parent,proposal)>self.config.max_distance:
                rejections['max_distance']+=1;continue
            if 'partner_typed_invariant' in info and key==typed_key(TypedInvariant.from_dict(info['partner_typed_invariant'])):
                rejections['interpolation_reproduces_partner']+=1;continue
            if self.config.method=='moment_preserving' and typed_moments(proposal)!=typed_moments(parent):
                raise AssertionError('Typed moment replacement changed required second moments.')
            if witness is None:
                witness,diag=self._realize(proposal,rng,rejections)
                if witness is None:continue
            else:
                if self._local_errors(proposal): rejections['local_feasibility_or_vocabulary']+=1;continue
                if not nx.is_connected(witness): rejections['edge_relocation_disconnected']+=1;continue
                if not typed_invariant_matches_graph(witness,proposal): raise AssertionError('Bad relocation witness.')
                if self.endpoint_compatible is not None and any(
                    not self.endpoint_compatible(witness.nodes[u][proposal.node_attribute],witness.nodes[v][proposal.node_attribute],d[proposal.edge_attribute])
                    or not self.endpoint_compatible(witness.nodes[v][proposal.node_attribute],witness.nodes[u][proposal.node_attribute],d[proposal.edge_attribute])
                    for u,v,d in witness.edges(data=True)):
                    rejections['endpoint_compatibility']+=1;continue
                if self.graph_validator is not None and not self.graph_validator(witness):
                    rejections['domain_validator_rejected_witness']+=1;continue
            return proposal,checks,info,None
        return None,checks,{},self._proposal_failure or ('candidate_budget_exhausted' if checks>=self.config.max_attempts else 'no_valid_'+self.config.method)

    def sample(self,rng=None):
        rng=self.parent_rng if rng is None else rng
        return self.perturb_parent(int(rng.integers(len(self.invariants))))

    def perturb_parent(self,parent_index: int,*,requested: bool | None = None):
        if parent_index<0 or parent_index>=len(self.invariants): raise IndexError('Eligible training parent index out of range.')
        idx=len(self.records);cfg=self.config;parent=self.invariants[parent_index]
        coin=np.random.default_rng(np.random.SeedSequence(self.seed,spawn_key=(18,idx,0)))
        rng=np.random.default_rng(np.random.SeedSequence(self.seed,spawn_key=(18,idx,1)))
        selected=bool(coin.random()<cfg.probability) if requested is None else bool(requested)
        current=parent; visited={typed_key(parent)}; operations=[];failure=None;checks=0;rejections=Counter()
        if selected:
            for _ in range(cfg.steps):
                try:
                    candidate,count,info,failure=self._one_step(current,parent,visited,rng,rejections)
                except DegreePerturbationError as exc:
                    candidate,count,info,failure=None,self._step_checks,{},'catalogue_budget_exceeded: '+str(exc)
                checks+=count
                if candidate is None:break
                current=candidate;visited.add(typed_key(current));operations.append(info)
        failed=selected and failure is not None
        if failed:current=parent
        # Unchanged samples/fallbacks need their own generated feasibility witness.
        output_failure=None
        if typed_key(current)==typed_key(parent):
            graph,diag=self._realize(current,rng,rejections)
            if graph is None:output_failure=diag['failure_reason']
        changed=typed_key(current)!=typed_key(parent)
        record={'sample_index':idx,'method':cfg.method,'parent_train_index':self.train_indices[parent_index],
            'parent_eligible_index':parent_index,'parent_typed_invariant':parent.to_dict(),'typed_invariant':current.to_dict(),
            'parent_degree_sequence':parent.degree_sequence,'degree_sequence':current.degree_sequence,
            'requested':selected,'changed':changed,'ordinary_degrees_changed':parent.degree_sequence!=current.degree_sequence,
            'novel_vs_training':typed_key(current) not in self.training_set,'failure_reason':failure,
            'output_failure':output_failure,'fallback_used':bool(failed and cfg.failure_policy=='keep_original'),
            'failure_policy':cfg.failure_policy,'candidate_checks':checks,'proposal_rejections':dict(rejections),
            'accepted_steps':len(operations) if not failed else 0,'rolled_back_steps':len(operations) if failed else 0,
            'distance_half_l1':typed_distance(parent,current),'joint_realization_verified':output_failure is None,
            'preserved_node_categories':tuple(s.node_type for s in parent.signatures)==tuple(s.node_type for s in current.signatures),
            'preserved_edge_type_counts':typed_totals(parent)==typed_totals(current),
            'preserved_second_moments':typed_moments(parent)==typed_moments(current),
            'node_count':current.num_nodes,'edge_type_totals':list(typed_totals(current)),
            'operations':operations,'repair_used':False,'attempts_used':1}
        self.records.append(record)
        if output_failure or (failed and cfg.failure_policy=='error'):
            raise DegreePerturbationError(f'Typed {cfg.method} failed for training parent {self.train_indices[parent_index]}: '
                f'{output_failure or failure}. No parent redraw, degree repair, or untyped fallback was performed.')
        summary=degree_summary(current.degree_sequence)
        summary.update(typed_invariant=current.to_dict(),sampling_diagnostics=record)
        return summary

    def report(self):
        rows=self.records;n=len(rows);requested=sum(r['requested'] for r in rows);changed=sum(r['changed'] for r in rows)
        failures=Counter(r['failure_reason'] for r in rows if r['failure_reason']);rejections=Counter()
        for row in rows:rejections.update(row['proposal_rejections'])
        mean=lambda key:float(np.mean([r[key] for r in rows])) if rows else 0.0
        return {'format':'empirical_typed_degree_perturbation_v1','method':self.config.method,'training_only':True,
            'config':asdict(self.config),'constructor_config':asdict(self.constructor_config),
            'checkpoint_signature_mask_enabled':self.allowed_signatures is not None,'domain_validator_enabled':self.graph_validator is not None,
            'num_training_graphs':self.original_training_count,'num_eligible_training_graphs':len(self.invariants),
            'parent_exclusions':self.exclusions,'training_typed_fingerprint':self.training_fingerprint,
            'parent_typed_fingerprint':typed_fingerprint([TypedInvariant.from_dict(r['parent_typed_invariant']) for r in rows]),
            'sampled_typed_fingerprint':typed_fingerprint([TypedInvariant.from_dict(r['typed_invariant']) for r in rows]),
            'num_samples':n,'num_requested':requested,'num_changed':changed,
            'changed_fraction':changed/n if n else 0.0,'success_given_requested':changed/requested if requested else None,
            'novel_typed_fraction':mean('novel_vs_training'),'ordinary_degree_changed_fraction':mean('ordinary_degrees_changed'),
            'num_identity_fallbacks':sum(r['fallback_used'] for r in rows),
            'failure_reasons':dict(failures),'proposal_rejections':dict(rejections),
            'mean_distance_half_l1':mean('distance_half_l1'),
            'all_joint_realizations_verified':all(r['joint_realization_verified'] for r in rows),
            'all_preserve_node_categories':all(r['preserved_node_categories'] for r in rows),
            'all_preserve_edge_type_counts':all(r['preserved_edge_type_counts'] for r in rows),
            'all_preserve_second_moments':all(r['preserved_second_moments'] for r in rows),'records':rows}
