"""Audits and separately named attributed-feature MMDs; no repairs or ORCA replacement."""
from __future__ import annotations
import json
import pickle
from pathlib import Path
from collections import Counter
import numpy as np
import networkx as nx
from scipy.spatial.distance import cdist
from grapher.models.gdsm_simple.wrapper import _sha256
from grapher.rewiring_mlp.attributed.data import GraphCategoryVocabulary
from .data import TypedGraphlets3,encode_graph,typed_counts
from .training_basis import basis_digest


def load_pickle(path):
    # Only use project-generated or otherwise trusted pickle files.
    with Path(path).open('rb') as handle:
        return pickle.load(handle)


def audit(generated_dir):
    root=Path(generated_dir);manifest=json.loads((root/'manifest.json').read_text())
    if manifest.get('format')!='grapher_gdsm_spectral_categorical_generation_v1':
        raise ValueError('Expected a managed spectral-categorical generation directory')
    for name,digest in manifest['artifact_sha256'].items():
        if _sha256(root/name)!=digest:raise ValueError(f'Artifact checksum mismatch: {name}')
    schema=json.loads((root/'categorical_schema.json').read_text())
    vocab=GraphCategoryVocabulary.from_dict(schema['category_vocabulary'])
    final=load_pickle(root/'base_graphs.pkl');initial=load_pickle(root/'initial_graphs.pkl')
    pre=load_pickle(root/'final_pre_rewire_graphs.pkl');degrees=load_pickle(root/'sampled_degree_sequences.pkl')
    vals=load_pickle(root/'final_adjacency_eigenvalues.pkl');vectors=load_pickle(root/'final_eigenvectors.pkl')
    n=len(final)
    if not n or any(len(a)!=n for a in (initial,pre,degrees,vals,vectors)):
        raise ValueError('Empty or mismatched output batches')
    spectral_mode=manifest.get('decode',{}).get('edge_existence_authority')=='spectral_topology_decoder'
    fixed_degrees=bool(manifest.get('decode',{}).get('exact_indexed_degree_guarantee',False))
    predictions=load_pickle(root/'predicted_summaries.pkl') if spectral_mode else None
    trajectories=load_pickle(root/'categorical_trajectories.pkl') if (root/'categorical_trajectories.pkl').is_file() else None
    if predictions is not None and len(predictions)!=n:
        raise AssertionError('Predicted spectral supports and graph counts differ')
    if trajectories is not None and len(trajectories)!=n:
        raise AssertionError('Trajectory and graph counts differ')
    fixed_basis=manifest.get('decode',{}).get('basis_source')=='training_bank'
    degree_trace=load_pickle(root/'degree_trajectories.pkl') if (root/'degree_trajectories.pkl').is_file() else None
    if degree_trace is not None:
        if degree_trace.get('timesteps')!=manifest['sampling_timesteps'] or len(degree_trace['indexed_degrees'])!=n:
            raise AssertionError('Degree trace timesteps or graph count disagree with manifest')
    if manifest.get('degree_trace',{}).get('saved') and degree_trace is None:
        raise AssertionError('Manifest requests a degree trace but the artifact is missing')
    sampled_bases=basis_records=checkpoint_bank=None
    sampled_basis_matches_checkpoint=None
    max_proposal_error=0.
    if fixed_basis:
        sampled_bases=load_pickle(root/'sampled_training_eigenvectors.pkl')
        basis_records=load_pickle(root/'sampled_training_basis_records.pkl')
        if len(sampled_bases)!=n or len(basis_records)!=n:
            raise AssertionError('Fixed-basis artifact counts differ from generated graphs')
        if not fixed_degrees or not manifest['decode'].get('fixed_initial_eigenbasis'):
            raise AssertionError('Training-bank decoding requires fixed basis and indexed-degree constraint')
        checkpoint_path=Path(manifest['checkpoint']['path'])
        if checkpoint_path.is_file():
            if _sha256(checkpoint_path)!=manifest['checkpoint']['sha256']:
                raise ValueError('Checkpoint checksum mismatch during basis provenance audit')
            # Only load the trusted managed checkpoint referenced by this run.
            import torch
            checkpoint_state=torch.load(checkpoint_path,map_location='cpu',weights_only=False)
            checkpoint_bank={int(k):v for k,v in checkpoint_state['basis_bank'].items()}
            del checkpoint_state
            sampled_basis_matches_checkpoint=True
    prior_match=initial_match=indexed_match=connected=0;max_error=0.;max_orthogonal_error=0.
    for i,(g,g0,gpre,d,z,u) in enumerate(zip(final,initial,pre,degrees,vals,vectors)):
        if len(g)!=len(g0) or len(g)!=len(gpre) or len(g)!=len(d):raise AssertionError('Node count changed')
        x,e=encode_graph(g,vocab,len(g));xp,ep=encode_graph(gpre,vocab,len(g))
        x0,e0=encode_graph(g0,vocab,len(g))
        if list(g.nodes())!=list(range(len(g))):raise AssertionError('Final node indices must be 0..n-1')
        np.testing.assert_array_equal(x,xp)
        np.testing.assert_array_equal((e>0).sum(1),(ep>0).sum(1))
        for k in range(1,vocab.num_edge_categories):
            np.testing.assert_array_equal((e==k).sum(1),(ep==k).sum(1))
        error=float(np.max(np.abs((u*(np.asarray(z)*np.sqrt(len(g)))[None])@u.T-(e>0))))
        orth=float(np.max(np.abs(u.T@u-np.eye(len(g)))))
        max_error=max(max_error,error);max_orthogonal_error=max(max_orthogonal_error,orth)
        if error>2e-4 or orth>2e-4:raise AssertionError(f'Final eigenpairs do not represent retained adjacency #{i}')
        indexed_match+=int(np.array_equal((e>0).sum(1),d))
        if spectral_mode:
            support=np.asarray(predictions[i]['topology_support'],dtype=bool)
            np.testing.assert_array_equal(ep>0,support,err_msg='Final bond draw altered spectral support')
            if manifest['decode']['bond_head_includes_no_edge']:
                raise AssertionError('Spectral topology cannot use a no-edge bond class')
        if fixed_degrees:
            np.testing.assert_array_equal((e0>0).sum(1),d,err_msg='Initial indexed degrees differ from prior')
            np.testing.assert_array_equal((e>0).sum(1),d,err_msg='Final indexed degrees differ from prior')
            if trajectories is not None:
                for step in trajectories[i]:
                    labels=np.asarray(step['edge_categories'])
                    if not np.array_equal(labels,labels.T) or np.any(np.diag(labels)):
                        raise AssertionError('Invalid categorical topology in trajectory')
                    np.testing.assert_array_equal((labels>0).sum(1),d,err_msg='Intermediate indexed degrees changed')
        if degree_trace is not None:
            saved=np.asarray(degree_trace['indexed_degrees'][i])
            if saved.shape!=(len(manifest['sampling_timesteps']),len(g)) or saved.dtype.kind not in 'iu':
                raise AssertionError('Invalid indexed-degree trace shape/dtype')
            if np.any(saved<0) or np.any(saved>=len(g)):
                raise AssertionError('Impossible degree value in trajectory')
            np.testing.assert_array_equal(saved[0],(e0>0).sum(1))
            np.testing.assert_array_equal(saved[-1],(e>0).sum(1))
            if fixed_degrees:
                np.testing.assert_array_equal(saved,np.broadcast_to(d,saved.shape),
                                              err_msg='Saved intermediate indexed degrees changed')
            if trajectories is not None:
                if [step['t'] for step in trajectories[i]]!=manifest['sampling_timesteps'][1:]:
                    raise AssertionError('Full graph trajectory timesteps disagree')
                for j,step in enumerate(trajectories[i],1):
                    np.testing.assert_array_equal(saved[j],(np.asarray(step['edge_categories'])>0).sum(1))
        if fixed_basis:
            donor_u=np.asarray(sampled_bases[i]);record=basis_records[i]
            if donor_u.shape!=(len(g),len(g)) or not np.isfinite(donor_u).all():
                raise AssertionError('Invalid sampled donor basis')
            np.testing.assert_allclose(donor_u.T@donor_u,np.eye(len(g)),atol=2e-5)
            if basis_digest(donor_u)!=record['basis_sha256']:
                raise AssertionError('Sampled basis and provenance digest disagree')
            if record['sample_index']!=i or record['num_nodes']!=len(g) or record['fallback']:
                raise AssertionError('Invalid donor provenance sample/size/fallback')
            if record['source']!='checkpoint_training_bank' or not record['fixed_for_entire_trajectory']:
                raise AssertionError('Decoder did not use a fixed training basis')
            if record['checkpoint_sha256']!=manifest['checkpoint']['sha256']:
                raise AssertionError('Donor checkpoint provenance disagrees')
            if record.get('training_split_sha256')!=manifest['dataset'].get('split_sha256',{}).get('train'):
                raise AssertionError('Donor training split provenance disagrees')
            donor_index=int(record['bank_index_within_size'])
            if checkpoint_bank is not None:
                choices=checkpoint_bank.get(len(g),[])
                if not 0<=donor_index<len(choices):
                    raise AssertionError('Donor index is outside checkpoint bank')
                np.testing.assert_array_equal(donor_u,choices[donor_index],
                                              err_msg='Sampled decoder basis differs from checkpoint training bank')
            # Check the actual final decoder proposal, not the final graph
            # eigenvectors (which generally differ after degree projection).
            prediction=predictions[i]
            if not prediction.get('fixed_training_basis'):
                raise AssertionError('Prediction does not carry the fixed-basis contract')
            coefficients=np.asarray(prediction['spectrum'])*np.sqrt(len(g))
            expected=(donor_u*coefficients[None])@donor_u.T
            expected=.5*(expected+expected.T);np.fill_diagonal(expected,0.)
            actual_scores=np.asarray(prediction['spectral_scores'])
            np.testing.assert_allclose(actual_scores,expected,atol=2e-5,rtol=2e-5,
                                       err_msg='Spectral proposal was not built from the saved training eigenbasis')
            max_proposal_error=max(max_proposal_error,float(np.max(np.abs(expected-actual_scores))))
        prior_match+=int(np.array_equal(np.sort((e>0).sum(1)),np.sort(d)))
        initial_match+=int(np.array_equal((e>0).sum(1),(e0>0).sum(1)))
        connected+=int(nx.is_connected(g))
    diag=json.loads((root/'rewiring_diagnostics.json').read_text())['aggregate']
    if spectral_mode and diag['categorical_degree_change_steps_mean']!=0:
        raise AssertionError('Bond sampling is recorded as changing node degrees')
    if fixed_degrees and diag['spectral_degree_change_steps_mean']!=0:
        raise AssertionError('Degree-preserving spectral decoder is recorded as changing degrees')
    if fixed_basis:
        if diag.get('basis_updates_per_graph')!=1 or diag.get('decoder_basis_updates_after_initialization')!=0:
            raise AssertionError('Fixed decoder basis has recorded updates')
        if diag.get('sampled_basis_fallbacks')!=0:
            raise AssertionError('Unexpected training-basis fallback')
    if manifest.get('decode',{}).get('connectivity_guarantee',False) and connected!=n:
        raise AssertionError('A connectivity-constrained decoder returned a disconnected graph')
    final_acceptance=manifest.get('final_sample_acceptance',{})
    if final_acceptance.get('require_connected',False) and connected!=n:
        raise AssertionError('Final connected-sample acceptance is enabled but a returned graph is disconnected')
    if int(manifest.get('rejected_final_graphs',0))!=int(diag.get('rejected_disconnected_final_graphs',0)):
        raise AssertionError('Manifest/refinement diagnostics disagree on final connectivity rejections')
    return {'status':'passed','num_graphs':n,'artifact_hashes_verified':True,
            'node_counts_preserved':True,'final_local_node_types_preserved':True,
            'final_local_indexed_degrees_and_typed_degrees_preserved':True,
            'prior_degree_preservation_rate':prior_match/n,'initial_degree_preservation_rate':initial_match/n,
            'connectedness_rate':connected/n,'raw_final_connectedness_rate':diag.get('raw_final_connectedness_rate',connected/n),
            'generation_yield':diag.get('generation_yield',1.0),
            'rejected_disconnected_final_graphs':diag.get('rejected_disconnected_final_graphs',0),
            'max_eigenpair_reconstruction_error':max_error,
            'max_eigenvector_orthogonality_error':max_orthogonal_error,
            'recorded_basis_updates_per_graph':diag['basis_updates_per_graph'],
            'recorded_categorical_degree_change_steps_mean':diag['categorical_degree_change_steps_mean'],
            'degree_changes_are_allowed_not_required':not fixed_degrees,
            'spectral_topology_mode':spectral_mode,
            'bond_support_verified_against_final_pre_rewire_graph':spectral_mode,
            'indexed_degree_guaranteed':fixed_degrees,
            'prior_indexed_degree_preservation_rate':indexed_match/n,
            'saved_intermediate_degrees_verified':fixed_degrees and (trajectories is not None or degree_trace is not None),
            'saved_intermediate_degree_vectors_verified':fixed_degrees and degree_trace is not None,
            'saved_full_intermediate_graphs_verified':fixed_degrees and trajectories is not None,
            'fixed_training_eigenbasis':fixed_basis,
            'sampled_basis_matches_checkpoint':sampled_basis_matches_checkpoint,
            'sampled_basis_checkpoint_check':('verified' if sampled_basis_matches_checkpoint else
                                             'checkpoint_unavailable' if fixed_basis else 'not_applicable'),
            'final_spectral_proposal_matches_saved_training_basis':fixed_basis,
            'max_fixed_basis_proposal_error':max_proposal_error if fixed_basis else None,
            'recorded_decoder_basis_updates_after_initialization':diag.get('decoder_basis_updates_after_initialization')}


def rbf_mmd2(a,b,sigma=1.,block=256):
    """Biased nonnegative RBF MMD^2, unit-Euclidean histograms, blocked pair memory."""
    a=np.asarray(a,np.float64);b=np.asarray(b,np.float64)
    if not len(a) or not len(b) or not np.isfinite(sigma) or sigma<=0:raise ValueError('Empty samples or invalid sigma')
    def average(x,y):
        total=0.
        for i in range(0,len(x),block):
            for j in range(0,len(y),block):
                total+=np.exp(-cdist(x[i:i+block],y[j:j+block],metric='sqeuclidean')/(2*sigma*sigma)).sum()
        return total/(len(x)*len(y))
    return float(max(0.,average(a,a)+average(b,b)-2*average(a,b)))


def evaluate(generated_dir,reference_graphs,*,sigma=1.,max_reference=None,seed=42):
    root=Path(generated_dir);schema=json.loads((root/'categorical_schema.json').read_text())
    if schema.get('graphlet_schema_version')==2:
        from .multiscale_evaluation import evaluate_multi
        return evaluate_multi(generated_dir,reference_graphs,sigma=sigma,max_reference=max_reference,seed=seed)
    vocab=GraphCategoryVocabulary.from_dict(schema['category_vocabulary'])
    generated=load_pickle(root/'base_graphs.pkl');reference=load_pickle(reference_graphs)
    if not isinstance(reference,(list,tuple)) or not reference:raise ValueError('Reference must be a nonempty graph list')
    if max_reference is not None:
        if max_reference<1:raise ValueError('max_reference must be positive')
        if max_reference<len(reference):
            idx=np.random.default_rng(seed).choice(len(reference),max_reference,replace=False)
            reference=[reference[i] for i in idx]
    all_rows=[];keys=set()
    for graphs in (generated,reference):
        rows=[]
        for g in graphs:
            x,e=encode_graph(g,vocab,max(1,len(g)));counts=typed_counts(x,e);keys.update(counts)
            rows.append((x,e,counts))
        all_rows.append(rows)
    # External evaluation distinguishes all observed typed classes; it does NOT
    # modify the checkpoint's training-only vocabulary or collapse unseen types.
    basis=TypedGraphlets3(sorted(keys));train_basis=TypedGraphlets3(schema['graphlet_keys'])
    features=[];overflow=[]
    for rows in all_rows:
        f={k:[] for k in ('node_category_histogram','edge_category_histogram','typed_graphlet3_joint_mass','connected_triple_mass')}
        ov=[]
        for x,e,counts in rows:
            nh=np.bincount(x,minlength=vocab.num_node_categories).astype(float)/len(x)
            ij=np.triu_indices(len(x),1);eh=np.bincount(e[ij],minlength=vocab.num_edge_categories).astype(float)/max(len(ij[0]),1)
            h,m=basis.encode_counts(counts,len(x))
            f['node_category_histogram'].append(nh);f['edge_category_histogram'].append(eh)
            # Unconditional mass per triple plus disconnected mass. Avoids
            # arbitrary normalized histograms when no connected triple exists.
            f['typed_graphlet3_joint_mass'].append(np.r_[h[:-1]*m,1-m])
            f['connected_triple_mass'].append([m])
            ov.append(float(train_basis.encode_counts(counts,len(x))[0][-1]))
        features.append(f);overflow.append(float(np.mean(ov)))
    return {'num_generated':len(generated),'num_reference':len(reference),'reference_path':str(Path(reference_graphs).resolve()),
            'reference_sha256':_sha256(Path(reference_graphs)),'reference_subsample_seed':seed if max_reference is not None else None,
            'protocol':'biased_RBF_MMD_squared_Euclidean','sigma':sigma,
            'not_standard_graphRNN_ORCA_or_NSPDK':True,'raw_graphs_no_sanitization_or_filter':True,
            'external_typed_graphlet3_class_count':len(keys),
            'mmd2':{k:rbf_mmd2(features[0][k],features[1][k],sigma) for k in features[0]},
            'mean_conditional_overflow_against_training_vocabulary':{'generated':overflow[0],'reference':overflow[1]}}
