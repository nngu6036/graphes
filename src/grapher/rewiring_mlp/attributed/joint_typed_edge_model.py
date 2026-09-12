"""Joint typed-DH-VAE and permutation-equivariant soft-bond endpoint predictor."""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any
import torch
from torch import nn
import torch.nn.functional as F

from grapher.models.dhvae_hh.typed_degree_vae import TypedSignatureHistogramVAE, TypedSignatureVectorizer
from grapher.rewiring_mlp.generic.joint_degree_model import degree_consistent_orbits, orbit_identity_residual
from grapher.rewiring_mlp.attributed.soft_edge_bridge import (
    pair_mask, center_edges, edge_probabilities, labels_to_logits, bridge_edges, bridge_spectra, project_spectra,
)
from grapher.utils.device import resolve_torch_device

from grapher.rewiring_mlp.generic.induced_graphlets import (
    InducedGraphletSpec, prediction_and_loss as induced_prediction_loss, mask_prediction,
)
from grapher.rewiring_mlp.attributed.data import GraphletBasis
from grapher.rewiring_mlp.attributed.induced_graphlets import (
    prediction_and_loss as attributed_induced_prediction_loss,
    mask_prediction as attributed_mask_prediction,
    metadata as attributed_graphlet_metadata,
)

from .adjacency_diffusion import (
    LEGACY_MODE, ADJACENCY_MODE, validate_views, validate_weights, NORMALIZATIONS,
    spectral_features as adjacency_features, features_from_logits, view_names,
    state_metadata, validate_model_config,
)

FORMAT='joint_typed_soft_edge_grapher_v1'
ADJACENCY_FORMAT='joint_typed_adjacency_grapher_v1' 


def mlp(in_dim, hidden, out_dim):
    return nn.Sequential(nn.Linear(in_dim,hidden),nn.SiLU(),nn.Linear(hidden,out_dim))


class DenseTypedEdgeLayer(nn.Module):
    """Symmetric pair updates and equivariant all-neighbor message passing."""
    def __init__(self, dim):
        super().__init__()
        self.pair=mlp(3*dim,2*dim,dim)
        self.msg=mlp(dim,dim,dim); self.neighbor=nn.Linear(dim,dim)
        self.node=mlp(2*dim,2*dim,dim)
        self.pair_norm=nn.LayerNorm(dim); self.node_norm=nn.LayerNorm(dim)

    def forward(self,h,e,mask):
        left=h[:,:,None,:]; right=h[:,None,:,:]
        e=self.pair_norm(e+self.pair(torch.cat([e,left+right,(left-right).abs()],-1)))
        e=e*pair_mask(mask)[...,None]
        msg=(self.msg(e)*self.neighbor(h)[:,None,:,:])*pair_mask(mask)[...,None]
        pooled=msg.sum(2)/(mask.sum(1)-1).clamp_min(1)[:,None,None]
        h=self.node_norm(h+self.node(torch.cat([h,pooled],-1)))*mask[...,None]
        return h,e


class JointTypedEdgePredictor(nn.Module):
    def __init__(self, *, typed_model_config: dict, vectorizer: dict,
                 atom_types: list[int], hidden_dim=128, num_layers=4,
                 spectral_layers=2, spectral_heads=4, spectral_enabled=True,
                 histogram_bins=100, orbit_enabled=True, smoothing=0.01,
                 induced_graphlet_k=None, induced_graphlet_scope="all",
                 induced_graphlet_catalogue_fingerprint=None,
                 induced_graphlet_basis=None,
                 spectral_mode=LEGACY_MODE, adjacency_views=None,
                 adjacency_bond_weights=None, adjacency_normalization="size_bound",
                 adjacency_output_spectra=True):
        super().__init__()
        self.model_config=deepcopy(dict(typed_model_config=typed_model_config,vectorizer=vectorizer,
            atom_types=list(atom_types),hidden_dim=int(hidden_dim),num_layers=int(num_layers),
            spectral_layers=int(spectral_layers),spectral_heads=int(spectral_heads),
            spectral_enabled=bool(spectral_enabled),histogram_bins=int(histogram_bins),
            orbit_enabled=bool(orbit_enabled),smoothing=float(smoothing)))
        if hidden_dim<4 or num_layers<1 or not 0<smoothing<1:
            raise ValueError('Invalid model dimensions/smoothing.')
        if histogram_bins not in (0,) and histogram_bins<2: raise ValueError('Histogram bins must be 0 or >=2.')
        self.degree_model=TypedSignatureHistogramVAE(**typed_model_config)
        self.vectorizer=TypedSignatureVectorizer.from_dict(vectorizer)
        self.atom_types=tuple(atom_types); self.edge_types=self.vectorizer.vocabulary.edge_types
        self.categories=len(self.edge_types)+1; d=int(hidden_dim)
        v=self.vectorizer; dm=self.degree_model
        expected=torch.tensor([s.edge_degrees for s in v.vocabulary.signatures],dtype=torch.float32)
        if dm.input_dim!=v.input_dim or not torch.equal(dm.signature_incidence.cpu(),expected):
            raise ValueError('Typed model/vocabulary mismatch; ordering is part of the checkpoint.')
        if (dm.min_nodes,dm.max_nodes)!=(v.min_nodes,v.max_nodes): raise ValueError('Size support mismatch.')
        self.smoothing=float(smoothing); self.histogram_bins=int(histogram_bins)
        self.orbit_enabled=bool(orbit_enabled); self.spectral_enabled=bool(spectral_enabled)
        if spectral_mode not in (LEGACY_MODE, ADJACENCY_MODE):
            raise ValueError(f"Unknown spectral_mode: {spectral_mode!r}.")
        self.spectral_mode = spectral_mode
        self.adjacency_output_spectra = bool(adjacency_output_spectra) if spectral_mode == ADJACENCY_MODE else False
        if spectral_mode == ADJACENCY_MODE:
            self.adjacency_views = validate_views(adjacency_views or ["topology", "bond_weighted"])
            self.adjacency_bond_weights = validate_weights(adjacency_bond_weights or [])
            if len(self.adjacency_bond_weights) != len(self.edge_types):
                raise ValueError("Adjacency bond weights must follow the typed vocabulary's edge ordering.")
            if adjacency_normalization not in NORMALIZATIONS:
                raise ValueError("Invalid adjacency normalization.")
            self.adjacency_normalization = adjacency_normalization
            self.adjacency_channels = len(view_names(self.adjacency_views, self.edge_types))
            self.model_config.update(spectral_mode=spectral_mode,
                adjacency_views=list(self.adjacency_views),
                adjacency_bond_weights=list(self.adjacency_bond_weights),
                adjacency_normalization=self.adjacency_normalization,
                adjacency_output_spectra=self.adjacency_output_spectra)
        self.typed_condition=mlp(2*dm.hidden_dim+dm.latent_dim+v.signature_dim+len(self.edge_types)+1,d,d)
        self.atom_embed=nn.Embedding(len(atom_types),d)
        self.degree_embed=nn.Linear(len(self.edge_types),d)
        self.time_embed=mlp(3,d,d)
        self.edge_embed=mlp(2*self.categories,d,d)
        self.layers=nn.ModuleList([DenseTypedEdgeLayer(d) for _ in range(num_layers)])
        if self.spectral_enabled:
            if d % spectral_heads: raise ValueError('hidden_dim must divide spectral_heads.')
            spectral_inputs = 2*self.adjacency_channels + 2 if self.spectral_mode == ADJACENCY_MODE else 6
            self.spec_embed=mlp(spectral_inputs,d,d)
            layer=nn.TransformerEncoderLayer(d,spectral_heads,4*d,dropout=0.0,
                                             batch_first=True,activation='gelu')
            self.spec_encoder=nn.TransformerEncoder(layer,spectral_layers,enable_nested_tensor=False)
            if self.spectral_mode == LEGACY_MODE:
                self.spectrum_head=mlp(2*d,d,2)
        self.edge_head=mlp(3*d,2*d,self.categories)
        if self.histogram_bins: self.histogram_head=mlp(d,d,self.histogram_bins)
        if self.orbit_enabled: self.orbit_head=mlp(d,d,15)
        self.induced_graphlet_basis = (
            None if induced_graphlet_basis is None else GraphletBasis.from_dict(induced_graphlet_basis)
        )
        self.induced_graphlet_spec = (
            None
            if (self.induced_graphlet_basis is not None or induced_graphlet_k is None)
            else InducedGraphletSpec(induced_graphlet_k, induced_graphlet_scope)
        )
        if self.induced_graphlet_basis is not None:
            fingerprint = attributed_graphlet_metadata(self.induced_graphlet_basis)["fingerprint"]
            if induced_graphlet_catalogue_fingerprint not in (None, fingerprint):
                raise ValueError("Induced graphlet checkpoint catalogue fingerprint mismatch.")
            self.induced_graphlet_head = mlp(d, d, self.induced_graphlet_basis.width)
            self.model_config.update(
                induced_graphlet_basis=self.induced_graphlet_basis.to_dict(),
                induced_graphlet_catalogue_fingerprint=fingerprint,
            )
        elif self.induced_graphlet_spec is not None:
            fingerprint = self.induced_graphlet_spec.metadata()["fingerprint"]
            if induced_graphlet_catalogue_fingerprint not in (None, fingerprint):
                raise ValueError("Induced graphlet checkpoint catalogue fingerprint mismatch.")
            self.induced_graphlet_head = mlp(d, d, self.induced_graphlet_spec.width)
            self.model_config.update(induced_graphlet_k=self.induced_graphlet_spec.k,
                induced_graphlet_scope=self.induced_graphlet_spec.scope,
                induced_graphlet_catalogue_fingerprint=fingerprint)
        self._degree_frozen=False

    def adjacency_features(self, probabilities, mask):
        if self.spectral_mode != ADJACENCY_MODE:
            raise ValueError("Adjacency features requested from a Laplacian model.")
        return adjacency_features(probabilities, mask, self.adjacency_bond_weights,
                                  self.adjacency_views, self.adjacency_normalization)

    def diffusion_metadata(self):
        return state_metadata(self)

    def induced_graphlet_metadata(self):
        if self.induced_graphlet_basis is not None:
            return attributed_graphlet_metadata(self.induced_graphlet_basis)
        if self.induced_graphlet_spec is not None:
            return self.induced_graphlet_spec.metadata()
        return None

    def set_degree_trainable(self,enabled:bool):
        self._degree_frozen=not enabled
        self.degree_model.requires_grad_(enabled)
        self.degree_model.train(self.training and enabled)

    def train(self,mode=True):
        super().train(mode)
        if getattr(self,'_degree_frozen',False): self.degree_model.eval()
        return self

    def degree_condition(self,batch):
        dm=self.degree_model
        hidden=dm.encoder(batch['typed_features']); mu=dm.mu(hidden)
        size=dm.size_encoder(dm._size_features(batch['n']))
        # Share decoder hidden layers AND decoded probabilities. Structure losses
        # reach the signature-output layer as well as its feature extractor.
        decoder_hidden=dm.signature_decoder.net[:-1](torch.cat([mu,size],-1))
        decoded=dm.decode(mu,batch['n'])
        p=decoded['signature_logits'].softmax(-1)
        features=torch.cat([hidden,mu,decoder_hidden,p,decoded['expected_incidence'],
                            batch['n'].float()[:,None]/max(dm.max_nodes,1)],-1)
        return self.typed_condition(features)

    def forward(self,batch):
        mask=batch['mask']; B,N=mask.shape
        cond=self.degree_condition(batch)
        t=batch['time']; time=self.time_embed(torch.stack([t,torch.sin(torch.pi*t),torch.cos(torch.pi*t)],-1))
        global_cond=cond+time
        tokens=None
        input_adjacency = None
        if self.spectral_enabled:
            rank=torch.arange(N,device=mask.device)[None,:]/(batch['n'][:,None]-1).clamp_min(1)
            if self.spectral_mode == ADJACENCY_MODE:
                # Recompute from the actual categorical state. Never accept an
                # independently supplied spectral_state as an adjacency view.
                input_adjacency = self.adjacency_features(edge_probabilities(batch['edge_state'], mask), mask)
                source_logits = labels_to_logits(batch['source_labels'], self.categories, mask, self.smoothing)
                source_adjacency = self.adjacency_features(edge_probabilities(source_logits, mask), mask)
                state_spectrum = input_adjacency['normalized']
                source_spectrum = source_adjacency['normalized']
                spectrum_mask = input_adjacency['mask']
            else:
                state_spectrum = batch['spectral_state']
                source_spectrum = batch['source_spectra']/batch['spectral_scale'][...,None]
                spectrum_mask = mask
            spec_in=torch.cat([state_spectrum.transpose(1,2), source_spectrum.transpose(1,2),
               rank[...,None].expand(B,N,1), t[:,None,None].expand(B,N,1)],-1)
            tokens=self.spec_encoder(self.spec_embed(spec_in)+global_cond[:,None,:],src_key_padding_mask=~spectrum_mask)
            # Spectral ranks are not node identities: only pooled spectral
            # information is broadcast to the permutation-equivariant node path.
            global_cond=global_cond+(tokens*spectrum_mask[...,None]).sum(1)/batch['n'][:,None]
        h=self.atom_embed(batch['atom'])+self.degree_embed(batch['typed_degrees']/max(self.vectorizer.max_nodes-1,1))
        h=(h+global_cond[:,None,:])*mask[...,None]
        source_logits=labels_to_logits(batch['source_labels'],self.categories,mask,self.smoothing)
        e=self.edge_embed(torch.cat([batch['edge_state'],source_logits],-1))*pair_mask(mask)[...,None]
        for layer in self.layers: h,e=layer(h,e,mask)
        left=h[:,:,None,:]; right=h[:,None,:,:]
        logits=center_edges(self.edge_head(torch.cat([e,left+right,(left-right).abs()],-1)),mask)
        pooled=(h*mask[...,None]).sum(1)/batch['n'][:,None]
        out={'clean_edge_logits':logits,'clean_edge_probabilities':edge_probabilities(logits,mask)}
        if self.spectral_enabled and self.spectral_mode == LEGACY_MODE:
            raw=self.spectrum_head(torch.cat([tokens,pooled[:,None,:].expand(B,N,-1)],-1)).transpose(1,2)
            out['clean_spectra']=project_spectra(raw,batch['spectral_trace'],mask)
        if input_adjacency is not None:
            out['input_adjacency_spectra'] = input_adjacency['spectra']
        if self.spectral_mode == ADJACENCY_MODE and self.adjacency_output_spectra:
            # A derived output, not a second trainable spectral head.
            clean_views = self.adjacency_features(out['clean_edge_probabilities'], mask)
            out['clean_adjacency_spectra'] = clean_views['spectra']
        if self.induced_graphlet_basis is not None or self.induced_graphlet_spec is not None:
            graphlet_logits = self.induced_graphlet_head(pooled)
            out['clean_induced_graphlet_histogram_logits'] = graphlet_logits
            if self.induced_graphlet_basis is not None:
                out['clean_induced_graphlet_histogram'] = attributed_mask_prediction(
                    graphlet_logits.softmax(-1), batch['n'], self.induced_graphlet_basis)
            else:
                out['clean_induced_graphlet_histogram'] = mask_prediction(
                    graphlet_logits.softmax(-1), batch['n'], self.induced_graphlet_spec)
        if self.histogram_bins: out['clean_clustering_histogram']=self.histogram_head(pooled).softmax(-1)
        if self.orbit_enabled:
            raw=F.softplus(self.orbit_head(pooled)).clamp_max(15).expm1()
            out['unconstrained_orbit_summary']=raw
            out['clean_orbit_summary']=degree_consistent_orbits(raw,batch['orbit_totals'])
        return out


def noisy_batch(batch,model,config,*,generator=None,endpoint_only=False):
    out=dict(batch); B=len(batch['n']); dev=batch['n'].device
    diff=config['edge_diffusion']
    validate_model_config(model, config)
    if endpoint_only: t=torch.zeros(B,device=dev)
    else:
        t=torch.rand(B,device=dev,generator=generator)
        # Explicit source and clean endpoints avoid leaving t=0 or t=1 untrained.
        modes=torch.rand(B,device=dev,generator=generator)
        f=float(diff.get('endpoint_fraction',0.1))
        t=torch.where(modes<f/2,torch.zeros_like(t),torch.where(modes<f,torch.ones_like(t),t))
    s=labels_to_logits(batch['source_labels'],model.categories,batch['mask'],model.smoothing)
    target=labels_to_logits(batch['target_labels'],model.categories,batch['mask'],model.smoothing)
    out['edge_state']=bridge_edges(s,target,t,batch['mask'],float(diff.get('sigma',1)),generator=generator)
    if model.spectral_enabled and model.spectral_mode == LEGACY_MODE:
        scale=batch['spectral_scale'][...,None]
        out['spectral_state']=bridge_spectra(batch['source_spectra']/scale,batch['target_spectra']/scale,
            t,batch['mask'],float(diff.get('spectral_sigma',0.15)),generator=generator)
    elif model.spectral_mode == ADJACENCY_MODE:
        out.pop('spectral_state', None)  # No separate stochastic spectral state.
    out['time']=t
    return out


def mean_per_graph(values,mask):
    return (values*mask).sum(tuple(range(1,values.ndim)))/mask.sum(tuple(range(1,mask.ndim))).clamp_min(1)


def structural_loss(outputs,batch,model,weights):
    """Every scalar is first normalized per graph, then averaged across graphs.

    CE includes all unordered pairs, including no-bond; a separate Gaussian
    endpoint regression term trains the logits used by the continuous bridge.
    """
    pm=pair_mask(batch['mask'],upper=True); logits=outputs['clean_edge_logits']
    target_logits=labels_to_logits(batch['target_labels'],model.categories,batch['mask'],model.smoothing)
    ce=F.cross_entropy(logits.permute(0,3,1,2),batch['target_labels'],reduction='none')
    edge_ce=mean_per_graph(ce,pm).mean()
    edge_mse=mean_per_graph((logits-target_logits).square().mean(-1),pm).mean()
    p=outputs['clean_edge_probabilities']*pair_mask(batch['mask'])[...,None]
    expected=p[...,1:].sum(2)
    incidence=((expected-batch['typed_degrees'])/batch['n'][:,None,None].clamp_min(1)).square().mean(-1)
    typed_loss=mean_per_graph(incidence,batch['mask']).mean()
    losses={'edge_ce':edge_ce,'edge_logit':edge_mse,'typed_consistency':typed_loss}
    metrics={'edge_accuracy':mean_per_graph((logits.argmax(-1)==batch['target_labels']).float(),pm).mean(),
             'edge_bond_accuracy':mean_per_graph((logits.argmax(-1)==batch['target_labels']).float(),
                                                pm & (batch['target_labels']>0)).mean(),
             'soft_typed_degree_rmse':mean_per_graph((expected-batch['typed_degrees']).square().mean(-1),batch['mask']).sqrt().mean()}
    if model.spectral_enabled and model.spectral_mode == LEGACY_MODE:
        delta=(outputs['clean_spectra']-batch['target_spectra'])/batch['spectral_scale'][...,None]
        losses['spectrum']=mean_per_graph(F.smooth_l1_loss(delta,torch.zeros_like(delta),reduction='none').mean(1),batch['mask']).mean()
        metrics['spectral_nrmse']=mean_per_graph(delta.square().mean(1),batch['mask']).sqrt().mean()
    if model.spectral_mode == ADJACENCY_MODE:
        if float(weights.get('spectrum', 0)) != 0:
            raise ValueError("The Laplacian spectrum loss is inactive; use adjacency_spectrum.")
        if float(weights.get('adjacency_spectrum', 0)) > 0 and not model.adjacency_output_spectra:
            raise ValueError("Enable adjacency spectral outputs before using the spectral loss.")
        if model.adjacency_output_spectra:
            # Reference has precisely the same smoothing and category semantics
            # as the clean edge-logit target, not the hard graph's Laplacian.
            target = model.adjacency_features(edge_probabilities(target_logits, batch['mask']), batch['mask'])
            predicted = outputs.get('clean_adjacency_spectra')
            if predicted is None:
                predicted = model.adjacency_features(outputs['clean_edge_probabilities'], batch['mask'])['spectra']
            delta=(predicted-target['spectra'])/target['scale'][...,None]
            sm=target['mask']
            losses['adjacency_spectrum']=mean_per_graph(
                F.smooth_l1_loss(delta,torch.zeros_like(delta),reduction='none').mean(1),sm).mean()
            metrics['adjacency_spectral_nrmse']=mean_per_graph(delta.square().mean(1),sm).sqrt().mean()
            metrics['adjacency_spectral_trace_max_abs']=predicted.sum(-1).abs().max()
            names=view_names(model.adjacency_views, model.edge_types)
            for c,name in enumerate(names):
                metrics['adjacency_'+name+'_nrmse']=mean_per_graph(delta[:,c].square(),sm).sqrt().mean()
    if model.histogram_bins:
        cdf=(outputs['clean_clustering_histogram']-batch['histogram']).cumsum(-1)[:,:-1]
        losses['clustering_histogram']=cdf.square().sum(-1).mean()/model.histogram_bins
        metrics['clustering_histogram_w1']=cdf.abs().sum(-1).mean()/model.histogram_bins
    if model.orbit_enabled:
        log_delta=outputs['clean_orbit_summary'].log1p()-batch['orbit'].log1p()
        losses['orbit_summary']=F.smooth_l1_loss(log_delta,torch.zeros_like(log_delta))
        metrics['orbit_summary_log_rmse']=log_delta.square().mean(-1).sqrt().mean()
        metrics['orbit_identity_max_abs']=orbit_identity_residual(outputs['clean_orbit_summary'],batch['orbit_totals']).max()
    if model.induced_graphlet_basis is not None or model.induced_graphlet_spec is not None:
        if 'induced_histogram' not in batch:
            raise ValueError('Induced graphlet supervision enabled but target is missing.')
        logits = outputs.get('clean_induced_graphlet_histogram_logits')
        if logits is None:
            logits = outputs['clean_induced_graphlet_histogram'].clamp_min(1e-30).log()
        if model.induced_graphlet_basis is not None:
            brier, ce, measured = attributed_induced_prediction_loss(
                logits, batch['induced_histogram'], batch['n'], model.induced_graphlet_basis)
        else:
            brier, ce, measured = induced_prediction_loss(logits, batch['induced_histogram'], batch['n'], model.induced_graphlet_spec)
        losses['induced_graphlet_histogram'] = brier
        losses['induced_graphlet_histogram_ce'] = ce
        metrics.update(measured)
    total=sum(float(weights.get(k,0))*value for k,value in losses.items())
    metrics.update({k+'_loss':v for k,v in losses.items()}); metrics['structure_loss']=total
    return total,{k:float(v.detach().cpu()) for k,v in metrics.items()}


def save_checkpoint(path,model,config,metrics,**metadata):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    checkpoint_format = ADJACENCY_FORMAT if model.spectral_mode == ADJACENCY_MODE else FORMAT
    payload={'format':checkpoint_format,'model_config':model.model_config,'model_state_dict':model.state_dict(),
             'config':deepcopy(config),'metrics':metrics,**metadata}
    temp=path.with_name(path.name+'.tmp'); torch.save(payload,temp); temp.replace(path)


def load_checkpoint(path,device='cpu'):
    dev=resolve_torch_device(device)
    # Payloads contain tensors + primitive metadata; no Python model pickles.
    ckpt=torch.load(Path(path),map_location=dev,weights_only=True)
    if ckpt.get('format') not in (FORMAT, ADJACENCY_FORMAT):
        raise ValueError('Not a joint typed soft-edge/adjacency checkpoint.')
    mode=ckpt['model_config'].get('spectral_mode',LEGACY_MODE)
    expected=ADJACENCY_FORMAT if mode == ADJACENCY_MODE else FORMAT
    if ckpt['format'] != expected:
        raise ValueError('Checkpoint format and spectral semantics disagree.')
    model=JointTypedEdgePredictor(**ckpt['model_config']).to(dev)
    model.load_state_dict(ckpt['model_state_dict'],strict=True); model.eval()
    return model,ckpt
