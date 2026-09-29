"""Joint spectral-rank transformer and permutation-equivariant categorical graph net.

Node indices are never confused with spectral ranks. The categorical edge state
is the adjacency in legacy mode. In bond-only mode the external degree-exact
topology owns support; edge zero is an INPUT sentinel, never an output category.
Only the actual current binary topology supplies eigenvectors.
"""
from __future__ import annotations
import inspect
import torch
from torch import nn
from torch.nn import functional as F
from grapher.models.gdsm_simple.model import sinusoidal_time_embedding
from .noise import pair_mask
from .spectral import eigenpairs, spectral_proposal


def mlp(input_dim, hidden, output_dim):
    return nn.Sequential(nn.Linear(input_dim,hidden), nn.SiLU(), nn.Linear(hidden,output_dim))


def mean_nodes(value, mask):
    return (value*mask[...,None]).sum(1)/mask.sum(1,keepdim=True).clamp_min(1)


def mean_pairs(value, pairs):
    return (value*pairs[...,None]).sum((1,2))/pairs.sum((1,2))[:,None].clamp_min(1)


class GraphLayer(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.edge = mlp(hidden*3,hidden,hidden)
        self.message = mlp(hidden*2,hidden,hidden)
        self.node = mlp(hidden*2,hidden,hidden)
        self.norm_e, self.norm_v = nn.LayerNorm(hidden), nn.LayerNorm(hidden)

    def forward(self,h,pair,mask):
        active=pair_mask(mask)
        hi,hj=h[:,:,None,:],h[:,None,:,:]
        p=self.norm_e(pair+self.edge(torch.cat((pair,hi+hj,(hi-hj).abs()),-1)))
        p=p*active[...,None]
        messages=self.message(torch.cat((p,hj.expand_as(p)),-1))*active[...,None]
        pooled=messages.sum(2)/active.sum(2)[...,None].clamp_min(1)
        h=self.norm_v(h+self.node(torch.cat((h,pooled),-1)))*mask[...,None]
        return h,p


class SpectralCategoricalDenoiser(nn.Module):
    def __init__(self, *, node_classes, edge_classes, graphlet_classes,
                 hidden_dim=128, num_layers=3, num_heads=4, ff_dim=256,
                 dropout=0., clustering_bins=100, spectral_conditioning=True, max_nodes=38,
                 graphlet_block_sizes=None, graphlet_orders=None, graphlet_size_weights=None, bond_only=False):
        super().__init__()
        if hidden_dim < 4 or hidden_dim % num_heads or num_layers < 1:
            raise ValueError("Invalid hidden dimension, head count, or layer count")
        self.node_classes,self.edge_classes=node_classes,edge_classes
        self.bond_only=bool(bond_only)
        self.output_edge_classes=edge_classes-int(self.bond_only)
        if self.output_edge_classes<1:
            raise ValueError('At least one real edge category is required')
        self.hidden_dim,self.max_nodes=hidden_dim,max_nodes
        self.spectral_conditioning=bool(spectral_conditioning)
        self.time=mlp(hidden_dim,hidden_dim,hidden_dim)
        self.node_input=nn.Linear(node_classes+2,hidden_dim)
        self.edge_input=nn.Linear(edge_classes,hidden_dim)
        self.pre=GraphLayer(hidden_dim)
        self.spec_input=nn.Linear(4,hidden_dim) # z_t, anchor, spectral rank, graph size
        self.graph_to_spectral=nn.Linear(hidden_dim,hidden_dim)
        layer=nn.TransformerEncoderLayer(hidden_dim,num_heads,ff_dim,dropout,batch_first=True,activation="gelu")
        # Older baseline environments predate the nested-tensor option.
        encoder_options = {}
        if "enable_nested_tensor" in inspect.signature(nn.TransformerEncoder).parameters:
            encoder_options["enable_nested_tensor"] = False
        self.spectral=nn.TransformerEncoder(layer,num_layers,**encoder_options)
        self.spec_out=nn.Sequential(nn.LayerNorm(hidden_dim),nn.Linear(hidden_dim,1))
        self.spectral_to_graph=nn.Linear(hidden_dim,hidden_dim)
        self.proposal_input=mlp(1,hidden_dim,hidden_dim)
        self.layers=nn.ModuleList(GraphLayer(hidden_dim) for _ in range(num_layers))
        self.node_head=nn.Linear(hidden_dim,node_classes)
        # Unattributed graphs have one deterministic real edge type: no classifier.
        self.edge_head=(None if self.bond_only and self.output_edge_classes==1 else
                        mlp(hidden_dim+2*node_classes,hidden_dim,self.output_edge_classes))
        self.summary=mlp(3*hidden_dim+node_classes+self.output_edge_classes,hidden_dim,hidden_dim)
        self.graphlet_head=nn.Linear(hidden_dim,graphlet_classes)
        self.graphlet_block_sizes=tuple(graphlet_block_sizes or ())
        self.graphlet_orders=tuple(graphlet_orders or ())
        self.graphlet_size_weights=tuple(graphlet_size_weights or ())
        if self.graphlet_block_sizes:
            if sum(self.graphlet_block_sizes)!=graphlet_classes or len(self.graphlet_orders)!=len(self.graphlet_block_sizes) or len(self.graphlet_size_weights)!=len(self.graphlet_orders):
                raise ValueError('Inconsistent multiscale graphlet head dimensions')
        self.mass_head=nn.Linear(hidden_dim,len(self.graphlet_orders) or 1)
        self.clustering_head=nn.Linear(hidden_dim,clustering_bins)
        self.orbit_head=nn.Linear(hidden_dim,4)

    def _bond_logits(self,pair,px,query):
        """Only selected unordered pairs reach the real-bond classifier."""
        upper=torch.nonzero(torch.triu(query,diagonal=1),as_tuple=True)
        edge_logits=pair.new_zeros((*query.shape,self.output_edge_classes))
        if self.edge_head is not None and len(upper[0]):
            b,i,j=upper
            features=torch.cat((.5*(pair[b,i,j]+pair[b,j,i]),
                                px[b,i]+px[b,j],(px[b,i]-px[b,j]).abs()),-1)
            logits=self.edge_head(features)
            edge_logits[b,i,j]=logits;edge_logits[b,j,i]=logits
        return edge_logits

    def query_bonds(self,pred,support):
        """Reuse encoded spectral/graph features after the topology is selected.

        This does not rerun the spectral transformer or change predicted
        structural targets. Output probabilities exist only on `support`.
        """
        if not self.bond_only or '_bond_features' not in pred:
            raise ValueError('Bond queries require a bond-only model prediction')
        pair,px,active=pred['_bond_features']
        if support.dtype!=torch.bool or support.shape!=active.shape or not torch.equal(support,support.transpose(1,2)) or (support & ~active).any():
            raise ValueError('Bond query must be valid symmetric boolean support')
        out=dict(pred)
        out['edge_logits']=self._bond_logits(pair,px,support)
        out['edge_prediction_mask']=support
        return out

    def forward(self,x,e,z,t,anchor,mask,diffusion_steps,*,current_pairs=None,edge_query_mask=None):
        if mask.ndim!=2 or z.shape!=mask.shape or x.shape!=mask.shape or anchor.shape!=mask.shape:
            raise ValueError("Expected node, mask, spectrum and anchor shapes [B,N]")
        if e.shape!=(len(mask),mask.shape[1],mask.shape[1]) or not torch.equal(e,e.transpose(1,2)):
            raise ValueError("Categorical edges must be symmetric [B,N,N]")
        n=mask.shape[1]; sizes=mask.sum(1)
        if not (sizes>0).all() or n>self.max_nodes:
            raise ValueError("Empty or oversized graph batch")
        active=pair_mask(mask)
        one_x=F.one_hot(x,self.node_classes).float()*mask[...,None]
        one_e=F.one_hot(e,self.edge_classes).float()*active[...,None]
        time=self.time(sinusoidal_time_embedding(t.float()/diffusion_steps,self.hidden_dim))
        degree=((e>0)&active).sum(2).float()/((sizes-1).clamp_min(1)[:,None])
        size=sizes.float()[:,None].expand(-1,n)/self.max_nodes
        h=self.node_input(torch.cat((one_x,degree[...,None],size[...,None]),-1))+time[:,None,:]
        h=h*mask[...,None]; pair=self.edge_input(one_e)*active[...,None]
        h,pair=self.pre(h,pair,mask)
        graph_context=mean_nodes(h,mask)
        rank=torch.arange(n,device=z.device,dtype=z.dtype)[None].expand_as(z)/((sizes-1).clamp_min(1)[:,None])
        sh=self.spec_input(torch.stack((z,anchor,rank,size),-1))+time[:,None,:]+self.graph_to_spectral(graph_context)[:,None,:]
        sh=self.spectral(sh,src_key_padding_mask=~mask)
        spectrum=self.spec_out(sh).squeeze(-1)*mask
        # Adjacency spectra have zero trace. No prior degree/moment is imposed.
        spectrum=(spectrum-spectrum.sum(1,keepdim=True)/sizes[:,None])*mask
        spectrum=spectrum.masked_fill(~mask,float('inf')).sort(-1).values.masked_fill(~mask,0)
        bound=(sizes-1).float()/sizes.float().sqrt()
        # Uniform rescaling preserves sorting and zero trace (individual clipping would not).
        scale=(bound/spectrum.abs().amax(1).clamp_min(1e-12)).clamp(max=1.)
        spectrum=spectrum*scale[:,None]*mask
        spectral_context=mean_nodes(sh,mask)
        if current_pairs is None:
            current_pairs=eigenpairs(e,mask)
        values,vectors=current_pairs
        scores=spectral_proposal(spectrum,values,vectors,mask)
        if self.spectral_conditioning:
            pair=pair+self.proposal_input(scores[...,None])*active[...,None]
            h=h+self.spectral_to_graph(spectral_context)[:,None,:]*mask[...,None]
            summary_spectral=spectral_context
        else:
            # True categorical-only decoder ablation, including summary heads.
            summary_spectral=torch.zeros_like(spectral_context)
        for layer in self.layers:
            h,pair=layer(h,pair,mask)
        node_logits=self.node_head(h)*mask[...,None]
        px=node_logits.softmax(-1)*mask[...,None]
        pi,pj=px[:,:,None,:],px[:,None,:,:]
        if self.bond_only:
            # Query masks select output rows AFTER graph/spectral encoding. In
            # training, querying clean positive pairs cannot leak the target
            # adjacency into the spectral or structural feature computation.
            support=(e>0)&active
            query=support
            if edge_query_mask is not None:
                if edge_query_mask.dtype!=torch.bool or edge_query_mask.shape!=e.shape or not torch.equal(edge_query_mask,edge_query_mask.transpose(1,2)) or (edge_query_mask & ~active).any():
                    raise ValueError('edge_query_mask must be valid symmetric boolean support')
                query=query|edge_query_mask
            edge_logits=self._bond_logits(pair,px,query)
            pe=edge_logits.softmax(-1)*support[...,None]
            edge_pool=mean_pairs(pe,support)
        else:
            if edge_query_mask is not None:
                raise ValueError('edge_query_mask is only supported by the bond-only head')
            edge_logits=self.edge_head(torch.cat((pair,pi+pj,(pi-pj).abs()),-1))
            edge_logits=.5*(edge_logits+edge_logits.transpose(1,2))
            edge_logits=edge_logits*active[...,None]
            pe=edge_logits.softmax(-1)*active[...,None]
            edge_pool=mean_pairs(pe,active)
        pooled=self.summary(torch.cat((mean_nodes(h,mask),mean_pairs(pair,active),summary_spectral,
                                       mean_nodes(px,mask),edge_pool),-1))
        result={"clean_spectrum":spectrum,"node_logits":node_logits,"edge_logits":edge_logits,
                "graphlet_logits":self.graphlet_head(pooled),"graphlet_mass":self.mass_head(pooled).sigmoid(),
                "clustering_logits":self.clustering_head(pooled),"orbit_log_mean":F.softplus(self.orbit_head(pooled)),
                "spectral_scores":scores}
        if self.bond_only:
            result['bond_only']=True
            result['edge_prediction_mask']=query
            result['_bond_features']=(pair,px,active)
        if self.graphlet_block_sizes:
            result['graphlet_block_sizes']=self.graphlet_block_sizes
            result['graphlet_orders']=self.graphlet_orders
            result['graphlet_size_weights']=self.graphlet_size_weights
        else:
            result['graphlet_mass']=result['graphlet_mass'].squeeze(-1)
        return result


def losses(pred,target,weights):
    mask=target['mask']; active=pair_mask(mask)
    upper=active & torch.ones_like(active).triu(1)
    def masked_ce(logits,labels,valid):
        if not valid.any(): return logits.sum()*0
        return F.cross_entropy(logits[valid],labels[valid])
    hist=target['histogram'];order_parts={}
    if pred.get('graphlet_block_sizes'):
        weights_k=pred['graphlet_size_weights'];blocks=pred['graphlet_block_sizes']
        graphlet=pred['graphlet_logits'].sum()*0;mass_loss=pred['graphlet_mass'].sum()*0
        offset=0
        for i,(width,w) in enumerate(zip(blocks,weights_k)):
            logh=pred['graphlet_logits'][:,offset:offset+width].log_softmax(-1)
            available=target['graphlet_order_mask'][:,i]
            has_mass=(target['mass'][:,i]>0)&available
            gh=(F.kl_div(logh[has_mass],hist[has_mass,offset:offset+width],reduction='batchmean')
                if has_mass.any() else logh.sum()*0)
            gm=(F.mse_loss(pred['graphlet_mass'][available,i],target['mass'][available,i])
                if available.any() else pred['graphlet_mass'][:,i].sum()*0)
            graphlet=graphlet+w*gh;mass_loss=mass_loss+w*gm
            order=pred['graphlet_orders'][i]
            order_parts[f'graphlet_{order}']=gh;order_parts[f'mass_{order}']=gm
            offset+=width
    else:
        logh=pred['graphlet_logits'].log_softmax(-1)
        has_mass=target['mass']>0
        graphlet=(F.kl_div(logh[has_mass],hist[has_mass],reduction='batchmean')
                  if has_mass.any() else logh.sum()*0)
        mass_loss=F.mse_loss(pred['graphlet_mass'],target['mass'])
    logc=pred['clustering_logits'].log_softmax(-1)
    parts={
        'spectral':((pred['clean_spectrum']-target['z']).square()*mask).sum()/mask.sum(),
        'node':masked_ce(pred['node_logits'],target['x'],mask),
        'edge':masked_ce(pred['edge_logits'],(target['e']-1).clamp_min(0),
                         upper & (target['e']>0) & pred['edge_prediction_mask'])
               if pred.get('bond_only',False) else masked_ce(pred['edge_logits'],target['e'],upper),
        'graphlet':graphlet,
        'mass':mass_loss,
        'clustering':F.kl_div(logc,target['clustering'],reduction='batchmean')+F.mse_loss(logc.exp().cumsum(-1),target['clustering'].cumsum(-1)),
        'orbit':F.mse_loss(pred['orbit_log_mean'],target['orbit']),
    }
    total=sum(parts[k]*float(weights[k]) for k in parts)
    return total,{**parts,**order_parts}


def predictions_numpy(pred,index,n):
    multi=bool(pred.get('graphlet_block_sizes'))
    if multi:
        blocks=pred['graphlet_logits'][index].split(pred['graphlet_block_sizes'])
        histogram=torch.cat([h.softmax(-1) for h in blocks]).detach().cpu().numpy()
        mass=pred['graphlet_mass'][index].detach().cpu().numpy()
    else:
        histogram=pred['graphlet_logits'][index].softmax(-1).detach().cpu().numpy()
        mass=float(pred['graphlet_mass'][index])
    result={
        'spectrum':pred['clean_spectrum'][index,:n].detach().cpu().numpy(),
        'node_probs':pred['node_logits'][index,:n].softmax(-1).detach().cpu().numpy(),
        'edge_probs':pred['edge_logits'][index,:n,:n].softmax(-1).detach().cpu().numpy(),
        'histogram':histogram,
        'mass':mass,
        'clustering':pred['clustering_logits'][index].softmax(-1).detach().cpu().numpy(),
        'orbit':pred['orbit_log_mean'][index].detach().cpu().numpy(),
    }

    if pred.get('bond_only',False):
        # K real categories; do not silently prepend a learned no-edge channel.
        result['bond_only']=True
        result['bond_probs']=result.pop('edge_probs')
        result['bond_prediction_mask']=pred['edge_prediction_mask'][index,:n,:n].detach().cpu().numpy()
        result['bond_probs']*=result['bond_prediction_mask'][...,None]
    if multi:
        result['graphlet_orders']=list(pred['graphlet_orders'])
        result['graphlet_block_sizes']=list(pred['graphlet_block_sizes'])
    return result
