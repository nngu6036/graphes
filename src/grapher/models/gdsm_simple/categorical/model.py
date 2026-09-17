"""Joint spectral-rank transformer and permutation-equivariant categorical graph net.

Node indices are never confused with spectral ranks. The categorical edge state
is the sole adjacency; spectral scores are decoder features, never a competing
threshold graph. Only the actual current binary topology supplies eigenvectors.
"""
from __future__ import annotations
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
                 dropout=0., clustering_bins=100, spectral_conditioning=True, max_nodes=38):
        super().__init__()
        if hidden_dim < 4 or hidden_dim % num_heads or num_layers < 1:
            raise ValueError("Invalid hidden dimension, head count, or layer count")
        self.node_classes,self.edge_classes=node_classes,edge_classes
        self.hidden_dim,self.max_nodes=hidden_dim,max_nodes
        self.spectral_conditioning=bool(spectral_conditioning)
        self.time=mlp(hidden_dim,hidden_dim,hidden_dim)
        self.node_input=nn.Linear(node_classes+2,hidden_dim)
        self.edge_input=nn.Linear(edge_classes,hidden_dim)
        self.pre=GraphLayer(hidden_dim)
        self.spec_input=nn.Linear(4,hidden_dim) # z_t, anchor, spectral rank, graph size
        self.graph_to_spectral=nn.Linear(hidden_dim,hidden_dim)
        layer=nn.TransformerEncoderLayer(hidden_dim,num_heads,ff_dim,dropout,batch_first=True,activation="gelu")
        self.spectral=nn.TransformerEncoder(layer,num_layers,enable_nested_tensor=False)
        self.spec_out=nn.Sequential(nn.LayerNorm(hidden_dim),nn.Linear(hidden_dim,1))
        self.spectral_to_graph=nn.Linear(hidden_dim,hidden_dim)
        self.proposal_input=mlp(1,hidden_dim,hidden_dim)
        self.layers=nn.ModuleList(GraphLayer(hidden_dim) for _ in range(num_layers))
        self.node_head=nn.Linear(hidden_dim,node_classes)
        self.edge_head=mlp(hidden_dim+2*node_classes,hidden_dim,edge_classes)
        self.summary=mlp(3*hidden_dim+node_classes+edge_classes,hidden_dim,hidden_dim)
        self.graphlet_head=nn.Linear(hidden_dim,graphlet_classes)
        self.mass_head=nn.Linear(hidden_dim,1)
        self.clustering_head=nn.Linear(hidden_dim,clustering_bins)
        self.orbit_head=nn.Linear(hidden_dim,4)

    def forward(self,x,e,z,t,anchor,mask,diffusion_steps,*,current_pairs=None):
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
        edge_logits=self.edge_head(torch.cat((pair,pi+pj,(pi-pj).abs()),-1))
        edge_logits=.5*(edge_logits+edge_logits.transpose(1,2))
        edge_logits=edge_logits*active[...,None]
        pe=edge_logits.softmax(-1)*active[...,None]
        pooled=self.summary(torch.cat((mean_nodes(h,mask),mean_pairs(pair,active),summary_spectral,
                                       mean_nodes(px,mask),mean_pairs(pe,active)),-1))
        return {"clean_spectrum":spectrum,"node_logits":node_logits,"edge_logits":edge_logits,
                "graphlet_logits":self.graphlet_head(pooled),"graphlet_mass":self.mass_head(pooled).sigmoid().squeeze(-1),
                "clustering_logits":self.clustering_head(pooled),"orbit_log_mean":F.softplus(self.orbit_head(pooled)),
                "spectral_scores":scores}


def losses(pred,target,weights):
    mask=target['mask']; active=pair_mask(mask)
    upper=active & torch.ones_like(active).triu(1)
    def masked_ce(logits,labels,valid):
        if not valid.any(): return logits.sum()*0
        return F.cross_entropy(logits[valid],labels[valid])
    hist=target['histogram']; logh=pred['graphlet_logits'].log_softmax(-1)
    has_mass=target['mass']>0
    graphlet=(F.kl_div(logh[has_mass],hist[has_mass],reduction='batchmean')
              if has_mass.any() else logh.sum()*0)
    logc=pred['clustering_logits'].log_softmax(-1)
    parts={
        'spectral':((pred['clean_spectrum']-target['z']).square()*mask).sum()/mask.sum(),
        'node':masked_ce(pred['node_logits'],target['x'],mask),
        'edge':masked_ce(pred['edge_logits'],target['e'],upper),
        'graphlet':graphlet,
        'mass':F.mse_loss(pred['graphlet_mass'],target['mass']),
        'clustering':F.kl_div(logc,target['clustering'],reduction='batchmean')+F.mse_loss(logc.exp().cumsum(-1),target['clustering'].cumsum(-1)),
        'orbit':F.mse_loss(pred['orbit_log_mean'],target['orbit']),
    }
    total=sum(parts[k]*float(weights[k]) for k in parts)
    return total,parts


def predictions_numpy(pred,index,n):
    return {
        'spectrum':pred['clean_spectrum'][index,:n].detach().cpu().numpy(),
        'node_probs':pred['node_logits'][index,:n].softmax(-1).detach().cpu().numpy(),
        'edge_probs':pred['edge_logits'][index,:n,:n].softmax(-1).detach().cpu().numpy(),
        'histogram':pred['graphlet_logits'][index].softmax(-1).detach().cpu().numpy(),
        'mass':float(pred['graphlet_mass'][index]),
        'clustering':pred['clustering_logits'][index].softmax(-1).detach().cpu().numpy(),
        'orbit':pred['orbit_log_mean'][index].detach().cpu().numpy(),
    }
