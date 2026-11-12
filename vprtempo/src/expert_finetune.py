"""MIT licensed relation fine-tuning of existing linear VPRTempo expert features."""
import hashlib
import json
import random

import torch
from torch import nn
from torch.nn import functional as F


class ExpertFeature(nn.Module):
    """Keep each original linear map and concatenate before a single L2 norm."""
    def __init__(self, weights):
        super().__init__()
        self.weights=nn.ParameterList([nn.Parameter(w.detach().clone()) for w in weights])
        for i,w in enumerate(weights):
            self.register_buffer(f'signs_{i}',w.sign().to(torch.int8).clone())

    def raw(self,x):return torch.cat([F.linear(x,w) for w in self.weights],dim=1)

    def encode(self,x):
        h=self.raw(x);norm=h.norm(dim=1,keepdim=True)
        return h/norm.clamp_min(1e-8),h,norm[:,0]>1e-8

    def mask_gradient(self,gradients):
        return [g*(getattr(self,f'signs_{i}')!=0) for i,g in enumerate(gradients)]

    @torch.no_grad()
    def project_signs(self):
        count=0
        for i,w in enumerate(self.weights):
            signs=getattr(self,f'signs_{i}')
            count+=int(((signs>0)&(w<0)).sum()+((signs<0)&(w>0)).sum())
            w.copy_(torch.where(signs>0,w.clamp_min(0),torch.where(signs<0,w.clamp_max(0),torch.zeros_like(w))))
        return count


def tensor_digest(state):
    h=hashlib.sha256()
    for name,value in sorted(state.items()):
        value=value.detach().cpu().contiguous()
        h.update(json.dumps([name,str(value.dtype),list(value.shape)]).encode());h.update(value.numpy().tobytes())
    return h.hexdigest()


def gradients(model,a,p,n,margin,method):
    """Exact full or anchor-stop-gradient derivative of the same cosine hinge."""
    if method not in ('full','anchor'):raise ValueError('unknown update method')
    if method=='full':
        da,_,va=model.encode(a);dp,_,vp=model.encode(p);dn,_,vn=model.encode(n)
        valid=va&vp&vn;count=int(valid.sum())
        if count:
            gap=(da*dp).sum(1)-(da*dn).sum(1)
            loss=F.relu(margin-gap[valid]).mean()
            result=torch.autograd.grad(loss,tuple(model.weights))
            active=int((gap[valid]<margin).sum())
        else:
            result=[torch.zeros_like(w) for w in model.weights];loss=a.new_tensor(0.);active=0
    else:
        with torch.no_grad():
            da,ha,va=model.encode(a);dp,_,vp=model.encode(p);dn,_,vn=model.encode(n)
            valid=va&vp&vn;count=int(valid.sum())
            gap=(da*dp).sum(1)-(da*dn).sum(1)
            active_mask=valid&(gap<margin);active=int(active_mask.sum())
            loss=F.relu(margin-gap[valid]).mean() if count else a.new_tensor(0.)
            direction=dn-dp
            feedback=(direction-(direction*da).sum(1,keepdim=True)*da)/ha.norm(dim=1,keepdim=True).clamp_min(1e-8)
            feedback*=active_mask[:,None]
            all_grad=(feedback.T@a)/max(count,1)
            result=list(all_grad.split([w.shape[0] for w in model.weights],dim=0))
    result=model.mask_gradient(result)
    if not torch.isfinite(loss) or any(not torch.isfinite(g).all() for g in result):
        raise FloatingPointError('nonfinite loss/gradient; no update applied')
    return result,{'loss':float(loss.detach()),'valid':count,'active':active}


@torch.no_grad()
def weight_norm(model):return float(torch.sqrt(sum(w.square().sum() for w in model.weights)))


@torch.no_grad()
def relative_drift(model,initial):
    return float(torch.sqrt(sum((w-v).square().sum() for w,v in zip(model.weights,initial))))/max(weight_norm_initial(initial),1e-30)


def weight_norm_initial(weights):return float(torch.sqrt(sum(w.square().sum() for w in weights)))


@torch.no_grad()
def apply_gradients(model,gs,lr,max_relative_step):
    current=weight_norm(model);gnorm=float(torch.sqrt(sum(g.square().sum() for g in gs)))
    scale=min(1.,max_relative_step*current/max(lr*gnorm,1e-30))
    before=[w.clone() for w in model.weights]
    for w,g in zip(model.weights,gs):w.add_(g,alpha=-lr*scale)
    projected=model.project_signs()
    if any(not torch.isfinite(w).all() for w in model.weights):
        for w,b in zip(model.weights,before):w.copy_(b)
        raise FloatingPointError('nonfinite update restored previous state')
    delta=float(torch.sqrt(sum((w-b).square().sum() for w,b in zip(model.weights,before))))
    return {'gradient_norm':gnorm,'step_scale':scale,'projected_connections':projected,
            'actual_relative_step':delta/max(current,1e-30)}


def make_schedule(initial,epochs,seed,gap=50):
    """Uniform / initial-feature hard negatives; only spring/fall are accepted."""
    if set(initial)!={'spring','fall'}:raise ValueError('schedule accepts training seasons only')
    n=len(initial['spring']);rng=random.Random(seed);rows=[];hard=[];candidates=[]
    if n<2*gap:raise ValueError('map too small for negative exclusion')
    for epoch in range(epochs):
        anchors=[(s,p) for s in (0,1) for p in range(n)];rng.shuffle(anchors)
        for i,(s,p) in enumerate(anchors):
            left=max(0,p-gap+1);right=min(n,p+gap);legal=left+n-right
            def place(rank):return rank if rank<left else right+rank-left
            if i%2==0:rows.append([s,p,place(rng.randrange(legal))])
            else:
                hard.append(len(rows));rows.append([s,p,-1])
                candidates.append(sorted(place(r) for r in rng.sample(range(legal),min(16,legal))))
    plan=torch.tensor(rows,dtype=torch.long)
    device=initial['spring'].device
    # Batch only training candidates; float reduction and lower-ID tie policy are fixed.
    for s in (0,1):
        selected=[i for i,idx in enumerate(hard) if rows[idx][0]==s]
        own=initial['spring' if s==0 else 'fall'];other=initial['fall' if s==0 else 'spring']
        for start in range(0,len(selected),128):
            batch=selected[start:start+128];positions=[hard[i] for i in batch]
            aa=torch.tensor([rows[p][1] for p in positions],device=device)
            cc=torch.tensor([candidates[i] for i in batch],device=device)
            scores=(other[cc]*own[aa,None,:]).sum(-1)
            chosen=cc.gather(1,scores.argmax(1,keepdim=True))[:,0].cpu()
            plan[positions,2]=chosen
    plan=plan.reshape(epochs,2*n,3)
    identity={'seed':seed,'epochs':epochs,'gap':gap,'policy':'half_uniform_half_hard16_initial_train_features',
              'shape':list(plan.shape),'tensor_sha256':hashlib.sha256(plan.numpy().tobytes()).hexdigest()}
    return plan,identity


def triplet_batch(train,rows,device):
    # train is [2,N,input_dim], and excludes every query season.
    s,p,n=rows.unbind(1)
    return (train[s,p].to(device),train[1-s,p].to(device),train[1-s,n].to(device))


def calibrate(model,train,plan,batch_size,device,cfg):
    """Choose common Full/Anchor settings using training batches only, without updates."""
    batches=[triplet_batch(train,r,device) for r in plan[0][:batch_size*cfg['calibration_batches']].split(batch_size)]
    before=tensor_digest(model.state_dict());profiles=[];chosen=None
    for margin in cfg['margin_candidates']:
        valid=active=0
        with torch.no_grad():
            for a,p,n in batches:
                da,_,va=model.encode(a);dp,_,vp=model.encode(p);dn,_,vn=model.encode(n)
                vv=va&vp&vn;valid+=int(vv.sum())
                active+=int((vv&(((da*dp).sum(1)-(da*dn).sum(1))<margin)).sum())
        profiles.append({'margin':margin,'valid':valid,'active':active,'active_fraction':active/max(valid,1)})
        if valid and active/valid>=cfg['min_calibration_active_fraction']:
            chosen=margin;break
    if chosen is None:raise RuntimeError('no active training objective at registered margin candidates')
    norms=[];wnorm=weight_norm(model)
    for method in ('full','anchor'):
        for a,p,n in batches:
            gs,diag=gradients(model,a,p,n,chosen,method)
            gnorm=float(torch.sqrt(sum(g.square().sum() for g in gs)))
            norms.append({'method':method,**diag,'gradient_norm':gnorm,'unit_lr_relative_step':gnorm/max(wnorm,1e-30)})
    peak=max(v['unit_lr_relative_step'] for v in norms)
    lr=min(cfg['max_lr'],cfg['max_relative_step']/max(peak,1e-30))
    if lr<cfg['min_lr']:raise RuntimeError('required safe learning rate below registered floor')
    if tensor_digest(model.state_dict())!=before:raise RuntimeError('calibration mutated the initial state')
    return {'margin':chosen,'lr':lr,'profiles':profiles,'gradient_profiles':norms,
            'initial_weight_norm':wnorm,'uses_query_data':False,'both_methods_share_recipe':True}
