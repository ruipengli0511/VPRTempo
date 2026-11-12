"""MIT licensed fixed-checkpoint Full/Anchor fine-tuning and locked evaluation."""
import argparse
import json
from pathlib import Path
import statistics
import sys
import time

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(Path(__file__).resolve().parent))
sys.dont_write_bytecode=True
from vprtempo.src.run_provenance import code_identity,environment_identity,resolve_locations,runtime_identity
from vprtempo.src.benchmark2700_common import sha_file,write_json
from vprtempo.src.expert_finetune import (ExpertFeature,tensor_digest,make_schedule,triplet_batch,
                                         calibrate,gradients,apply_gradients,relative_drift,probe_diagnostics)
from eval_attribution2700 import official_weights
from eval_locked2700 import evaluate


def read(p):return json.loads(Path(p).read_bytes())
def json_sha(x):
    import hashlib
    return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def checked_json(path,expected):
    if sha_file(path)!=expected:raise ValueError('source JSON changed: '+str(path))
    return read(path)


def load_source(item,device):
    cp=Path(item['checkpoint'])
    if sha_file(cp)!=item['checkpoint_sha256']:raise ValueError('original weights changed')
    state=torch.load(cp,map_location='cpu',weights_only=True)
    pairs=official_weights(state)
    model=ExpertFeature([w for w,o in pairs]).to(device)
    # No projection, renormalization or clipping is applied when loading the source.
    for w,(expected,_) in zip(model.weights,pairs):
        if not torch.equal(w.cpu(),expected):raise ValueError('source initialization changed')
    return model


def load_inputs(pointer,development):
    path=Path(pointer['pixel_cache'])
    if sha_file(path)!=pointer['pixel_cache_sha256']:raise ValueError('pixel input cache changed')
    pixels=torch.load(path,map_location='cpu',weights_only=True)['pixels']
    if development:
        pixels={'spring':pixels['spring'],'fall':pixels['fall'],'summer':pixels['summer'][:500].clone()}
    expected=500 if development else 2700
    if any(pixels[s].shape!=(3300,784) for s in ('spring','fall')) or pixels['summer'].shape!=(expected,784):
        raise ValueError('input shape mismatch')
    return pixels


@torch.no_grad()
def extract(model,pixels,device,normalized=False):
    result={}
    for season,values in pixels.items():
        parts=[]
        for x in values.split(256):
            if normalized:parts.append(model.encode(x.to(device))[0])
            else:parts.append(model.raw(x.to(device)).cpu())
        result[season]=torch.cat(parts)
    return result


@torch.no_grad()
def score(model,pixels,ids,device):
    before=tensor_digest(model.state_dict())
    result=evaluate(None,extract(model,pixels,device),ids)
    if tensor_digest(model.state_dict())!=before:raise RuntimeError('evaluation mutated weights')
    return result


def baseline_parity(result,item,development):
    summary=checked_json(item['feature_summary'],item['feature_summary_sha256'])
    previous=checked_json(item['feature_predictions'],item['feature_predictions_sha256'])
    if development:previous=previous[:500]
    current=result['predictions']
    if len(current)!=len(previous):raise ValueError('initial comparison count mismatch')
    largest=0.
    for a,b in zip(current,previous):
        if a['query_place_id']!=b['query_place_id'] or [v['place'] for v in a['place_top5']]!=[v['place'] for v in b['place_top5']]:
            raise ValueError('initial feature ranking differs from B05')
        largest=max(largest,max(abs(x['score']-y['score']) for x,y in zip(a['place_top5'],b['place_top5'])))
    if largest>1e-6:raise ValueError('initial feature scores differ from B05')
    subset='development' if development else 'overall'
    for k in (1,5):
        if abs(result['subsets'][subset][f'R@{k}']-summary['subsets'][subset][f'R@{k}'])>1e-12:
            raise ValueError('initial metric differs from B05')
    return {'queries':len(current),'top5_exact_match':True,'max_abs_score_diff':largest,'B05_summary_sha256':item['feature_summary_sha256']}


def save_checkpoint(out,model,epoch,metadata):
    p=out/f'checkpoint_epoch_{epoch}.pth'
    torch.save({'model':{k:v.detach().cpu().clone() for k,v in model.state_dict().items()},
                'metadata':{**metadata,'epoch':epoch}},p)
    return {'epoch':epoch,'path':str(p),'sha256':sha_file(p)}


def resolve_probe(cfg,source,plan,seed):
    count=cfg.get('probe_triplets',0)
    if source.get('fixed_probe'):
        ref=source['fixed_probe'];saved=checked_json(ref['path'],ref['sha256'])
        rows=torch.tensor(saved['samples'],dtype=torch.long)
        if tuple(rows.shape)!=(count,3):raise ValueError('common probe shape mismatch')
        if (not ((rows[:,0]==0)|(rows[:,0]==1)).all() or (rows[:,1:]<0).any()
                or (rows[:,1:]>=cfg['n_places']).any()
                or ((rows[:,1]-rows[:,2]).abs()<cfg['negative_min_gap']).any()):
            raise ValueError('common probe contains invalid training identities')
        digest=tensor_digest({'triplets':rows})
        if digest!=saved['tensor_sha256'] or digest!=ref['tensor_sha256']:
            raise ValueError('common probe tensor changed')
        identity={**saved,'source':'B07 immutable common spring/fall probe','evaluated_seed':seed,
                  'source_identity_sha256':ref['sha256']}
        return rows,identity
    rows=plan[0][:count].clone()
    import hashlib
    return rows,{'seed':seed,'count':len(rows),'source':'first fixed epoch training triplets',
        'plan_sha256':hashlib.sha256(plan.cpu().numpy().tobytes()).hexdigest(),'tensor_sha256':tensor_digest({'triplets':rows}),
        'fields':['anchor_season_0spring_1fall','place','negative_place'],'samples':rows.tolist()}


def train(cfg,source,pointer,out,provenance,method,seed):
    item=next(v for v in source['models'] if v['seed']==seed)
    device=torch.device(cfg['device']);pixels=load_inputs(pointer,True)
    model=load_source(item,device);initial=[w.detach().clone() for w in model.weights]
    initial_hash=tensor_digest(model.state_dict())
    pre=score(model,pixels,list(range(600,1100)),device)
    parity=baseline_parity(pre,item,True);write_json(out/'initial_parity.json',parity)
    initial_features=extract(model,{k:pixels[k] for k in ('spring','fall')},device,normalized=True)
    plan,plan_identity=make_schedule(initial_features,cfg['epochs'],seed,cfg['negative_min_gap'])
    del initial_features
    pairing=source.get('pairing_references',{}).get(str(seed),source.get('pairing_reference'))
    if pairing:
        if seed!=pairing['seed'] or plan_identity!=pairing['schedule_identity'] or initial_hash!=pairing['initial_state_sha256']:
            raise ValueError('initial state or negative schedule differs from B06 reference')
    training=torch.stack((pixels['spring'],pixels['fall']))
    calibration=calibrate(model,training,plan,cfg['batch_size'],device,cfg)
    if 'requested_learning_rate' in cfg and abs(calibration['lr']-cfg['requested_learning_rate'])>1e-12:
        raise RuntimeError('requested learning-rate arm reduced by initial safety check; do not silently relabel')
    if tensor_digest(model.state_dict())!=initial_hash:raise RuntimeError('preparation changed initial state')
    torch.save({'plan':plan,'identity':plan_identity},out/'training_plan.pt')
    write_json(out/'calibration.json',calibration);write_json(out/'schedule_identity.json',plan_identity)
    effective={**cfg,'seed':seed,'method':method,'lr':calibration['lr'],'margin':calibration['margin'],
               'feature_dim':4704,'forward':'three_original_linear_maps_concat_global_L2',
               'sign_projection':'preserve_initial_sign_and_zero_connections; zero floor, no upper bound',
               'source_checkpoint':item,'initial_state_sha256':initial_hash}
    write_json(out/'effective_config.json',{'config':effective,**provenance})
    metadata={'seed':seed,'method':method,'source_checkpoint_sha256':item['checkpoint_sha256'],
              'initial_state_sha256':initial_hash,'code_commit':provenance['code']['commit'],
              'schedule_sha256':plan_identity['tensor_sha256'],'calibration_sha256':json_sha(calibration),
              'config_sha256':json_sha(effective)}
    epoch_results=[];checkpoints=[];history=[];probe_history=[]
    probe_rows,identity=resolve_probe(cfg,source,plan,seed)
    if len(probe_rows):
        write_json(out/'probe_identity.json',identity)
        metadata['probe_sha256']=identity['tensor_sha256']
    def record_probe(epoch,complete=True):
        if not len(probe_rows):return
        diagnostic=probe_diagnostics(model,training,probe_rows,calibration['margin'],method,calibration['lr'],
                                     cfg['max_relative_step'],cfg['batch_size'],device)
        diagnostic.update(epoch=epoch,epoch_complete=complete,probe_sha256=identity['tensor_sha256'])
        write_json(out/f'probe_epoch_{epoch}.json',diagnostic);probe_history.append(diagnostic)
    def record(epoch,result=None,eligible=True):
        r=result or score(model,pixels,list(range(600,1100)),device)
        write_json(out/f'development_epoch_{epoch}_predictions.json',r.pop('predictions'))
        checkpoints.append(save_checkpoint(out,model,epoch,{**metadata,'epoch_complete':eligible}))
        epoch_results.append({'epoch':epoch,'development':r['subsets']['development'],
                              'relative_drift':relative_drift(model,initial),'eligible_for_selection':eligible})
    record_probe(0);record(0,pre);started=time.monotonic();stop_reason='budget_complete'
    log=(out/'updates.jsonl').open('x')
    try:
        for epoch,rows in enumerate(plan,1):
            total={'loss':0.,'valid':0,'active':0,'batches':0,'step_capped':0,'projected_connections':0,'max_relative_step':0.,
                   'sum_unmasked_gradient_norm':0.,'sum_masked_gradient_norm':0.,'sum_pre_projection_step_norm':0.,
                   'sum_actual_step_norm':0.,'sum_projection_change_norm':0.,'sum_projection_retained_step_ratio':0.}
            numerical_stop=False
            for batch,indices in enumerate(rows.split(cfg['batch_size'])):
                a,p,n=triplet_batch(training,indices,device)
                try:
                    gs,diag=gradients(model,a,p,n,calibration['margin'],method)
                    step=apply_gradients(model,gs,calibration['lr'],cfg['max_relative_step'])
                except FloatingPointError as exc:
                    stop_reason='numerical_stop:'+str(exc);numerical_stop=True;break
                total['loss']+=diag['loss']*diag['valid'];total['valid']+=diag['valid'];total['active']+=diag['active']
                total['batches']+=1;total['step_capped']+=int(step['step_scale']<1)
                total['projected_connections']+=step['projected_connections']
                total['max_relative_step']=max(total['max_relative_step'],step['actual_relative_step'])
                for key in ['unmasked_gradient_norm','masked_gradient_norm']:total['sum_'+key]+=diag[key]
                for key in ['pre_projection_step_norm','actual_step_norm','projection_change_norm','projection_retained_step_ratio']:
                    total['sum_'+key]+=step[key]
                if batch%20==0:log.write(json.dumps({'epoch':epoch,'batch':batch,**diag,**step})+'\n');log.flush()
            total['mean_loss']=total.pop('loss')/max(total['valid'],1)
            total['active_fraction']=total['active']/max(total['valid'],1)
            total['relative_drift']=relative_drift(model,initial);total['epoch']=epoch;history.append(total)
            for key in ['unmasked_gradient_norm','masked_gradient_norm','pre_projection_step_norm','actual_step_norm','projection_change_norm','projection_retained_step_ratio']:
                total['mean_'+key]=total['sum_'+key]/max(total['batches'],1)
            print(json.dumps({'epoch':epoch,**total}),flush=True)
            if total['relative_drift']>=cfg['max_relative_drift']:stop_reason='drift_guard'
            record_probe(epoch,complete=not numerical_stop)
            if epoch in cfg['evaluation_epochs'] or numerical_stop or stop_reason=='drift_guard':record(epoch,eligible=not numerical_stop)
            if numerical_stop or stop_reason=='drift_guard':break
    finally:log.close()
    write_json(out/'summary.json',{'status':'complete','phase':'training','method':method,'seed':seed,
       'stop_reason':stop_reason,'epoch_results':epoch_results,'epoch_diagnostics':history,'checkpoints':checkpoints,
       'calibration':calibration,'schedule_identity':plan_identity,'initial_parity':parity,
       'probe_history':probe_history,'probe_identity':identity if len(probe_rows) else None,
       'requested_learning_rate':cfg.get('requested_learning_rate'),
       'training_s':time.monotonic()-started,'source':item,'initial_state_sha256':initial_hash,
       'max_allocated_gpu_bytes':torch.cuda.max_memory_allocated() if device.type=='cuda' else None,
       'retained_queries_used_for_selection':False,**provenance})


def choose_locked(source,training_runs,expected_commit,out):
    """Lock one epoch per method across seeds, using development R@1 only."""
    groups={m:[] for m in ('full','anchor')}
    for item in training_runs:
        root=Path(item['run_dir']);s=read(root/'artifacts/summary.json')
        producer=item.get('expected_commit',expected_commit)
        if s['status']!='complete' or s['phase']!='training' or s['code']['commit']!=producer:
            raise ValueError('incomplete or wrong-version training source')
        bindings={'summary_sha256':root/'artifacts/summary.json',
                  'result_manifest_sha256':root/'result_manifest.json',
                  'effective_config_sha256':root/'artifacts/effective_config.json'}
        if producer!=expected_commit and (not item.get('reuse') or not all(item.get(k) for k in bindings)):
            raise ValueError('cross-version reuse requires frozen summary/config/result-manifest hashes')
        for key,path in bindings.items():
            if item.get(key) and sha_file(path)!=item[key]:raise ValueError('reused evidence changed: '+str(path))
        if s['method']!=item.get('method',s['method']) or s['seed']!=item.get('seed',s['seed']):
            raise ValueError('training role differs from fixed manifest')
        if item.get('learning_rate') is not None and s['calibration']['lr']!=item['learning_rate']:
            raise ValueError('training learning rate differs from fixed manifest')
        pairing=source.get('pairing_references',{}).get(str(s['seed']))
        if pairing and (s['schedule_identity']!=pairing['schedule_identity'] or s['initial_state_sha256']!=pairing['initial_state_sha256']):
            raise ValueError('training initialization/schedule differs from registered seed')
        if source.get('fixed_probe') and (s.get('probe_identity') or {}).get('tensor_sha256')!=source['fixed_probe']['tensor_sha256']:
            raise ValueError('training diagnostic probe differs from registered common probe')
        if read(root/'trial_outcome.json')['status']!='complete':raise ValueError('training runner not complete')
        s['_training_commit']=producer;s['_training_run_id']=root.name
        s['_summary_sha256']=sha_file(root/'artifacts/summary.json')
        groups[s['method']].append(s)
    locks=[];selection=[]
    for method,rows in groups.items():
        rows.sort(key=lambda v:v['seed'])
        if [v['seed'] for v in rows]!=[0,1,2]:raise ValueError('three seeds required for common-epoch locking')
        epochs=set.intersection(*(set(v['epoch'] for v in s['epoch_results'] if v['eligible_for_selection']) for s in rows))
        counts={}
        for e in epochs:
            hits=[]
            for s in rows:
                metric=next(v['development'] for v in s['epoch_results'] if v['epoch']==e)
                n=metric.get('queries',500);h=round(metric['R@1']*n)
                if n!=500 or abs(h/n-metric['R@1'])>1e-9:raise ValueError('development hit count mismatch')
                hits.append(h)
            counts[e]=sum(hits)
        scores={e:v/1500 for e,v in counts.items()}
        epoch=min(epochs,key=lambda e:(-counts[e],e))
        selection.append({'method':method,'epoch':epoch,'development_epoch_scores':scores,'development_total_correct':counts})
        for s in rows:
            cp=next(v for v in s['checkpoints'] if v['epoch']==epoch)
            if sha_file(cp['path'])!=cp['sha256']:raise ValueError('selected weight changed')
            locks.append({'method':method,'seed':s['seed'],'epoch':epoch,'checkpoint':cp,
                          'source':s['source'],'schedule_identity':s['schedule_identity'],
                          'calibration_sha256':json_sha(s['calibration']),'initial_state_sha256':s['initial_state_sha256'],
                          'training_commit':s['_training_commit'],'training_run_id':s['_training_run_id'],
                          'training_summary_sha256':s['_summary_sha256'],
                          'probe_sha256':(s.get('probe_identity') or {}).get('tensor_sha256')})
    for seed in range(3):
        f=next(v for v in locks if v['method']=='full' and v['seed']==seed)
        a=next(v for v in locks if v['method']=='anchor' and v['seed']==seed)
        if any(f[k]!=a[k] for k in ('schedule_identity','initial_state_sha256','calibration_sha256')):
            raise ValueError('Full/Anchor pair did not share preparation')
    probe_hashes=[v['probe_sha256'] for v in locks]
    if any(probe_hashes) and len(set(probe_hashes))!=1:raise ValueError('common probe differs across seeds')
    value={'selection':'one epoch per method, mean development500 across3 seeds, earliest tie; epoch0 allowed',
           'selection_results':selection,'models':locks}
    write_json(out/'locked_models.json',value)
    return value


def aggregate_rows(rows):
    output={}
    for subset in ('overall','development','buffer','retained'):
        result={}
        for k in (1,5):
            values=[v['subsets'][subset][f'R@{k}'] for v in rows]
            result[f'mean_R@{k}']=statistics.mean(values)
            result[f'std_R@{k}']=statistics.stdev(values) if len(values)>1 else None
        output[subset]=result
    return output


def paired_rows(left,right):
    paired=[]
    for row in left:
        ref=next(v for v in right if v['seed']==row['seed'])
        subsets={s:{f'R@{k}':row['subsets'][s][f'R@{k}']-ref['subsets'][s][f'R@{k}'] for k in (1,5)}
                 for s in ('overall','development','buffer','retained')}
        paired.append({'seed':row['seed'],'subsets':subsets})
    summary=aggregate_rows(paired)
    for subset in summary:
        for k in (1,5):
            vals=[v['subsets'][subset][f'R@{k}'] for v in paired]
            summary[subset][f'positive_seeds_R@{k}']=sum(v>1e-12 for v in vals)
            summary[subset][f'negative_seeds_R@{k}']=sum(v<-1e-12 for v in vals)
    return {'per_seed':paired,'summary':summary}


def final_evaluation(cfg,source,pointer,out,provenance,training_runs):
    locked=choose_locked(source,training_runs,provenance['code']['commit'],out)
    # Retained query tensors are passed through the model only after model locking.
    pixels=load_inputs(pointer,False);device=torch.device(cfg['device']);ids=list(range(600,3300))
    initial_parity=[];frozen=[]
    for item in source['models']:
        model=load_source(item,device);r=score(model,pixels,ids,device)
        initial_parity.append({'seed':item['seed'],**baseline_parity(r,item,False)})
        write_json(out/f'frozen_seed{item["seed"]}_predictions.json',r.pop('predictions'))
        frozen.append({'seed':item['seed'],'source_checkpoint_sha256':item['checkpoint_sha256'],**r})
    write_json(out/'frozen_initial_full_parity.json',initial_parity)
    rows=[]
    for lock in locked['models']:
        cp=lock['checkpoint'];payload=torch.load(cp['path'],map_location='cpu',weights_only=True)
        md=payload['metadata']
        if (md['epoch'],md['seed'],md['method'],md['code_commit'])!=(lock['epoch'],lock['seed'],lock['method'],lock['training_commit']):
            raise ValueError('selected checkpoint metadata mismatch')
        if md['source_checkpoint_sha256']!=lock['source']['checkpoint_sha256'] or md['initial_state_sha256']!=lock['initial_state_sha256']:
            raise ValueError('selected checkpoint initialization mismatch')
        if md['schedule_sha256']!=lock['schedule_identity']['tensor_sha256'] or md['calibration_sha256']!=lock['calibration_sha256']:
            raise ValueError('selected checkpoint preparation differs from locked evidence')
        if lock['probe_sha256'] and md.get('probe_sha256')!=lock['probe_sha256']:
            raise ValueError('selected checkpoint probe identity mismatch')
        model=load_source(lock['source'],device);model.load_state_dict(payload['model']);model.requires_grad_(False)
        r=score(model,pixels,ids,device)
        write_json(out/f'{lock["method"]}_seed{lock["seed"]}_predictions.json',r.pop('predictions'))
        row={'method':lock['method'],'seed':lock['seed'],'epoch':lock['epoch'],'checkpoint_sha256':cp['sha256'],
             'training_commit':lock['training_commit'],'training_run_id':lock['training_run_id'],**r}
        rows.append(row)
    old=checked_json(source['reference_b05_comparison'],source['reference_b05_comparison_sha256'])
    aggregate=[]
    for method in ('full','anchor'):
        rr=[v for v in rows if v['method']==method];subsets=aggregate_rows(rr)
        deltas=[]
        for row in rr:
            baseline=next(v for v in old['all_rows'] if v['method']=='original_feature_retrieval' and v['seed']==row['seed'])
            deltas.append({'seed':row['seed'],'overall':row['subsets']['overall']['R@1']-baseline['subsets']['overall']['R@1'],
                           'retained':row['subsets']['retained']['R@1']-baseline['subsets']['retained']['R@1']})
        aggregate.append({'method':method,'subsets':subsets,'paired_delta_from_frozen':deltas,
                          'paired_R1_R5_from_frozen':paired_rows(rr,frozen)})
    write_json(out/'effective_config.json',{'config':cfg,'locked_models_sha256':sha_file(out/'locked_models.json'),**provenance})
    write_json(out/'summary.json',{'status':'complete','phase':'locked_evaluation','results':rows,'aggregate':aggregate,
       'frozen_results':frozen,'frozen_aggregate':aggregate_rows(frozen),
       'full_minus_anchor':paired_rows([v for v in rows if v['method']=='full'],[v for v in rows if v['method']=='anchor']),
       'reference_B05':old['aggregate'],'locked_models_sha256':sha_file(out/'locked_models.json'),
       'scope':'previously observed benchmark; training selection uses only registered development500',**provenance})


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,required=True);p.add_argument('--run-id',required=True)
    p.add_argument('--mode',choices=['train','evaluate'],required=True)
    p.add_argument('--method',choices=['full','anchor']);p.add_argument('--seed',type=int)
    p.add_argument('--learning-rate',type=float,choices=[0.01,0.1])
    p.add_argument('--sources',type=Path,required=True);p.add_argument('--sources-sha',required=True)
    p.add_argument('--training-runs',type=Path);p.add_argument('--training-runs-sha')
    for name in ('data-dir','cache-dir','run-root','deployment-manifest','deployment-manifest-sha','environment-manifest'):
        p.add_argument('--'+name)
    a=p.parse_args();cfg=read(a.config);cfg['run_id']=a.run_id
    if a.learning_rate is not None:
        cfg['max_lr']=a.learning_rate;cfg['requested_learning_rate']=a.learning_rate
    if cfg.get('query_role')=='development_only_no_retained_evaluation':
        if a.mode!='train' or a.seed!=0 or a.learning_rate is None:
            raise ValueError('B07 accepts only registered seed0 training arms, no full-query evaluation')
    if cfg.get('experiment_stage')=='B08':
        if a.mode=='train' and (a.seed not in (1,2) or a.learning_rate!=0.1):
            raise ValueError('B08 trains only seed1/2 at0.1; seed0 must be reused')
    for name in ('data_dir','cache_dir','run_root','deployment_manifest','deployment_manifest_sha','environment_manifest'):
        if getattr(a,name):cfg[name]=getattr(a,name)
    if cfg['n_places']!=3300 or cfg['development_ids']!=[600,1099]:raise ValueError('fixed protocol mismatch')
    code=code_identity(ROOT,cfg.get('deployment_manifest'),cfg.get('deployment_manifest_sha'))
    if code['dirty']:raise RuntimeError('dirty fine-tuning source')
    deployed=code['source']=='verified_deployment_snapshot'
    env=environment_identity(cfg.get('environment_manifest'),ROOT/'pixi.lock',required=deployed)
    provenance={'code':code,'environment':env,'runtime':runtime_identity(torch,np)}
    torch.backends.cuda.matmul.allow_tf32=False
    source=checked_json(a.sources,a.sources_sha)
    pointer=checked_json(source['prepared_inputs'],source['prepared_inputs_sha256'])
    if pointer['identity']['query_ids']!=list(range(600,3300)):raise ValueError('wrong source query identities')
    out,_=resolve_locations(cfg,ROOT,deployed);out.mkdir(parents=True,exist_ok=False)
    write_json(out/'requested_config.json',{'config':cfg,'mode':a.mode,'method':a.method,'seed':a.seed,**provenance})
    if a.mode=='train':
        if a.seed not in (0,1,2) or a.method is None:raise ValueError('training method/seed required')
        train(cfg,source,pointer,out,provenance,a.method,a.seed)
    else:
        runs=checked_json(a.training_runs,a.training_runs_sha)
        final_evaluation(cfg,source,pointer,out,provenance,runs['runs'])


if __name__=='__main__':main()
