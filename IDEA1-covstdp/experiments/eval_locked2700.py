"""MIT licensed evaluation-only adapter. Never selects or updates a model."""
import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
sys.dont_write_bytecode=True
from vprtempo.src.run_provenance import code_identity,environment_identity,resolve_locations,runtime_identity
from vprtempo.src.task_feature import (_datasets,TaskFeature,CONV_STATE,digest_json,
                                       digest_file,load_gabor,descriptors)
from vprtempo.src.task_protocol import cached_tensor
from vprtempo.src.metrics import recallAtK
from vprtempo.src.benchmark2700_common import sha_file,write_json,score_report


def prepare(cfg,cache):
    n=cfg['n_places']; ids=list(range(cfg['query_start'],cfg['query_start']+cfg['query_count']))
    datasets,manifest,csv=_datasets(ROOT,Path(cfg['data_dir']),n,8,ids)
    old=json.loads(Path(cfg['reuse_data_identity']).read_text())
    if old['csv_sha256']!=csv:
        raise ValueError('CSV version differs from trained models')
    old_pixels=torch.load(old['pixel_cache'],map_location='cpu',weights_only=True)['pixels']
    old_records={(r['season'],r['place']):r for r in old['manifest']}
    old_indices={s:{r['place']:i for i,r in enumerate(r for r in old['manifest'] if r['season']==s)}
                 for s in ('spring','fall','summer')}
    identity={'dataset':'Nordland','n_places':n,'query_ids':ids,'query_role':'locked_benchmark',
              'dims':[28,28],'patches':7,'patch_norm':True,'filter':8,
              'csv_sha256':csv,'manifest':manifest,'manifest_sha256':digest_json(manifest),
              'data_dir':cfg['data_dir'],'reused_pixel_source_sha256':sha_file(old['pixel_cache'])}
    key=digest_json({'manifest':identity['manifest_sha256'],'source':sha_file(__file__),
                     'old':identity['reused_pixel_source_sha256']})
    pixel_path=cache/f'benchmark_pixels_{key[:16]}.pt'
    def build():
        pixels={}
        for season in ('spring','fall','summer'):
            places=ids if season=='summer' else list(range(n)); items=[]
            rows=[r for r in manifest if r['season']==season]
            for i,(place,row) in enumerate(zip(places,rows)):
                earlier=old_records.get((season,place))
                if earlier and all(earlier[k]==row[k] for k in ('file','csv_index','size','mtime_ns')):
                    items.append(old_pixels[season][old_indices[season][place]].clone())
                else:
                    items.append(datasets[season][i][0].float())
            pixels[season]=torch.stack(items)
        return {'pixels':pixels}
    pixels=cached_tensor(pixel_path,build)['pixels']
    identity['pixel_cache']=str(pixel_path)
    inputs,identity=load_gabor(ROOT,pixels,cache,identity)
    return pixels,inputs,identity


@torch.no_grad()
def evaluate(model,inputs,ids):
    if model is None:
        feats=inputs; diagnostics={}
    else:
        batches={s:descriptors(model,x) for s,x in inputs.items()}
        feats={s:v[0] for s,v in batches.items()}; diagnostics={s:v[2] for s,v in batches.items()}
    db=torch.cat((feats['spring'],feats['fall'])).float(); q=feats['summer'].float()
    valid_db=db.norm(dim=1)>1e-8; valid_q=q.norm(dim=1)>1e-8
    db=torch.nn.functional.normalize(db,dim=1);q=torch.nn.functional.normalize(q,dim=1)
    score=(db@q.T).numpy();score[~valid_db.numpy()]=-np.inf
    n=len(inputs['spring']); places=np.maximum(score[:n],score[n:])
    r=score_report(places,ids,recallAtK,valid_q.numpy())
    r['invalid_reference_images']=int((~valid_db).sum());r['activity']=diagnostics
    return r


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,required=True)
    p.add_argument('--mode',choices=['prepare','task'],required=True)
    p.add_argument('--run-id',required=True)
    p.add_argument('--model-id')
    p.add_argument('--locked-models',type=Path)
    p.add_argument('--locked-models-sha')
    p.add_argument('--prepared-inputs',type=Path)
    p.add_argument('--prepared-inputs-sha')
    p.add_argument('--allow-dirty',action='store_true')
    for key in ('data-dir','cache-dir','run-root','deployment-manifest','deployment-manifest-sha','environment-manifest'):
        p.add_argument('--'+key)
    a=p.parse_args();cfg=json.loads(a.config.read_text())
    for key in ('data_dir','cache_dir','run_root','deployment_manifest','deployment_manifest_sha','environment_manifest'):
        if getattr(a,key):cfg[key]=getattr(a,key)
    cfg['run_id']=a.run_id
    code=code_identity(ROOT,cfg.get('deployment_manifest'),cfg.get('deployment_manifest_sha'))
    if code['dirty'] and not a.allow_dirty:raise RuntimeError('dirty evaluation code')
    deployed=code['source']=='verified_deployment_snapshot'
    env=environment_identity(cfg.get('environment_manifest'),ROOT/'pixi.lock',required=deployed)
    runtime=runtime_identity(torch,np);out,cache=resolve_locations(cfg,ROOT,deployed)
    out.mkdir(parents=True,exist_ok=False)
    provenance={'code':code,'environment':env,'runtime':runtime}
    write_json(out/'effective_config.json',{'config':cfg,**provenance})
    ids=list(range(cfg['query_start'],cfg['query_start']+cfg['query_count']))
    started=time.monotonic()
    if a.mode=='prepare':
        pixels,gabor,ident=prepare(cfg,cache)
        pointer={'identity':ident,'pixel_cache':ident['pixel_cache'],'gabor_cache':ident['gabor_cache'],
                 'pixel_cache_sha256':sha_file(ident['pixel_cache']),
                 'gabor_cache_sha256':sha_file(ident['gabor_cache']),**provenance}
        write_json(out/'prepared_inputs.json',pointer)
        for name,inputs in [('gray_direct',pixels),('gabor_direct',gabor)]:
            r=evaluate(None,inputs,ids);write_json(out/(name+'_predictions.json'),r.pop('predictions'))
            write_json(out/(name+'_summary.json'),r)
        write_json(out/'summary.json',{'status':'complete','mode':'prepare','elapsed_s':time.monotonic()-started,**provenance})
        return
    if not a.prepared_inputs or sha_file(a.prepared_inputs)!=a.prepared_inputs_sha:
        raise ValueError('prepared input identity mismatch')
    pointer=json.loads(a.prepared_inputs.read_text());ident=pointer['identity']
    if ident['query_ids']!=ids or ident['n_places']!=cfg['n_places']:raise ValueError('query protocol mismatch')
    if not a.locked_models or sha_file(a.locked_models)!=a.locked_models_sha:raise ValueError('model lock manifest mismatch')
    lock=json.loads(a.locked_models.read_text());item=next(x for x in lock['models'] if x['model_id']==a.model_id)
    cp=Path(item['checkpoint'])
    if sha_file(cp)!=item['checkpoint_sha256']:raise ValueError('checkpoint changed')
    state=torch.load(cp,map_location='cpu',weights_only=True);meta=state['metadata']
    if meta['git']['commit']!=item['training_commit'] or meta['epoch']!=item['epoch']:
        raise ValueError('training identity/locked epoch mismatch')
    if (meta['config']['seed']!=item['seed'] or meta['config']['lr']!=item['lr']
            or meta['config']['frontend']!=item['frontend'] or meta['config']['n_places']!=cfg['n_places']):
        raise ValueError('locked training configuration mismatch')
    if item['frontend']=='gabor' and item['gabor_state_sha256']!=ident['gabor_state_sha256']:
        raise ValueError('frozen Gabor state changed')
    source=pointer['gabor_cache'] if item['frontend']=='gabor' else pointer['pixel_cache']
    expected=pointer['gabor_cache_sha256'] if item['frontend']=='gabor' else pointer['pixel_cache_sha256']
    if sha_file(source)!=expected:raise ValueError('fixed input cache changed')
    data=torch.load(source,map_location='cpu',weights_only=True)
    inputs=data['features'] if item['frontend']=='gabor' else data['pixels']
    model=TaskFeature(1152 if item['frontend']=='gabor' else 784,2304,item['seed'],torch.device(cfg['device']))
    model.load_state_dict(state['model']);model.requires_grad_(False);model.eval()
    r=evaluate(model,inputs,ids)
    if any(not torch.equal(value.cpu(),state['model'][name]) for name,value in model.state_dict().items()):
        raise RuntimeError('evaluation unexpectedly changed model state')
    write_json(out/'predictions.json',r.pop('predictions'))
    r.update({'status':'complete','model':item,'query_role':'locked_benchmark','data_identity':{
       'manifest_sha256':ident['manifest_sha256'],'csv_sha256':ident['csv_sha256']},
       'model_state_unchanged':True,'elapsed_s':time.monotonic()-started,**provenance})
    write_json(out/'summary.json',r)
    print(json.dumps({'model':a.model_id,'subsets':r['subsets']}),flush=True)


if __name__=='__main__':main()
