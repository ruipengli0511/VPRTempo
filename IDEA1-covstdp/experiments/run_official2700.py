"""MIT licensed engineering adapter around unchanged upstream VPRTempo methods."""
import argparse
import json
import logging
import os
from pathlib import Path
import random
import shutil
import sys
import time
from types import SimpleNamespace

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT));sys.dont_write_bytecode=True
from vprtempo.src.run_provenance import code_identity,environment_identity,resolve_locations,runtime_identity
from vprtempo.src.benchmark2700_common import sha_file,write_json,score_report
from vprtempo.src.dataset import CustomImageDataset,ProcessImage
from vprtempo.VPRTempoTrain import VPRTempoTrain,train_new_model
from vprtempo.VPRTempo import VPRTempo
from vprtempo.src.metrics import recallAtK
import importlib
TRAIN=importlib.import_module('vprtempo.VPRTempoTrain')

PIXELS={};LOOKUP={}
class CachedOfficialDataset(CustomImageDataset):
    """Keep official selection/labels and replace deterministic decoding only."""
    def __getitem__(self,index):
        row=self.img_labels.iloc[index];path=Path(row['file_path'])
        season=path.parent.name;place=LOOKUP[season][path.name]
        image=PIXELS[season][place].clone();label=row.iloc[1]
        if self.target_transform:label=self.target_transform(label)
        return image,label


def verify_core():
    record=json.loads((ROOT/'vprtempo/src/official_source_identity.json').read_text())
    for item in record['algorithm_files']:
        if sha_file(ROOT/item['path'])!=item['sha256']:
            raise ValueError('official algorithm changed: '+item['path'])
    return record


def train_models(args,seed,logger):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)
    count=args.database_places//args.max_module
    models=[]
    for i in range(count):
        model=VPRTempoTrain(args,[28,28],logger,count,args.max_module)
        model.to(torch.device('cpu'));models.append(model)
    train_new_model(models,'official_model.pth')
    return models


def main():
    global PIXELS,LOOKUP
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,required=True);p.add_argument('--run-id',required=True)
    p.add_argument('--seed',type=int,required=True)
    p.add_argument('--prepared-inputs',type=Path,required=True);p.add_argument('--prepared-inputs-sha',required=True)
    p.add_argument('--allow-dirty',action='store_true')
    for key in ('data-dir','cache-dir','run-root','deployment-manifest','deployment-manifest-sha','environment-manifest'):
        p.add_argument('--'+key)
    a=p.parse_args();cfg=json.loads(a.config.read_text());cfg['run_id']=a.run_id;cfg['seed']=a.seed
    for key in ('data_dir','cache_dir','run_root','deployment_manifest','deployment_manifest_sha','environment_manifest'):
        if getattr(a,key):cfg[key]=getattr(a,key)
    version=code_identity(ROOT,cfg.get('deployment_manifest'),cfg.get('deployment_manifest_sha'))
    if version['dirty'] and not a.allow_dirty:raise RuntimeError('dirty baseline adapter')
    deployed=version['source']=='verified_deployment_snapshot'
    env=environment_identity(cfg.get('environment_manifest'),ROOT/'pixi.lock',required=deployed)
    runtime=runtime_identity(torch,np);provenance={'code':version,'environment':env,'runtime':runtime}
    official=verify_core()
    if sha_file(a.prepared_inputs)!=a.prepared_inputs_sha:raise ValueError('prepared pointer changed')
    pointer=json.loads(a.prepared_inputs.read_text());ident=pointer['identity']
    ids=list(range(cfg['query_start'],cfg['query_start']+cfg['query_count']))
    if ids!=ident['query_ids'] or cfg['n_places']!=ident['n_places']:raise ValueError('data protocol mismatch')
    if sha_file(pointer['pixel_cache'])!=pointer['pixel_cache_sha256']:raise ValueError('pixel payload changed')
    PIXELS=torch.load(pointer['pixel_cache'],map_location='cpu',weights_only=True)['pixels']
    for season in ('spring','fall','summer'):
        rows=[r for r in ident['manifest'] if r['season']==season]
        LOOKUP[season]={row['file']:i for i,row in enumerate(rows)}
        if sha_file(ROOT/f'vprtempo/dataset/nordland-{season}.csv')!=ident['csv_sha256'][season]:
            raise ValueError('official/current CSV mismatch')
    out,cache=resolve_locations(cfg,ROOT,deployed);out.mkdir(parents=True,exist_ok=False)
    write_json(out/'effective_config.json',{'config':cfg,**provenance,'upstream':official})
    work=out/'official_work';(work/'vprtempo/dataset').mkdir(parents=True)
    (work/'vprtempo/models').mkdir()
    for season in ('spring','fall','summer'):
        shutil.copy2(ROOT/f'vprtempo/dataset/nordland-{season}.csv',work/f'vprtempo/dataset/nordland-{season}.csv')
    os.chdir(work)
    ns=SimpleNamespace(dataset='nordland',data_dir=cfg['data_dir'],database_places=cfg['n_places'],
       query_places=len(ids),max_module=cfg['max_module'],database_dirs='spring,fall',query_dir='summer',
       GT_tolerance=0,skip=0,filter=8,epoch=cfg['epochs'],patches=7,dims='28,28',
       train_new_model=True,quantize=False,model_name=None,PR_curve=False,sim_mat=False,run_demo=False)
    if ns.database_places%ns.max_module:raise ValueError('baseline protocol requires equal expert modules')
    logger=logging.getLogger('official-benchmark');logger.setLevel(logging.INFO)
    logger.addHandler(logging.StreamHandler())
    # Verify runtime cache against the actual official decoding path, without changing RNG.
    check=CachedOfficialDataset(['./vprtempo/dataset/nordland-spring.csv','./vprtempo/dataset/nordland-fall.csv'],
           cfg['data_dir'],['spring','fall'],transform=ProcessImage([28,28],7),filter=8,
           max_samples=cfg['n_places'],test=False)
    parity=[]
    for index in sorted(set([0,len(check)//2-1,len(check)//2,len(check)-1])):
        original,label=CustomImageDataset.__getitem__(check,index)
        cached,label2=check[index]
        difference=float((original-cached).abs().max())
        if difference!=0 or label!=label2:raise ValueError('cached official input parity failed')
        parity.append({'index':index,'max_abs_diff':difference,'label':int(label)})
    write_json(out/'input_parity.json',parity)
    TRAIN.CustomImageDataset=CachedOfficialDataset
    started=time.monotonic()
    models=train_models(ns,a.seed,logger)
    train_seconds=time.monotonic()-started
    checkpoint=work/'vprtempo/models/official_model.pth'
    checkpoint_hash=sha_file(checkpoint)
    del models
    # Identical official inference path: feature linear -> output linear, then concatenate experts.
    ns.skip=cfg['query_start']*8;ns.train_new_model=False
    count=ns.database_places//ns.max_module
    infer=[]
    for i in range(count):
        model=VPRTempo(ns,[28,28],logger,count,str(out),ns.max_module)
        model.to(torch.device('cpu'));model.eval();infer.append(model)
    infer[0].load_model(infer,str(checkpoint))
    device=torch.device(infer[0].device)
    infer[0].inferences=[torch.nn.Sequential(m.feature_layer.w,m.output_layer.w).to(device) for m in infer]
    scores=[]
    evaluation_started=time.monotonic()
    with torch.no_grad():
        for x in PIXELS['summer'].split(1):
            scores.append(infer[0].forward(x.to(device)).cpu())
    matrix=torch.cat(scores).numpy().T
    result=score_report(matrix,ids,recallAtK)
    write_json(out/'predictions.json',result.pop('predictions'))
    result.update({'status':'complete','method':'original_vprtempo','seed':a.seed,
       'training_s':train_seconds,'evaluation_s':time.monotonic()-evaluation_started,
       'checkpoint':str(checkpoint),'checkpoint_sha256':checkpoint_hash,
       'official_core_identity':official,'data_manifest_sha256':ident['manifest_sha256'],**provenance})
    write_json(out/'summary.json',result)
    print(json.dumps({'seed':a.seed,'training_s':train_seconds,'subsets':result['subsets']}),flush=True)


if __name__=='__main__':main()
