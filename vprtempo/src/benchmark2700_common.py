"""MIT licensed shared exact-place evaluation for original and new VPR systems."""
import hashlib
import json
import os
from pathlib import Path

import numpy as np


def sha_file(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1048576),b''):
            h.update(block)
    return h.hexdigest()


def write_json(path,value):
    path=Path(path)
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n')
    os.replace(temp,path)


def score_report(scores,query_ids,native_recall,query_valid=None):
    """Use official argsort ranking and verify against official Recall@K."""
    scores=np.asarray(scores)
    ids=np.asarray(query_ids,dtype=np.int64)
    if scores.ndim!=2 or scores.shape[1]!=len(ids) or len(set(ids.tolist()))!=len(ids):
        raise ValueError('score/query identity mismatch')
    if not len(ids) or np.any(ids<0) or np.any(ids>=scores.shape[0]) or np.isnan(scores).any():
        raise ValueError('invalid score or ground truth')
    valid=np.ones(len(ids),dtype=bool) if query_valid is None else np.asarray(query_valid,dtype=bool)
    order=scores.argsort(0)[::-1]
    top=order[:5]
    matches={k:(order[:k]==ids[None,:]).any(0)&valid for k in (1,5,10,15,20,25)}
    gt=np.zeros(scores.shape,dtype=bool)
    gt[ids,np.arange(len(ids))]=True
    native={k:float(native_recall(scores,gt,K=k)) for k in (1,5)}
    if valid.all():
        assert all(abs(matches[k].mean()-native[k])<1e-12 for k in (1,5))
    groups={'overall':np.ones(len(ids),dtype=bool),
            'development':(ids>=600)&(ids<=1099),
            'buffer':(ids>=1100)&(ids<=1149),
            'retained':(ids>=1150)&(ids<=3299)}
    report={}
    for name,mask in groups.items():
        if mask.any():
            report[name]={'queries':int(mask.sum()),**{f'R@{k}':float(v[mask].mean()) for k,v in matches.items()},
                          'invalid_queries':int((~valid[mask]).sum())}
    predictions=[{'query':i,'query_place_id':int(place),'gt':int(place),
                  'invalid_query':not bool(valid[i]),
                  'place_top5':[{'place':int(p),'score':float(scores[p,i])}
                                for p in top[:,i] if np.isfinite(scores[p,i])]} for i,place in enumerate(ids)]
    tied=int(np.sum(scores[order[0],np.arange(len(ids))]==scores[order[1],np.arange(len(ids))]))
    return {'subsets':report,'official_native_R@1':native[1],'official_native_R@5':native[5],
            'top1_tied_queries':tied,'rank_policy':'official numpy argsort descending',
            'predictions':predictions}
