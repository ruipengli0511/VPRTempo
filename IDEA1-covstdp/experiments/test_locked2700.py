"""MIT licensed exact-GT/official-metric regression for locked evaluation."""
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from vprtempo.src.benchmark2700_common import score_report
from vprtempo.src.metrics import recallAtK

s=np.zeros((6,2),dtype=np.float32);s[5,0]=2;s[1,1]=3
r=score_report(s,[5,1],recallAtK)
assert r['subsets']['overall']['R@1']==1 and r['official_native_R@1']==1
assert [p['place_top5'][0]['place'] for p in r['predictions']]==[5,1]
r=score_report(s,[5,1],recallAtK,[True,False])
assert r['subsets']['overall']['R@1']==.5 and r['official_native_R@1']==1
s=np.zeros((3300,3),dtype=np.float32)
for i,p in enumerate([600,1100,1150]):s[p,i]=1
r=score_report(s,[600,1100,1150],recallAtK)
assert all(r['subsets'][name]['queries']==1 for name in ('development','buffer','retained'))
assert r['subsets']['retained']['R@1']==1
print('official Recall@K parity, explicit query GT, invalid reporting and region splits passed')
