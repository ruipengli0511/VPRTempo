"""MIT licensed checks of warm-start identity, gradients and immutable controls."""
from pathlib import Path
import sys
import unittest
import json
import hashlib
import tempfile

import torch
from torch.nn import functional as F

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from vprtempo.src.expert_finetune import (ExpertFeature,tensor_digest,gradients,apply_gradients,
                                        make_schedule,calibrate)


class WarmStartChecks(unittest.TestCase):
    def model(self):
        torch.manual_seed(7)
        ww=[torch.randn(5,4,dtype=torch.float64) for _ in range(3)]
        for w in ww:w[0,0]=0
        return ExpertFeature(ww),ww

    def test_source_values_and_linear_forward_are_unchanged(self):
        model,ww=self.model();ww[0][1,1]=12.;model=ExpertFeature(ww)
        x=torch.randn(6,4,dtype=torch.float64)
        self.assertEqual(model.weights[0][1,1],12.)
        expected=torch.cat([F.linear(x,w) for w in ww],1)
        torch.testing.assert_close(model.raw(x),expected,rtol=0,atol=0)
        self.assertTrue(bool((expected<0).any()))
        torch.testing.assert_close(model.encode(x)[0],F.normalize(expected,dim=1),rtol=0,atol=0)

    def test_anchor_manual_equals_detached_autograd(self):
        model,_=self.model();torch.manual_seed(11)
        a,p,n=[torch.randn(8,4,dtype=torch.float64) for _ in range(3)]
        actual,_=gradients(model,a,p,n,2.,'anchor')
        da,_,_=model.encode(a);dp=model.encode(p)[0].detach();dn=model.encode(n)[0].detach()
        loss=F.relu(2.-(da*dp).sum(1)+(da*dn).sum(1)).mean()
        expected=model.mask_gradient(torch.autograd.grad(loss,tuple(model.weights)))
        for x,y in zip(actual,expected):torch.testing.assert_close(x,y,rtol=1e-10,atol=1e-12)

    def test_full_equals_three_path_reference(self):
        model,_=self.model();torch.manual_seed(12)
        a,p,n=[torch.randn(8,4,dtype=torch.float64) for _ in range(3)]
        actual,_=gradients(model,a,p,n,2.,'full')
        da=model.encode(a)[0];dp=model.encode(p)[0];dn=model.encode(n)[0]
        loss=F.relu(2.-(da*dp).sum(1)+(da*dn).sum(1)).mean()
        expected=model.mask_gradient(torch.autograd.grad(loss,tuple(model.weights)))
        for x,y in zip(actual,expected):torch.testing.assert_close(x,y,rtol=0,atol=0)

    def test_projection_and_relative_step_limit(self):
        model,_=self.model();original=[w.detach().clone() for w in model.weights]
        diag=apply_gradients(model,[w.detach().clone()*1e6 for w in model.weights],1.,1e-4)
        self.assertLessEqual(diag['actual_relative_step'],1.000001e-4)
        for w,old in zip(model.weights,original):
            self.assertTrue(bool((w[old>0]>=0).all()));self.assertTrue(bool((w[old<0]<=0).all()))
            self.assertTrue(bool((w[old==0]==0).all()))
        model.project_signs()
        restored=ExpertFeature(original);restored.load_state_dict(model.state_dict())
        self.assertEqual(tensor_digest(restored.state_dict()),tensor_digest(model.state_dict()))

    def test_schedule_replays_and_never_accepts_queries(self):
        initial={'spring':F.normalize(torch.randn(120,15),dim=1),'fall':F.normalize(torch.randn(120,15),dim=1)}
        a,ia=make_schedule(initial,2,0,50);b,ib=make_schedule(initial,2,0,50)
        self.assertTrue(torch.equal(a,b));self.assertEqual(ia,ib)
        self.assertTrue(bool(((a[:,:,1]-a[:,:,2]).abs()>=50).all()))
        with self.assertRaises(ValueError):make_schedule({**initial,'summer':initial['spring']},2,0,50)

    def test_calibration_is_method_independent_and_does_not_update(self):
        model,_=self.model();torch.manual_seed(8)
        train=torch.randn(2,120,4,dtype=torch.float64)
        initial={s:model.encode(train[i])[0].detach() for i,s in enumerate(['spring','fall'])}
        plan,_=make_schedule(initial,1,2,50)
        cfg={'calibration_batches':2,'margin_candidates':[2.],'min_calibration_active_fraction':0.05,
             'max_lr':0.001,'min_lr':1e-7,'max_relative_step':1e-4}
        before=tensor_digest(model.state_dict());one=calibrate(model,train,plan,16,torch.device('cpu'),cfg)
        two=calibrate(model,train,plan,16,torch.device('cpu'),cfg)
        self.assertEqual(one,two);self.assertEqual(before,tensor_digest(model.state_dict()))

    def test_selection_uses_common_development_epoch_and_excludes_partial_epoch(self):
        sys.path.insert(0,str(Path(__file__).resolve().parent))
        from run_expert_finetune import choose_locked
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);runs=[]
            for method in ['full','anchor']:
                for seed in range(3):
                    d=root/f'{method}{seed}';(d/'artifacts').mkdir(parents=True)
                    values=[0.7,([0.9,0.5,0.5][seed] if method=='full' else 0.8),0.8,1.0]
                    cps=[]
                    for e in range(4):
                        p=d/f'cp{e}';p.write_bytes(f'{method}{seed}{e}'.encode())
                        cps.append({'epoch':e,'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
                    summary={'status':'complete','phase':'training','code':{'commit':'test'},'method':method,'seed':seed,
                       'source':{'seed':seed},'calibration':{'lr':0.001},'schedule_identity':{'seed':seed},
                       'initial_state_sha256':str(seed),'checkpoints':cps,
                       'epoch_results':[{'epoch':e,'eligible_for_selection':e!=3,'development':{'R@1':v}} for e,v in enumerate(values)]}
                    (d/'artifacts/summary.json').write_text(json.dumps(summary))
                    (d/'trial_outcome.json').write_text(json.dumps({'status':'complete'}));runs.append({'run_dir':str(d)})
            result=choose_locked({},runs,'test',root)
            selected={v['method']:v['epoch'] for v in result['selection_results']}
            self.assertEqual(selected,{'full':2,'anchor':1})


if __name__=='__main__':unittest.main()
