"""MIT licensed checks of warm-start identity, gradients and immutable controls."""
from pathlib import Path
import sys
import unittest
import json
import hashlib
import tempfile
import importlib.util
import subprocess

import torch
from torch.nn import functional as F

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from vprtempo.src.expert_finetune import (ExpertFeature,tensor_digest,gradients,apply_gradients,
                                        make_schedule,calibrate,probe_diagnostics,preview_gradients)


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

    def test_probe_and_preview_do_not_mutate_or_consume_rng(self):
        model,_=self.model();torch.manual_seed(19)
        train=torch.randn(2,120,4,dtype=torch.float64)
        initial={s:model.encode(train[i])[0].detach() for i,s in enumerate(['spring','fall'])}
        plan,_=make_schedule(initial,1,0,50)
        state=tensor_digest(model.state_dict());rng=torch.get_rng_state().clone()
        result=probe_diagnostics(model,train,plan[0][:128],0.1,'anchor',0.1,1e-4,16,torch.device('cpu'))
        self.assertEqual(state,tensor_digest(model.state_dict()));self.assertTrue(torch.equal(rng,torch.get_rng_state()))
        self.assertEqual(result['triplets'],128);self.assertEqual(result['valid'],128)
        self.assertFalse(result['uses_query_data']);self.assertTrue(result['model_state_unchanged'])

    def test_preview_step_matches_actual_and_diagnostics_decompose_projection(self):
        model,_=self.model();gs=[w.detach().clone()*2 for w in model.weights]
        before=tensor_digest(model.state_dict())
        preview=preview_gradients(model,gs,1.,2.)
        self.assertEqual(before,tensor_digest(model.state_dict()))
        actual=apply_gradients(model,gs,1.,2.)
        for key in preview:self.assertEqual(preview[key],actual[key])
        self.assertGreater(actual['projection_change_norm'],0.)
        self.assertLessEqual(actual['actual_step_norm'],actual['pre_projection_step_norm'])

    def test_instrumentation_matches_frozen_B06_updates(self):
        # The reference is the committed B06 implementation, not a rewritten formula.
        source=subprocess.check_output(['git','-C',str(ROOT),'show',
            '2d2527da1536840460c7e7d87c8f313e616b0f27:vprtempo/src/expert_finetune.py'],text=True)
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'b06_reference.py';path.write_text(source)
            spec=importlib.util.spec_from_file_location('frozen_b06_reference',path)
            old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
            for method in ['full','anchor']:
                new,ww=self.model();legacy=old.ExpertFeature(ww)
                torch.manual_seed(12);triplets=[torch.randn(8,4,dtype=torch.float64) for _ in range(3)]
                for _ in range(4):
                    gnew,_=gradients(new,*triplets,0.1,method);gold,_=old.gradients(legacy,*triplets,0.1,method)
                    for x,y in zip(gnew,gold):torch.testing.assert_close(x,y,rtol=0,atol=0)
                    apply_gradients(new,gnew,0.1,1e-4);old.apply_gradients(legacy,gold,0.1,1e-4)
                    self.assertEqual(tensor_digest(new.state_dict()),old.tensor_digest(legacy.state_dict()))


if __name__=='__main__':unittest.main()
