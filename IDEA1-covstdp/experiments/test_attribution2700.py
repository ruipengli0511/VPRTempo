"""MIT licensed checks for exact frozen-state and readout semantics."""
from pathlib import Path
import sys
import unittest

import torch
from torch import nn

sys.path.insert(0,str(Path(__file__).resolve().parent))
from eval_attribution2700 import (assert_same_states, state_digest, official_weights,
                                 feature_linear, output_linear, check_original_output, evaluate)


class FrozenControls(unittest.TestCase):
    def test_full_anchor_initial_pair_requires_all_state(self):
        a={'weight':torch.tensor([1.,-2.]),'mask':torch.tensor([True,False])}
        b={k:v.clone() for k,v in a.items()}
        assert_same_states(a,b)
        self.assertEqual(state_digest(a),state_digest(b))
        b['mask'][1]=True
        with self.assertRaises(ValueError):assert_same_states(a,b)
        self.assertNotEqual(state_digest(a),state_digest(b))

    def test_official_feature_is_original_linear_intermediate(self):
        torch.manual_seed(9); state={}; modules=[]
        for i in range(3):
            first=nn.Linear(4,5,bias=False);second=nn.Linear(5,3,bias=False)
            state[f'model_{i}']={'feature_layer.w.weight':first.weight.detach().clone(),
                                'output_layer.w.weight':second.weight.detach().clone()}
            modules.append(nn.Sequential(first,second))
        weights=official_weights(state,4,5,3)
        x=torch.randn(7,4)*10
        features=feature_linear(x,weights)
        self.assertEqual(tuple(features.shape),(7,15))
        self.assertTrue(bool((features<0).any()))
        self.assertTrue(bool((features>0.9).any()))
        expected=torch.cat([m(x) for m in modules],dim=1)
        torch.testing.assert_close(output_linear(x,weights),expected,rtol=0,atol=0)
        preds=[]
        for row in output_linear(x,weights).numpy():
            order=row.argsort()[::-1][:5]
            preds.append({'place_top5':[{'place':int(p),'score':float(row[p])} for p in order]})
        parity=check_original_output(weights,x,preds,torch.device('cpu'))
        self.assertTrue(parity['top5_exact_match'])
        self.assertLess(parity['max_abs_score_diff'],1e-6)

    def test_place_retrieval_merges_seasons_before_topk(self):
        # Both seasonal images for place0 must occupy a single place candidate.
        inputs={'spring':torch.tensor([[1.,0.],[0.,1.],[1.,1.],[1.,-1.],[-1.,1.],[-1.,-1.]]),
                'fall':torch.tensor([[1.,0.],[0.,1.],[1.,1.],[1.,-1.],[-1.,1.],[-1.,-1.]]),
                'summer':torch.tensor([[1.,0.],[0.,1.]])}
        result=evaluate(None,inputs,[0,1])
        self.assertEqual(result['subsets']['overall']['R@1'],1.)
        self.assertEqual(result['subsets']['overall']['R@5'],1.)
        self.assertEqual(len(result['predictions'][0]['place_top5']),5)
        self.assertEqual(len({p['place'] for p in result['predictions'][0]['place_top5']}),5)


if __name__=='__main__':unittest.main()
