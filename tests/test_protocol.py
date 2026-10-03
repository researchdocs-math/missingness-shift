import sys
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from run_benchmark import make_mask, Network

def test_matched_counts_and_observed_anchor():
    r=np.random.default_rng(2);X=r.normal(size=(601,11));y=r.normal(size=601)
    plus=make_mask(X,y,'y',.3,np.random.default_rng(9),(X,y),3,True)
    minus=make_mask(X,y,'y',.3,np.random.default_rng(9),(X,y),-3,True)
    assert np.all(plus.sum(0)==minus.sum(0))
    assert np.all(plus.sum(0)[1:]==round(.3*601))
    assert not plus[:,0].any() and not minus[:,0].any()
    assert y[plus[:,1]].mean()>y[minus[:,1]].mean()

def test_mcar_independence_and_mnar_direction():
    r=np.random.default_rng(12);X=r.normal(size=(50000,4));y=r.normal(size=50000)
    m=make_mask(X,y,'mcar',.3,np.random.default_rng(4),(X,y))
    assert abs(m[:,1:].mean()-.3)<.01
    assert abs(np.corrcoef(m[:,1],y)[0,1])<.02
    p=make_mask(X,y,'y',.3,np.random.default_rng(4),(X,y))
    assert np.corrcoef(p[:,1],y)[0,1]>.45

def test_complete_neumise_passes_covariates_to_head():
    torch.manual_seed(1);n=Network(4,'NeuMISE');n.eval();x=torch.randn(20,4)
    assert torch.equal(n(x),n.head(x).squeeze(-1))

def test_all_models_have_finite_outputs_with_missing_inputs():
    x=torch.randn(20,5);x[:10,1:]=float('nan')
    for kind in ['Mean-MLP','Mask-MLP','Aug-MLP','NeuMiss','NeuMISE']:
        n=Network(5,kind);n.eval()
        assert torch.isfinite(n(x)).all()

def test_binary_selection_risk_identity():
    for source in [-.8,0,.8]:
        for target in [-.8,0,.8]:
            risk=.25*(1+target)*(1-source)**2+.25*(1-target)*(-1-source)**2
            assert np.isclose(risk,.5*(1-target**2+(source-target)**2))

if __name__=='__main__':
    import inspect
    checks=[(name,fn) for name,fn in list(globals().items()) if name.startswith('test_') and inspect.isfunction(fn)]
    for name,fn in checks:fn();print('PASS',name)
    print(f'{len(checks)} protocol checks passed.')
