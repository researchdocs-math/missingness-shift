"""Paired missingness-shift benchmark. All model selection uses source validation."""
import os
for _v in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_v]='1'
from pathlib import Path
import argparse, copy, json, time, platform, hashlib
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
import pandas as pd
from scipy.special import expit, logit
from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingRegressor, ExtraTreesRegressor
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
import torch
from torch import nn
ROOT=Path(__file__).resolve().parents[1]
METHODS=['Ridge+mask','ExtraTrees+mask','HistGB','Mean-MLP','Mask-MLP','Aug-MLP','Iterative-MLP','NeuMiss','NeuMISE']
SCENARIOS=['ID','MCAR10','MCAR50','MCAR70','MAR+','MAR-','Self+','Self-','Y+','Y-','Y+matched','Y-matched','Block30']

def intercept(z, rate, strength):
    lo,hi=-30.,30.
    for _ in range(60):
        mid=(lo+hi)/2
        if expit(mid+strength*z).mean()>rate:hi=mid
        else:lo=mid
    return (lo+hi)/2

def make_mask(X,y,kind,rate,rng,reference,strength=3.,matched=False):
    """Feature zero is always observed. Calibration references use training data only.
    Matched masks fix featurewise test counts by ranking logistic latent scores.
    All target mask mechanisms are simulator interventions, not learner inputs.
    """
    n,p=X.shape;u=np.clip(rng.uniform(size=(n,p)),1e-9,1-1e-9)
    if kind=='mcar': z=np.zeros((n,p)); zr=np.zeros((len(reference[0]),p))
    elif kind=='mar':z=np.repeat(X[:,0,None],p,axis=1);zr=np.repeat(reference[0][:,0,None],p,axis=1)
    elif kind=='y':z=np.repeat(y[:,None],p,axis=1);zr=np.repeat(reference[1][:,None],p,axis=1)
    elif kind=='self':z=X;zr=reference[0]
    elif kind=='block':
        a=u[:,[0]]<rate;m=np.repeat(a,p,axis=1);m[:,0]=False;return m
    else:raise ValueError(kind)
    if matched:
        score=strength*z+logit(u)
        m=np.zeros((n,p),dtype=bool);k=int(round(rate*n))
        if k:m[np.argsort(score,axis=0)[-k:,:],np.arange(p)[None,:]]=True
    else:
        a=np.array([intercept(zr[:,j],rate,strength) for j in range(p)])
        m=u<expit(a[None,:]+strength*z)
    m[:,0]=False
    return m

def corrupt(X,m):return np.where(m,np.nan,X).astype(np.float32)
def dense(X,mask=True):
    x=np.nan_to_num(X)
    return np.concatenate([x,np.isnan(X).astype(np.float32)],axis=1) if mask else x

class Network(nn.Module):
    def __init__(self,p,kind,width=64,depth=5):
        super().__init__();self.kind=kind;self.depth=depth
        if kind in ('NeuMiss','NeuMISE'):
            self.linear=nn.Linear(p,p,bias=False);nn.init.xavier_uniform_(self.linear.weight,gain=0.5)
            if kind=='NeuMiss':self.mu=nn.Parameter(torch.zeros(p))
            else:self.bn=nn.BatchNorm1d(p)
        inp=2*p if kind in ('Mask-MLP','Aug-MLP') else p
        self.head=nn.Sequential(nn.Linear(inp,width),nn.ReLU(),nn.Linear(width,width),nn.ReLU(),nn.Linear(width,1))
    def forward(self,x):
        m=torch.isnan(x);v=torch.nan_to_num(x)
        if self.kind=='NeuMiss':
            v=v-(~m)*self.mu;h=v
            for _ in range(self.depth):h=self.linear(h)*(~m)+v
            v=h
        elif self.kind=='NeuMISE':
            # Reimplementation of the published imputation-like recursion.
            h=v
            for _ in range(self.depth):h=self.bn(self.linear(h))*m+v
            v=h
        elif self.kind in ('Mask-MLP','Aug-MLP'):v=torch.cat([v,m.float()],dim=1)
        return self.head(v).squeeze(-1)

def fit_neural(X,y,V,v,kind,seed,lr,device,epochs=250):
    torch.set_num_threads(1);torch.manual_seed(seed);np.random.seed(seed)
    net=Network(X.shape[1],kind).to(device)
    tx=torch.as_tensor(X,device=device);ty=torch.as_tensor(y,dtype=torch.float32,device=device)
    vx=torch.as_tensor(V,device=device);vy=torch.as_tensor(v,dtype=torch.float32,device=device)
    opt=torch.optim.AdamW(net.parameters(),lr=lr,weight_decay=1e-4)
    best=float('inf');saved=None;stale=0;bestepoch=0
    gen=torch.Generator(device='cpu').manual_seed(seed+773)
    for epoch in range(epochs):
        net.train();order=torch.randperm(len(y),generator=gen)
        for ix in order.split(256):
            # BatchNorm requires at least two rows.
            if len(ix)<2:continue
            xb=tx[ix.to(device)];yb=ty[ix.to(device)]
            if kind=='Aug-MLP':
                # Additional independent erasure of observed features, fresh each batch.
                extra=torch.rand(xb.shape,device=device)<0.2;extra[:,0]=False
                xb=xb.masked_fill(extra,float('nan'))
            opt.zero_grad(set_to_none=True);loss=(net(xb)-yb).square().mean()
            if not torch.isfinite(loss):return None,float('inf'),0
            loss.backward();nn.utils.clip_grad_norm_(net.parameters(),5.);opt.step()
        net.eval()
        with torch.no_grad():score=(net(vx)-vy).square().mean().item()
        if score<best-1e-6:
            best=score;saved=copy.deepcopy(net.state_dict());stale=0;bestepoch=epoch+1
        else:stale+=1
        if stale>=25:break
    net.load_state_dict(saved);net.eval()
    return net,best,bestepoch

def predict(net,X,device):
    if isinstance(net,nn.Module):
        with torch.no_grad():return net(torch.as_tensor(X,device=device)).cpu().numpy()
    return net.predict(X)

def job(dataset,seed,source,device='cpu',epochs=250,methods=None):
    torch.set_num_threads(1)
    stamp=time.time();out=ROOT/'results'/f'{dataset}__{source}__{seed}.json'
    if out.exists():return str(out)+' cached'
    z=np.load(ROOT/'data'/f'{dataset}.npz');rawX=z['X'];rawy=z['y'];n=len(rawy)
    order=np.random.default_rng(seed).permutation(n)
    nt=int(.55*n);nv=int(.15*n);nc=int(.15*n)
    splits=[order[:nt],order[nt:nt+nv],order[nt+nv:nt+nv+nc],order[nt+nv+nc:]]
    # Full covariates are used by the simulator to define a latent standardized DGP.
    # Learner feature normalization below is recomputed from observed training cells.
    xm=rawX[splits[0]].mean(0);xs=rawX[splits[0]].std(0);xs[xs<1e-10]=1
    ym=rawy[splits[0]].mean();ys=rawy[splits[0]].std()
    X=(rawX-xm)/xs;y=(rawy-ym)/ys
    ref=(X[splits[0]],y[splits[0]])
    skind='mcar' if source=='MCAR' else 'y'
    ms=[make_mask(X[ix],y[ix],skind,.3,np.random.default_rng(seed*1009+70+i),ref) for i,ix in enumerate(splits)]
    observed=corrupt(rawX[splits[0]],ms[0])
    obsmean=np.nanmean(observed,axis=0);obsstd=np.nanstd(observed,axis=0);obsstd[obsstd<1e-8]=1
    standardized=(rawX-obsmean)/obsstd
    a,b,c,t=[corrupt(standardized[ix],m) for ix,m in zip(splits,ms)]
    ya,yb,yc,yt=[y[ix].astype(np.float32) for ix in splits]
    targets={};masks={};ti=splits[-1]
    for scen in SCENARIOS:
        if scen=='ID':m=ms[-1]
        else:
            kind,rate,strength,matched='mcar',.3,3.,False
            if scen.startswith('MCAR'):rate=int(scen[4:])/100
            elif scen.startswith('MAR'):kind='mar';strength=3. if '+' in scen else -3.
            elif scen.startswith('Self'):kind='self';strength=3. if '+' in scen else -3.
            elif scen.startswith('Y'):kind='y';strength=3. if '+' in scen else -3.;matched='matched' in scen
            elif scen.startswith('Block'):kind='block'
            # Common random numbers pair all test interventions within a split.
            m=make_mask(X[ti],y[ti],kind,rate,np.random.default_rng(seed*1009+400),ref,strength,matched)
        targets[scen]=corrupt(standardized[ti],m);masks[scen]=m
    rows=[];selected=[];predictions={};chosen=methods or METHODS
    for k,method in enumerate(chosen):
        start=time.time();best=float('inf');net=None;bestparams={};transform=None
        if method=='Ridge+mask':
            for alpha in [0.1,1.,10.,100.]:
                model=Ridge(alpha=alpha).fit(dense(a),ya);val=np.mean((model.predict(dense(b))-yb)**2)
                if val<best:best=val;net=model;bestparams={'alpha':alpha}
            transform=lambda q:dense(q)
        elif method=='ExtraTrees+mask':
            for leaf in [2,5,10]:
                model=ExtraTreesRegressor(n_estimators=300,min_samples_leaf=leaf,max_features=1.,n_jobs=1,random_state=seed).fit(dense(a),ya)
                val=np.mean((model.predict(dense(b))-yb)**2)
                if val<best:best=val;net=model;bestparams={'min_samples_leaf':leaf}
            transform=lambda q:dense(q)
        elif method=='HistGB':
            for leaf in [7,15,31]:
                model=HistGradientBoostingRegressor(max_iter=250,max_leaf_nodes=leaf,learning_rate=.05,l2_regularization=1.,early_stopping=False,random_state=seed).fit(a,ya)
                val=np.mean((model.predict(b)-yb)**2)
                if val<best:best=val;net=model;bestparams={'max_leaf_nodes':leaf}
            transform=lambda q:q
        else:
            aa,bb=a,b
            if method=='Iterative-MLP':
                imp=IterativeImputer(max_iter=10,random_state=seed,sample_posterior=False,skip_complete=True).fit(a)
                aa=imp.transform(a).astype(np.float32);bb=imp.transform(b).astype(np.float32)
                transform=lambda q:imp.transform(q).astype(np.float32)
            else:transform=lambda q:q
            for lr in [0.0003,0.001,0.003]:
                model,val,ep=fit_neural(aa,ya,bb,yb,method,seed+10000*k,lr,device,epochs)
                if val<best:best=val;net=model;bestparams={'lr':lr,'best_epoch':ep}
        if net is None:raise RuntimeError(f'No valid model {dataset} {seed} {method}')
        cp=predict(net,transform(c),device);res=np.sort(np.abs(cp-yc));rank=min(len(res),int(np.ceil((len(res)+1)*.9)));q=float(res[rank-1])
        preds={sc:predict(net,transform(v),device) for sc,v in targets.items()}
        for sc,pred in preds.items():
            err=pred-yt;m=masks[sc];few=m[:,1:].mean(1)>.5
            row=dict(dataset=dataset,seed=seed,source=source,method=method,scenario=sc,n_test=len(yt),nrmse=float(np.sqrt(np.mean(err**2))),mae=float(np.abs(err).mean()),coverage=float((np.abs(err)<=q).mean()),width=2*q,missing_rate=float(m[:,1:].mean()),high_missing_n=int(few.sum()),high_missing_mse=float(np.mean(err[few]**2)) if few.any() else None)
            rows.append(row)
        predictions[method]=preds
        selected.append(dict(method=method,validation_mse=float(best),parameters=bestparams,fit_and_evaluation_seconds=time.time()-start))
    predf={f'{m}__{s}':v for m,d in predictions.items() for s,v in d.items()};predf['y_test']=yt;predf['test_indices']=ti
    for sc,m in masks.items():predf['mask__'+sc]=m
    np.savez_compressed(out.with_suffix('.npz'),**predf)
    record=dict(dataset=dataset,seed=seed,source=source,device=device,n=n,p=rawX.shape[1],split_sizes=list(map(len,splits)),split_sha256=hashlib.sha256(order.tobytes()).hexdigest(),elapsed_seconds=time.time()-stamp,selected=selected,rows=rows)
    temp=out.with_suffix('.tmp');temp.write_text(json.dumps(record,indent=2));temp.replace(out)
    return f'{dataset} {source} {seed}: {time.time()-stamp:.1f}s'

def aggregate():
    rows=[];selections=[]
    for f in sorted((ROOT/'results').glob('*.json')):
        d=json.loads(f.read_text())
        if 'rows' not in d:continue
        rows.extend(d['rows'])
        for s in d['selected']:selections.append(dict(dataset=d['dataset'],source=d['source'],seed=d['seed'],**s))
    if rows:pd.DataFrame(rows).to_csv(ROOT/'results/metrics.csv',index=False)
    (ROOT/'results/model_selection.json').write_text(json.dumps(selections,indent=2))
    print(f'Aggregated {len(rows)} evaluation rows.',flush=True)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--workers',type=int,default=6);ap.add_argument('--seeds',type=int,nargs='+',default=[11,23,37,51,71]);ap.add_argument('--datasets',nargs='+',default=['wine_red','wine_white','concrete','energy','airfoil','abalone','synthetic']);ap.add_argument('--sources',nargs='+',default=['MCAR','Y']);ap.add_argument('--device',default='cpu',choices=['cpu','mps']);ap.add_argument('--epochs',type=int,default=250);ap.add_argument('--aggregate-only',action='store_true');a=ap.parse_args()
    (ROOT/'results').mkdir(exist_ok=True)
    if a.aggregate_only:aggregate();return
    (ROOT/'results/runtime_environment.json').write_text(json.dumps(dict(python=platform.python_version(),platform=platform.platform(),torch=torch.__version__,numpy=np.__version__,workers=a.workers,device=a.device,epochs=a.epochs,seeds=a.seeds),indent=2))
    jobs=[(d,s,src,a.device,a.epochs) for d in a.datasets for src in a.sources for s in a.seeds]
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        fs={ex.submit(job,*j):j for j in jobs}
        for f in as_completed(fs):print(f.result(),flush=True)
    aggregate()
if __name__=='__main__':main()
