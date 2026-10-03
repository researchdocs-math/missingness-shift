"""Download UCI datasets and store numeric regression arrays with provenance."""
from pathlib import Path
import io, zipfile, hashlib, json
import requests
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'data'
SPECS={
 'wine_red':(186,'wine+quality','winequality-red.csv'),
 'wine_white':(186,'wine+quality','winequality-white.csv'),
 'concrete':(165,'concrete+compressive+strength','Concrete_Data.xls'),
 'energy':(242,'energy+efficiency','ENB2012_data.xlsx'),
 'airfoil':(291,'airfoil+self+noise','airfoil_self_noise.dat'),
 'abalone':(1,'abalone','abalone.data')}
def main():
 DATA.mkdir(exist_ok=True)
 manifest=[]
 cache={}
 for name,(uid,slug,filename) in SPECS.items():
  url=f'https://archive.ics.uci.edu/static/public/{uid}/{slug}.zip'
  if uid not in cache:
   r=requests.get(url,timeout=120);r.raise_for_status();cache[uid]=r.content
  raw=cache[uid];z=zipfile.ZipFile(io.BytesIO(raw))
  member=next(k for k in z.namelist() if k.endswith(filename))
  content=z.read(member)
  (DATA/filename).write_bytes(content)
  if name.startswith('wine'): df=pd.read_csv(io.BytesIO(content),sep=';'); X=df.iloc[:,:-1].values;y=df.iloc[:,-1].values
  elif name=='concrete': df=pd.read_excel(io.BytesIO(content));X=df.iloc[:,:-1].values;y=df.iloc[:,-1].values
  elif name=='energy': df=pd.read_excel(io.BytesIO(content)).iloc[:768,:10];X=df.iloc[:,:8].values;y=df.iloc[:,8].values
  elif name=='airfoil': df=pd.read_csv(io.BytesIO(content),sep=r'\s+',header=None);X=df.iloc[:,:5].values;y=df.iloc[:,5].values
  else:
   df=pd.read_csv(io.BytesIO(content),header=None)
   # Seven numeric measurements; sex is not used in this numerical benchmark.
   X=df.iloc[:,1:8].values;y=df.iloc[:,8].values
  X=np.asarray(X,dtype=np.float64);y=np.asarray(y,dtype=np.float64)
  assert np.isfinite(X).all() and np.isfinite(y).all()
  np.savez_compressed(DATA/f'{name}.npz',X=X,y=y)
  manifest.append(dict(name=name,uci_id=uid,url=url,member=member,sha256=hashlib.sha256(content).hexdigest(),n=len(y),p=X.shape[1],license='CC BY 4.0',target='heating load' if name=='energy' else ('rings' if name=='abalone' else 'last column')))
 rng=np.random.default_rng(8128);n=5000;p=20
 cov=0.6**np.abs(np.arange(p)[:,None]-np.arange(p)[None,:]);X=rng.multivariate_normal(np.zeros(p),cov,n)
 y=2*np.sin(X[:,0])+X[:,1]*X[:,2]+0.5*X[:,3]**2+X[:,4]+rng.normal(0,0.5,n)
 np.savez_compressed(DATA/'synthetic.npz',X=X,y=y)
 manifest.append(dict(name='synthetic',n=n,p=p,seed=8128,license='MIT',target='2 sin(x0)+x1*x2+0.5*x3^2+x4+Normal(0,0.5^2)'))
 (DATA/'manifest.json').write_text(json.dumps(manifest,indent=2))
 print(json.dumps(manifest,indent=2))
if __name__=='__main__':main()
