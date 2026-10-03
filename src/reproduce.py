"""Rerun the fixed protocol in a fresh directory and compare archived predictions."""
import os
for _v in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_v]='1'
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
import argparse,json,shutil,platform,importlib.metadata,time
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
DATASETS=['wine_red','wine_white','concrete','energy','airfoil','abalone','synthetic']
SEEDS=[11,23,37,51,71]
def rerun_task(destination,dataset,seed,source):
    import run_benchmark as bench
    bench.ROOT=Path(destination)
    return bench.job(dataset,seed,source,device='cpu',epochs=250)
def compare(destination, tasks, atol):
    checked=0;largest=0.;errors=[]
    for dataset,seed,source in tasks:
        name=f'{dataset}__{source}__{seed}.npz'
        with np.load(ROOT/'results'/name) as ref,np.load(destination/'results'/name) as new:
            if set(ref.files)!=set(new.files):errors.append(name+': array names differ');continue
            for key in ref.files:
                a,b=ref[key],new[key];checked+=1
                if a.shape!=b.shape:errors.append(name+': '+key+' shape');continue
                if a.dtype.kind in 'biu':ok=np.array_equal(a,b)
                else:
                    largest=max(largest,float(np.max(np.abs(a-b))))
                    ok=np.allclose(a,b,atol=atol,rtol=0,equal_nan=False)
                if not ok:errors.append(name+': '+key)
    report=dict(environments=len(tasks),arrays_checked=checked,absolute_tolerance=atol,maximum_absolute_difference=largest,passed=not errors,mismatches=errors)
    (destination/'comparison.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
    if errors:raise SystemExit('Prediction comparison failed; see comparison.json.')
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,default=ROOT/'reproduced');p.add_argument('--workers',type=int,default=6);p.add_argument('--quick',action='store_true',help='One energy/MCAR/seed-11 environment, all nine methods.');p.add_argument('--atol',type=float,default=1e-6);a=p.parse_args()
    if a.workers<1:p.error('--workers must be positive')
    dst=a.output.resolve()
    if dst.exists():p.error('Choose a fresh output directory; existing results are preserved.')
    dst.mkdir(parents=True);shutil.copytree(ROOT/'data',dst/'data');(dst/'results').mkdir()
    tasks=[('energy',11,'MCAR')] if a.quick else [(d,s,src) for d in DATASETS for src in ['MCAR','Y'] for s in SEEDS]
    start=time.time()
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        futures=[pool.submit(rerun_task,str(dst),*task) for task in tasks]
        for f in as_completed(futures):print(f.result(),flush=True)
    import run_benchmark as bench
    bench.ROOT=dst;bench.aggregate()
    if not a.quick:
        import analyze
        analyze.ROOT=dst;analyze.main()
    env=dict(python=platform.python_version(),platform=platform.platform(),workers=a.workers,device='cpu',elapsed_seconds=time.time()-start,packages={k:importlib.metadata.version(k) for k in ['numpy','scipy','pandas','scikit-learn','torch']})
    (dst/'environment.json').write_text(json.dumps(env,indent=2));compare(dst,tasks,a.atol)
if __name__=='__main__':main()
