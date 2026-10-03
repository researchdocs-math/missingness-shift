"""Summarize archived predictions with paired, dataset-balanced statistics."""
from pathlib import Path
import json,numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parents[1]

def bootstrap(values,n=20000,seed=6021):
    """Resample datasets, then paired seeds within each sampled dataset."""
    rng=np.random.default_rng(seed);a=np.asarray(values);D,S=a.shape
    ds=rng.integers(0,D,size=(n,D));ss=rng.integers(0,S,size=(n,D,S))
    v=a[ds[:,:,None],ss].mean(axis=(1,2));return np.quantile(v,[.025,.975]).tolist()

def main():
    d=pd.read_csv(ROOT/'results/metrics.csv');real=d[d.dataset!='synthetic'].copy();real['mse']=real.nrmse**2
    table=real.groupby(['method','source','scenario'])[['nrmse','mse','coverage','width']].mean().reset_index()
    table.to_csv(ROOT/'results/summary.csv',index=False)
    contrasts=[]
    for method,x in real.groupby('method'):
        p=x.pivot(index=['dataset','seed'],columns=['source','scenario'],values='mse')
        delta=p[('Y','Y-matched')]-p[('Y','Y+matched')]
        interaction=delta-(p[('MCAR','Y-matched')]-p[('MCAR','Y+matched')])
        ci=bootstrap(interaction.unstack().values)
        contrasts.append(dict(method=method,matched_mse_increase=float(delta.mean()),interaction=float(interaction.mean()),interaction_ci_low=ci[0],interaction_ci_high=ci[1],positive_datasets=int((interaction.groupby('dataset').mean()>0).sum())))
    pd.DataFrame(contrasts).to_csv(ROOT/'results/paired_contrasts.csv',index=False)
    per_dataset=real.groupby(['dataset','method','source','scenario'])[['nrmse','mse','coverage','width']].mean().reset_index();per_dataset.to_csv(ROOT/'results/dataset_summary.csv',index=False)
    checks=[]
    for file in (ROOT/'results').glob('*.npz'):
        a=np.load(file);pm=a['mask__Y+matched'];mm=a['mask__Y-matched'];checks.append(np.array_equal(pm.sum(0),mm.sum(0)))
    report=dict(evaluation_rows=len(d),datasets=d.dataset.nunique(),real_datasets=real.dataset.nunique(),seeds=sorted(d.seed.unique().tolist()),methods=d.method.nunique(),scenarios=d.scenario.nunique(),source_mechanisms=d.source.nunique(),matched_count_checks=len(checks),all_matched_counts_identical=bool(all(checks)))
    (ROOT/'results/analysis_validation.json').write_text(json.dumps(report,indent=2));print(report);print(pd.DataFrame(contrasts).round(3).to_string(index=False))
if __name__=='__main__':main()
