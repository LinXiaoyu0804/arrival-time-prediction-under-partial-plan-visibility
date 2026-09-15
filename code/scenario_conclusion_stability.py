from pathlib import Path
import time,json,shutil
import numpy as np
import pandas as pd
R=Path(__file__).resolve().parents[1];O=R/'outputs/TRE_scenario_alignment_20260913'
freeze={'time':time.strftime('%Y-%m-%d %H:%M:%S'),'type':'Follow-up conclusion-stability audit; no model fitting or tuning','natural_mismatch_groups':'Within-route fraction of delivery tasks whose recorded planned stage differs from eventual execution stage: <=10%,(10%,30%],>30%. Evaluation-only strata never enter selectors.','driver_groups':'Previously seen in ETA training or calibration vs unseen','robustness':'Compare gain against all four fixed simple rules under every predefined response scenario. Both direction and paired95% intervals reported. No claim all real-world shifts are covered.','stress_extension':'Uniform arbitrary-stage misreport10/20 percent, using exact expectation over all other stage labels. This includes non-adjacent errors; fixed choices.','interpretation':'Evidence that method ordering persists under tested discrepancies, not proof that software plan equals human intent.'}
(O/'stability_audit_protocol.json').write_text(json.dumps(freeze,ensure_ascii=False,indent=2),encoding='utf8');shutil.copy2(__file__,O/'code/scenario_conclusion_stability.py')
t=pd.read_parquet(O/'tables/all_losses.parquet');extras=[];metas=[]
for k in [3,5]:
    for origin in [13,19,25]:
        q=pd.read_parquet(O/f'cache/w{origin}_k{k}_evaluation.parquet')
        # Reuse stored model for alternative-label losses; no retraining.
        import joblib
        bundle=joblib.load(O/f'models/w{origin}_k{k}.joblib');features=list(bundle['aware'].feature_name_)
        y=q.target_minutes.to_numpy();pred=[]
        for label in range(k):
            x=q.copy();x['answer_phase']=label;pred.append(bundle['aware'].predict(x[features]))
        err=abs(np.stack(pred,axis=1)-y[:,None]);ix=np.arange(len(q));true=q.planned_phase.to_numpy();clean=err[ix,true];wrong=(err.sum(axis=1)-clean)/(k-1)
        codes,routes=pd.factorize(q['Route ID'],sort=True);n=np.bincount(codes)
        def means(x):return np.bincount(codes,weights=x)/n
        meta=q.groupby('Route ID').agg(driver=('Driver ID','first'),mismatch=('proxy_matches_future',lambda x:1-x.mean())).reindex(routes);meta.index.name='Route ID';meta['origin']=origin;meta['k']=k;metas.append(meta.reset_index())
        choices=pd.read_parquet(O/f'tables/w{origin}_k{k}_choices.parquet')
        for method in ['random','gain','uncertainty','disagreement','latest_eta']:
            if method!='random':selected=q._rowkey.isin(set(choices[(choices.method==method)&choices.selected]._rowkey)).to_numpy()
            for p in [.1,.2]:
                gain=q.base_error.to_numpy()-((1-p)*clean+p*wrong)
                reduction=means(gain)/n if method=='random' else means(gain*selected)
                z=meta.copy();z['mae']=means(q.base_error.to_numpy())-reduction;z['base_mae']=means(q.base_error.to_numpy());z['method']=method;z['condition']=f'arbitrary_error{int(p*100)}';z['answers']=1.;z['attempts']=1;extras.append(z.reset_index())
ex=pd.concat(extras,ignore_index=True);ex.to_parquet(O/'tables/arbitrary_error_routes.parquet',index=False)
ex.groupby(['k','method','condition'],as_index=False).mae.mean().to_csv(O/'tables/arbitrary_error_summary.csv',index=False)
meta=pd.concat(metas,ignore_index=True);meta['mismatch_group']=pd.cut(meta.mismatch,[-1,.1,.3,1.0001],labels=['0_to_10pct','10_to_30pct','over_30pct']).astype(str)
t=t.merge(meta[['origin','k','Route ID','mismatch','mismatch_group']],on=['origin','k','Route ID'],validate='many_to_one')
ex=ex.merge(t[['origin','k','Route ID','seen_driver','mismatch_group']].drop_duplicates(),on=['origin','k','Route ID'],validate='many_to_one')
t=pd.concat([t,ex],ignore_index=True)
refs=['random','uncertainty','disagreement','latest_eta']
rows=[];summaries=[]
def compare(z,label):
    ids=z[['origin','Route ID','driver']].drop_duplicates().sort_values(['origin','Route ID']);idx=pd.MultiIndex.from_frame(ids[['origin','Route ID']]);codes,drivers=pd.factorize(ids.driver)
    if len(drivers)<2:return
    boot=np.random.default_rng(20260913).multinomial(len(drivers),np.ones(len(drivers))/len(drivers),size=2000);cnt=np.bincount(codes)
    pivot=z.pivot(index=['origin','Route ID'],columns='method',values='mae').reindex(idx)
    for ref in refs:
        delta=(pivot.gain-pivot[ref]).to_numpy();dist=(boot@np.bincount(codes,weights=delta))/(boot@cnt);lo,hi=np.quantile(dist,[.025,.975]);rows.append(dict(k=int(z.k.iloc[0]),scope=label,condition=z.condition.iloc[0],reference=ref,difference=delta.mean(),low=lo,high=hi,routes=len(ids),drivers=len(drivers),mean_direction=bool(delta.mean()<0),interval_support=bool(hi<0)))
    s=z.groupby('method',as_index=False).agg(mae=('mae','mean'),answers=('answers','mean'));s['scope']=label;s['k']=int(z.k.iloc[0]);s['condition']=z.condition.iloc[0];s['routes']=len(ids);summaries.append(s)
for k in [3,5]:
    for condition in ['proxy_clean','independent_refusal25','independent_refusal50','entropy_refusal25','entropy_refusal50','adjacent_error10','adjacent_error20','arbitrary_error10','arbitrary_error20']:
        compare(t[(t.k==k)&(t.condition==condition)],'all')
    clean=t[(t.k==k)&(t.condition=='proxy_clean')]
    for group in ['0_to_10pct','10_to_30pct','over_30pct']:compare(clean[clean.mismatch_group==group],'mismatch_'+group)
    for value in [False,True]:compare(clean[clean.seen_driver==value],'driver_seen_'+str(value))
tab=pd.DataFrame(rows);tab.to_csv(O/'tables/conclusion_stability_intervals.csv',index=False)
pd.concat(summaries,ignore_index=True).to_csv(O/'tables/stability_group_summary.csv',index=False)
matrix=tab[tab.scope.eq('all')].groupby(['k','condition'],as_index=False).agg(comparisons=('reference','size'),favorable_mean_count=('mean_direction','sum'),supported_interval_count=('interval_support','sum'));matrix.to_csv(O/'tables/stability_matrix.csv',index=False)
print(matrix.to_string(index=False));print(tab[(tab.scope!='all')&(tab.reference.isin(['disagreement','uncertainty']))].to_string(index=False));print(meta.groupby(['k','mismatch_group']).size().to_string())
