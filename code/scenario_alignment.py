import os,sys,time,json,hashlib,shutil
from pathlib import Path
os.environ['OMP_NUM_THREADS']='6';sys.dont_write_bytecode=True
import numpy as np
import pandas as pd
import joblib
from lightgbm import LGBMRegressor,LGBMClassifier
R=Path(__file__).resolve().parents[1];O=R/'outputs/TRE_scenario_alignment_20260913'
P=R/'data/source'
sys.path.insert(0,str(R/'code'));import planned_actual_core as core
for d in ['tables','models','cache','code']:(O/d).mkdir(parents=True,exist_ok=True)
def save(n,x):(O/n).write_text(json.dumps(x,ensure_ascii=False,indent=2,default=str),encoding='utf8')
def log(s):print(time.strftime('%H:%M:%S'),s,flush=True)
protocol={'date':'2026-09-13','scope':'Scenario bridge only: all learning uses past completed deliveries, not recorded historical plans. Evaluation queries reveal recorded SOFTWARE planned stages as a proxy, NOT observed driver predeparture intent. Actual evaluation sequence is an unattainable diagnostic only.',
 'windows':[[0,6,7,12,13,18],[0,12,13,18,19,24],[0,18,19,24,25,31]],
 'history':'Historical IndexP and DistanceP removed before all learning. Reuse core history encoding by mapping historical execution rank to its legacy rank slot, then rename features execution_rank. Both upstream supervision and informed ETA stage labels use historical actual order. Calibration selector gain labels also use past completed actual-order stages.',
 'variants':'K=3 and K=5, both fixed and fully reported. 3 route OOF upstream folds, original 300-tree rank,60-round classifier,500-tree ETA and 200-tree gain regression; no tuning.',
 'budget':'One attempted query per route for gain,entropy,stage-disagreement,latest-ETA and uniform random exact expectation. Refusal gives baseline prediction; no second query.',
 'response_tests':'Perfect proxy response; independent25/50 percent refusal; refusal for normalized entropy above calibration75th/50th percentile; adjacent-stage error10/20 percent. Exact expectations. These are scenarios, not empirical human error/cost estimates.',
 'references':'Unqueried baseline; all recorded-plan-stage labels available; direct full recorded-plan ETA trained on original-plan history, explicitly privileged historical-information reference; future execution-stage oracle diagnostic.',
 'checks':'All-route time boundaries, route OOF isolation, no historical-plan input dependency, current forbidden-field perturbation plus row shuffle for feature/prediction invariance; stages for evaluation only accessed after scores frozen.',
 'decision':'Primary feasibility is actual-history-only gain selector vs fixed simple rules with SOFTWARE plan proxy answers. No claim of human feasibility; all variants reported without selecting on evaluation. Retain any failure.'}
if (O/'protocol.json').exists():assert json.loads((O/'protocol.json').read_text())==protocol
else:save('protocol.json',protocol)
shutil.copy2(__file__,O/'code/scenario_alignment.py')
rename={c:c.replace('plan_rank','execution_rank') for c in core.GENERATOR_FEATURES if 'plan_rank' in c}
GF=[rename.get(c,c) for c in core.GENERATOR_FEATURES];FF=GF+core.ORDER_FEATURES
def fold(x):return int(hashlib.sha256(('scenario_route|'+str(x)).encode()).hexdigest()[:12],16)%3
def historical(raw):
    z=raw.drop(columns=['IndexP','DistanceP'],errors='ignore').copy()
    # IndexA is known for completed historical deliveries only.
    z['IndexP']=z.IndexA
    return z
def history(q,enc,training=False):
    z=core.add_history_features(q,enc,training_crossfit=training).rename(columns=rename)
    z['execution_rank_target']=z.IndexA/(z.route_stop_count-1).clip(lower=1)
    return z
def stage(z,col,k):
    assert not z.duplicated(['Route ID',col]).any()
    rank=z.groupby('Route ID')[col].rank(method='first')-1
    denom=(z.groupby('Route ID')[col].transform('size')-1).clip(lower=1)
    return np.floor(np.clip((rank/denom).to_numpy(),0,1-1e-12)*k).astype(int)
def fit_up(q,k):
    p={**core.GENERATOR_MODEL_PARAMS,'n_jobs':6}
    a=LGBMRegressor(**p).fit(q[GF],q.execution_rank_target)
    b=LGBMClassifier(**{**p,'objective':'multiclass','num_class':k,'n_estimators':60}).fit(q[GF],stage(q,'IndexA',k))
    return a,b
def up(q,a,b,k):
    z=q.copy();pr=b.predict_proba(z[GF]);assert pr.shape[1]==k
    z['pred_hard_phase']=pr.argmax(axis=1);z['pred_expected_phase']=pr@np.arange(k);z['pred_rank_score']=a.predict(z[GF])
    for i in range(k):z[f'prob_{i}']=pr[:,i]
    keys=['Route ID','pred_rank_score','latest_rel','earliest_rel','Address ID','Stop ID','Delivery','Depot'];assert not z.duplicated(keys).any()
    order=z.sort_values(keys,kind='stable');z.loc[order.index,'generated_rank']=order.groupby('Route ID').cumcount().to_numpy()
    z['pred_rank_phase']=np.floor(np.clip(z.generated_rank/(z.route_stop_count-1).clip(lower=1),0,1-1e-12)*k).astype(int)
    return z
def fit_pipeline(raw,k,name):
    path=O/f'models/{name}.joblib'
    if path.exists():return joblib.load(path)
    start=time.perf_counter();q=historical(raw);folds=q['Route ID'].map(fold);parts=[]
    alt=raw.copy();alt['IndexP']=-12345;alt['DistanceP']=999999
    pd.testing.assert_frame_equal(historical(raw),historical(alt))
    for f in range(3):
        tr=q[folds.ne(f)].copy();held=q[folds.eq(f)].copy();assert not set(tr['Route ID'])&set(held['Route ID'])
        enc=core.fit_history_encoders(tr);a,b=fit_up(history(tr,enc,True),k);parts.append(up(history(held,enc),a,b,k))
    enc=core.fit_history_encoders(q);tr=history(q,enc,True);oof=pd.concat(parts).set_index('_rowkey')
    cols=['pred_hard_phase','pred_expected_phase','pred_rank_score','generated_rank','pred_rank_phase']+[f'prob_{i}' for i in range(k)]
    for c in cols:tr[c]=tr._rowkey.map(oof[c])
    assert tr[cols].notna().all().all()
    a,b=fit_up(tr,k);tr['answer_phase']=stage(tr,'IndexA',k);tr=core.add_sequence_features(tr,'generated_rank',None);tr=tr[tr.Delivery.eq(1)]
    pars={**core.FIXED_MODEL_PARAMS,'n_jobs':6}
    base=LGBMRegressor(**pars).fit(tr[FF],tr.target_minutes);aware=LGBMRegressor(**pars).fit(tr[FF+['answer_phase']],tr.target_minutes)
    m={'enc':enc,'rank':a,'cls':b,'base':base,'aware':aware,'k':k,'train_routes':list(q['Route ID'].unique()),'seconds':time.perf_counter()-start}
    joblib.dump(m,path);log(f'Fit {name}: {m["seconds"]:.1f}s');return m
def current_features(raw,m):
    assert not set(raw['Route ID'])&set(m['train_routes'])
    # Legacy core requires unique IndexP for bookkeeping. Use arbitrary local row numbers, not a true order.
    z=raw.drop(columns=['IndexP','DistanceP'],errors='ignore').copy();z['IndexP']=z.groupby('Route ID').cumcount()
    z=up(history(z,m['enc']),m['rank'],m['cls'],m['k']);z=core.add_sequence_features(z,'generated_rank',None);z=z[z.Delivery.eq(1)].copy().reset_index(drop=True)
    z['base_eta']=m['base'].predict(z[FF]);pr=z[[f'prob_{i}' for i in range(m['k'])]].to_numpy();z['entropy']=-(pr*np.log(np.clip(pr,1e-12,1))).sum(axis=1)/np.log(m['k'])
    return z
raw=[]
for s in ['training','validation','confirmatory_evaluation']:
    z=pd.read_parquet(P/f'work_experiments/planned_actual_v1/partitions/{s}.parquet');ok=z['_strict_primary'].astype(str).str.lower().eq('true');raw.append(z[ok].copy())
raw=pd.concat(raw,ignore_index=True);raw['_rowkey']=raw['Route ID'].astype(str)+'|'+raw.IndexP.astype(str);assert raw._rowkey.is_unique
weeks=raw.groupby('Route ID')['Week ID'].agg(['min','max'])
for origin,end in [(13,19),(19,25),(25,32)]:
    tr=raw[raw['Route ID'].isin(weeks.index[weeks['max']<origin-6])].copy();ca=raw[raw['Route ID'].isin(weeks.index[(weeks['min']>=origin-6)&(weeks['max']<origin)])].copy();ev=raw[raw['Route ID'].isin(weeks.index[(weeks['min']>=origin)&(weeks['max']<end)])].copy()
    assert tr['Week ID'].max()<ca['Week ID'].min()<ev['Week ID'].min()
    for k in [3,5]:
        name=f'w{origin}_k{k}'
        if (O/f'{name}_complete.json').exists():continue
        m=fit_pipeline(tr,k,name);cal=current_features(ca,m)
        truth=history(historical(ca),m['enc']);truth['actual_phase']=stage(truth,'IndexA',k);truth=truth.set_index('_rowkey');cal['answer_phase']=cal._rowkey.map(truth.actual_phase)
        cal['gain']=abs(cal.base_eta-cal.target_minutes)-abs(m['aware'].predict(cal[FF+['answer_phase']])-cal.target_minutes)
        sf=FF+['pred_rank_score','pred_expected_phase']+[f'prob_{i}' for i in range(k)]+['base_eta'];assert not any('plan_rank' in c for c in sf)
        p=dict(objective='regression',n_estimators=200,learning_rate=.04,num_leaves=15,min_child_samples=100,reg_lambda=10,random_state=20260913,n_jobs=6,deterministic=True,force_col_wise=True,verbosity=-1)
        w=1/cal.groupby('Route ID')['Route ID'].transform('size');sel=LGBMRegressor(**p).fit(cal[sf],cal.gain,sample_weight=w/w.mean())
        joblib.dump({'model':sel,'features':sf},O/f'models/{name}_selector.joblib')
        thresholds={p:float(cal.entropy.quantile(p)) for p in [.5,.75]}
        save(f'{name}_freeze.json',{'time':time.strftime('%Y-%m-%d %H:%M:%S'),'no_historical_plan_fields':True,'calibration_gain_source':'historical actual execution stage','refusal_thresholds':thresholds,'train_routes':tr['Route ID'].nunique(),'cal_routes':ca['Route ID'].nunique()})
        q=current_features(ev,m);q['gain_score']=sel.predict(q[sf]);q['disagreement']=abs(q.pred_hard_phase-q.pred_rank_phase);q['latest_eta']=q.base_eta
        alt=ev.sample(frac=1,random_state=20260913).copy();alt['IndexP']=-999;alt['IndexA']=-333;alt['Arrived Time']=888888;alt['DistanceA']=555555;alt['DistanceP']=666666
        qc=current_features(alt,m).set_index('_rowkey').reindex(q._rowkey)
        assert np.allclose(qc[FF].to_numpy(),q[FF].to_numpy(),equal_nan=True,atol=0,rtol=0)
        assert np.array_equal(qc.base_eta.to_numpy(),q.base_eta.to_numpy())
        assert np.array_equal(sel.predict(qc[sf]),q.gain_score.to_numpy())
        # Only after decision scores freeze do we derive software-plan proxy answers and future-stage diagnostics.
        truth=core.add_route_fields(ev);truth['planned_phase']=stage(truth,'IndexP',k);truth['actual_phase']=stage(truth,'IndexA',k);truth=truth.set_index('_rowkey')
        q['planned_phase']=q._rowkey.map(truth.planned_phase);q['actual_phase']=q._rowkey.map(truth.actual_phase)
        yp=[]
        for i in range(k):
            z=q[FF].copy();z['answer_phase']=i;yp.append(m['aware'].predict(z))
        yp=np.stack(yp,axis=1);y=q.target_minutes.to_numpy();base=abs(q.base_eta.to_numpy()-y);ix=np.arange(len(q));plan=q.planned_phase.to_numpy();act=q.actual_phase.to_numpy()
        clean=abs(yp[ix,plan]-y);oracle=abs(yp[ix,act]-y)
        l=np.maximum(plan-1,0);r=np.minimum(plan+1,k-1);adj=(abs(yp[ix,l]-y)+abs(yp[ix,r]-y))/2;adj=np.where(plan==0,abs(yp[:,1]-y),np.where(plan==k-1,abs(yp[:,k-2]-y),adj))
        conditions={'proxy_clean':(base-clean,np.ones(len(q))),'independent_refusal25':(.75*(base-clean),np.full(len(q),.75)),'independent_refusal50':(.5*(base-clean),np.full(len(q),.5)),'entropy_refusal25':((base-clean)*(q.entropy.to_numpy()<=thresholds[.75]),(q.entropy.to_numpy()<=thresholds[.75]).astype(float)),'entropy_refusal50':((base-clean)*(q.entropy.to_numpy()<=thresholds[.5]),(q.entropy.to_numpy()<=thresholds[.5]).astype(float)),'adjacent_error10':(base-(.9*clean+.1*adj),np.ones(len(q))),'adjacent_error20':(base-(.8*clean+.2*adj),np.ones(len(q))),'future_stage_oracle_DIAGNOSTIC':(base-oracle,np.ones(len(q)))}
        rc,routes=pd.factorize(q['Route ID'],sort=True);n=np.bincount(rc);groups=[np.flatnonzero(rc==j) for j in range(len(routes))]
        def means(v):return np.bincount(rc,weights=v)/n
        meta=q.groupby('Route ID').agg(driver=('Driver ID','first')).reindex(routes);meta.index.name='Route ID';meta['seen_driver']=meta.driver.isin(set(tr['Driver ID'])|set(ca['Driver ID']))
        tie=np.array([int(hashlib.sha256((str(a)+'|'+str(b)+'|'+str(c)).encode()).hexdigest()[:12],16) for a,b,c in zip(q['Route ID'],q['Stop ID'],q['Address ID'])]);records=[];choices=[]
        for method,col in [('random',None),('gain','gain_score'),('uncertainty','entropy'),('disagreement','disagreement'),('latest_eta','latest_eta')]:
            chosen=np.zeros(len(q),bool)
            if col:
                for ids in groups:chosen[ids[np.lexsort((tie[ids],-q[col].to_numpy()[ids]))[0]]]=True
                assert np.bincount(rc,weights=chosen).tolist()==[1]*len(routes)
                choices.append(pd.DataFrame({'_rowkey':q._rowkey,'method':method,'selected':chosen}))
            for condition,(gain,response) in conditions.items():
                reduction=means(gain)/n if col is None else means(gain*chosen)
                answers=means(response) if col is None else np.bincount(rc,weights=response*chosen)
                z=meta.copy();z['mae']=means(base)-reduction;z['base_mae']=means(base);z['method']=method;z['condition']=condition;z['answers']=answers;z['attempts']=1;z['origin']=origin;z['k']=k;records.append(z.reset_index())
        for method,error in [('all_proxy_stage',clean),('no_query',base),('all_future_stage_DIAGNOSTIC',oracle)]:
            z=meta.copy();z['mae']=means(error);z['base_mae']=means(base);z['method']=method;z['condition']='reference';z['answers']=n if method!='no_query' else 0;z['attempts']=z.answers;z['origin']=origin;z['k']=k;records.append(z.reset_index())
        q['proxy_matches_future']=(plan==act);q['base_error']=base;q['proxy_error']=clean;q['oracle_error']=oracle
        q.to_parquet(O/f'cache/{name}_evaluation.parquet',index=False);pd.concat(choices).to_parquet(O/f'tables/{name}_choices.parquet',index=False);pd.concat(records).to_parquet(O/f'tables/{name}_losses.parquet',index=False)
        save(f'{name}_complete.json',{'time':time.strftime('%Y-%m-%d %H:%M:%S'),'eval_routes':len(routes),'actual_history_only_adapter_invariance':True,'current_forbidden_fields_and_shuffle_full_features_invariance':True,'raw_proxy_execution_agreement':float(np.mean(plan==act))});log('Completed '+name)
    # Information-rich comparator: train on recorded historical full plan, evaluation full plan disclosed.
    ep=core.fit_history_encoders(tr);a=core.add_history_features(tr,ep,training_crossfit=True);a,ff=core.regime_frame(a,'P');b=core.add_history_features(ev,ep,training_crossfit=False);b,_=core.regime_frame(b,'P')
    ref=LGBMRegressor(**{**core.FIXED_MODEL_PARAMS,'n_jobs':6}).fit(a.loc[a.Delivery.eq(1),ff],a.loc[a.Delivery.eq(1),'target_minutes']);b=b[b.Delivery.eq(1)].copy();b['error']=abs(ref.predict(b[ff])-b.target_minutes)
    refrows=b.groupby('Route ID').agg(mae=('error','mean'),driver=('Driver ID','first')).reset_index();refrows['origin']=origin;refrows.to_csv(O/f'tables/w{origin}_privileged_full_plan.csv',index=False);joblib.dump({'model':ref,'features':ff,'encoders':ep},O/f'models/w{origin}_privileged_full_plan.joblib')
save('completion.json',{'time':time.strftime('%Y-%m-%d %H:%M:%S'),'windows':3,'stage_variants':[3,5],'primary_answer':'software plan proxy, not human report','no_historical_plans_in_primary_models':True})
allrows=pd.concat([pd.read_parquet(O/f'tables/w{o}_k{k}_losses.parquet') for o in [13,19,25] for k in [3,5]],ignore_index=True);allrows.to_parquet(O/'tables/all_losses.parquet',index=False)
print(allrows[allrows.condition.eq('proxy_clean')].groupby(['origin','k','method']).mae.mean().unstack().round(5).to_string())
