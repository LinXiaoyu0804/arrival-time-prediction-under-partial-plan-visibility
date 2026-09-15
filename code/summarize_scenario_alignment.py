from pathlib import Path
import json,shutil
import numpy as np
import pandas as pd
R=Path(__file__).resolve().parents[1];O=R/'outputs/TRE_scenario_alignment_20260913'
t=pd.read_parquet(O/'tables/all_losses.parquet');shutil.copy2(__file__,O/'code/summarize_scenario_alignment.py')
assert not t.duplicated(['origin','k','Route ID','method','condition']).any()
s=t.groupby(['k','method','condition'],as_index=False).agg(mae=('mae','mean'),base_mae=('base_mae','mean'),answers=('answers','mean'),routes=('Route ID','size'));s.to_csv(O/'tables/summary.csv',index=False)
window=t.groupby(['origin','k','method','condition'],as_index=False).mae.mean();window.to_csv(O/'tables/window_summary.csv',index=False)
ids=t[['origin','Route ID','driver']].drop_duplicates().sort_values(['origin','Route ID']);assert ids['Route ID'].is_unique
idx=pd.MultiIndex.from_frame(ids[['origin','Route ID']]);codes,drivers=pd.factorize(ids.driver);counts=np.bincount(codes);boot=np.random.default_rng(20260913).multinomial(len(drivers),np.ones(len(drivers))/len(drivers),size=2000)
def ci(d):
    samples=(boot@np.bincount(codes,weights=d))/(boot@counts)
    return float(np.mean(d)),*np.quantile(samples,[.025,.975]).tolist()
rows=[]
for k in [3,5]:
    for condition in t.condition.unique():
        if condition=='reference':continue
        q=t[(t.k==k)&(t.condition==condition)];p=q.pivot(index=['origin','Route ID'],columns='method',values='mae').reindex(idx)
        p['no_query']=q.groupby(['origin','Route ID']).base_mae.first().reindex(idx)
        for ref in ['random','uncertainty','disagreement','latest_eta','no_query']:
            d,lo,hi=ci((p.gain-p[ref]).to_numpy());rows.append(dict(k=k,condition=condition,reference=ref,difference=d,low=lo,high=hi))
ints=pd.DataFrame(rows);ints.to_csv(O/'tables/paired_intervals.csv',index=False)
p=t[(t.method=='gain')&(t.condition=='proxy_clean')].pivot(index=['origin','Route ID'],columns='k',values='mae').reindex(idx);d,lo,hi=ci((p[3]-p[5]).to_numpy())
stageci={'difference_3_minus_5':d,'low':lo,'high':hi};(O/'stage_comparison.json').write_text(json.dumps(stageci,indent=2))
full=pd.concat([pd.read_csv(O/f'tables/w{x}_privileged_full_plan.csv') for x in [13,19,25]])
full.groupby('origin').mae.mean().to_csv(O/'tables/full_plan_summary.csv')
agreement=[]
for k in [3,5]:
    q=pd.concat([pd.read_parquet(O/f'cache/w{x}_k{k}_evaluation.parquet',columns=['Route ID','proxy_matches_future']) for x in [13,19,25]])
    agreement.append(dict(k=k,event_agreement=q.proxy_matches_future.mean(),route_agreement=q.groupby('Route ID').proxy_matches_future.mean().mean()))
pd.DataFrame(agreement).to_csv(O/'tables/proxy_execution_agreement.csv',index=False)
def val(k,m,c='proxy_clean',field='mae'):return float(s[(s.k==k)&(s.method==m)&(s.condition==c)][field].iloc[0])
names={'no_query':'不询问','random':'随机','uncertainty':'不确定性','disagreement':'阶段分歧','latest_eta':'预计最晚完成','gain':'收益选择','all_proxy_stage':'全部软件计划阶段可用','all_future_stage_DIAGNOSTIC':'全部未来执行阶段（不可部署诊断）'}
table=[]
for m,n in names.items():
    c='reference' if m in ['no_query','all_proxy_stage','all_future_stage_DIAGNOSTIC'] else 'proxy_clean'
    table.append(f'| {n} | {val(3,m,c):.3f} | {val(5,m,c):.3f} |')
ct=[]
for _,r in ints[ints.condition.eq('proxy_clean')].iterrows():ct.append(f'| {r.k} | {names[r.reference]} | {r.difference:.3f} | [{r.low:.3f}, {r.high:.3f}] |')
stressnames={'proxy_clean':'代理答案无附加错误','independent_refusal25':'独立25%拒答','independent_refusal50':'独立50%拒答','entropy_refusal25':'高熵拒答：校准75分位阈值','entropy_refusal50':'高熵拒答：校准50分位阈值','adjacent_error10':'10%相邻阶段误报','adjacent_error20':'20%相邻阶段误报'}
stress=[]
for c,label in stressnames.items():
    for k in [3,5]:stress.append(f'| {k} | {label} | {val(k,"gain",c):.3f} | {val(k,"random",c):.3f} | {val(k,"uncertainty",c):.3f} | {val(k,"disagreement",c):.3f} | {val(k,"gain",c,"answers"):.1%} |')
out=f'''# 场景对齐验证：只用历史实际配送记录

2026-09-13。本轮完成三个时间窗口、三阶段与五阶段、五种查询策略及七种回答情境。目标是检验方法能否减少对历史完整软件计划的依赖，不是将公开软件计划数据冒充司机访谈数据。

## 结论

历史计划依赖已在本轮主模型中移除：收益选择在三组时间窗口、两种阶段数下的平均MAE均低于随机、不确定性、阶段分歧和预计最晚完成规则。合并区间见下表。

**这补上了历史输入的场景一致性；司机是否能准确、低负担地回答仍没有实测。** 训练中的阶段来自过去实际完成记录，评估中的回答来自当前软件计划，两者存在来源差异。本轮正是在这种差异下检验迁移，不将它写成已经观测到的人工意图预测。

## 1. 历史计划如何被彻底移出主流程

历史IndexP及DistanceP在主模型训练前删除。为复用核心处理函数，将过去已完成的IndexA映射到旧函数的排序输入槽，再将相关特征显式重命名为execution_rank。排序监督、阶段分类监督、包含阶段的ETA训练及校准期收益标签，都使用过去的实际完成顺序。

这不是只删除两个历史均值特征：上游监督目标和下游收益标签来源也一并重建。过去实际顺序是已完成记录，不是当前路线未来信息。当前IndexA、到达时间及距离在预测阶段不进入模型输入。

训练三个窗口沿用0—6/7—12/13—18周、0—12/13—18/19—24周、0—18/19—24/25—31周，并保留整条路线边界隔离。六套流程均使用三折路线折外上游预测及固定参数，没有按本轮结果调参。

身份匹配仍保留原行键作为元数据；它不进入模型特征或查询评分。当前核心函数所需的IndexP唯一键由无顺序含义的行计数替代。打乱行序并改写当前计划、实际顺序、到达时间和距离后，整套允许特征、基线预测和收益评分保持不变。改写历史计划值后，历史训练适配输入完全相同。

## 2. 查询与回答的准确含义

每条路线尝试询问一个任务；拒答后回到原预测，不补问第二个任务。收益选择器在看到当前回答和实际结果前冻结。

评估回答由数据中的软件计划顺序IndexP分成三段或五段，作为“可能获取的前瞻计划信息”的代理。它并不是司机真实报告。分段表示整趟停靠序列的相对位置，包含数据中的仓库/取送停靠，不是上午、中午、下午，也不等于等长时间段。现场语言与停靠计数规则须先统一。

未来实际执行阶段IndexA仅用于不可部署诊断，绝不作为在线回答输入。软件计划与实际执行阶段不一致，也不等于司机回答错误，因为计划可能在执行中发生改变。

## 3. 无附加拒答/误报时的结果

4112条互不重复的评估路线，261个司机簇。MAE按路线等权，单位分钟。

| 策略 | 三阶段 | 五阶段 |
|---|---:|---:|
'''+ '\n'.join(table)+f'''

三阶段收益选择减五阶段收益选择为{d:.3f}分钟，95%区间[{lo:.3f},{hi:.3f}]。该区间用于比较本轮整体流程，不证明两种标签等效；两套阶段分类器及ETA也分别训练。不能把两方案的差异全部归于人更容易回答。

额外保留了历史及当前完整软件计划可用的特权参考，合并MAE为{full.mae.mean():.3f}分钟。这项参考拥有主模型不具备的历史计划和当前完整顺序/距离信息，用于显示信息充分时的表现，不是同预算公平基线。其成绩不能被隐藏，也不能用于声称少量查询已替代完整数字化。

## 4. 配对区间

下表为收益选择减对应对照，负值有利。261个司机簇配对bootstrap2000次，同一司机跨窗口一起抽样；普通探索性95%区间，未作多模型选择后的确认性声明。

| 阶段数 | 对照 | 差值 | 95%区间 |
|---|---|---:|---|
'''+ '\n'.join(ct)+'''

## 5. 无法回答与误报压力测试

独立拒答按给定概率精确求期望。高熵拒答的阈值只用校准期确定，表示模型不确定性较高的任务更可能无法获得回答；它不是已观测的人类回应机制。不同方法问到的任务不同，实际回答率因此不同，必须同时看预测结果与回答率。

| 阶段数 | 情境 | 收益选择MAE | 随机 | 不确定性 | 阶段分歧 | 收益选择有效回答率 |
|---|---|---:|---:|---:|---:|---:|
'''+ '\n'.join(stress)+'''

相邻误报只是一类噪声。若错误与未记录偏好、计划变化或任务收益相关，当前压力测试不能穷尽其影响。拒答与模型不确定性相关时，也不能把不同实际回答量的差异写成“同等有效信息量”下的优势；这里公平固定的是尝试询问次数。

## 6. 现场之前与现场之后的分界

已经完成：历史计划依赖移除、三/五阶段比较、拒答与误报、完整信息参考、时间隔离、特征与评分扰动审计。

尚未观测：司机是否事先有意图、能否使用统一阶段定义回答、回答多久、答错的结构、计划后续如何改变、完整顺序提交所需负担。另一份《现场核查方案.md》规定了这些问题的记录方式，不要求把未来实际顺序当成回答真值。

本轮没有访问或联系案例企业，没有开展访谈。现有结果应定位为真实业务动因下的离线桥接证据，而非现场验证。

## 归档

protocol.json记录运行前规则；每个窗口阶段组合的freeze/complete文件记录阈值与检查；models/保存主模型及明确标识的完整计划特权参考；cache/保存逐任务输出；tables/保存全部情境、查询、区间与参考。code/保存复用入口。
'''
(O/'场景对齐验证结论.md').write_text(out,encoding='utf8')
print(s[(s.condition=='proxy_clean')|((s.condition=='reference')&(s.method=='no_query'))].to_string(index=False));print(ints[ints.condition=='proxy_clean'].to_string(index=False));print(stageci);print('full plan',full.mae.mean());print(s[s.method.eq('gain')].to_string(index=False))
