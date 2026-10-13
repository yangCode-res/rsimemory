"""Run baselines, train-only reflection, three code branches and validation."""
from __future__ import annotations
import argparse
import hashlib
import inspect
import json
from pathlib import Path
import subprocess
import sys
from .data import load_medmemory, patient_splits, representative_queries
from .evolution import reflect, generate, new_node, backprop, pareto
from .fixtures import FixtureClient, dataset as fixture_dataset
from .model import ModelClient
from .policy import PlainPolicy, GeneratedPolicy
from .runtime import replay, dump, intervention, call_cost


def commit_policy(repo, code, label, parent=None):
    repo.mkdir(exist_ok=True)
    if not (repo/'.git').exists():
        subprocess.run(['git','init','-q',str(repo)],check=True)
    source=repo/(label+'.py'); source.write_text(code)
    def git(args,data=None):
        return subprocess.check_output(['git','-C',str(repo),'-c','user.name=RSI Experiment',
                                        '-c','user.email=rsi@local',*args],input=data).decode().strip()
    blob=git(['hash-object','-w',str(source.resolve())])
    tree=git(['mktree'],f'100644 blob {blob}\tpolicy.py\n'.encode())
    args=['commit-tree',tree,'-m',label]
    if parent: args.extend(['-p',parent])
    commit=git(args);git(['update-ref','refs/heads/'+label,commit])
    return commit


def comparison_metric(report):
    cost=report['cost']['policy']
    def seconds(row):
        if 'sandbox_seconds' in row: return row['sandbox_seconds']
        return row.get('ingest_seconds',0)+row.get('search_seconds',0)
    return {'AVG':report['AVG'],'policy_tokens':0,
            'policy_seconds':sum(seconds(r) for r in cost)+report['cost'].get('search_seconds',0.),
            'policy_bytes':max([r.get('peak_storage_bytes',r.get('storage_bytes',0)) for r in cost] or [0])}


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--dataset')
    p.add_argument('--output',required=True)
    p.add_argument('--language',choices=['zh','en'],default='zh')
    p.add_argument('--fixture',action='store_true')
    p.add_argument('--base-url',default='http://127.0.0.1:7860/v1')
    p.add_argument('--judge-url')
    p.add_argument('--model',default='qwen3.8-27b')
    p.add_argument('--judge-model')
    p.add_argument('--checkpoint',type=int)
    p.add_argument('--patients-per-split',type=int)
    p.add_argument('--per-type',type=int)
    p.add_argument('--evolve',action='store_true')
    p.add_argument('--no-diagnosis',action='store_true')
    p.add_argument('--semantic-model-path')
    p.add_argument('--max-interventions',type=int,default=2)
    args=p.parse_args()
    out=Path(args.output)
    if out.exists(): p.error('Use a new output directory; never overwrite runs')
    if not args.fixture and not args.dataset: p.error('--dataset is required')
    config={'chunk_chars':1200,'chunk_overlap':120,'top_k':5,'max_steps':4,'context_chars':16000,
            'mode':'Efficient','seed':20261013,'language':args.language,
            'fixture':args.fixture,'model':args.model,'judge_model':args.judge_model or args.model,
            'speculation':'disabled','policy_auxiliary_model':'disabled',**vars(args)}
    config['judge_model']=args.judge_model or args.model
    config['actor_max_tokens']=1024;config['optimizer_max_tokens']=3072
    try:
        repo=Path(__file__).resolve().parent.parent
        config['implementation_commit']=subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],stderr=subprocess.DEVNULL).decode().strip()
        config['implementation_dirty']=bool(subprocess.check_output(['git','-C',str(repo),'status','--porcelain']).strip())
    except subprocess.CalledProcessError:
        config['implementation_commit']=None;config['implementation_dirty']=True
    if args.dataset:
        manifest=Path(args.dataset)/'download_manifest.json'
        config['data_revision']=json.loads(manifest.read_text()).get('revision') if manifest.exists() else 'unverified'
    if args.semantic_model_path:
        manifest=Path(args.semantic_model_path)/'rsi-model-manifest.json'
        config['semantic_manifest']=json.loads(manifest.read_text()) if manifest.exists() else {'revision':'unverified'}
    if args.fixture:
        sessions,queries=fixture_dataset();actor_client=FixtureClient();judge_client=FixtureClient();optimizer=FixtureClient()
    else:
        sessions,queries=load_medmemory(args.dataset,args.language)
        actor_client=ModelClient(args.base_url,args.model,max_tokens=1024)
        judge_client=ModelClient(args.judge_url or args.base_url,args.judge_model or args.model,max_tokens=1024)
        optimizer=ModelClient(args.base_url,args.model,max_tokens=3072)
    split=patient_splits([s['persona_id'] for s in sessions],config['seed'])
    # Diagnostic subset keeps test sealed and stays within each original split.
    run_ids={k:v[:args.patients_per_split] if args.patients_per_split else v for k,v in split.items() if k!='test'}
    run_queries={k:representative_queries(queries,v,args.checkpoint,args.per_type) for k,v in run_ids.items()}
    out.mkdir(parents=True)
    dump(out/'config.json',config);dump(out/'splits.json',split);dump(out/'run_patients.json',run_ids)
    dump(out/'config-hash.json',{'sha256':hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest()})
    root_code=inspect.getsource(PlainPolicy)
    root_commit=commit_policy(out/'policy-git',root_code,'plain-bm25')
    tree={'algorithm':'MCTS/UCT single expansion','nodes':{'plain-bm25':new_node('plain-bm25')},'c':1.414}
    tree['nodes']['plain-bm25']['commit']=root_commit
    train,train_rows=replay(sessions,run_queries['train'],run_ids['train'],lambda:PlainPolicy(config),
                            actor_client,judge_client,config,out/'plain-bm25'/'train')
    validation,_=replay(sessions,run_queries['validation'],run_ids['validation'],lambda:PlainPolicy(config),
                        actor_client,judge_client,config,out/'plain-bm25'/'validation')
    tree['nodes']['plain-bm25']['status']='evaluated' if train['AVG'] is not None and validation['AVG'] is not None else 'incomplete_scoring'
    comparisons={'plain-bm25':comparison_metric(validation)}
    semantic={'status':'skipped','reason':'no frozen semantic model supplied'}
    if args.semantic_model_path:
        sc=dict(config,semantic_model_path=args.semantic_model_path,semantic_weight=1.,bm25_weight=0.)
        try:
            st,_=replay(sessions,run_queries['train'],run_ids['train'],lambda:PlainPolicy(sc),actor_client,judge_client,sc,out/'plain-semantic'/'train')
            sv,_=replay(sessions,run_queries['validation'],run_ids['validation'],lambda:PlainPolicy(sc),actor_client,judge_client,sc,out/'plain-semantic'/'validation')
            comparisons['plain-semantic']=comparison_metric(sv)
            semantic={'status':'evaluated' if st['AVG'] is not None and sv['AVG'] is not None else 'incomplete_scoring'}
            comparisons['plain-semantic']['valid']=semantic['status']=='evaluated'
        except Exception as exc: semantic={'status':'failed','reason':str(exc)}
    dump(out/'semantic-status.json',semantic)
    lookup={q['query_id']:q for q in run_queries['train']}
    branches=[];seen_code={}
    if args.evolve:
        interventions=intervention(train_rows,lookup,actor_client,judge_client,sessions,args.max_interventions) if args.max_interventions>0 else []
        dump(out/'interventions.json',interventions)
        try:
            diagnosis=reflect(optimizer,train_rows,lookup,interventions,root_code,not args.no_diagnosis)
            dump(out/'reflexion.json',diagnosis)
        except Exception as exc:
            diagnosis={'proposals': [], 'status':'failed'}
            dump(out/'reflexion-failure.json',{'reason':f'{type(exc).__name__}: {exc}'})
        for i,proposal in enumerate(diagnosis['proposals']):
            name=f'candidate-{i+1}';node=new_node(name,'plain-bm25');node['proposal']=proposal;tree['nodes'][name]=node
            branch=out/'branches'/name;branch.mkdir(parents=True)
            dump(branch/'proposal.json',proposal)
            try:
                source,cost=generate(optimizer,proposal,root_code)
                (branch/'policy.py').write_text(source);dump(branch/'codegen-cost.json',cost)
                code_hash=hashlib.sha256(source.encode()).hexdigest()
                if code_hash in seen_code and not args.fixture:
                    raise ValueError('Duplicate generated code: '+seen_code[code_hash])
                seen_code[code_hash]=name
                node['commit']=commit_policy(out/'policy-git',source,name,root_commit)
                factory=lambda source=source:GeneratedPolicy(source,config,timeout_s=5.,memory_bytes=256*1024*1024)
                # Constructor verifies mandatory OS isolation before any replay.
                factory()
                t,_=replay(sessions,run_queries['train'],run_ids['train'],factory,actor_client,judge_client,config,branch/'train')
                v,_=replay(sessions,run_queries['validation'],run_ids['validation'],factory,actor_client,judge_client,config,branch/'validation')
                node['status']='evaluated' if t['AVG'] is not None and v['AVG'] is not None else 'incomplete_scoring'
                node['train_AVG']=t['AVG'];node['validation_AVG']=v['AVG']
                if node['status']=='evaluated': backprop(tree,name,t['AVG'])
                comparisons[name]=comparison_metric(v)
                comparisons[name]['valid']=node['status']=='evaluated'
            except Exception as exc:
                if hasattr(exc,'candidate_source'):
                    (branch/'policy.py').write_text(exc.candidate_source)
                    dump(branch/'codegen-cost.json',exc.codegen_cost)
                    node['commit']=commit_policy(out/'policy-git',exc.candidate_source,name,root_commit)
                node['status']='invalid';node['failure']=f'{type(exc).__name__}: {exc}'
                dump(branch/'failure.json',{'reason':node['failure']})
            dump(out/'tree.json',tree)
            branches.append(node)
    dump(out/'tree.json',tree)
    summary={'fixture':args.fixture,'scientific_result':False if args.fixture else 'engineering rubric; uncalibrated',
             'comparison':comparisons,'pareto':pareto(comparisons),'branches':branches,
             'semantic':semantic,'cost':{'optimizer':call_cost(optimizer.calls),
                                       'intervention':call_cost([c for r in (interventions if args.evolve else []) for c in r['call_log']])},
             'test_evaluated':False,'no_diagnosis_control':args.no_diagnosis}
    dump(out/'summary.json',summary)
    lines=['# 工程运行报告', '', '此运行用于协议和工程验证，未经过论文级 judge 校准。', '',
           '|版本|验证 AVG|Policy 秒数|Policy 峰值字节|', '|---|---:|---:|---:|']
    for name,r in comparisons.items():
        lines.append(f'|{name}|{r["AVG"]}|{r["policy_seconds"]:.3f}|{r["policy_bytes"]}|')
    lines.extend(['', '合成 fixture：'+str(args.fixture), '测试集评估：False',
                  '语义基线状态：'+semantic['status'], '保留的 Pareto 候选：'+', '.join(summary['pareto'])])
    for node in branches:
        lines.append(node['id']+': '+node['status']+(' — '+node['failure'] if 'failure' in node else ''))
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n')
    # Console exposes only aggregate status, never medical records or patient IDs.
    print(json.dumps({'run':str(out),'fixture':args.fixture,'branches':len(branches),
                      'valid_candidates':sum(n['status']=='evaluated' for n in branches),
                      'test_evaluated':False}))


if __name__=='__main__': main()
