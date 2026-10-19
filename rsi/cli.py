"""Run baselines, train-only reflection, three code branches and validation."""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import signal
from pathlib import Path
import subprocess
import sys
from importlib.metadata import version, PackageNotFoundError
from urllib.error import URLError
from .data import load_medmemory, patient_splits, representative_queries, SPLIT_VERSION
from .evolution import reflect, generate, new_node, backprop, pareto
from .fixtures import FixtureClient, dataset as fixture_dataset
from .model import ModelClient, ModelServiceUnavailable
from .policy import PlainPolicy, GeneratedPolicy
from .policy_source import baseline_source, source_manifest, SOURCE_VERSION
from .runtime import replay, dump, intervention, call_cost
from .tracing import trace_run, trace_span, write_failure_count


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
    return {'AVG':report['AVG'], 'E2E_AVG':report.get('E2E_AVG'),
            'abnormal_rate':report.get('abnormal_rate'),
            'unanswered_as_zero':report.get('counts',{}).get('unanswered_as_zero'),
            'policy_tokens':0,
            'policy_seconds':sum(seconds(r) for r in cost)+report['cost'].get('search_seconds',0.),
            'policy_bytes':max([r.get('peak_storage_bytes',r.get('storage_bytes',0)) for r in cost] or [0])}


def _interrupt_replay(signum, frame):
    # The supervisor stops infrastructure failures through normal exception
    # cleanup, so an active replay retains its failed progress and flushed rows.
    raise ModelServiceUnavailable('Replay interrupted by supervisor')


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
    p.add_argument('--candidate-timeout-s',type=float,default=30.)
    p.add_argument('--candidate-memory-mb',type=int,default=2048)
    p.add_argument('--reflexion-max-tokens',type=int,default=8192)
    p.add_argument('--codegen-max-tokens',type=int,default=16384)
    p.add_argument('--reflexion-temperature',type=float,default=0.7)
    p.add_argument('--codegen-temperature',type=float,default=0.4)
    p.add_argument('--optimizer-timeout-s',type=float,default=900.)
    args=p.parse_args()
    if min(args.reflexion_max_tokens,args.codegen_max_tokens)<=0:
        p.error('Optimizer token budgets must be positive')
    if not all(math.isfinite(t) and 0<=t<=2 for t in
               (args.reflexion_temperature,args.codegen_temperature)):
        p.error('Optimizer temperatures must be finite and between 0 and 2')
    if not math.isfinite(args.optimizer_timeout_s) or args.optimizer_timeout_s<=0:
        p.error('Optimizer timeout must be finite and positive')
    signal.signal(signal.SIGTERM, _interrupt_replay)
    out=Path(args.output)
    if out.exists(): p.error('Use a new output directory; never overwrite runs')
    if not args.fixture and not args.dataset: p.error('--dataset is required')
    config={'contract_version':'v0.3','interface_version':'v0.3','state_serializer':'cloudpickle-base64-v1','chunk_chars':1200,'chunk_overlap':120,'top_k':5,'max_steps':4,'context_chars':16000,
            'mode':'Efficient','seed':20261013,'split_version':SPLIT_VERSION,
            'split_fractions':{'train':.4,'validation':.2,'test':.4},'language':args.language,
            'fixture':args.fixture,'model':args.model,'judge_model':args.judge_model or args.model,
            'speculation':'disabled','policy_auxiliary_model':'disabled',**vars(args)}
    config['judge_model']=args.judge_model or args.model
    config['evaluation_metric_version']='strict-plus-e2e-v1'
    config['actor_max_tokens']=1024
    config['optimizer_generation_settings']={
        'reflexion':{'max_tokens':args.reflexion_max_tokens,'temperature':args.reflexion_temperature,
                     'timeout_s':args.optimizer_timeout_s},
        'codegen':{'max_tokens':args.codegen_max_tokens,'temperature':args.codegen_temperature,
                   'timeout_s':args.optimizer_timeout_s}}
    config['observability_schema']='local-spans-v1';config['rationale_schema']='evidence-v3'
    config['evolution_prompt_version']='explore-v2'
    config['parent_source_version']=SOURCE_VERSION
    config['python_version']=sys.version.split()[0]
    try: config['cloudpickle_version']=version('cloudpickle')
    except PackageNotFoundError: config['cloudpickle_version']=None
    config['worker_isolation']='Landlock-read-only-runtime + seccomp'
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
        optimizer=ModelClient(args.base_url,args.model,
                              role_settings=config['optimizer_generation_settings'])
    split=patient_splits([s['persona_id'] for s in sessions],config['seed'])
    config['split_patient_counts']={k:len(v) for k,v in split.items()}
    # Diagnostic subset keeps test sealed and stays within each original split.
    run_ids={k:v[:args.patients_per_split] if args.patients_per_split else v for k,v in split.items() if k!='test'}
    run_queries={k:representative_queries(queries,v,args.checkpoint,args.per_type) for k,v in run_ids.items()}
    out.mkdir(parents=True)
    with trace_run(out/'traces.jsonl', run_id=out.name,
                   metadata={'implementation_commit':config['implementation_commit'],
                             'contract_version':config['contract_version'], 'fixture':args.fixture,
                             'rationale_schema':config['rationale_schema']}) as round_span:
        round_span.update(input={'split_patient_counts':config['split_patient_counts'],
                                 'objective':'improve memory policy under a frozen actor',
                                 'language':args.language})
        summary=_run_round(args,out,config,split,run_ids,run_queries,sessions,actor_client,judge_client,optimizer)
        round_span.update(output=summary, metadata={'trace_write_failures':write_failure_count()})


def _run_round(args,out,config,split,run_ids,run_queries,sessions,actor_client,judge_client,optimizer):
    root_code=baseline_source()
    parent_manifest=source_manifest(root_code, config['language'])
    config['parent_source_sha256']=parent_manifest['sha256']
    dump(out/'config.json',config);dump(out/'splits.json',split);dump(out/'run_patients.json',run_ids)
    dump(out/'config-hash.json',{'sha256':hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest()})
    (out/'parent-policy.py').write_text(root_code)
    dump(out/'parent-source.json',parent_manifest)
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
        except ModelServiceUnavailable:
            raise
        except Exception as exc: semantic={'status':'failed','reason':str(exc)}
    dump(out/'semantic-status.json',semantic)
    lookup={q['query_id']:q for q in run_queries['train']}
    branches=[];seen_code={}
    if args.evolve:
        interventions=intervention(train_rows,lookup,actor_client,judge_client,sessions,args.max_interventions) if args.max_interventions>0 else []
        dump(out/'interventions.json',interventions)
        try:
            context={'active_parent_config':{
                'retrieval_mode':'lexical_bm25','semantic_weight':0.,
                'language':config['language'],
                'baseline_tokenization':parent_manifest['baseline_tokenization'],
                'parent_source_version':parent_manifest['version'],
                'policy_auxiliary_model':config['policy_auxiliary_model'],
                **{k:config[k] for k in ('top_k','max_steps','context_chars','chunk_chars','chunk_overlap')}},
                'parent_train_metrics':{'comparison':comparison_metric(train),
                    'type_accuracy':train.get('type_accuracy',{}),'counts':train.get('counts',{})}}
            diagnosis=reflect(optimizer,train_rows,lookup,interventions,root_code,
                              not args.no_diagnosis,policy_context=context)
            dump(out/'reflexion.json',diagnosis)
        except ModelServiceUnavailable:
            raise
        except Exception as exc:
            diagnosis={'proposals': [], 'status':'failed'}
            dump(out/'reflexion-failure.json',{'reason':f'{type(exc).__name__}: {exc}'})
        for i,proposal in enumerate(diagnosis['proposals']):
            name=f'candidate-{i+1}';node=new_node(name,'plain-bm25');node['proposal']=proposal;tree['nodes'][name]=node
            with trace_span('evolution.proposal', input=proposal,
                            metadata={'policy':name, 'parent_policy':'plain-bm25',
                                      'rationale_status':'model_reported_hypothesis'}) as observation:
                observation.update(output={'basis':proposal.get('basis'), 'rationale':proposal.get('rationale'),
                                           'mechanism':proposal.get('mechanism'),
                                           'falsification_test':proposal.get('falsification_test'),
                                           'supporting_case_ids':proposal.get('supporting_case_ids',[]),
                                           'supporting_cluster_labels':proposal.get('supporting_cluster_labels',[]),
                                           'uncertainty':proposal.get('uncertainty')})
            branch=out/'branches'/name;branch.mkdir(parents=True)
            dump(branch/'proposal.json',proposal)
            stage='codegen'
            try:
                with trace_span('candidate.codegen', metadata={'policy':name, 'parent_policy':'plain-bm25'},
                                input={'proposal':proposal}) as observation:
                    source,cost=generate(optimizer,proposal,root_code,
                                         policy_context=context['active_parent_config'])
                    observation.update(output={'source_sha256':hashlib.sha256(source.encode()).hexdigest()})
                (branch/'policy.py').write_text(source);dump(branch/'codegen-cost.json',cost)
                code_hash=hashlib.sha256(source.encode()).hexdigest()
                if code_hash in seen_code and not args.fixture:
                    raise ValueError('Duplicate generated code: '+seen_code[code_hash])
                seen_code[code_hash]=name
                node['commit']=commit_policy(out/'policy-git',source,name,root_commit)
                factory=lambda source=source:GeneratedPolicy(source,config,timeout_s=args.candidate_timeout_s,memory_bytes=args.candidate_memory_mb*1024*1024)
                # Constructor verifies mandatory OS isolation before any replay.
                stage='initialize'
                factory()
                stage='train'
                t,_=replay(sessions,run_queries['train'],run_ids['train'],factory,actor_client,judge_client,config,branch/'train')
                stage='validation'
                v,_=replay(sessions,run_queries['validation'],run_ids['validation'],factory,actor_client,judge_client,config,branch/'validation')
                node['status']='evaluated' if t['AVG'] is not None and v['AVG'] is not None else 'incomplete_scoring'
                node['train_AVG']=t['AVG'];node['validation_AVG']=v['AVG']
                if node['status']=='evaluated': backprop(tree,name,t['AVG'])
                comparisons[name]=comparison_metric(v)
                comparisons[name]['valid']=node['status']=='evaluated'
            except ModelServiceUnavailable:
                raise
            except Exception as exc:
                if hasattr(exc,'candidate_source'):
                    (branch/'policy.py').write_text(exc.candidate_source)
                    dump(branch/'codegen-cost.json',exc.codegen_cost)
                    node['commit']=commit_policy(out/'policy-git',exc.candidate_source,name,root_commit)
                node['status']='invalid';node['failure']=f'{type(exc).__name__}: {exc}'
                node['failure_category']=getattr(exc,'category','environment' if isinstance(exc,URLError) else 'contract' if stage=='codegen' else 'runtime')
                node['failure_stage']=stage
                dump(branch/'failure.json',{'reason':node['failure'],'category':node['failure_category'],
                                           'stage':stage,'policy_cost':getattr(exc,'policy_cost',None)})
            dump(out/'tree.json',tree)
            with trace_span('candidate.result', metadata={'policy':name}) as observation:
                observation.update(output=node)
            branches.append(node)
    dump(out/'tree.json',tree)
    summary={'contract_version':config['contract_version'],'fixture':args.fixture,'scientific_result':False if args.fixture else 'engineering rubric; uncalibrated',
             'comparison':comparisons,'pareto':pareto(comparisons),'branches':branches,
             'semantic':semantic,'cost':{'optimizer':call_cost(optimizer.calls),
                                       'intervention':call_cost([c for r in (interventions if args.evolve else []) for c in r['call_log']])},
             'test_evaluated':False,'no_diagnosis_control':args.no_diagnosis}
    dump(out/'summary.json',summary)
    with trace_span('round.result') as observation:
        observation.update(output=summary)
    lines=['# 工程运行报告', '', '此运行用于协议和工程验证，未经过论文级 judge 校准。',
           'E2E AVG 将 actor 未作答计为 0，保留六类等权平均；judge 异常、缺类仍为 null。严格 AVG、UCT/Pareto 规则保持原口径。', '',
           '|版本|严格验证 AVG|E2E 验证 AVG|异常率|Policy 秒数|Policy 峰值字节|',
           '|---|---:|---:|---:|---:|---:|']
    for name,r in comparisons.items():
        lines.append(f'|{name}|{r["AVG"]}|{r["E2E_AVG"]}|{r["abnormal_rate"]}|{r["policy_seconds"]:.3f}|{r["policy_bytes"]}|')
    lines.extend(['', '合成 fixture：'+str(args.fixture), '测试集评估：False',
                  '语义基线状态：'+semantic['status'], '保留的 Pareto 候选：'+', '.join(summary['pareto'])])
    for node in branches:
        lines.append(node['id']+': '+node['status']+(' — '+node['failure'] if 'failure' in node else ''))
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n')
    # Console exposes only aggregate status, never medical records or patient IDs.
    print(json.dumps({'run':str(out),'fixture':args.fixture,'branches':len(branches),
                      'valid_candidates':sum(n['status']=='evaluated' for n in branches),
                      'test_evaluated':False}))
    return summary


if __name__=='__main__': main()
