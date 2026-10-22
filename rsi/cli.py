"""Run baselines, train-only reflection, three code branches and validation."""
from __future__ import annotations
import argparse
import copy
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextvars import copy_context
import threading
import time
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
from .evolution import reflect, generate, new_node, backprop, pareto, REFLECTION_PROMPT_VERSION
from .diagnostics import build_training_diagnostics
from .fixtures import FixtureClient, dataset as fixture_dataset
from .model import ModelClient, ModelServiceUnavailable
from .policy import PlainPolicy, GeneratedPolicy
from .policy_source import baseline_source, source_manifest, SOURCE_VERSION
from .runtime import replay, dump, intervention, call_cost
from .tracing import trace_run, trace_span, write_failure_count
from .resume import baseline_runtime_fingerprint, verify_baseline_resume, materialize_baseline_resume


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


def _set_stage(out, stage):
    dump(out/'run-status.json', {'status':'running', 'stage':stage, 'test_evaluated':False})


def _record_round_failure(out, exc, optimizer=None):
    status_path=out/'run-status.json'
    current=json.loads(status_path.read_text()) if status_path.exists() else {}
    stage=getattr(exc,'stage',current.get('stage','initialization'))
    failure={'status':'failed','stage':stage,'exception_type':type(exc).__name__,
             'category':getattr(exc,'category','execution_error'),'test_evaluated':False}
    dump(status_path,failure)
    summary_path=out/'summary.json'
    summary=json.loads(summary_path.read_text()) if summary_path.exists() else {'comparison':{},'branches':[]}
    summary.update(status='failed',failure=failure,test_evaluated=False)
    if optimizer is not None:
        summary.setdefault('cost',{})['optimizer']=call_cost(optimizer.calls)
    dump(summary_path,summary)
    with (out/'REPORT.md').open('a') as stream:
        stream.write('\n运行失败：'+stage+' / '+failure['exception_type']+'；未完成进化，不代表零个有效提案的成功实验。\n')
    return summary


class _CancellableClient:
    def __init__(self, client, cancelled):
        self.client, self.cancelled = client, cancelled

    @property
    def calls(self):
        return self.client.calls

    @property
    def base_url(self):
        return getattr(self.client,'base_url',None)

    def complete(self, *args, **kwargs):
        if self.cancelled.is_set():
            raise ModelServiceUnavailable('Candidate execution cancelled after a peer failure')
        return self.client.complete(*args, **kwargs)


def _candidate_clients(args, config, endpoint, actor_client, judge_client, cancelled):
    if args.candidate_workers==1 and not args.candidate_base_urls:
        actor,judge=actor_client,judge_client
    elif args.fixture:
        actor,judge=FixtureClient(),FixtureClient()
    elif config.get('llm_roles'):
        from .llm_roles import RoutedClient
        actor,judge=RoutedClient(config,endpoint),RoutedClient(config,endpoint)
    else:
        actor=ModelClient(endpoint,config['model'],max_tokens=config['actor_max_tokens'])
        judge=ModelClient(endpoint,config['judge_model'],max_tokens=config['actor_max_tokens'])
    return _CancellableClient(actor,cancelled),_CancellableClient(judge,cancelled)


def _evaluate_candidate(job,args,config,sessions,run_queries,run_ids,
                        actor_client,judge_client,endpoint,cancelled):
    name,branch=job['name'],job['branch']
    node=copy.deepcopy(job['node'])
    node['inference_endpoint']=endpoint
    stage='initialize';metrics=None;started=time.perf_counter()
    actor,judge=_candidate_clients(args,config,endpoint,actor_client,judge_client,cancelled)
    actor_call_start,judge_call_start=len(actor.calls),len(judge.calls)
    try:
        with trace_span('candidate.execution',metadata={'policy':name,'inference_endpoint':endpoint,
                        'execution_mode':config['candidate_execution']['mode']}) as span:
            _set_stage(branch,stage)
            factory=lambda:GeneratedPolicy(job['source'],config,timeout_s=args.candidate_timeout_s,
                                            memory_bytes=args.candidate_memory_mb*1024*1024)
            if cancelled.is_set(): raise ModelServiceUnavailable('Candidate execution cancelled')
            factory()  # Verify isolation before replay; every patient still gets fresh state.
            stage='train';_set_stage(branch,stage)
            t,_=replay(sessions,run_queries['train'],run_ids['train'],factory,actor,judge,config,branch/'train')
            if cancelled.is_set(): raise ModelServiceUnavailable('Candidate execution cancelled')
            stage='validation';_set_stage(branch,stage)
            v,_=replay(sessions,run_queries['validation'],run_ids['validation'],factory,actor,judge,config,branch/'validation')
            node['status']='evaluated' if t['AVG'] is not None and v['AVG'] is not None else 'incomplete_scoring'
            node.update(train_AVG=t['AVG'],validation_AVG=v['AVG'])
            metrics=comparison_metric(v);metrics['valid']=node['status']=='evaluated'
            span.update(output={'status':node['status'],'train_AVG':t['AVG'],'validation_AVG':v['AVG']})
    except ModelServiceUnavailable as exc:
        cancelled.set()
        exc.stage=name+'.'+stage
        exc.category='service_unavailable'
        dump(branch/'run-status.json',{'status':'failed','stage':stage,'category':'service_unavailable'})
        raise
    except Exception as exc:
        node.update(status='invalid',failure=f'{type(exc).__name__}: {exc}',failure_stage=stage,
                    failure_category=getattr(exc,'category','runtime'))
        dump(branch/'failure.json',{'reason':node['failure'],'category':node['failure_category'],
                                   'stage':stage,'policy_cost':getattr(exc,'policy_cost',None)})
    else:
        dump(branch/'run-status.json',{'status':'completed','stage':'candidate.finished',
                                      'scoring_status':node['status'],'test_evaluated':False})
    finally:
        dump(branch/'execution.json',{'endpoint':endpoint,'mode':config['candidate_execution']['mode'],
             'wall_seconds':time.perf_counter()-started,
             'model_cost':call_cost(actor.calls[actor_call_start:]+judge.calls[judge_call_start:]),
             'test_evaluated':False})
    if node['status']=='invalid':
        dump(branch/'run-status.json',{'status':'failed','stage':stage,'category':node['failure_category']})
    with trace_span('candidate.result',metadata={'policy':name}) as span:
        span.update(output=node)
    return name,node,metrics


def _run_candidate_jobs(jobs,args,config,sessions,run_queries,run_ids,
                        actor_client,judge_client,accept):
    cancelled=threading.Event()
    endpoints=args.candidate_base_urls or [args.base_url]
    def run(job,index):
        return _evaluate_candidate(job,args,config,sessions,run_queries,run_ids,
                                    actor_client,judge_client,
                                    endpoints[job.get('replica_index',index)%len(endpoints)],cancelled)
    if args.candidate_workers==1:
        for index,job in enumerate(jobs): accept(run(job,index))
        return
    # Each worker owns model clients, policy state, progress and trace context.
    # Shared tree/Git/report writes remain on the coordinator thread.
    with ThreadPoolExecutor(max_workers=args.candidate_workers,thread_name_prefix='rsi-candidate') as pool:
        futures=[pool.submit(copy_context().run,run,job,index) for index,job in enumerate(jobs)]
        try:
            for future in as_completed(futures): accept(future.result())
        except BaseException:
            cancelled.set()
            for future in futures: future.cancel()
            raise


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
    p.add_argument('--reflexion-max-corrections',type=int,default=1)
    p.add_argument('--resume-baselines-from')
    p.add_argument('--candidate-workers',type=int,choices=(1,2,3),default=1)
    p.add_argument('--candidate-base-urls',nargs='+')
    p.add_argument('--llm-config',help='Versioned YAML workflow-node model configuration')
    p.add_argument('--search-profile',help='Explicitly opt into a versioned search configuration')
    p.add_argument('--continuous',action='store_true',help='Bounded multi-generation UCT frontier search')
    p.add_argument('--max-candidates',type=int,default=6)
    p.add_argument('--children-per-expansion',type=int,default=2)
    p.add_argument('--max-depth',type=int,default=3)
    p.add_argument('--search-c',type=float,default=1.414)
    p.add_argument('--search-max-hours',type=float,default=0.,help='Active-time soft limit checked between expansions; 0 disables it')
    p.add_argument('--resume-search',action='store_true',help='Resume a continuous checkpoint with its frozen configuration')
    args=p.parse_args()
    if args.search_profile and not args.resume_search:
        profile=json.loads(Path(args.search_profile).read_text())
        if profile.get('version')!='continuous-v1' or profile.get('reward')!='development_validation_E2E_AVG':
            p.error('Unsupported search profile/objective')
        args=p.parse_args(profile['cli_flags']+sys.argv[1:])
    if args.resume_search:
        existing=Path(args.output)/'config.json'
        if not existing.exists(): p.error('Resume requires an existing config.json')
        saved=json.loads(existing.read_text())
        if not saved.get('continuous'): p.error('Only continuous searches can resume')
        for action in p._actions:
            if action.dest in saved and action.dest not in ('help','resume_search'):
                setattr(args,action.dest,saved[action.dest])
    if args.continuous:
        if not args.evolve: p.error('--continuous requires --evolve')
        if min(args.max_candidates,args.children_per_expansion,args.max_depth)<=0: p.error('Search budgets must be positive')
        if not 1<=args.children_per_expansion<=3: p.error('Use 1 to 3 children per expansion')
        if not math.isfinite(args.search_c) or args.search_c<0: p.error('Invalid exploration coefficient')
        if not math.isfinite(args.search_max_hours) or args.search_max_hours<0: p.error('Invalid active-time limit')
        if args.max_interventions: args.max_interventions=0  # Not yet parent-policy aware; explicitly recorded below.

    if args.candidate_base_urls and len(args.candidate_base_urls)!=args.candidate_workers:
        p.error('Provide exactly one candidate endpoint per worker')
    if args.candidate_workers>1 and not args.candidate_base_urls:
        p.error('Parallel candidates require independent --candidate-base-urls')
    if args.candidate_base_urls and len(set(args.candidate_base_urls))!=len(args.candidate_base_urls):
        p.error('Candidate replica endpoints must be distinct')
    if args.candidate_workers>1 and args.judge_url:
        p.error('Parallel candidate actor/judge use each replica; omit --judge-url')
    if not 0 <= args.reflexion_max_corrections <= 2:
        p.error('Reflexion corrections must be between 0 and 2')
    if args.resume_baselines_from and not args.evolve:
        p.error('--resume-baselines-from requires --evolve')
    if min(args.reflexion_max_tokens,args.codegen_max_tokens)<=0:
        p.error('Optimizer token budgets must be positive')
    if not all(math.isfinite(t) and 0<=t<=2 for t in
               (args.reflexion_temperature,args.codegen_temperature)):
        p.error('Optimizer temperatures must be finite and between 0 and 2')
    if not math.isfinite(args.optimizer_timeout_s) or args.optimizer_timeout_s<=0:
        p.error('Optimizer timeout must be finite and positive')
    signal.signal(signal.SIGTERM, _interrupt_replay)
    out=Path(args.output)
    if out.exists() and not args.resume_search: p.error('Use a new output directory; never overwrite runs')
    if not args.fixture and not args.dataset: p.error('--dataset is required')
    config={'contract_version':'v0.3','interface_version':'v0.3','state_serializer':'cloudpickle-base64-v1','chunk_chars':1200,'chunk_overlap':120,'top_k':5,'max_steps':4,'context_chars':16000,
            'mode':'Efficient','seed':20261013,'split_version':SPLIT_VERSION,
            'split_fractions':{'train':.4,'validation':.2,'test':.4},'language':args.language,
            'fixture':args.fixture,'model':args.model,'judge_model':args.judge_model or args.model,
            'speculation':'disabled','policy_auxiliary_model':'disabled',**vars(args)}
    if args.continuous:
        config['continuous_search']={'version':'continuous-v1','max_candidates':args.max_candidates,
            'children_per_expansion':args.children_per_expansion,'max_depth':args.max_depth,
            'reward':'development_validation_E2E_AVG','validation_role':'development',
            'exploration_c':args.search_c,'expansion_policy':'unexpanded_frontier_once',
            'max_active_hours':args.search_max_hours,'interventions':'disabled_until_parent_aware'}
    config['judge_model']=args.judge_model or args.model
    config['evaluation_metric_version']='strict-plus-e2e-v1'
    config['actor_max_tokens']=1024
    config['optimizer_generation_settings']={
        'reflexion':{'max_tokens':args.reflexion_max_tokens,'temperature':args.reflexion_temperature,
                     'timeout_s':args.optimizer_timeout_s},
        'codegen':{'max_tokens':args.codegen_max_tokens,'temperature':args.codegen_temperature,
                   'timeout_s':args.optimizer_timeout_s}}
    if args.llm_config:
        from .llm_roles import install
        if args.resume_search and saved.get('llm_roles'):
            if hashlib.sha256(Path(args.llm_config).read_bytes()).hexdigest()!=saved['llm_roles']['source_sha256']:
                p.error('LLM YAML changed; start a new experiment')
            config['llm_roles']=saved['llm_roles']
        else:
            install(config,args.llm_config)
        config['llm_baseline_roles']={key:config['llm_roles']['resolved'][key] for key in ('actor','judge')}
        config['llm_baseline_roles']['router_sha256']=hashlib.sha256((Path(__file__).parent/'llm_roles.py').read_bytes()).hexdigest()
    config['observability_schema']='local-spans-v1';config['rationale_schema']='evidence-v4-structured-diagnosis'
    config['reflexion_stages']=['proposals'] if args.no_diagnosis else ['diagnosis','proposals']
    config['baseline_runtime_sha256']=baseline_runtime_fingerprint()
    config['evolution_prompt_version']=REFLECTION_PROMPT_VERSION
    config['diagnostic_selection_version']='training-diagnostics-coverage-v1'
    config['reflexion_input_budget']=None
    config['reflexion_case_targets']={'failures':12,'successes':4}
    config['candidate_execution']={'mode':'parallel' if args.candidate_workers>1 else 'serial',
        'workers':args.candidate_workers,'endpoints':args.candidate_base_urls or [args.base_url],
        'merge_order':'proposal_order','state':'fresh_replay_per_candidate',
        'version':'candidate-concurrent-v1','codegen_order':'serial_then_candidate_replays'}
    config['candidate_runtime_sha256']={name:hashlib.sha256(
        (Path(__file__).resolve().parent/name).read_bytes()).hexdigest()
        for name in ('sandbox.py','tracing.py')}
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
    if args.resume_search:
        config['parent_source_sha256']=saved['parent_source_sha256']
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
        if config.get('llm_roles'):
            from .llm_roles import RoutedClient
            actor_client=RoutedClient(config,args.base_url)
            judge_client=RoutedClient(config,args.base_url)
            optimizer=RoutedClient(config,args.base_url)
        else:
            actor_client=ModelClient(args.base_url,args.model,max_tokens=1024)
            judge_client=ModelClient(args.judge_url or args.base_url,args.judge_model or args.model,max_tokens=1024)
            optimizer=ModelClient(args.base_url,args.model,
                                  role_settings=config['optimizer_generation_settings'])
    split=patient_splits([s['persona_id'] for s in sessions],config['seed'])
    config['split_patient_counts']={k:len(v) for k,v in split.items()}
    # Diagnostic subset keeps test sealed and stays within each original split.
    run_ids={k:v[:args.patients_per_split] if args.patients_per_split else v for k,v in split.items() if k!='test'}
    run_queries={k:representative_queries(queries,v,args.checkpoint,args.per_type) for k,v in run_ids.items()}
    out.mkdir(parents=True,exist_ok=args.resume_search)
    with trace_run(out/'traces.jsonl', run_id=(out.parent.name if out.name == 'result' else out.name),
                   metadata={'implementation_commit':config['implementation_commit'],
                             'contract_version':config['contract_version'], 'fixture':args.fixture,
                             'rationale_schema':config['rationale_schema']}) as round_span:
        round_span.update(input={'split_patient_counts':config['split_patient_counts'],
                                 'objective':'improve memory policy under a frozen actor',
                                 'language':args.language})
        try:
            summary=_run_round(args,out,config,split,run_ids,run_queries,sessions,actor_client,judge_client,optimizer)
        except BaseException as exc:
            if args.resume_search and isinstance(exc,(ValueError,RuntimeError)):
                raise  # Refused resume must not overwrite the existing search status.
            summary=_record_round_failure(out,exc,optimizer)
            round_span.update(output=summary,metadata={'trace_write_failures':write_failure_count()})
            raise
        round_span.update(output=summary, metadata={'trace_write_failures':write_failure_count()})


def _run_round(args,out,config,split,run_ids,run_queries,sessions,actor_client,judge_client,optimizer):
    if args.continuous and args.resume_search:
        from .continuous import run as continuous_run
        return continuous_run(args,out,config,split,run_ids,run_queries,sessions,actor_client,judge_client,optimizer)
    _set_stage(out,'baseline.verification' if args.resume_baselines_from else 'baseline.train')
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
    reused=None
    if args.resume_baselines_from:
        plan=verify_baseline_resume(args.resume_baselines_from,config,split,run_ids,run_queries,
                                    parent_manifest,code_root=Path(__file__).resolve().parent.parent)
        reused=materialize_baseline_resume(plan,out)
        train,train_rows,validation=plan['train_report'],plan['train_rows'],plan['validation_report']
        with trace_span('baseline.reused',metadata={'actual_model_execution':False}) as observation:
            observation.update(output=reused)
    else:
        train,train_rows=replay(sessions,run_queries['train'],run_ids['train'],lambda:PlainPolicy(config),
                                actor_client,judge_client,config,out/'plain-bm25'/'train')
        _set_stage(out,'baseline.validation')
        validation,_=replay(sessions,run_queries['validation'],run_ids['validation'],lambda:PlainPolicy(config),
                            actor_client,judge_client,config,out/'plain-bm25'/'validation')
    tree['nodes']['plain-bm25']['status']='evaluated' if train['AVG'] is not None and validation['AVG'] is not None else 'incomplete_scoring'
    comparisons={'plain-bm25':comparison_metric(validation)}
    semantic={'status':'skipped','reason':'no frozen semantic model supplied'}
    if reused:
        semantic=plan['semantic_status']
        if args.semantic_model_path:
            comparisons['plain-semantic']=comparison_metric(plan['semantic_validation_report'])
            comparisons['plain-semantic']['valid']=semantic['status']=='evaluated'
    elif args.semantic_model_path:
        _set_stage(out,'baseline.semantic')
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
    dump(out/'tree.json',tree)
    dump(out/'summary.json',{'status':'running','comparison':comparisons,'branches':[],
                            'baseline_reuse':reused,'test_evaluated':False})
    if args.continuous:
        from .continuous import run as continuous_run
        return continuous_run(args,out,config,split,run_ids,run_queries,sessions,actor_client,judge_client,optimizer,
            initial=(root_code,root_commit,train,validation,comparisons,semantic,reused))
    lookup={q['query_id']:q for q in run_queries['train']}
    branches=[];seen_code={};candidate_jobs=[]
    if args.evolve:
        _set_stage(out,'diagnostics')
        diagnostics = None
        if not args.no_diagnosis:
            diagnostics=build_training_diagnostics(train_rows,lookup,seed=config['seed'])
            directory=out/'diagnostics'
            directory.mkdir()
            with (directory/'index.jsonl').open('w') as stream:
                for row in diagnostics['index']:
                    stream.write(json.dumps(row,ensure_ascii=False)+'\n')
            dump(directory/'summary.json',diagnostics['summary'])
            dump(directory/'selection.json',diagnostics['selection'])
            dump(directory/'examples.json',diagnostics['examples'])
        interventions=intervention(train_rows,lookup,actor_client,judge_client,sessions,args.max_interventions) if not args.no_diagnosis and args.max_interventions>0 else []
        dump(out/'interventions.json',interventions)
        _set_stage(out,'reflexion')
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
                              not args.no_diagnosis,policy_context=context,diagnostics=diagnostics,
                              max_corrections=args.reflexion_max_corrections,artifact_dir=out/'reflexion')
            dump(out/'reflexion.json',diagnosis)
        except Exception as exc:
            dump(out/'reflexion-failure.json',{'exception_type':type(exc).__name__,
                 'stage':getattr(exc,'stage','reflexion'), 'category':getattr(exc,'category','execution_error'),
                 'attempt_artifacts':'reflexion/', 'status':'failed'})
            raise
        for i,proposal in enumerate(diagnosis['proposals']):
            name=f'candidate-{i+1}';node=new_node(name,'plain-bm25');node['proposal']=proposal;tree['nodes'][name]=node
            dump(out/'tree.json',tree)
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
            _set_stage(out,name+'.'+stage)
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
                dump(out/'tree.json',tree)
                candidate_jobs.append({'name':name,'node':copy.deepcopy(node),'source':source,
                                       'branch':branch,'replica_index':i})
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
            with trace_span('candidate.result' if node['status']=='invalid' else 'candidate.prepared',
                            metadata={'policy':name}) as observation:
                observation.update(output=node)
            branches.append(node)
            dump(out/'summary.json',{'status':'running','comparison':comparisons,'branches':branches,
                                    'baseline_reuse':reused,'test_evaluated':False})
        _set_stage(out,'candidates.parallel' if args.candidate_workers>1 else 'candidates.serial')
        def accept(result):
            name,node,metrics=result
            tree['nodes'][name]=node
            for index,branch_node in enumerate(branches):
                if branch_node['id']==name: branches[index]=node
            if metrics is not None: comparisons[name]=metrics
            dump(out/'tree.json',tree)
            dump(out/'summary.json',{'status':'running','comparison':comparisons,'branches':branches,
                                    'execution':config['candidate_execution'],
                                    'baseline_reuse':reused,'test_evaluated':False})
        _run_candidate_jobs(candidate_jobs,args,config,sessions,run_queries,run_ids,
                            actor_client,judge_client,accept)
        # Stable proposal-order aggregation, independent of completion order.
        for node in branches:
            if node['status']=='evaluated': backprop(tree,node['id'],node['train_AVG'])
    dump(out/'tree.json',tree)
    summary={'status':'completed','baseline_reuse':reused,'execution':config['candidate_execution'],
             'evolution_outcome':('evaluated_candidates' if any(n['status']=='evaluated' for n in branches) else
                                  'no_fully_scored_candidates') if args.evolve else 'not_requested',
             'contract_version':config['contract_version'],'fixture':args.fixture,'scientific_result':False if args.fixture else 'engineering rubric; uncalibrated',
             'comparison':comparisons,'pareto':pareto(comparisons),'branches':branches,
             'semantic':semantic,'cost':{'optimizer':call_cost(optimizer.calls),
                                       'intervention':call_cost([c for r in (interventions if args.evolve else []) for c in r['call_log']])},
             'test_evaluated':False,'no_diagnosis_control':args.no_diagnosis}
    dump(out/'summary.json',summary)
    dump(out/'run-status.json',{'status':'completed','stage':'round.finished',
                              'evolution_outcome':summary['evolution_outcome'],'test_evaluated':False})
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
