"""Budgeted UCT frontier search; validation is explicitly a development set."""
import copy
import fcntl
import hashlib
import json
import math
import os
import time
from pathlib import Path
from .evolution import reflect, generate, new_node, backprop
from .diagnostics import build_training_diagnostics
from .tracing import trace_span
from .model import ModelServiceUnavailable


def read(path):
    return json.loads(Path(path).read_text())


def atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    os.replace(temporary, path)


def fingerprint(config, split, run_queries, sessions=None):
    # Check actual selected questions as well as the declared data revision.
    excluded = {'resume_search', 'implementation_dirty', 'implementation_commit'}
    return hashlib.sha256(json.dumps({
        'config': {k:v for k,v in config.items() if k not in excluded},
        'split':split, 'queries':run_queries, 'sessions':sessions,
        'runtime': {p.name:hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in sorted(Path(__file__).parent.glob('*.py'))}}, sort_keys=True).encode()).hexdigest()


def objective(report):
    value = report.get('E2E_AVG')
    return value if isinstance(value, (int, float)) and math.isfinite(value) else None


def select_parent(tree, max_depth):
    """Expand each eligible version once; deterministic tie break by creation order."""
    entries = []
    for identifier,node in tree['nodes'].items():
        if not node.get('search_eligible') or node.get('expanded') or node['generation'] >= max_depth:
            continue
        parent = tree['nodes'].get(node.get('parent'), {})
        n = node['visits']; pn = parent.get('visits', max(1,n))
        bonus = tree['c'] * math.sqrt(math.log(max(1,pn)) / n) if n else None
        entries.append({'policy':identifier, 'Q':node['Q'], 'visits':n,
                        'parent_visits':pn, 'exploration_bonus':bonus,
                        'uct':node['Q'] + bonus if n else None,
                        'priority':'finite' if n else 'unvisited'})
    if not entries:
        return None,entries
    # max preserves the first entry for ties.
    chosen = max(entries, key=lambda x:(x['priority']=='unvisited', x['uct'] or 0))['policy']
    return chosen,entries


def run(args, out, config, split, run_ids, run_queries, sessions, actor, judge, optimizer, initial=None):
    from .cli import commit_policy, comparison_metric, _run_candidate_jobs, _set_stage
    out = Path(out); checkpoint = out/'search-state.json'
    selected_ids=set(run_ids['train'])|set(run_ids['validation'])
    signature = fingerprint(config,split,run_queries,[s for s in sessions if s['persona_id'] in selected_ids])
    with (out/'search.lock').open('a') as lock:
        try:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError('An active driver already owns this search')
        if checkpoint.exists():
            state = read(checkpoint)
            if state['fingerprint'] != signature:
                raise ValueError('Config, split, questions or runtime changed; resume refused')
        else:
            if initial is None:
                raise ValueError('No search checkpoint exists')
            code,commit,train,validation,comparisons,semantic,reused = initial
            reward = objective(validation)
            root = new_node('plain-bm25')
            root.update(commit=commit,generation=0,expanded=False,search_eligible=reward is not None,
                        status='evaluated' if train['AVG'] is not None and validation['AVG'] is not None else 'incomplete_scoring',
                        train_AVG=train['AVG'],validation_AVG=validation['AVG'])
            tree = {'algorithm':'bounded UCT frontier search', 'c':args.search_c,
                    'reward':'development_validation_E2E_AVG', 'validation_role':'development',
                    'nodes':{'plain-bm25':root}}
            if reward is not None:
                backprop(tree,'plain-bm25',reward)
            state = {'version':'continuous-v1', 'fingerprint':signature, 'tree':tree,
                     'expansions':[], 'comparisons':comparisons, 'semantic':semantic,
                     'baseline_reuse':reused, 'next_candidate':1, 'active_expansion':None,
                     'active_seconds':0., 'stop_reason':None, 'artifact_hashes':{}}
        session_seconds = state['active_seconds']; started = time.monotonic()
        def policy_path(name):
            return out/'plain-bm25' if name=='plain-bm25' else out/'branches'/name
        def source(name):
            return (out/'parent-policy.py').read_text() if name=='plain-bm25' else (policy_path(name)/'policy.py').read_text()
        def remember(path):
            if path.exists():
                state['artifact_hashes'][str(path.relative_to(out))] = hashlib.sha256(path.read_bytes()).hexdigest()
        for relative,expected in state['artifact_hashes'].items():
            path=out/relative
            if not path.exists() or hashlib.sha256(path.read_bytes()).hexdigest()!=expected:
                raise ValueError('Checkpoint artifact changed: '+relative)
        def save():
            state['active_seconds'] = session_seconds + time.monotonic()-started
            # Completed checkpoints need no further model calls.
            atomic(checkpoint,state); atomic(out/'tree.json',state['tree'])
            branches=[n for k,n in state['tree']['nodes'].items() if k!='plain-bm25']
            atomic(out/'summary.json',{'status':'completed' if state['stop_reason'] else 'running',
                'branches':branches, 'comparison':state['comparisons'], 'search':{
                    'version':state['version'], 'reward':'development_validation_E2E_AVG',
                    'validation_role':'development', 'stop_reason':state['stop_reason'],
                    'expansions':state['expansions'], 'active_seconds':state['active_seconds']},
                'baseline_reuse':state['baseline_reuse'], 'test_evaluated':False, 'fixture':args.fixture})
        for f in ['parent-policy.py','plain-bm25/train/report.json','plain-bm25/train/answers.jsonl',
                  'plain-bm25/validation/report.json']:
            remember(out/f)
        save()
        seen={hashlib.sha256(source(k).encode()).hexdigest() for k in state['tree']['nodes']
              if k=='plain-bm25' or (policy_path(k)/'policy.py').exists()}
        while not state['stop_reason']:
            if args.search_max_hours and state['active_seconds'] >= args.search_max_hours*3600:
                state['stop_reason']='active_wall_time_budget'; break
            active=state['active_expansion']
            if active is None:
                remaining=args.max_candidates-(state['next_candidate']-1)
                if remaining<=0:
                    state['stop_reason']='candidate_budget'; break
                parent,entries=select_parent(state['tree'],args.max_depth)
                if parent is None:
                    state['stop_reason']='no_expandable_nodes'; break
                active={'id':f"expansion-{len(state['expansions'])+1:03d}", 'parent':parent,
                        'count':min(args.children_per_expansion,remaining), 'selection_snapshot':entries,
                        'status':'selected', 'children':[]}
                state['expansions'].append(active); state['active_expansion']=active
                state['tree']['nodes'][parent]['expanded']=True
                save()
                with trace_span('search.selection',input={'frontier':entries},metadata={'expansion':active['id']}) as span:
                    span.update(output={'selected_parent':parent,'reward':'development_validation_E2E_AVG'})
            # Deserializing a checkpoint creates separate dicts for this reference.
            active=next(e for e in state['expansions'] if e['id']==active['id'])
            state['active_expansion']=active
            parent=active['parent']; parent_node=state['tree']['nodes'][parent]
            directory=out/'expansions'/active['id']; directory.mkdir(parents=True,exist_ok=True)
            parent_train=read(policy_path(parent)/'train/report.json')
            rows=[json.loads(line) for line in (policy_path(parent)/'train/answers.jsonl').read_text().splitlines() if line]
            parent_code=source(parent); lookup={q['query_id']:q for q in run_queries['train']}
            context={'active_parent_config':{'retrieval_mode':'lexical_bm25' if parent=='plain-bm25' else 'generated_policy',
                'language':config['language'], 'policy_auxiliary_model':config['policy_auxiliary_model'],
                **{k:config[k] for k in ('top_k','max_steps','context_chars','chunk_chars','chunk_overlap')}},
                'parent_train_metrics':{'comparison':comparison_metric(parent_train),
                    'type_accuracy':parent_train.get('type_accuracy',{}),'counts':parent_train.get('counts',{})}}
            if not (directory/'reflexion.json').exists():
                _set_stage(out,active['id']+'.diagnostics')
                diagnostics=build_training_diagnostics(rows,lookup,seed=config['seed']) if not args.no_diagnosis else None
                if diagnostics:
                    for name in ('summary','selection','examples'):
                        atomic(directory/'diagnostics'/(name+'.json'),diagnostics[name])
                reflection=reflect(optimizer,rows,lookup,[],parent_code,not args.no_diagnosis,
                    policy_context=context,diagnostics=diagnostics,max_corrections=args.reflexion_max_corrections,
                    artifact_dir=directory/'reflexion',proposal_count=active['count'])
                atomic(directory/'reflexion.json',reflection)
                if active['id']=='expansion-001':
                    atomic(out/'reflexion.json',reflection)
                    if diagnostics:
                        for name in ('summary','selection','examples'):
                            atomic(out/'diagnostics'/(name+'.json'),diagnostics[name])
                remember(directory/'reflexion.json'); save()
            reflection=read(directory/'reflexion.json')
            if not active['children']:
                for proposal in reflection['proposals']:
                    name=f"candidate-{state['next_candidate']:04d}"; state['next_candidate']+=1
                    node=new_node(name,parent)
                    node.update(proposal=proposal,generation=parent_node['generation']+1,expansion=active['id'],
                                expanded=False,search_eligible=False,reward_applied=False)
                    state['tree']['nodes'][name]=node; active['children'].append(name)
                save()
            jobs=[]
            for index,name in enumerate(active['children']):
                node=state['tree']['nodes'][name]; branch=policy_path(name); branch.mkdir(parents=True,exist_ok=True)
                if node.get('reward_applied') or node['status']=='invalid' or node.get('evaluation_complete'):
                    continue
                atomic(branch/'proposal.json',node['proposal'])
                if not (branch/'policy.py').exists():
                    _set_stage(out,name+'.codegen')
                    try:
                        with trace_span('candidate.codegen', input={'proposal':node['proposal']},
                                        metadata={'policy':name,'parent_policy':parent,'expansion':active['id']}) as span:
                            code,cost=generate(optimizer,node['proposal'],parent_code,policy_context=context['active_parent_config'])
                            span.update(output={'source_sha256':hashlib.sha256(code.encode()).hexdigest()})
                        code_hash=hashlib.sha256(code.encode()).hexdigest()
                        if code_hash in seen and not args.fixture:
                            raise ValueError('Duplicate generated policy')
                        (branch/'policy.py').write_text(code); atomic(branch/'codegen-cost.json',cost); seen.add(code_hash)
                    except ModelServiceUnavailable:
                        raise
                    except Exception as exc:
                        node.update(status='invalid',failure_stage='codegen',failure_category=type(exc).__name__)
                        if hasattr(exc,'candidate_source'):
                            (branch/'rejected-policy.py').write_text(exc.candidate_source)
                            atomic(branch/'codegen-cost.json',getattr(exc,'codegen_cost',{}))
                        atomic(branch/'failure.json',{'stage':'codegen','category':type(exc).__name__})
                        save(); continue
                code=(branch/'policy.py').read_text()
                if not node.get('commit'):
                    node['commit']=commit_policy(out/'policy-git',code,name,parent_node['commit'])
                remember(branch/'policy.py'); save()
                reports=[branch/fold/'report.json' for fold in ('train','validation')]
                if all(f.exists() for f in reports):
                    t,v=map(read,reports)
                    node.update(status='evaluated' if t['AVG'] is not None and v['AVG'] is not None else 'incomplete_scoring',
                                train_AVG=t['AVG'],validation_AVG=v['AVG'],evaluation_complete=True)
                    state['comparisons'][name]=comparison_metric(v); continue
                # Partial model rollouts cannot be appended safely; preserve and replay fresh.
                for fold in ('train','validation'):
                    existing=branch/fold
                    if existing.exists():
                        attempts=branch/'interrupted-attempts'; attempts.mkdir(exist_ok=True)
                        existing.rename(attempts/(fold+'-'+str(time.time_ns())))
                jobs.append({'name':name,'node':copy.deepcopy(node),'source':code,'branch':branch,'replica_index':index})
            def accept(result):
                name,node,metrics=result; node['evaluation_complete']=True
                state['tree']['nodes'][name]=node
                if metrics is not None:
                    state['comparisons'][name]=metrics
                save()
            _set_stage(out,active['id']+'.candidates')
            _run_candidate_jobs(jobs,args,config,sessions,run_queries,run_ids,actor,judge,accept)
            for name in active['children']:
                node=state['tree']['nodes'][name]
                if node.get('reward_applied') or node['status']=='invalid':
                    continue
                for fold in ('train','validation'):
                    for file in ('report.json','answers.jsonl'):
                        remember(policy_path(name)/fold/file)
                reward=objective(read(policy_path(name)/'validation/report.json'))
                if reward is not None:
                    node['search_eligible']=True; backprop(state['tree'],name,reward)
                node.update(reward=reward,reward_applied=True)
            active['status']='completed'; state['active_expansion']=None; save()
        save()
        atomic(out/'run-status.json',{'status':'completed','stage':'search.finished',
                                    'stop_reason':state['stop_reason'],'test_evaluated':False})
        (out/'REPORT.md').write_text('# Continuous search\nReward: validation E2E AVG (development set).\nStop: '+str(state['stop_reason'])+'\nTest remains sealed.\n')
        return read(out/'summary.json')
