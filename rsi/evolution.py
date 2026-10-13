"""Training-only hypothesis/code generation, tree backpropagation and Pareto."""
from __future__ import annotations
import copy
import hashlib
import json
import math
from .sandbox import validate_source

REFLECTION_SCHEMA = {
    'type':'object', 'additionalProperties':False, 'required':['clusters','proposals'],
    'properties': {
        'clusters': {'type':'array','maxItems':6,'items':{
            'type':'object','additionalProperties':False,
            'required':['label','case_ids','attribution','uncertainty'],
            'properties':{'label':{'type':'string'},'case_ids':{'type':'array','items':{'type':'string'}},
                          'attribution':{'type':'string'},'uncertainty':{'enum':['low','medium','high']}}}},
        'proposals': {'type':'array','minItems':3,'maxItems':3,'items':{
            'type':'object','additionalProperties':False,
            'required':['direction','hypothesis','prediction','risk'],
            'properties':{k:{'type':'string'} for k in ['direction','hypothesis','prediction','risk']}}}}}


def compact_example(value, text_limit=500):
    """Bound diagnostic excerpts without producing broken JSON or hidden data."""
    if isinstance(value, str):
        return value if len(value) <= text_limit else value[:text_limit]+' [excerpt truncated]'
    if isinstance(value, list):
        return [compact_example(v, text_limit) for v in value[:5]]
    if isinstance(value, dict):
        return {k: compact_example(v, text_limit) for k, v in value.items()}
    return value


def reflect(client, answers, query_lookup, interventions, parent_code, with_diagnosis=True):
    examples = []
    if with_diagnosis:
        # Keep failed and successful cases, never validation item-level results.
        chosen = [r for r in answers if r['score'] == 0][:4]+[r for r in answers if r['score'] == 1][:2]
        for r in chosen:
            q = query_lookup[r['query_id']]
            examples.append({k: r[k] for k in ['query_id','query_type','question','answer','score','reason','evidence','trajectory']})
            examples[-1]['reference_answers'] = q['positive_answers']
            examples[-1]['source_evidence'] = q.get('evidence', [])
    prompt = {'task': 'Improve only memory management; answering system is frozen.',
              'parent_code': parent_code, 'examples': compact_example(examples),
              'interventions': compact_example(interventions) if with_diagnosis else [],
              'diagnostic_excerpts': True,
              'required_output': {'clusters': [{'label':'string','case_ids':[], 'attribution':'string','uncertainty':'low/medium/high'}],
                                  'proposals':[{'direction':'string','hypothesis':'string','prediction':'string','risk':'string'}]},
              'constraints': 'Return exactly three distinct proposals. If no reliable failure attribution exists, say so. '
                             'Do not insert references or answers into policy code. Changes only affect memory state and retrieval.'}
    # Keep complete code; reduce only diagnostic excerpts uniformly before
    # dropping examples. This engineering character cap leaves output headroom
    # for the deployed 32k context; it is not an exact tokenizer budget.
    for limit in (500, 300, 150, 75):
        prompt['examples'] = compact_example(examples, limit)
        if len(json.dumps(prompt, ensure_ascii=False)) <= 24000: break
    while len(json.dumps(prompt, ensure_ascii=False)) > 24000 and prompt['examples']:
        prompt['examples'].pop()
    serialized = json.dumps(prompt, ensure_ascii=False)
    if len(serialized) > 24000:
        raise ValueError('Reflexion parent/config exceeds engineering input budget')
    response = client.complete([{'role':'system','content':'You diagnose memory-management errors offline. Excerpts can be truncated; express attribution uncertainty. Return valid JSON, no markdown. Keep each text field concise: at most 150 characters. Produce exactly three distinct proposals.'},
                                {'role':'user','content':serialized}],role='reflexion',json_mode=True,json_schema=REFLECTION_SCHEMA)
    obj = json.loads(response['text'])
    if not isinstance(obj,dict) or not isinstance(obj.get('clusters'),list) or len(obj.get('proposals',[])) != 3:
        raise ValueError('Reflexion must return clusters and exactly three proposals')
    for proposal in obj['proposals']:
        if not isinstance(proposal,dict) or not all(isinstance(proposal.get(k),str) for k in ['direction','hypothesis','prediction','risk']):
            raise ValueError('Invalid proposal schema')
    for cluster in obj['clusters']:
        if not isinstance(cluster,dict) or not isinstance(cluster.get('case_ids'),list) or \
                cluster.get('uncertainty') not in ('low','medium','high') or \
                not all(isinstance(cluster.get(k),str) for k in ('label','attribution')):
            raise ValueError('Invalid cluster schema')
    obj['call'] = {k: response[k] for k in ['usage','latency_s','role']}
    obj['input_chars'] = len(serialized)
    obj['diagnostic_example_count'] = len(prompt['examples'])
    return obj


def generate(client, proposal, parent_code):
    instruction = '''Produce a standalone restricted Python memory policy. Return JSON {"code":"..."}.
Exactly three entry functions: initialize(config), ingest(state,session), search(state,query,filters).
Input session is a chunk with id,text,session_id,persona_id. search returns original registered IDs, at most 5.
JSON state can organize, merge, prune and index data. No answering, no reference answers, no external state.
Syntax: functions, assignments, if, for, return, expressions, lists, dicts, sets. NO imports, classes, lambdas,
comprehensions, decorators, annotations, try, while, global, underscore names, files, network, reflection.
Allowed builtins: sorted,len,min,max,sum,range,set,list,dict,str,int,float,bool,enumerate,zip,abs,round.
Allowed methods: lower,strip,split,join,replace,startswith,endswith,find,count,casefold,append,extend,insert,
pop,remove,reverse,sort,copy,index,get,keys,values,items,update,setdefault. Helper functions are allowed.
Use only passed config/state/session/query/filters. Each function must have explicit arguments without defaults.
All function and variable names must start with a Latin letter, NEVER an underscore. Use tokens, rank, helper;
never _tokens, _rank, __init__, _ or self. JSON keys such as session_id are allowed.
The parent is trusted host class code in a DIFFERENT dialect. Do not copy its class methods as functions.
Rewrite the algorithm as these three JSON-state entry functions using the supplied legal starter.
Prefer simple loop-based code. Do not include scores, patient-specific constants, or evaluation questions.'''
    response = client.complete([{'role':'system','content':instruction},
                                {'role':'user','content':json.dumps({'proposal':proposal,'parent_code':parent_code,
                                                                   'legal_starter':PARENT_CODE},ensure_ascii=False)}],
                               role='codegen',json_mode=True)
    obj = json.loads(response['text']); code = obj.get('code')
    cost = {k: response[k] for k in ['usage','latency_s','role']}
    try:
        validate_source(code)
    except Exception as exc:
        # Rejected proposals remain reviewable without running their code.
        if isinstance(code,str): exc.candidate_source = code
        exc.codegen_cost = cost
        raise
    return code, cost


PARENT_CODE = '''def initialize(config):
    return {"records": []}
def ingest(state, session):
    state["records"].append(session)
    return state
def search(state, query, filters):
    scored = []
    for r in state["records"]:
        score = 0
        for i in range(len(query)-1):
            if query[i:i+2] in r["text"]:
                score += 1
        scored.append([score, r["session_id"], r["id"]])
    ranked = sorted(scored)
    ranked.reverse()
    ids = []
    for r in ranked[:5]:
        ids.append(r[2])
    return ids
'''


def uct(node, parent_visits, c=1.414):
    if node.get('status') in ('invalid','runtime_failed'): return -math.inf
    if not node['visits']: return math.inf
    return node['Q']+c*math.sqrt(math.log(max(1,parent_visits))/node['visits'])


def backprop(tree, node_id, reward):
    if not isinstance(reward,(int,float)) or not math.isfinite(reward): return
    nodes = tree['nodes']; node = nodes[node_id]
    while node:
        node['visits'] += 1; node['value_sum'] += reward; node['Q'] = node['value_sum']/node['visits']
        node = nodes.get(node['parent'])


def new_node(node_id, parent=None):
    return {'id':node_id,'parent':parent,'visits':0,'value_sum':0.,'Q':None,'status':'pending'}


def pareto(reports):
    # Separate comparable metrics; null AVG or failed runs are never eligible.
    valid = {k:v for k,v in reports.items() if v.get('AVG') is not None and v.get('valid',True)}
    keep = []
    for k,a in valid.items():
        dominated = False
        for j,b in valid.items():
            if j == k: continue
            av = [-a['AVG'],a['policy_tokens'],a['policy_seconds'],a['policy_bytes']]
            bv = [-b['AVG'],b['policy_tokens'],b['policy_seconds'],b['policy_bytes']]
            if all(x<=y for x,y in zip(bv,av)) and any(x<y for x,y in zip(bv,av)):
                dominated=True; break
        if not dominated: keep.append(k)
    return keep
