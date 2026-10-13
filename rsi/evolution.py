"""Training-only hypothesis/code generation, tree backpropagation and Pareto."""
from __future__ import annotations
import copy
import hashlib
import json
import math
from .sandbox import validate_source


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
              'parent_code': parent_code, 'examples': examples,
              'interventions': interventions if with_diagnosis else [],
              'required_output': {'clusters': [{'label':'string','case_ids':[], 'attribution':'string','uncertainty':'low/medium/high'}],
                                  'proposals':[{'direction':'string','hypothesis':'string','prediction':'string','risk':'string'}]},
              'constraints': 'Return exactly three distinct proposals. If no reliable failure attribution exists, say so. '
                             'Do not insert references or answers into policy code. Changes only affect memory state and retrieval.'}
    response = client.complete([{'role':'system','content':'You diagnose memory-management errors offline. Return valid JSON, no markdown.'},
                                {'role':'user','content':json.dumps(prompt,ensure_ascii=False)}],role='reflexion',json_mode=True)
    obj = json.loads(response['text'])
    if not isinstance(obj,dict) or not isinstance(obj.get('clusters'),list) or len(obj.get('proposals',[])) != 3:
        raise ValueError('Reflexion must return clusters and exactly three proposals')
    for proposal in obj['proposals']:
        if not isinstance(proposal,dict) or not all(isinstance(proposal.get(k),str) for k in ['direction','hypothesis','prediction','risk']):
            raise ValueError('Invalid proposal schema')
    obj['call'] = {k: response[k] for k in ['usage','latency_s','role']}
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
Prefer simple loop-based code. Do not include scores, patient-specific constants, or evaluation questions.'''
    response = client.complete([{'role':'system','content':instruction},
                                {'role':'user','content':json.dumps({'proposal':proposal,'parent_code':parent_code},ensure_ascii=False)}],
                               role='codegen',json_mode=True)
    obj = json.loads(response['text']); code = obj.get('code')
    validate_source(code)
    return code, {k: response[k] for k in ['usage','latency_s','role']}


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
