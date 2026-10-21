"""Training-only hypothesis/code generation, tree backpropagation and Pareto."""
from __future__ import annotations
import copy
import hashlib
import json
import math
from .sandbox import validate_source
from .tracing import trace_span
from .diagnostics import build_training_diagnostics

REFLECTION_PROMPT_VERSION = 'explore-v3-full-diagnostics'

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
            'required':['direction','hypothesis','prediction','risk','basis','rationale',
                        'mechanism','falsification_test',
                        'supporting_case_ids','supporting_cluster_labels','uncertainty'],
            'properties':{**{k:{'type':'string'} for k in ['direction','hypothesis','prediction','risk','rationale',
                                                         'mechanism','falsification_test']},
                          'basis':{'enum':['training_cases','parent_code','exploratory']},
                          'supporting_case_ids':{'type':'array','items':{'type':'string'}},
                          'supporting_cluster_labels':{'type':'array','items':{'type':'string'}},
                          'uncertainty':{'enum':['low','medium','high']}}}}}}


def reflect(client, answers, query_lookup, interventions, parent_code, with_diagnosis=True,
            policy_context=None, diagnostics=None):
    with trace_span('reflexion', metadata={'role':'reflexion', 'evidence_scope':'train_only'}) as observation:
        result = _reflect(client, answers, query_lookup, interventions, parent_code, with_diagnosis,
                          policy_context, diagnostics)
        observation.update(output=result)
        return result


def _reflect(client, answers, query_lookup, interventions, parent_code, with_diagnosis=True,
             policy_context=None, diagnostics=None):
    bundle = ((diagnostics if diagnostics is not None else
               build_training_diagnostics(answers, query_lookup)) if with_diagnosis else None)
    examples = bundle['examples'] if bundle else []
    context = dict(policy_context or {})
    if not with_diagnosis:
        context.pop('parent_train_metrics',None)
    prompt = {'task': 'Improve memory-policy AVG across all task families under measured costs; answering system is frozen.',
              'prompt_version':REFLECTION_PROMPT_VERSION,
              'parent_code': parent_code, 'parent_context':context, 'examples': examples,
              'training_diagnostics':bundle['summary'] if bundle else None,
              'diagnostic_selection':bundle['selection'] if bundle else None,
              'interventions': interventions if with_diagnosis else [],
              'diagnostic_excerpts': False,
              'diagnostic_input_budget': None,
              'required_output': {'clusters': [{'label':'string','case_ids':[], 'attribution':'string','uncertainty':'low/medium/high'}],
                                  'proposals':[{'direction':'string','hypothesis':'string','prediction':'string','risk':'string',
                                                'basis':'training_cases/parent_code/exploratory','rationale':'brief observable basis',
                                                'mechanism':'how the proposed memory operations change evidence availability or measured cost',
                                                'falsification_test':'observation or comparison that would refute the hypothesis',
                                                'supporting_case_ids':[], 'supporting_cluster_labels':[],
                                                'uncertainty':'low/medium/high'}]},
              'search_guidance': 'The baseline implementation is a starting point, not a preferred architecture. '
                  'Explore any memory representation, indexing, retrieval/ranking, or ingest-time update, '
                  'consolidation and forgetting mechanism expressible through the existing policy interface. '
                  'These are examples, not a fixed menu. Seek mechanism-diverse proposals, not three variants of a minor patch. '
                  'Include at least one substantial structural redesign when plausible; mark unsupported structural ideas '
                  'exploratory with high uncertainty instead of inventing evidence. Do not prefer low risk, few changed lines, '
                  'or memory savings over the AVG/cost frontier. Connect changes to observed memory failures when evidence permits. '
                  'Check active_parent_config before targeting a code path: dormant optional embedding code is not evidence '
                  'of an active cache bottleneck. The standalone parent includes its imports, text-processing helpers '
                  'and entrypoints. Use the declared workload language and actual helper behavior when diagnosing '
                  'retrieval; Chinese text need not contain word-separating spaces. Alternative tokenizers and '
                  'representations remain allowed; preserve useful evidence retrieval when redesigning internals. '
                  'Cost-only proposals need a relevant measured bottleneck or must be '
                  'explicitly exploratory. Describe the operational mechanism and a falsification test.',
              'constraints': 'Return exactly three distinct proposals. If no reliable failure attribution exists, say so. '
                             'Do not insert references or answers into policy code. Changes only affect memory state and retrieval. '
                             'For each proposal report a concise evidence-based rationale, its basis, and uncertainty. '
                             'Only cite case IDs actually supplied in examples and exact cluster labels. '
                             'Cluster attribution must explain the observed failure and inferred mechanism, not merely low/medium/high. '
                             'Do not attach every case to every cluster without case-specific support. '
                             'Code-inspection and exploratory ideas may use empty case links; say the support is unmeasured. '
                             'Do not claim measured memory leaks, latency, or causation without supplied measurements. '
                             'Use the all-training distributions to assess how representative each selected case is. '
                             'Patient, task, time and observable-error groups are coverage axes, not verified causal clusters. '
                             'Source-session hits are proxies: they do not prove the required facts were retrieved. '
                             'Compare failures with the supplied successful controls; do not assume every wrong answer '
                             'has the same memory cause. Selected example fields and intervention records are not '
                             'excerpted by this prompt builder. These are explicit hypotheses for audit, not requests '
                             'for private chain-of-thought.'}
    # No application-level input budget: preserve the full parent and every
    # selected case, list and intervention. Physical provider limits are errors,
    # never a reason to silently truncate or drop diagnostic controls.
    serialized = json.dumps(prompt, ensure_ascii=False)
    with trace_span('diagnostic-input', input=prompt,
                    metadata={'actual_model_input':True, 'input_chars':len(serialized),
                              'diagnostic_example_count':len(prompt['examples']),
                              'parent_code_sha256':hashlib.sha256(parent_code.encode()).hexdigest(),
                              'diagnostic_input_budget':None,
                              'diagnostic_selection_version':bundle['selection']['version'] if bundle else None}) as observation:
        observation.update(output={'selected_case_ids':[e['query_id'] for e in prompt['examples']],
                                   'selection':bundle['selection'] if bundle else None,
                                   'training_summary':bundle['summary'] if bundle else None,
                                   'input_truncated':False})
    response = client.complete([{'role':'system','content':
                                'You design and diagnose memory-management policies offline. The answering agent is frozen; '
                                'policy internals may be redesigned substantially. Seek testable gains in the AVG/cost frontier, '
                                'not minimum edit distance or minimum risk. Produce exactly three mechanism-diverse proposals. '
                                'Ground claims in the supplied active configuration and training evidence, distinguish observation '
                                'from inference, and permit ambitious exploratory hypotheses with explicit uncertainty. '
                                'The all-training summary and representative failures/successes are supplied together. '
                                'Observed error groups do not by themselves establish causal mechanisms. '
                                'Explain enough for an independent reviewer to test each mechanism; '
                                'there is no arbitrary 150-character field limit. Return valid JSON, no markdown.'},
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
        if cluster['attribution'].strip().lower() in ('low','medium','high'):
            raise ValueError('Cluster attribution needs an explanation, not only a confidence level')
    actual_ids = {e['query_id'] for e in prompt['examples']}
    labels = {c['label'] for c in obj['clusters']}
    for cluster in obj['clusters']:
        if not set(cluster['case_ids']).issubset(actual_ids):
            raise ValueError('Cluster cites a case absent from actual training input')
    for proposal in obj['proposals']:
        if proposal.get('basis') not in ('training_cases','parent_code','exploratory') or \
                not all(isinstance(proposal.get(k),str) and proposal[k].strip()
                        for k in ('rationale','mechanism','falsification_test')) or \
                proposal.get('uncertainty') not in ('low','medium','high') or \
                not isinstance(proposal.get('supporting_case_ids'),list) or \
                not isinstance(proposal.get('supporting_cluster_labels'),list):
            raise ValueError('Proposal needs an auditable rationale, basis, links and uncertainty')
        if not set(proposal['supporting_case_ids']).issubset(actual_ids) or \
                not set(proposal['supporting_cluster_labels']).issubset(labels):
            raise ValueError('Proposal cites evidence absent from its actual reflection input')
        if proposal['basis']=='training_cases' and not proposal['supporting_case_ids']:
            raise ValueError('Training-case motivation needs at least one supplied case')
    obj['call'] = {k: response[k] for k in ['usage','latency_s','role']}
    obj['input_chars'] = len(serialized)
    obj['diagnostic_example_count'] = len(prompt['examples'])
    obj['diagnostic_case_ids'] = [e['query_id'] for e in prompt['examples']]
    obj['rationale_schema'] = 'evidence-v3'
    obj['prompt_version'] = REFLECTION_PROMPT_VERSION
    obj['diagnostic_input_budget'] = None
    obj['diagnostic_input_truncated'] = False
    obj['diagnostic_selection'] = bundle['selection'] if bundle else None
    obj['rationale_status'] = 'model_reported_hypotheses_not_verified_causes'
    return obj


def generate(client, proposal, parent_code, policy_context=None):
    with trace_span('codegen', input={'proposal':proposal}, metadata={'role':'codegen'}) as observation:
        source, cost = _generate(client, proposal, parent_code, policy_context)
        observation.update(output={'source':source, 'source_sha256':hashlib.sha256(source.encode()).hexdigest(),
                                   'cost':cost})
        return source, cost


def _generate(client, proposal, parent_code, policy_context=None):
    instruction = '''Produce a standalone normal Python memory-policy module. Return JSON {"code":"..."}.
Provide callable initialize(config), ingest(state,session), search(state,query,filters).
Normal Python classes, helpers, lambdas, annotations, exceptions, comprehensions, decorators, default
arguments, standard builtins, and imports of installed libraries are supported. No syntax/name whitelist.
State may be a custom class or any object that cloudpickle can snapshot. Host handles opaque snapshots;
do not add a serializable declaration or convert your state to JSON. Host snapshots your state and policy
module together. Each search restores an independent snapshot, including policy-module globals; its
mutations are discarded. Only ingest advances persistent memory. External module/process state is not
automatically persisted. Session is a raw chunk with id,text,session_id,persona_id.
Search must return registered raw IDs (at most 5), within the patient/checkpoint filters.
Filesystem access is read-only and limited to installed runtime libraries; no private dataset files,
network or process control. Use passed historical records as data. Do not modify the actor/evaluator,
produce answers, or embed reference answers, patient-specific constants, or evaluation questions.
The trusted parent is baseline host code; implement the same three public entrypoints with your own design.'''
    instruction += '''
Implement the proposal's mechanism faithfully, including substantial internal redesign when proposed.
The starter and parent are interface references, not architectural templates to preserve.
Return the complete executable module in one JSON object; do not abbreviate with ellipses,
omit helper implementations, or return a diff. Do not add unrelated behavior changes just to be novel.'''
    instruction += '''
The supplied parent is standalone and includes its actual imports and helper functions.
Use runtime_context to understand the language and active configuration. Chinese text may have no
word-separating spaces; preserve useful partial-fact retrieval when changing text processing.
Alternative tokenization, indexing and memory representations are allowed; no specific tokenizer
or baseline architecture is mandatory.'''

    response = client.complete([{'role':'system','content':instruction},
                                {'role':'user','content':json.dumps({'proposal':proposal,'parent_code':parent_code,
                                                                   'legal_starter':PARENT_CODE,
                                                                   'runtime_context':dict(policy_context or {})},ensure_ascii=False)}],
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
