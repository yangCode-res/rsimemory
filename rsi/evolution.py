"""Training-only hypothesis/code generation, tree backpropagation and Pareto."""
from __future__ import annotations
import copy
import hashlib
import json
import math
from .sandbox import validate_source
from .tracing import trace_span
from .diagnostics import build_training_diagnostics
from .reflexion_protocol import (DIAGNOSIS_SCHEMA, PROPOSAL_SCHEMA,
                                 call_validated_stage, validate_diagnosis,
                                 validate_proposals)

REFLECTION_PROMPT_VERSION = 'explore-v4-structured-reflexion'


def reflect(client, answers, query_lookup, interventions, parent_code, with_diagnosis=True,
            policy_context=None, diagnostics=None, max_corrections=1, artifact_dir=None, proposal_count=3):
    with trace_span('reflexion', metadata={'role':'reflexion', 'evidence_scope':'train_only'}) as observation:
        result = _reflect(client, answers, query_lookup, interventions, parent_code, with_diagnosis,
                          policy_context, diagnostics, max_corrections, artifact_dir, proposal_count)
        observation.update(output=result)
        return result


def _reflect(client, answers, query_lookup, interventions, parent_code, with_diagnosis=True,
             policy_context=None, diagnostics=None, max_corrections=1, artifact_dir=None, proposal_count=3):
    bundle = ((diagnostics if diagnostics is not None else
               build_training_diagnostics(answers, query_lookup)) if with_diagnosis else None)
    examples = bundle['examples'] if bundle else []
    context = dict(policy_context or {})
    if not with_diagnosis:
        context.pop('parent_train_metrics',None)
    prompt = {'task': 'Improve memory-policy AVG across all task families under measured costs; answering system is frozen.',
              'prompt_version':REFLECTION_PROMPT_VERSION, 'proposal_count':proposal_count,
              'parent_code': parent_code, 'parent_context':context, 'examples': examples,
              'training_diagnostics':bundle['summary'] if bundle else None,
              'diagnostic_selection':bundle['selection'] if bundle else None,
              'interventions': interventions if with_diagnosis else [],
              'diagnostic_excerpts': False,
              'diagnostic_input_budget': None,
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
              'constraints': 'Do not insert references or answers into policy code. Changes only affect memory state and retrieval. '
                             'Only cite case IDs actually supplied in examples and exact cluster labels. '
                             'Do not claim measured memory leaks, latency, or causation without supplied measurements. '
                             'Use the all-training distributions to assess how representative each selected case is. '
                             'Patient, task, time and observable-error groups are coverage axes, not verified causal clusters. '
                             'Source-session hits are proxies: they do not prove the required facts were retrieved. '
                             'Compare failures with supplied successful controls; do not assume every wrong answer has the same cause. '
                             'These are explicit hypotheses for audit, not requests for private chain-of-thought.'}
    # No application-level input budget: preserve the full parent and every
    # selected case, list and intervention. Physical provider limits are errors,
    # never a reason to silently truncate or drop diagnostic controls.
    serialized = json.dumps(prompt, ensure_ascii=False)
    with trace_span('diagnostic-bundle', input=prompt,
                    metadata={'actual_model_input':False, 'input_chars':len(serialized),
                              'diagnostic_example_count':len(prompt['examples']),
                              'parent_code_sha256':hashlib.sha256(parent_code.encode()).hexdigest(),
                              'diagnostic_input_budget':None,
                              'diagnostic_selection_version':bundle['selection']['version'] if bundle else None}) as observation:
        observation.update(output={'selected_case_ids':[e['query_id'] for e in prompt['examples']],
                                   'selection':bundle['selection'] if bundle else None,
                                   'training_summary':bundle['summary'] if bundle else None,
                                   'input_truncated':False})
    system = ('You design and diagnose memory-management policies offline. The answering agent is frozen. '
              'Treat supplied examples and measurements as the full evidence; distinguish observations from hypotheses. '
              'Observed error groups do not establish causes. Do not invent evidence or report causal truth as verified. '
              'Return valid JSON only. Explain enough for an independent reviewer to test each mechanism; '
              'there is no arbitrary field length limit.')
    actual_ids = [e['query_id'] for e in prompt['examples']]
    stage_records = {}
    stage_input_chars = {}

    def stage_messages(stage, extra=None):
        stage_payload = dict(prompt)
        stage_payload['stage'] = stage
        if stage == 'diagnosis':
            stage_payload['stage_instructions'] = (
                'Report concrete observed failures in observed_failure. Keep causes in cause_hypotheses, '
                'with supporting observations, counter-evidence, confidence and a discriminating test. '
                'When attribution is unsupported, use attribution_status unresolved and an empty hypotheses list. '
                'Never fill an explanation field with only low, medium, or high. Return only clusters.')
        else:
            stage_payload['stage_instructions'] = (
                f'Return exactly {proposal_count} mechanism-diverse proposals. Ground any training-case motivation in the '
                'supplied examples and diagnosis. Code inspection or exploratory proposals may have empty case links. '
                'Include an auditable rationale, mechanism, falsification test and uncertainty.')
            if extra is not None:
                stage_payload['diagnosis'] = extra
        serialized_stage = json.dumps(stage_payload, ensure_ascii=False)
        messages = [{'role': 'system', 'content': system}, {'role': 'user', 'content': serialized_stage}]
        stage_input_chars[stage] = len(serialized_stage)
        with trace_span('diagnostic-input' if stage == 'diagnosis' else 'proposal-input',
                        input=stage_payload, metadata={'actual_model_input':True,
                        'input_chars':len(serialized_stage), 'stage':stage,
                        'diagnostic_example_count':len(examples), 'diagnostic_input_budget':None}) as span:
            span.update(output={'selected_case_ids':actual_ids,'input_truncated':False})
        return messages

    diagnosis_obj = {'clusters': []}
    if with_diagnosis:
        def diagnosis_validator(value):
            validate_diagnosis(value, actual_ids)
        messages = stage_messages('diagnosis')
        with trace_span('reflexion.diagnosis', input=messages,
                        metadata={'role': 'reflexion', 'stage': 'diagnosis', 'evidence_scope': 'train_only'}) as span:
            diagnosis_obj, diagnosis_record = call_validated_stage(
                client, 'diagnosis', messages, DIAGNOSIS_SCHEMA, diagnosis_validator,
                max_corrections=max_corrections, artifact_dir=artifact_dir)
            stage_records['diagnosis'] = diagnosis_record
            span.update(output=diagnosis_record)

    cluster_labels = [cluster['label'] for cluster in diagnosis_obj['clusters']]
    proposal_messages = stage_messages('proposals', diagnosis_obj)
    with trace_span('reflexion.proposals', input=proposal_messages,
                    metadata={'role': 'reflexion', 'stage': 'proposals', 'evidence_scope': 'train_only'}) as span:
        proposal_schema = copy.deepcopy(PROPOSAL_SCHEMA)
        proposal_schema['properties']['proposals'].update(minItems=proposal_count, maxItems=proposal_count)
        proposal_obj, proposal_record = call_validated_stage(
            client, 'proposals', proposal_messages, proposal_schema,
            lambda value: validate_proposals(value, actual_ids, cluster_labels, proposal_count),
            max_corrections=max_corrections, artifact_dir=artifact_dir)
        stage_records['proposals'] = proposal_record
        span.update(output=proposal_record)

    obj = {'clusters': diagnosis_obj['clusters'], 'proposals': proposal_obj['proposals']}
    usage = {}
    latency_s = 0.0
    for stage_record in stage_records.values():
        for attempt in stage_record['attempts']:
            for key, value in (attempt.get('usage') or {}).items():
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    usage[key] = usage.get(key, 0) + value
            response = attempt.get('response')
            if isinstance(response, dict) and isinstance(response.get('latency_s'), (int, float)):
                latency_s += response['latency_s']
    obj['call'] = {'usage': usage, 'latency_s': latency_s, 'role': 'reflexion'}
    obj['stage_records'] = stage_records
    obj['stage_input_chars'] = stage_input_chars
    obj['input_chars'] = next(iter(stage_input_chars.values()))
    obj['diagnosis_status'] = 'completed' if with_diagnosis else 'disabled'
    obj['diagnostic_example_count'] = len(prompt['examples'])
    obj['diagnostic_case_ids'] = [e['query_id'] for e in prompt['examples']]
    obj['rationale_schema'] = 'evidence-v4-structured-diagnosis'
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
