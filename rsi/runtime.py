"""Frozen replay, tool authority, role accounting, and persisted evaluation."""
from __future__ import annotations
import copy
import hashlib
import json
from pathlib import Path
import time
from .actor import Actor, ACTOR_SYSTEM_PROMPT
from .data import chunk_session
from .evaluation import evaluate, aggregate


def dump(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2))


class HostTools:
    def __init__(self, policy, persona_id, checkpoint, top_k=5, context_chars=16000):
        self.policy = copy.deepcopy(policy)
        self.persona_id, self.checkpoint = persona_id, checkpoint
        self.top_k, self.context_chars = top_k, context_chars
        self.audit = []
        self.visible = {}
        self.registry = copy.deepcopy(getattr(policy, '_registry', getattr(policy, '_by_id', {})))

    def search(self, query, filters=None):
        filters = dict(filters or {})
        if set(filters)-{'session_id', 'min_session_id'}:
            raise PermissionError('Scope fields are owned by the host')
        filters.update(persona_id=self.persona_id, checkpoint=self.checkpoint)
        start = time.perf_counter()
        rows = self.policy.search(str(query)[:4000], filters)[:self.top_k]
        for r in rows:
            original = self.registry.get(r['id'])
            if original is None or original['persona_id'] != self.persona_id or original['session_id'] > self.checkpoint:
                raise PermissionError('Unregistered or future evidence')
            self.visible[r['id']] = original
        self.audit.append({'action': 'search', 'seconds': time.perf_counter()-start,
                           'ids': [r['id'] for r in rows]})
        return [{'id': r['id'], 'text': self.visible[r['id']]['text'][:450],
                 'session_id': self.visible[r['id']]['session_id']} for r in rows]

    def read(self, ids):
        if len(ids) > self.top_k or len(set(ids)) != len(ids):
            raise PermissionError('Invalid read cardinality')
        if any(cid not in self.visible for cid in ids):
            raise PermissionError('read requires an ID from this question search')
        budget, result = self.context_chars, []
        for cid in ids:
            r = self.visible[cid]; text = r['text'][:budget]; budget -= len(text)
            result.append({'id': cid, 'text': text, 'session_id': r['session_id']})
        self.audit.append({'action': 'read', 'ids': list(ids), 'bytes': sum(len(r['text'].encode()) for r in result)})
        return result


class InterventionTools:
    """Training-only oracle evidence injection; never a ranked policy or score."""
    def __init__(self, query, visible_source_sessions):
        self.rows = []
        self.rejected = 0
        for i, kp in enumerate(query.get('evidence', [])):
            if not isinstance(kp,dict) or not kp.get('content'):
                self.rejected += 1; continue
            try: sid = int(kp['session_id'])
            except (KeyError, ValueError, TypeError):
                self.rejected += 1; continue
            if sid > 0 and sid in visible_source_sessions and sid <= query['session_id']:
                self.rows.append({'id': f'oracle-{i}', 'text': str(kp['content']),
                                  'session_id': sid})
            else: self.rejected += 1
    def search(self, query, filters=None): return copy.deepcopy(self.rows)
    def read(self, ids):
        allowed = {r['id']: r for r in self.rows}
        if any(cid not in allowed for cid in ids): raise PermissionError('Unknown intervention ID')
        return [copy.deepcopy(allowed[cid]) for cid in ids]


def call_cost(calls):
    roles = {}
    for c in calls:
        row = roles.setdefault(c.get('role', 'unknown'), {'calls': 0, 'seconds': 0., 'prompt_tokens': 0, 'completion_tokens': 0})
        row['calls'] += 1; row['seconds'] += c.get('latency_s', 0.)
        for k in ['prompt_tokens', 'completion_tokens']: row[k] += c.get('usage', {}).get(k, 0)
    return roles


def replay(sessions, queries, ids, policy_factory, actor_client, judge_client, config, output):
    out = Path(output); out.mkdir(parents=True, exist_ok=False)
    answers, operations, costs = [], [], []
    begin = time.perf_counter()
    for pid in ids:
        policy = policy_factory()
        pending = {}
        for q in queries:
            if q['persona_id'] == pid: pending.setdefault(q['session_id'], []).append(q)
        for session in sessions:
            if session['persona_id'] != pid: continue
            if not pending or session['session_id'] > max(pending): break
            for chunk in chunk_session(session, config['chunk_chars'], config['chunk_overlap']):
                start = time.perf_counter(); policy.ingest(chunk)
                operations.append({'event': 'ingest', 'persona_id': pid, 'session_id': session['session_id'],
                                   'memory_id': chunk['id'], 'seconds': time.perf_counter()-start})
            for q in pending.get(session['session_id'], []):
                actor = Actor(actor_client, config['max_steps'], config['context_chars'])
                tools = HostTools(policy, pid, session['session_id'], config['top_k'], config['context_chars'])
                response = actor.answer(q['question'], tools)
                if response['status'] not in ('completed', 'forced_final'):
                    grade = {'score': None, 'status': 'actor_failed', 'reason': response['status'], 'call_log': []}
                else:
                    grade = evaluate(q, response['answer'], judge_client)
                answers.append({'persona_id': pid, 'query_id': q['query_id'], 'session_id': q['session_id'],
                                'query_type': q['query_type'], 'question': q['question'],
                                'answer': response['answer'], 'actor_status': response['status'],
                                'trajectory': response['trajectory'], 'evidence': response['evidence'],
                                'tool_audit': tools.audit, 'score': grade['score'], 'status': grade['status'],
                                'reason': grade['reason'], 'call_log': response['call_log']+grade['call_log']})
                operations.extend({'event': 'tool', 'query_id': q['query_id'], **r} for r in tools.audit)
        costs.append(dict(policy.cost))
    for name, rows in [('answers', answers), ('operations', operations)]:
        with (out/(name+'.jsonl')).open('w') as f:
            for row in rows: f.write(json.dumps(row, ensure_ascii=False)+'\n')
    report = aggregate(answers)
    search_seconds = sum(r.get('seconds',0.) for r in operations if r.get('action')=='search')
    report.update(mode='Efficient', grading_rubric='custom-engineering-v0.2-not-paper-verbatim',
                  contract_version=config.get('contract_version','v0.2'),
                  fixture=config.get('fixture',False), scientific_result=False,
                  wall_seconds=time.perf_counter()-begin,
                  cost={'policy': costs, 'model_roles': call_cost([c for r in answers for c in r['call_log']]),
                        'search_seconds': search_seconds,
                        'policy_auxiliary_model': {'status': 'disabled'}},
                  fixed_prompt_sha256=hashlib.sha256(ACTOR_SYSTEM_PROMPT.encode()).hexdigest())
    dump(out/'report.json', report)
    return report, answers


def intervention(answers, query_lookup, client, judge_client, sessions, max_cases=2):
    results = []
    for row in answers:
        if row['score'] != 0: continue
        q = query_lookup[row['query_id']]
        visible = {s['session_id'] for s in sessions if s['persona_id']==q['persona_id'] and s['session_id']<=q['session_id']}
        tools = InterventionTools(q, visible)
        if not tools.rows: continue
        answer = Actor(client).answer(q['question'], tools)
        grade = evaluate(q, answer['answer'], judge_client) if answer['status'] in ('completed','forced_final') else {'score':None,'call_log':[]}
        results.append({'query_id': q['query_id'], 'type': 'replace_with_annotated_evidence',
                        'original_score': 0, 'intervention_score': grade['score'],
                        'actor_status': answer['status'], 'rejected_source_points': tools.rejected,
                        'diagnosis': 'evidence sensitivity only; not proof of unique cause',
                        'call_log': answer['call_log']+grade['call_log']})
        if len(results) >= max_cases: break
    return results
