"""Frozen replay, tool authority, role accounting, and persisted evaluation."""
from __future__ import annotations
import copy
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time
from .actor import Actor, ACTOR_SYSTEM_PROMPT
from .data import chunk_session
from .evaluation import evaluate, aggregate
from .tracing import trace_span


def dump(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    # Readers and periodic archival see either the old complete metadata or
    # the new complete metadata, never a file truncated during rewriting.
    fd, temporary = tempfile.mkstemp(prefix='.rsi-atomic-', suffix='.tmp', dir=path.parent)
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as stream:
            stream.write(json.dumps(value, ensure_ascii=False, indent=2))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


class HostTools:
    def __init__(self, policy, persona_id, checkpoint, top_k=5, context_chars=16000,
                 on_operation=None):
        self.policy = copy.deepcopy(policy)
        self.persona_id, self.checkpoint = persona_id, checkpoint
        self.top_k, self.context_chars = top_k, context_chars
        self.audit = []
        self.on_operation = on_operation
        self.visible = {}
        self.registry = copy.deepcopy(getattr(policy, '_registry', getattr(policy, '_by_id', {})))

    def search(self, query, filters=None):
        with trace_span('memory.search', kind='retriever',
                        input={'query':query, 'filters':filters}) as observation:
            rows = self._search(query, filters)
            observation.update(output=rows)
            return rows

    def _search(self, query, filters=None):
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
        self._record({'action': 'search', 'seconds': time.perf_counter()-start,
                      'ids': [r['id'] for r in rows]})
        return [{'id': r['id'], 'text': self.visible[r['id']]['text'][:450],
                 'session_id': self.visible[r['id']]['session_id']} for r in rows]

    def read(self, ids):
        with trace_span('memory.read', kind='tool', input={'ids':ids}) as observation:
            rows = self._read(ids)
            observation.update(output=rows)
            return rows

    def _read(self, ids):
        if len(ids) > self.top_k or len(set(ids)) != len(ids):
            raise PermissionError('Invalid read cardinality')
        if any(cid not in self.visible for cid in ids):
            raise PermissionError('read requires an ID from this question search')
        budget, result = self.context_chars, []
        for cid in ids:
            r = self.visible[cid]; text = r['text'][:budget]; budget -= len(text)
            result.append({'id': cid, 'text': text, 'session_id': r['session_id']})
        self._record({'action': 'read', 'ids': list(ids), 'bytes': sum(len(r['text'].encode()) for r in result)})
        return result

    def _record(self, operation):
        self.audit.append(operation)
        if self.on_operation is not None:
            self.on_operation(operation)


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
    path = Path(output)
    with trace_span('replay', metadata={'fold':path.name, 'policy':path.parent.name}) as observation:
        result = _replay(sessions, queries, ids, policy_factory, actor_client, judge_client, config, output)
        observation.update(output=result[0])
        return result


def _replay(sessions, queries, ids, policy_factory, actor_client, judge_client, config, output):
    out = Path(output); out.mkdir(parents=True, exist_ok=False)
    answers, operations, costs = [], [], []
    begin = time.perf_counter()
    answer_path, operation_path = out/'answers.jsonl', out/'operations.jsonl'
    progress_path = out/'progress.json'
    ids = list(ids)
    id_set = set(ids)
    total_questions = sum(1 for q in queries if q['persona_id'] in id_set)
    progress = {'status': 'running', 'total_questions': total_questions,
                'completed': 0, 'scored': 0, 'abnormal': 0}

    def save_progress(status=None):
        if status is not None:
            progress['status'] = status
        dump(progress_path, progress)

    def append_jsonl(file, row):
        file.write(json.dumps(row, ensure_ascii=False, separators=(',', ':'))+'\n')
        file.flush()

    save_progress()
    active_query = False
    try:
        with answer_path.open('w', encoding='utf-8') as answer_file, \
                operation_path.open('w', encoding='utf-8') as operation_file:
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
                        operation = {'event': 'ingest', 'persona_id': pid, 'session_id': session['session_id'],
                                     'memory_id': chunk['id'], 'seconds': time.perf_counter()-start}
                        operations.append(operation)
                        append_jsonl(operation_file, operation)
                    for q in pending.get(session['session_id'], []):
                        active_query = True

                        def persist_tool_operation(row, query_id=q['query_id']):
                            operation = {'event': 'tool', 'query_id': query_id, **row}
                            operations.append(operation)
                            append_jsonl(operation_file, operation)

                        actor = Actor(actor_client, config['max_steps'], config['context_chars'])
                        tools = HostTools(policy, pid, session['session_id'], config['top_k'],
                                          config['context_chars'], on_operation=persist_tool_operation)
                        with trace_span('question', kind='agent', new_trace=True,
                                        input={'question':q['question']},
                                        metadata={'query_id':q['query_id'], 'checkpoint':q['session_id'],
                                                  'query_type':q['query_type']}) as question_span:
                            response = actor.answer(q['question'], tools)
                            if response['status'] not in ('completed', 'forced_final'):
                                grade = {'score': None, 'status': 'actor_failed', 'reason': response['status'], 'call_log': []}
                            else:
                                with trace_span('grade', kind='evaluator') as grade_span:
                                    grade = evaluate(q, response['answer'], judge_client)
                                    grade_span.update(output=grade)
                            question_span.update(output={'answer':response['answer'], 'score':grade['score'],
                                                         'status':grade['status'], 'reason':grade['reason']})
                        row = {'persona_id': pid, 'query_id': q['query_id'], 'session_id': q['session_id'],
                               'query_type': q['query_type'], 'question': q['question'],
                               'answer': response['answer'], 'actor_status': response['status'],
                               'trajectory': response['trajectory'], 'evidence': response['evidence'],
                               'tool_audit': tools.audit, 'score': grade['score'], 'status': grade['status'],
                               'reason': grade['reason'], 'call_log': response['call_log']+grade['call_log']}
                        answers.append(row)
                        append_jsonl(answer_file, row)
                        progress['completed'] += 1
                        if type(grade['score']) is int and grade['score'] in (0, 1):
                            progress['scored'] += 1
                        else:
                            progress['abnormal'] += 1
                        active_query = False
                        save_progress()
                costs.append(dict(policy.cost))
    except BaseException:
        if active_query:
            progress['abnormal'] += 1
        save_progress('failed')
        raise
    try:
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
        save_progress('completed')
        return report, answers
    except BaseException:
        save_progress('failed')
        raise


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
