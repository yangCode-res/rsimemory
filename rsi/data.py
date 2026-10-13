"""Separate public online views from private offline annotations."""
from __future__ import annotations
import hashlib
import json
import re
from pathlib import Path


def patient_splits(ids, seed=20261013):
    ids = sorted(set(ids), key=lambda x: hashlib.sha256(f'{seed}:{x}'.encode()).hexdigest())
    n = len(ids)
    if n < 3:
        raise ValueError('At least three patients required for disjoint splits')
    a, b = max(1, int(n*.6)), max(1, int(n*.2))
    return {'train': ids[:a], 'validation': ids[a:a+b], 'test': ids[a+b:]}


def normalize_query(row):
    q = dict(row)
    def decode(value, default):
        if isinstance(value, str):
            try: return json.loads(value)
            except ValueError: return default
        return value if value is not None else default
    answers = decode(q.get('answers'), [])
    def positive(entry):
        if not isinstance(entry, dict): return False
        flag = entry.get('is_correct')
        if hasattr(flag, 'item'): flag = flag.item()
        return flag is True or (type(flag) is int and flag == 1)
    positives = [str(a['content']) for a in answers if positive(a) and 'content' in a]
    q['positive_answers'] = positives
    q['reference_answers'] = positives
    q['evidence'] = decode(q.get('source_key_points'), [])
    metadata = decode(q.get('metadata'), {})
    q['causal_chain'] = metadata.get('reasoning_chain', [])
    q['required_facts'] = metadata.get('required_memory_nodes', [])
    option_text = q['question'].replace('\\n', '\n')
    options = dict(re.findall(r'(?:^|\n)\s*([A-Z])[.、．)]\s*([^\n]+)', option_text))
    if q.get('query_type') == 'multiple_choice':
        # Use the full option-labelled annotation set only in the private grader
        # view. This also handles inline options in the question text.
        for entry in answers:
            if isinstance(entry,dict) and isinstance(entry.get('content'),str):
                match=re.match(r'^\s*([A-Z])[.、．:：)]\s*(.*)',entry['content'],re.S)
                if match: options.setdefault(match.group(1),match.group(2).strip())
        q['options'] = options
        selected = []
        for gold in positives:
            prefix = re.match(r'^\s*([A-Z])(?:[.、．:：)]|\s|$)', gold)
            if prefix and prefix.group(1) in options:
                selected.append(prefix.group(1))
            for label, text in options.items():
                if gold.strip() == label or gold.strip() == text.strip(): selected.append(label)
            selected.extend(re.findall(r'(?<![A-Za-z])([A-Z])(?=$|[,、\s])', gold))
        # Some releases store the option letter in metadata, not content.
        explicit = metadata.get('correct_options', metadata.get('correct_option'))
        if explicit:
            selected.extend(explicit if isinstance(explicit, list) else [str(explicit)])
        q['correct_options'] = sorted(set(selected))
    return q


def load_medmemory(root, language='zh', mode='Efficient'):
    if mode != 'Efficient':
        raise NotImplementedError('Mixed injection mapping has not been verified; fail closed')
    import pandas as pd
    root = Path(root)/'data'/language
    d = pd.read_parquet(root/'dialogues.parquet')
    questions = pd.read_parquet(root/'queries.parquet')
    sessions = []
    for (pid, sid), group in d.groupby(['persona_id', 'session_id'], sort=True):
        turns = group.sort_values('turn', kind='stable')
        sessions.append({'persona_id': int(pid), 'session_id': int(sid),
                         'text': '\n'.join(str(r.role)+': '+str(r.content) for r in turns.itertuples())})
    return sessions, [normalize_query(q) for q in questions.to_dict('records')]


def chunk_session(session, size=1200, overlap=120):
    if size <= overlap or overlap < 0:
        raise ValueError('Invalid chunk settings')
    text = session['text']
    chunks = []
    for i, start in enumerate(range(0, len(text), size-overlap)):
        chunks.append({'id': f'p{session["persona_id"]}-s{session["session_id"]}-c{i}',
                       'persona_id': session['persona_id'], 'session_id': session['session_id'],
                       'text': text[start:start+size]})
        if start+size >= len(text): break
    return chunks


def representative_queries(queries, ids, max_checkpoint=None, per_type=None):
    eligible = [q for q in queries if q['persona_id'] in ids and
                (max_checkpoint is None or q['session_id'] <= max_checkpoint)]
    eligible.sort(key=lambda q: (q['persona_id'], q['session_id'], str(q['query_id'])))
    if per_type is None: return eligible
    counts, result = {}, []
    for q in eligible:
        kind = q['query_type']
        if counts.get(kind, 0) < per_type:
            result.append(q); counts[kind] = counts.get(kind, 0)+1
    return result
