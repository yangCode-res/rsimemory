"""Package the trusted baseline with its module-level dependencies and API."""
from __future__ import annotations

import ast
import hashlib
import inspect

from . import policy

SOURCE_VERSION = 'standalone-baseline-v1'


def baseline_source():
    """Follow static dependencies of PlainPolicy in its first-party module.

    Generated parent policies already have standalone source artifacts; this
    builder is specifically for the trusted initial baseline. Do not bundle
    the host GeneratedPolicy/sandbox wrapper or project-relative imports.
    """
    text = inspect.getsource(policy)
    tree = ast.parse(text)
    bindings = {}
    for index, node in enumerate(tree.body):
        names = []
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            names = [node.name]
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            names = [a.asname or (a.name.split('.')[0] if isinstance(node, ast.Import)
                                 else a.name) for a in node.names]
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            names = [n.id for target in targets for n in ast.walk(target)
                     if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)]
        for name in names:
            bindings[name] = index
    selected = set()
    pending = ['PlainPolicy']
    while pending:
        index = bindings[pending.pop()]
        if index in selected:
            continue
        selected.add(index)
        node = tree.body[index]
        if isinstance(node, ast.ImportFrom) and node.level:
            raise ValueError('Baseline source depends on a project-relative import')
        pending.extend(n.id for n in ast.walk(node)
                       if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)
                       and n.id in bindings and bindings[n.id] not in selected)
    selected.update(i for i, node in enumerate(tree.body)
                    if isinstance(node, ast.ImportFrom) and node.module == '__future__')
    pieces = [ast.get_source_segment(text, tree.body[i]) for i in sorted(selected)]
    pieces.append('''def initialize(config):
    return PlainPolicy(config)


def ingest(state, session):
    state.ingest(session)
    return state


def search(state, query, filters):
    return [row["id"] for row in state.search(query, filters)]
''')
    source = '\n\n'.join(pieces) + '\n'
    compile(source, '<baseline-policy>', 'exec')
    return source


def source_manifest(source, language):
    return {'version': SOURCE_VERSION,
            'sha256': hashlib.sha256(source.encode()).hexdigest(),
            'language': language,
            'baseline_tokenization': 'Unicode character bigrams; casefold and collapse whitespace',
            'entrypoints': ['initialize', 'ingest', 'search'],
            'implementation_freedom': 'Alternative language-appropriate memory mechanisms are allowed'}
