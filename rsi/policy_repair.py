"""A narrow, auditable tokenizer repair for the three historical candidates."""
from __future__ import annotations

import ast
import hashlib
import inspect

from .policy import _grams
from .sandbox import validate_source


def repair_tokenizer(source):
    tree = validate_source(source)
    targets = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == '_grams']
    if len(targets) != 1 or targets[0].decorator_list:
        raise ValueError('Expected one undecorated module-level _grams function')
    target = targets[0]
    helper = ast.parse(inspect.getsource(_grams)).body[0]
    # Keep the baseline helper dependency local; all other candidate bytes,
    # including imports, caching, recency and reranking, stay unchanged.
    helper.body.insert(1, ast.Import(names=[ast.alias(name='re')]))
    replacement = ast.unparse(ast.fix_missing_locations(helper)) + '\n'
    lines = source.splitlines(keepends=True)
    prefix = ''.join(lines[:target.lineno-1])
    suffix = ''.join(lines[target.end_lineno:])
    repaired = prefix + replacement + suffix
    validate_source(repaired)
    return repaired, {
        'method': 'replace_only_grams_with_baseline_character_bigrams',
        'source_sha256': hashlib.sha256(source.encode()).hexdigest(),
        'repaired_sha256': hashlib.sha256(repaired.encode()).hexdigest(),
        'original_function_lines': [target.lineno, target.end_lineno],
        'other_source_bytes_preserved': True,
    }
