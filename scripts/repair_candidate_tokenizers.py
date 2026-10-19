"""Create private, versioned fixes without editing the historical run."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rsi.cli import commit_policy
from rsi.policy import GeneratedPolicy
from rsi.policy_repair import repair_tokenizer


def chinese_smoke(source, config):
    policy = GeneratedPolicy(source, config, timeout_s=60)
    for sid in range(1, 22):
        text = ('user: 青松片引起皮疹，随后停用青松片。' if sid == 1
                else 'user: 今天讨论天气、饮食和运动安排。 ' + str(sid))
        policy.ingest({'id':f'synthetic-{sid}', 'persona_id':0, 'session_id':sid, 'text':text})
    result = policy.search('青松片皮疹停用', {'persona_id':0, 'checkpoint':21})
    return {'relevant_in_top5':any(r['id'] == 'synthetic-1' for r in result),
            'relevant_at_rank1':bool(result and result[0]['id'] == 'synthetic-1'),
            'returned_count':len(result)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run, out = args.run.resolve(), args.output.resolve()
    immutable = run.parent if (run.parent/'_archive-manifest.json').exists() else run
    if out.is_relative_to(immutable):
        raise ValueError('Repair artifacts must be outside the historical archive')
    if out.exists():
        raise FileExistsError('Use a fresh repair directory')
    os.umask(0o077)
    tree = json.loads((run/'tree.json').read_text())
    config = json.loads((run/'config.json').read_text())
    if config['language'] != 'zh':
        raise ValueError('This historical repair is scoped to the Chinese run')
    originals = {}
    for name in ('candidate-1', 'candidate-2', 'candidate-3'):
        source = (run/'branches'/name/'policy.py').read_text()
        parent = tree['nodes'][name]['commit']
        committed = subprocess.check_output(
            ['git', '-C', str(run/'policy-git'), 'show', parent+':policy.py'], text=True)
        if committed != source:
            raise ValueError('Historical source does not match its policy Git commit')
        originals[name] = (source, parent)
    out.mkdir(parents=True, mode=0o700)
    repo = out/'policy-git'
    subprocess.run(['git', 'clone', '--quiet', '--no-hardlinks', '--no-checkout',
                    str(run/'policy-git'), str(repo)], check=True)
    manifest = {'source_run':str(run), 'created_at':datetime.now(timezone.utc).isoformat(),
                'implementation_commit':subprocess.check_output(
                    ['git','-C',str(Path(__file__).resolve().parents[1]),'rev-parse','HEAD'],
                    text=True).strip(),
                'repair_version':'tokenizer-only-v1',
                'origin':'manual repair of historical candidates; not newly generated RSI proposals',
                'original_run_modified':False, 'model_calls':0, 'test_evaluated':False,
                'status':'validating', 'policies':{}}
    for name, (source, parent) in originals.items():
        repaired, metadata = repair_tokenizer(source)
        branch = name+'-tokenizer-fix'
        path = out/branch
        path.mkdir(mode=0o700)
        (path/'policy.py').write_text(repaired)
        before = chinese_smoke(source, config)
        after = chinese_smoke(repaired, config)
        if not after['relevant_in_top5']:
            raise RuntimeError('Repaired policy failed the synthetic Chinese retrieval check')
        commit = commit_policy(repo, repaired, branch, parent)
        metadata.update({'original_commit':parent, 'repair_commit':commit,
                         'source_path':str(path/'policy.py'),
                         'synthetic_before':before, 'synthetic_after':after,
                         'status':'smoke_validated_not_answer_evaluated'})
        manifest['policies'][branch] = metadata
        (out/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
        print(json.dumps({'policy':branch, **metadata}), flush=True)
    manifest['status'] = 'smoke_validated_not_answer_evaluated'
    (out/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')


if __name__ == '__main__':
    main()
