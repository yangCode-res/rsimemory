"""Offline synthetic probes and archived behavior comparison; no LLM calls."""
import argparse,json,hashlib
from pathlib import Path
from rsi.policy import GeneratedPolicy

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--run',required=True);a=ap.parse_args();r=Path(a.run)
 cfg=json.loads((r/'config.json').read_text());cfg['semantic_weight']=0
 chunks=[{'id':f'probe-{i}','persona_id':999,'session_id':i+1,'text':f'2024-01-{i+1:02d} blood pressure 血压 {110+i*5} mmHg 因为运动导致改善 option A coffee option B walking'} for i in range(8)]
 queries=['血压多少','2024-01-03 血压','具体数值 125 mmHg','Which option A coffee B walking','为什么运动导致改善','when blood pressure 2024-01-05']
 sources={'plain-bm25':r/'parent-policy.py',**{p.name:p/'policy.py' for p in sorted((r/'branches').iterdir()) if p.is_dir()}}
 outputs={}
 for n,p in sources.items():
  policy=GeneratedPolicy(p.read_text(),cfg)
  for c in chunks:policy.ingest(c)
  snap=policy.snapshot();seq=[]
  for q in queries:
   policy.restore(snap);seq.append([v['id'] for v in policy.search(q,{'persona_id':999,'checkpoint':8})])
  outputs[n]={'source_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'probe_ids':seq}
 tree=json.loads((r/'search-state.json').read_text())['tree']['nodes']
 for n,d in outputs.items():
  if n=='plain-bm25':continue
  parent=tree[n]['parent'];d['parent']=parent;d['synthetic_equal_to_parent']=sum(x==y for x,y in zip(d['probe_ids'],outputs[parent]['probe_ids']))
  d['archived']={}
  for fold in ['train','validation']:
   def load(name):
    p=r/(name if name=='plain-bm25' else 'branches/'+name)/fold/'answers.jsonl'
    return [json.loads(l) for l in p.read_text().splitlines()]
   x,y=load(parent),load(n);assert len(x)==len(y)
   assert all((a['persona_id'],a['query_id'])==(b['persona_id'],b['query_id']) for a,b in zip(x,y))
   d['archived'][fold]={'total':len(x),**{k+'_equal':sum(a[k]==b[k] for a,b in zip(x,y)) for k in ['score','answer','trajectory','evidence']}}
 dest=r/'audit-20261023/behavior-validation.json';dest.write_text(json.dumps({'synthetic_only':True,'probe_count':len(queries),'results':outputs},indent=2));print(dest)
 for n,d in outputs.items():print(n,{k:v for k,v in d.items() if k not in ['probe_ids','source_sha256']})
if __name__=='__main__':main()
