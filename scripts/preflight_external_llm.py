"""Check configured external LLM services using synthetic text before GPU warmup."""
import argparse,json,os,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from rsi.llm_roles import resolve
from rsi.model import ModelClient

def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);a=p.parse_args()
    config={'model':'qwen3.8-27b','judge_model':'qwen3.8-27b','actor_max_tokens':1024,
            'optimizer_generation_settings':{'reflexion':{},'codegen':{}}}
    profile=resolve(a.config,config);groups={}
    for role,v in profile['resolved'].items():
        if not v.get('base_url'):continue
        groups.setdefault((v['base_url'],v['model'],v.get('api_key_env')),[]).append(role)
    results=[]
    for (url,model,env),roles in groups.items():
        if env and not os.getenv(env):raise RuntimeError('External LLM credential environment variable is missing')
        client=ModelClient(url,model,api_key=os.getenv(env,'') if env else '',temperature=0,max_tokens=1024,timeout_s=90)
        schema={'type':'object','properties':{'ok':{'type':'boolean'}},'required':['ok'],'additionalProperties':False}
        response=client.complete([{'role':'user','content':'Synthetic connectivity check. Return only {"ok":true}.'}],
                                 role='reflexion',json_mode=True,json_schema=schema)
        if json.loads(response['text'])!={'ok':True}:raise ValueError('External model structured output check failed')
        results.append({'model':model,'base_url':url,'roles':roles,'usage':response.get('usage')})
    print(json.dumps({'status':'ok','input':'synthetic_only','checks':results}))
if __name__=='__main__':main()
