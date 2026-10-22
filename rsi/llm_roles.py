"""Versioned, secret-free YAML selection of the five LLM workflow nodes."""
import copy
import hashlib
import json
import math
import os
import re
from pathlib import Path
from urllib.parse import urlsplit
from .model import ModelClient

NODES = ('actor','judge','diagnosis','proposals','codegen')
FIELDS = {'model','base_url','temperature','max_tokens','timeout_s','api_key_env'}
DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / 'configs' / 'llm-roles.yaml'


def _validate(values):
    if set(values)-FIELDS:
        raise ValueError('Unknown LLM settings; credentials must use api_key_env, never plaintext')
    for key,value in values.items():
        if key=='model' and (not isinstance(value,str) or not value.strip()):
            raise ValueError('model must be a nonempty served model name')
        if key=='base_url' and value is not None:
            if not isinstance(value,str):raise ValueError('base_url must be a URL or null')
            url=urlsplit(value)
            if url.scheme not in ('http','https') or not url.hostname or url.username or url.password or url.query or url.fragment:
                raise ValueError('base_url must be HTTP(S), without credentials, query or fragment')
        if key=='api_key_env' and value is not None:
            if not isinstance(value,str) or not re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]*',value):
                raise ValueError('api_key_env must name an environment variable')
        if key=='max_tokens' and value is not None and (type(value) is not int or value<=0):
            raise ValueError('max_tokens must be a positive integer or null')
        if key in ('temperature','timeout_s'):
            if type(value) not in (int,float) or not math.isfinite(value):
                raise ValueError(key+' must be finite numeric')
            if key=='temperature' and not 0<=value<=2:raise ValueError('temperature must be between 0 and 2')
            if key=='timeout_s' and value<=0:raise ValueError('timeout_s must be positive')


def resolve(path,config):
    import yaml
    # Reject duplicate YAML keys rather than silently taking the last value.
    class Loader(yaml.SafeLoader):pass
    def mapping(loader,node,deep=False):
        result={}
        for key,value in node.value:
            k=loader.construct_object(key,deep=deep)
            if k in result:raise ValueError('Duplicate YAML key')
            result[k]=loader.construct_object(value,deep=deep)
        return result
    Loader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,mapping)
    raw=Path(path).read_bytes()
    try:spec=yaml.load(raw,Loader=Loader)
    except yaml.YAMLError:raise ValueError('Invalid YAML syntax in LLM configuration') from None
    if not isinstance(spec,dict) or spec.get('version')!='llm-roles-v1' or set(spec)-{'version','defaults','roles'}:
        raise ValueError('Expected llm-roles-v1 with defaults and roles')
    defaults=spec.get('defaults') or {};roles=spec.get('roles') or {}
    if not isinstance(defaults,dict) or not isinstance(roles,dict) or set(roles)-set(NODES):
        raise ValueError('Unknown workflow role or invalid defaults')
    _validate(defaults)
    resolved={}
    for node in NODES:
        role=roles.get(node) or {}
        if not isinstance(role,dict):raise ValueError('Role settings must be a mapping')
        _validate(role)
        generation=config['optimizer_generation_settings'].get('codegen' if node=='codegen' else 'reflexion',{}) if node in ('diagnosis','proposals','codegen') else {}
        value={'model':config['judge_model'] if node=='judge' else config['model'],
               'base_url':config.get('judge_url') if node=='judge' else None,
               'api_key_env':'RSI_API_KEY','temperature':0.,'max_tokens':config['actor_max_tokens'],'timeout_s':180.,
               **generation,**defaults,**role}
        _validate(value);resolved[node]=value
    return {'version':'llm-roles-v1','source_sha256':hashlib.sha256(raw).hexdigest(),
            'resolved':resolved,'resolved_sha256':hashlib.sha256(json.dumps(resolved,sort_keys=True).encode()).hexdigest()}


def install(config,path):
    profile=resolve(path,config)
    config['llm_roles']=profile
    # Existing report labels must remain truthful for actor and judge.
    config['model']=profile['resolved']['actor']['model']
    config['judge_model']=profile['resolved']['judge']['model']
    config['actor_max_tokens']=profile['resolved']['actor']['max_tokens']
    if 'judge_max_tokens' in config:
        config['judge_max_tokens']=profile['resolved']['judge']['max_tokens']
    return profile


class RoutedClient:
    """Maintain per-client call accounting while dispatching to independent models."""
    def __init__(self,config,endpoint):
        self.base_url=endpoint;self.calls=[];self.clients={}
        for node,settings in config['llm_roles']['resolved'].items():
            settings=copy.deepcopy(settings)
            env=settings.pop('api_key_env');key=os.getenv(env,'') if env else ''
            base=settings.pop('base_url') or endpoint
            model=settings.pop('model')
            self.clients[node]=ModelClient(base_url=base,model=model,api_key=key,**settings)

    def _complete(self,node,messages,**kwargs):
        client=self.clients[node];start=len(client.calls)
        from .tracing import trace_span
        try:
            with trace_span('llm.route',metadata={'workflow_node':node,'configured_model':client.model,
                                                'base_url':client.base_url}):
                return client.complete(messages,**kwargs)
        finally:self.calls.extend({**call,'workflow_node':node} for call in client.calls[start:])

    def complete(self,messages,role='actor',**kwargs):
        node=role if role in self.clients else 'proposals'
        return self._complete(node,messages,role=role,**kwargs)

    def for_stage(self,stage):
        if stage not in ('diagnosis','proposals'):raise ValueError('Unknown reflexion stage')
        owner=self
        class Stage:
            @property
            def calls(self):return owner.calls
            def complete(self,messages,**kwargs):
                return owner._complete(stage,messages,**kwargs)
        return Stage()
