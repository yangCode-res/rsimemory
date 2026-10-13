"""Synthetic fixtures for integration checks; not benchmark measurements."""
import json
from .evaluation import TYPES


def dataset():
    sessions, queries = [], []
    for pid in range(1,7):
        sessions.append({'persona_id':pid,'session_id':10,'text':'fixture value ALPHA. option A: ALPHA. option B: BETA.'})
        for i,kind in enumerate(TYPES):
            queries.append({'persona_id':pid,'session_id':10,'query_id':f'fixture-{pid}-{i}',
                            'query_type':kind,'question':'What is the fixture value?' if kind != 'multiple_choice' else 'Select the correct option.',
                            'positive_answers':['ALPHA'],'reference_answers':['ALPHA'],
                            'options':{'A':'ALPHA','B':'BETA'},'correct_options':['A'],
                            'evidence':[{'content':'fixture value ALPHA','session_id':10}]})
    return sessions, queries


class FixtureClient:
    """Predictable fake server exercises protocols only, without real data."""
    def __init__(self): self.calls=[]
    def complete(self,messages,role='actor',json_mode=False,json_schema=None):
        if role == 'actor':
            question=messages[1]['content']
            if not any(m['content'].startswith('search result:') for m in messages):
                result={'action':'search','query':question,'filters':{}}
            else: result={'action':'final','answer':'A' if 'option' in question else 'ALPHA'}
        elif role == 'judge': result={'score':1,'reason':'synthetic fixture evaluator only'}
        elif role == 'reflexion':
            result={'clusters':[{'label':'fixture','case_ids':[],'attribution':'synthetic protocol test','uncertainty':'high'}],
                    'proposals':[{'direction':f'fixture-{i}','hypothesis':'protocol check','prediction':'runs','risk':'not research evidence'} for i in range(3)]}
        elif role == 'codegen':
            from .evolution import PARENT_CODE
            result={'code':PARENT_CODE}
        else: raise ValueError(role)
        call={'text':json.dumps(result),'usage':{},'latency_s':0.,'role':role}
        self.calls.append(call)
        return call
