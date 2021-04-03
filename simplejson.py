#!/usr/bin/python
# Invoke-Expression "python .\simplejson.py"
import json
import pystache
from pathlib import Path

class BillingEntity:

    def __init__(self, d):
        self.id = d['id']
        self.code = d['code']
        self.name = d['name']
        self.upperName = self.name.upper().replace(' ', '_')
    
    def __repr__(self):
        return "%s(%r)" % (self.__class__, self.__dict__)

#data = [{"name": "Jane", "age": 17}, {"name": "Thomas", "age": 27}]

#json_data = json.dumps(data)
#print(repr(json_data))

with open('config.json') as f:
    config = json.load(f)
    for k,v in config.items():
        print(f'{k}: {v}')
        if k == 'display':
            print(f'{k}: {v // 8}')
        if k == 'countries':
            for c in v:
                print(c)
    t = 'Using {{theme}} in size {{size}} but is this shown {{#splashscreen}}Yes{{/splashscreen}}{{^splashscreen}}No{{/splashscreen}}'
    print(pystache.render(t, config))

    config['len'] = len(config['countries'])
    tmpl = Path('tmpl.mustache').read_text()
    output = pystache.render(tmpl, config)
    output = Path('tmpl.html').write_text(output)

with open('be.json') as f:
    data = json.load(f)
    xs = []
    for be in data['data']:
        print(be)
        a = BillingEntity(be)
        print(a)
        xs.append(a)
    
    #print(xs)
    tmpl = Path('comms.mustache').read_text()
    output = pystache.render(tmpl, {'data':xs})
    output = Path('comms.xml').write_text(output)
