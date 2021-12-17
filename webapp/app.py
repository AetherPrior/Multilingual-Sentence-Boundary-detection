import torch
import os
import yaml
import argparse
import json

from dotmap import DotMap
from autopunct.wrappers.BertPunctuatorWrapper import BertPunctuatorWrapper
from flask import render_template, request, Flask, session

parser = argparse.ArgumentParser(description='Webapp for SUD')
parser.add_argument('--docker',action='store_true',help='Specify if running inside docker container',default=False)

def get_config_from_yaml(yaml_file):
    with open(yaml_file, 'r') as config_file:
        config_yaml = yaml.load(config_file, Loader=yaml.Loader)
    # Using DotMap we will be able to reference nested parameters via attribute such as x.y instead of x['y']
    config = DotMap(config_yaml)
    return config

root_model_path = '../data-webapp'
dual_model_path = 'xlm-roberta-base-epoch-2.pth'
malay_model_path = 'xlm-roberta-base-ms.pth'
curr_path = 'dual'
ptype = 'all'





model = None

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('app.html')

@app.route('/lang_change',methods=['POST'])
def lang_change():
    """
    To offer a provision to change the language of the model
    :returns: The html page
    :effect: Model is changed 
    """
    global model, curr_path, root_model_path, malay_model_path, dual_model_path, ptype
    try: 
        ptype = request.form['punc_type']
    except KeyError:
        try:
            print(request.json)
            message = request.json['fav_language']
            ptype = request.json['punc_type']
        except Exception:
            print("Exception HIT!")
            checked = "checked" if curr_path == 'dual' else None
            checked_ms = "checked" if curr_path == 'ms' else None
            checked_all = "checked" if ptype == 'all' else None
            checked_end = "checked" if ptype == 'period' else None
            res = render_template('app.html',checked=checked, checked_ms=checked_ms,checked_all=checked_all,checked_end=checked_end)
            return res
    
    checked = "checked" if curr_path == 'dual' else None
    checked_ms = "checked" if curr_path == 'ms' else None
    checked_all = "checked" if ptype == 'all' else None
    checked_end = "checked" if ptype == 'period' else None

    print(checked_all,checked_end)
    res = render_template('app.html',checked=checked, checked_ms=checked_ms, checked_all=checked_all,checked_end=checked_end)
    return res

def sentenceCase(inSentence, lowercaseBefore):
    inSentence = '' if (inSentence is None) else inSentence
    if lowercaseBefore:
        inSentence = inSentence.lower()
    
    ## capitalize first letter
    words = inSentence.split()
    words[0] = words[0].capitalize() 

    ## finish the rest
    for i in range(0,len(words)-1):
        if words[i][-1] in ['.','?']:
            words[i+1] = words[i+1].capitalize()
    
    inSentence = " ".join(words)
    return inSentence    

@app.route('/predict', methods=['POST'])
def predict():
    try:
        message = request.form['message']
    except Exception:
        try:
            message = request.json['message'].strip()
        except Exception:
            return json.dumps({})

    if len(message) > 0:
        text = model.predict(message,ptype=ptype)
    else:
        text = ""
    res = render_template('app.html', prediction=sentenceCase(text,lowercaseBefore=True))
    return res

@app.route('/backend_predict',methods=['POST']) # content_type: application/json
def backend_predict():
    try:
        message = request.json['message'].strip()
    except Exception:
        return json.dumps({})
    text = model.predict(message,ptype=ptype)
    res = {'prediction': sentenceCase(text, lowercaseBefore=True)}
    return json.dumps(res)


if __name__ == '__main__':
    args = parser.parse_args()
    if args.docker:
        root_model_path = '/data'
    
    model = BertPunctuatorWrapper(get_config_from_yaml('./config-XLM-roberta-base-uncased.yaml'),torch.load(os.path.join(root_model_path,dual_model_path),map_location=torch.device('cpu')))  
    trace = torch.jit.trace(model._classifier,torch.LongTensor([[1,1,1,1,1]]))
    with open('saved_model.pth','wb') as f:
        torch.jit.save(trace,f)
    #from waitress import serve
    #serve(app,host='0.0.0.0',port=5000)
    #app.run(host="0.0.0.0",debug=True)

