import torch
import os
import yaml
import argparse

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
dual_model_path = 'xlm-roberta-base-epoch-3.pth'
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
    global model, curr_path, root_model_path, malay_model_path, dual_model_path, ptype
    try: 
        message = request.form['fav_language']
        ptype = request.form['punc_type']
    except KeyError:
        res = render_template('app.html',checked=None, checked_ms=None)
        return res
    
    if message == 'ms' and curr_path != 'ms':
        curr_path = 'ms'
        del model
        print('switching to malay')
        model = BertPunctuatorWrapper(get_config_from_yaml('./config-XLM-roberta-base-uncased.yaml'),torch.load(os.path.join(root_model_path,malay_model_path),map_location=torch.device('cpu')))  
    elif message == 'dual' and curr_path != 'dual':
        curr_path = 'dual'
        del model
        print('switching to eng_zh')
        model = BertPunctuatorWrapper(get_config_from_yaml('./config-XLM-roberta-base-uncased.yaml'),torch.load(os.path.join(root_model_path,dual_model_path),map_location=torch.device('cpu')))
 
    checked = "checked" if curr_path == 'dual' else None
    checked_ms = "checked" if curr_path == 'ms' else None

    print(checked, checked_ms)
    res = render_template('app.html',checked=checked, checked_ms=checked_ms)
    return res
    

@app.route('/predict', methods=['POST'])
def predict():
    try:
        message = request.form['message']
    except Exception:
        message = request.json['message'].strip()

    text = model.predict(message,ptype=ptype)
    res = render_template('app.html', prediction=text)
    return res


if __name__ == '__main__':
    args = parser.parse_args()
    if args.docker:
        root_model_path = '/data'
    
    model = BertPunctuatorWrapper(get_config_from_yaml('./config-XLM-roberta-base-uncased.yaml'),torch.load(os.path.join(root_model_path,dual_model_path),map_location=torch.device('cpu')))  
    app.run(host='0.0.0.0',debug=True)
