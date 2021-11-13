import torch
import numpy as np
import yaml
from dotmap import DotMap
from autopunct.wrappers.BertPunctuatorWrapper import BertPunctuatorWrapper
from flask import render_template, request, Flask

def get_config_from_yaml(yaml_file):
    with open(yaml_file, 'r') as config_file:
        config_yaml = yaml.load(config_file, Loader=yaml.Loader)
    # Using DotMap we will be able to reference nested parameters via attribute such as x.y instead of x['y']
    config = DotMap(config_yaml)
    return config

checkpoint = torch.load('.\\data\\xlm-roberta-base-epoch-3.pth')

model = BertPunctuatorWrapper(get_config_from_yaml('.\config-XLM-roberta-base-uncased.yaml'),checkpoint)  

app = Flask(__name__)

@app.route('/')
def home():
    """ This is the homepage of our API.
    It can be accessed by http://127.0.0.1:5000/
    """
    return render_template('app.html')


@app.route('/predict', methods=['POST'])
def predict():
    """
    To make a prediction on one sample of the text
    satire or fake news
    :return: a result of prediction in HTML page
    """

    message = request.form['message']
    text = model.predict(message)
    res = render_template('app.html', prediction=text)
    return res


if __name__ == '__main__':
    app.run(debug=True)
