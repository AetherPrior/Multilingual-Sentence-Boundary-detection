import torch
import argparse
import yaml 

from dotmap import DotMap
from autopunct.base.BaseWrapper import BaseWrapper
from autopunct.models.BertPunctuator import BertPunctuator

parser = argparse.ArgumentParser(description='Model trace export')
parser.add_argument('--path',type=str,help='Specify path to model with model file',required=True)

class BertPunctuatorWrapper(BaseWrapper):
    def __init__(self, config, checkpoint):
        super().__init__(config)

        self._classifier = BertPunctuator(config)
        self._classifier.load_state_dict(checkpoint['model_state_dict'])
        self._classifier.eval()

def get_config_from_yaml(yaml_file):
    with open(yaml_file, 'r') as config_file:
        config_yaml = yaml.load(config_file, Loader=yaml.Loader)
    # Using DotMap we will be able to reference nested parameters via attribute such as x.y instead of x['y']
    config = DotMap(config_yaml)
    return config




if __name__ == '__main__':
    argparser = parser.parse_args()
    model = BertPunctuatorWrapper(get_config_from_yaml('./config-XLM-roberta-base-uncased.yaml'),torch.load(argparser.path,map_location=torch.device('cpu')))
    with open('./saved_model.pth','wb') as f:
        trace = torch.jit.trace(model._classifier,torch.LongTensor([[1,1,1,1,1]]))
        torch.jit.save(trace,f)