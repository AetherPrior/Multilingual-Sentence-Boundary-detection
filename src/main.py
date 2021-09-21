from neural_punctuator.utils.data import get_config_from_yaml
from neural_punctuator.wrappers.BertPunctuatorWrapper import BertPunctuatorWrapper
import logging
import argparse
parser = argparse.ArgumentParser(description='arguments for the model')
parser.add_argument('--save-model', action='store_true',
                    help='save model or not')
parser.add_argument('--break-train-loop', action='store_true',
                    help='prevent training for debugging purposes')
parser.add_argument('--stage',type=str, default='')
parser.add_argument('--model-path',type=str,default='/content/drive/MyDrive/SUD_PROJECT/neural-punctuator/models-xlm-roberta/')
parser.add_argument('--data-path',type=str,default='/content/drive/MyDrive/SUD_PROJECT/neural-punctuator/dataset/dual/xlm-roberta-base/')
parser.add_argument('--num-epochs',type=int,default=-1)
parser.add_argument('--log-level',type=str,choices=['INFO','DEBUG','WARNING','ERROR'],default='INFO')
parser.add_argument('--save-n-steps',type=int,help='Save after n steps, default=1 epoch',default=-1)
parser.add_argument('--force-save',action='store_true',help='Force save, overriding all settings')

log_enum = {
    'INFO': logging.INFO,
    'DEBUG': logging.DEBUG,
    'WARNING': logging.WARNING,
    'ERROR': logging.ERROR
}

def override_arguments(config):
    args = parser.parse_args()
    
    logging.basicConfig(level=log_enum[args.log_level])
    
    if args.save_model:
        config.model.save_model=True
    
    if args.break_train_loop:
        if config.model.save_model:
            if args.force_save:
                logging.warning("Force saving despite break train loop!! Make sure you don't save too many epochs!!")
            else:
                logging.warning("Breaking train loop while save_model set to true. Overriding config to NOT save model")
                config.model.save_model=False
        config.debug.break_train_loop=True
    
    if args.num_epochs > 0:
        config.trainer.num_epochs = args.num_epochs
        logging.warning(f"config num_epochs overriden to {args.num_epochs}!!")
    
    if args.save_n_steps > 0:
        config.trainer.save_n_steps = args.save_n_steps
    
    config.data.data_path = args.data_path
    config.trainer.load_model = args.stage
    config.model.save_model_path = args.model_path
    
    logging.info("*******************************")
    logging.info(str(config))
    logging.info("*******************************")
    
    if args.model_path:
        logging.warning(f"Config model path has been overridden!!!! New path is: {config.model.save_model_path}")
    
    return config
    

if __name__ == '__main__':

    config = get_config_from_yaml(
        'neural_punctuator/configs/config-XLM-roberta-base-uncased.yaml')
    config = override_arguments(config)

    if config.model.save_model == False:
        print("WARNING, MODEL NOT BEING SAVED")
    if config.debug.break_train_loop == True:
        print("Warning, no training!")

    pipe = BertPunctuatorWrapper(config)
    pipe.train()

