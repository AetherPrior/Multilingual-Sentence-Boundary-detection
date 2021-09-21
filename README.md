# Multilingual SBD using BERT models

## Source: ##

BERT Framework borrowed from [here](https://github.com/attilanagy234/neural-punctuator/)  

Modifications have been made to the notebooks, the data preprocessing, and the model architecture.  

## Installation and Usage: ##
### Virtual environment:  
It is **HIGHLY RECOMMENDED** to use a virtualenv for this project  
1. Setup is fairly easy, open up a terminal and do:  
        - `python -m venv /path/to/root/dire/.../.../venv_name`    
2. Then everytime you want to run the program, just do  
	- `source ./venv/bin/activate`

### Libraries:  	
Install the required libraries via the requirements.txt file  
	1. Activate your venv, and then do `pip install requirements.txt`  

### Dataset: ##

#### Creation: 
Source for dataset TODO (check iwslt). The `notebooks` folder has the necessary notebooks for the dataset creation. 


#### Pre-built:  

Get the dataset here. Store the dataset in any folder and provide the necessary arguments to the model  

The dataset folder structure is as follows:  

```
dataset/
|
|-dual/
|   |
|   |-xlm-roberta-base/
|            |-- train.pkl
|            |-- valid.pkl
|            |-- test.pkl
|-en
| |-...
|
|-zh
```
Use the `--data-path` argument to set the data directory:  
```
  --data-path DATA_PATH
                        path to dataset directory
```

### Running the program

Run the `main.py` file:

```  
usage: main.py [-h] [--save-model] [--break-train-loop] [--stage STAGE]
               [--model-path MODEL_PATH] [--data-path DATA_PATH]
               [--num-epochs NUM_EPOCHS]
               [--log-level {INFO,DEBUG,WARNING,ERROR}]
               [--save-n-steps SAVE_N_STEPS] [--force-save]

arguments for the model

optional arguments:
  -h, --help            show this help message and exit
  --save-model          save model
  --break-train-loop    prevent training for debugging purposes
  --stage STAGE         load model from checkpoint stage
  --model-path MODEL_PATH
                        path to model directory
  --data-path DATA_PATH
                        path to dataset directory
  --num-epochs NUM_EPOCHS
                        no. of epochs to run the model
  --log-level {INFO,DEBUG,WARNING,ERROR}
                        Logging info to be displayed
  --save-n-steps SAVE_N_STEPS
                        Save after n steps, default=1 epoch
  --force-save          Force save, overriding all settings
```

Sample command:

```
python main.py \
--model-path  '/content/drive/MyDrive/SUD_PROJECT/neural-punctuator/models-xlm-roberta-dual/' \
--num-epochs 2 \
--data-path '/content/drive/MyDrive/SUD_PROJECT/neural-punctuator/dataset/en/xlm-roberta-base/'  \
--stage 'xlm-roberta-base-epoch-1.pth'
```




