# Web application 
## Deployment (with docker compose)

- Pull the image from dockerhub: https://hub.docker.com/repository/docker/aetherprior/sud
- Download the XLM-R traced model for the three languages from here: https://drive.google.com/drive/folders/15dXlprTXuvUVKmZRKd_1Pz8rCZr1BmPQ?usp=sharing

- Navigate to the webapp directory, and change the `../data-webapp` directory in the `docker-compose.yml` to the directory of the downloaded files
- In the terminal, type `docker compose up` . The image will be built and the webapp will be served at port 5000 at localhost

### Optional: Running your own trained model 
- Aside from the trained model, it is required to export the pytorch `jit trace` of the model, with a dummy input before running, to optimize on memory
- Export the trace of the trained model, using the `export_trace.py` script to the model directory 
  - Note that the model would be saved with the name `saved_model.pth` in the same directory. Move it to your volume mount directory
- Proceed with deployment as usual 

## Deployment (with docker)

- Pull the image from dockerhub: https://hub.docker.com/repository/docker/aetherprior/sud
- Download the XLM-R model for the three languages from here: https://drive.google.com/drive/folders/15dXlprTXuvUVKmZRKd_1Pz8rCZr1BmPQ?usp=sharing

Run the following command:  
```
docker run  -v /path/to/modeldata/dir:/data -p 127.0.0.1:5000:5000  <image name>
```  
The model should be served at port 5000 of localhost.  

## Send request to server

### To send a lang/punctuation change request

Send a POST request with the header as `{'Content-type': 'application/json'}` , and with the payload set as the following:  
```
{
  fav_language: 'dual' for eng+zh or 'ms' for malay,
  punc_type: 'all' or 'period'
}
```
Changing the model can take time; please wait.

### To send data to the model

Send a POST request with the header as `{'Content-type': 'application/json'}` , and with the payload set as the following:  
```
{
  message: '<your unpunctuated text>'
}
```
You will receive a response with the `prediction` field set to the punctuated text. 

### Sample code for backend interaction
```
import requests
import json
headers = {'Content-type': 'application/json'}
r = requests.post('http://localhost:5000/backend_predict',data=json.dumps({'message': 'hello this is abhinav here to help you'}), headers=headers)
print(r.content)
r = requests.post('http://localhost:5000/lang_change',data=json.dumps({'fav_language': 'dual', 'punc_type': 'all'}), headers=headers)
```

## Recommendations (if taken further)  
- Use ONNX instead of pytorch jit
    - Due to technical issues, we have decided to go with pytorch jit traces instead. 