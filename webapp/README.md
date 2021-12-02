# Web application 
## Deployment (with docker compose)

- Pull the image from dockerhub: https://hub.docker.com/repository/docker/aetherprior/sud
- Download the XLM-R models for both English + Zh and Malay from here: https://drive.google.com/drive/folders/1ZdKn4i6WXN9_rPot1lx-_0EJnTzw5HuD?usp=sharing
- Navigate to the webapp directory, and change the `../data-webapp` directory in the `docker-compose.yml` to the directory of the downloaded files
- In the terminal, type `docker compose up` . The image will be build and the webapp will be served at port 5000 at localhost

## Deployment (with docker)

- Pull the image from dockerhub: https://hub.docker.com/repository/docker/aetherprior/sud
- Download the XLM-R models for both English + Zh and Malay from here: https://drive.google.com/drive/folders/1ZdKn4i6WXN9_rPot1lx-_0EJnTzw5HuD?usp=sharing

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
