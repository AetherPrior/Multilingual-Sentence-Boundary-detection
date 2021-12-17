import requests
import json
headers = {'Content-type': 'application/json'}
r = requests.post('http://192.168.0.100:5000/backend_predict',data=json.dumps({'message': 'hello this is abhinav here to help you'}), headers=headers)
print(r.content)
r = requests.post('http://192.168.0.100:5000/lang_change',data=json.dumps({'fav_language': 'dual', 'punc_type': 'all'}), headers=headers)
