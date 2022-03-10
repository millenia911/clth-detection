from flask import Flask, request, Response
from flask_cors import CORS, cross_origin
import jsonpickle
import numpy as np
import cv2
import time
import datetime
import random
from inference import get_outfit_pred

app = Flask(__name__)
CORS(app)

server_id = int(100*random.random())

@app.route('/api/test', methods=['POST'])
@cross_origin()
def test():
    start=time.time()
    r = request
    nparr = np.frombuffer(r.data, np.uint8)
    img, image, bbox = None, None, None

    try:
      print(len(nparr))
      img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
      image, bbox = get_outfit_pred(img)
      _, image = cv2.imencode('.jpg', image)
    except Exception as e:
      print(e)
      print('failed to process image')
      pass
    
    response = {'message': 'image processed', 
                'size':'{}x{}'.format(img.shape[1], img.shape[0]) if type(img) != type(None) else "None",
                'image': image.tobytes(),
                'bbox':  bbox if len(bbox) else 'no object detected',
                'process time': time.time()-start,
                'server_id': server_id,
                'time_stamp': str(datetime.datetime.now())
                }

    response_pickled = jsonpickle.encode(response)

    return Response(response=response_pickled, status=200, mimetype="application/json")


app.run(host="0.0.0.0", port=8080, debug=True)
