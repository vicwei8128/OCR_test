import cv2
import imutils
import json
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import requests
import time
from base64 import b64encode
from IPython.display import Image
from pylab import rcParams

rcParams['figure.figsize'] = 10, 20


def makeImageData(imgpath):
    img_req = None
    with open(imgpath, 'rb') as f:
        ctxt = b64encode(f.read()).decode()
        img_req = {
            'image': {
                'content': ctxt
            },
            'features': [{
                'type': 'DOCUMENT_TEXT_DETECTION',
                'maxResults': 1
            }]
        }
    return json.dumps({"requests": img_req}).encode()


def requestOCR(url, api_key, imgpath):
    imgdata = makeImageData(imgpath)
    response = requests.post(ENDPOINT_URL,
                             data=imgdata,
                             params={'key': api_key},
                             headers={'Content-Type': 'applica tion/json'})
    return response


with open('superb.json') as f:
    data = json.load(f)

ENDPOINT_URL = 'https://vision.googleapis.com/v1/images:annotate'
api_key = data["api_key"]
img_loc = "Image.jpg"

Image(img_loc)

result = requestOCR(ENDPOINT_URL, api_key, img_loc)

if result.status_code != 200 or result.json().get('error'):
    print("Error")
else:
    result = result.json()['responses'][0]['textAnnotations']

result

for index in range(len(result)):
    print(result[index]["description"])


def gen_cord(result):
    cord_df = pd.DataFrame(result['boundingPoly']['vertices'])
    x_min, y_min = np.min(cord_df["x"]), np.min(cord_df["y"])
    x_max, y_max = np.max(cord_df["x"]), np.max(cord_df["y"])
    return result["description"], x_max, x_min, y_max, y_min


text, x_max, x_min, y_max, y_min = gen_cord(result[-1])
image = cv2.imread(img_loc)
cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
print("Text Detected = {}".format(text))
