import os
from google.cloud import vision
import matplotlib.pyplot as plt



os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = "key.json"
client = vision.ImageAnnotatorClient()

YOUR_PIC = './images/61187.jpg'

with open(YOUR_PIC, 'rb') as image_file:
    content = image_file.read()
image = vision.Image(content=content)

response = client.text_detection(image=image)
texts = response.text_annotations

print(response)

for text in texts:
    print(text.description)