import os
from google.cloud import vision
import matplotlib.pyplot as plt
from PIL import Image

os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = "key.json"
client = vision.ImageAnnotatorClient()

YOUR_PIC = './images/some_plate.jpg'
im = Image.open(YOUR_PIC)
plt.imshow(im)

with open(YOUR_PIC, 'rb') as image_file:
    content = image_file.read()
image = vision.Image(content=content)
response = client.document_text_detection(image=image)
texts = response.document_text_annotations

print(response)



for text in texts:
    if text.text == ("1", "9"):
        a = [(v.x, v.y) for v in text.symbols.vertices]
        a.append(a[0])
        x, y = zip(*a)
        plt.plot(x, y, color='yellow')
        plt.text(x[0], y[0], text.description, color='blue')
        if text.description == "OU":
            a = [(v.x, v.y) for v in text.bounding_poly.vertices]
            a.append(a[0])
            x, y = zip(*a)
            plt.plot(x, y, color='yellow')
            plt.text(x[0], y[0], text.description, color='blue')

    continue

plt.show()
