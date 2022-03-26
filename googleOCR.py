from google.cloud import vision

import os
import io

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "key.json"
client = vision.ImageAnnotatorClient()

path = os.path.abspath('images/40150.jpg')
with io.open(path, 'rb') as image_file:
    content = image_file.read()

image = vision.Image(content=content)

response = client.image_properties(image=image)
props = response.image_properties_annotation
print('Properties of the image:')

for color in props.dominant_colors.colors:
    print('Fraction: {}'.format(color.pixel_fraction))
    print('\tr: {}'.format(color.color.red))
    print('\tg: {}'.format(color.color.green))
    print('\tb: {}'.format(color.color.blue))