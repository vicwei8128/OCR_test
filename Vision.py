from google.cloud import vision

YOUR_PIC = 'YOUR_PIC'

client = vision.ImageAnnotatorClient()
with open(YOUR_PIC, 'rb') as image_file:
    content = image_file.read()
image = vision.Image(content=content)
response = client.label_detection(image=image)
print(response)
