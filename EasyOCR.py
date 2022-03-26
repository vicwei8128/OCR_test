import easyocr

reader = easyocr.Reader(['ch_tra', 'en'])
result = reader.readtext('images/61187.jpg', detail=0)
print(result)
