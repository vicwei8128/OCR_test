from PIL import Image  # version 3.2.0
import pytesseract as ocr  # version 0.2.5
import cv2.cv2 as cv2  # version 3.4.5
import numpy as np  # version 1.15.2
import easyocr


# 1.讀取影像
imgPath = "images/61187.jpg"
img = cv2.imread(imgPath, cv2.IMREAD_COLOR)
# img = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)

# 2.調整圖像大小
# img = cv2.resize(img, (428, 270), interpolation=cv2.INTER_CUBIC)

# 3.影像去噪
gray = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)
# gray = cv2.fastNlMeansDenoising(img, None, 10, 3, 3, 3)
coefficients = [0, 1, 1]
m = np.array(coefficients).reshape((1, 3))
# 旋轉圖片
gray = cv2.transform(gray, m)

# 4.閾值 180  maxval:255
ret, binary = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY_INV)
ele = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 10))

# 5.膨脹操作
dilation = cv2.dilate(binary, ele, iterations=1)

cv2.imshow('img', img)
cv2.waitKey(0)


image = Image.fromarray(img)
text = ocr.pytesseract.image_to_string(Image.open(img), lang="chi_tra")
print(text)

reader = easyocr.Reader(['ch_tra', 'en'])
result = reader.readtext(img, detail=0)
print(result)