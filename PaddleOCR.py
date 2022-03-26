# from paddleocr import PaddleOCR,draw_ocr
# # from matplotlib import pyplot as plt
# # import cv2 #opencv
# import os
#
# # Setup model
# ocr_model = PaddleOCR(lang='chinese_cht',use_gpu=False)
#
# img_path = os.path.join('.', 'images/38375.jpg')
# # Run the ocr method on the ocr model
# result = ocr_model.ocr(img_path)
#
# for res in result:
#     print(res[1][0])


import cv2.cv2 as cv2
import numpy as np
from paddleocr import PaddleOCR, draw_ocr
from opencc import OpenCC

ocr = PaddleOCR(use_angle_cls=True, lang="ch")
# chinese(sim)&English = ch; English = en;Chinese Traditioal=chinese_cht

# 1.讀取影像
imgPath = "images/61187.jpg"
img = cv2.imread(imgPath, cv2.IMREAD_COLOR)
# imgC = img.copy()
# img = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)



# 2.調整圖像大小等比例縮放
scale_percent = 50       # percent of original size
width = int(img.shape[1] * scale_percent / 100)
height = int(img.shape[0] * scale_percent / 100)
dim = (width, height)
# resize image
resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)

# 3.影像去噪
gray = cv2.fastNlMeansDenoisingColored(resized, None, 10, 10, 7, 21)
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

# cv2.imshow('img', imgC)
cv2.imshow('img1', resized)
cv2.waitKey(0)

# img_path = 'images/56401.jpg'
result = ocr.ocr(resized, cls=True)
result1 = ocr.ocr(img, cls=True)
for res in result:
    cc = OpenCC('s2t')
    # s2t = chinese.sim->tra
    print(cc.convert(res[1][0]))

print("--------------------------")

for res in result1:
    cc = OpenCC('s2t')
    # s2t = chinese.sim->tra
    print(cc.convert(res[1][0]))