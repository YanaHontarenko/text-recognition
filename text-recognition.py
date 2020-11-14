import cv2
import pytesseract
import numpy as np

img = cv2.imread('data\\big-image.png')

gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
gray, img_bin = cv2.threshold(gray,128,255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)
gray = cv2.bitwise_not(img_bin)
kernel = np.ones((2, 1), np.uint8)
img = cv2.erode(gray, kernel, iterations=1)
img = cv2.dilate(img, kernel, iterations=1)
cv2.imshow("image", img)
cv2.waitKey(0)
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
out_below = pytesseract.image_to_string(img, lang="eng", config="--psm 6")
print("OUTPUT:", out_below)