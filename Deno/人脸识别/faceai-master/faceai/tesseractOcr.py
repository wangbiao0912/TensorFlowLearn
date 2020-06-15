#coding=utf-8
#文字识别类
from PIL import Image
import pytesseract
import cv2

# path = "img/text-img.png"
path="/Users/wangbiao/Downloads/Video/视频内容分析/_00:03:30.jpg"
# path="/Users/wangbiao/Downloads/getVerificationCode.jpeg"
text = pytesseract.image_to_string(Image.open(path), lang='chi_sim+eng')
print(text+">>>>>")

img = cv2.imread(path)
cv2.imshow("Image", img)

cv2.waitKey(0)
cv2.destroyAllWindows()