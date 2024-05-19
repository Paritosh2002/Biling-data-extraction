from paddleocr import PaddleOCR, draw_ocr # main OCR dependencies
from matplotlib import pyplot as plt # for plotting maps
import cv2 #opencv for writing and reading the images
import os # folder directory navigation

ocr_model = PaddleOCR(lang='en',use_angle_cls=True)
img_path = 'sample_images/image_1.jpg'
# Running the ocr method on the ocr model
result = ocr_model.ocr(img_path)
# Extracting detected components
boxes = [res[0] for res in result] # 
texts = [res[1][0] for res in result]
scores = [res[1][1] for res in result]
img = cv2.imread(img_path) 

# reorders the color channels
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.figure(figsize=(100,15))

# draw annotations on image
annotated = draw_ocr(img, boxes, texts, scores) 

# show the image using matplotlib
plt.imshow(annotated)