import sys, site
print(sys.executable)
print(sorted(site.getsitepackages()+[site.getusersitepackages()]))
from sam2.sam2_image_predictor import SAM2ImagePredictor
print(SAM2ImagePredictor)
import cv2
img = cv2.imread(r"D:\Python\Python Project\nav_vlm\img\demo01.jpg")
print(img is None)