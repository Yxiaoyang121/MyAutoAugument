import cv2
import numpy as np


from AugumentMethods import apply_light_fluctuation, apply_reflectivity_variation

img_bgr = cv2.imread('../ImageSourceTest/boli.bmp')
aug_img = apply_reflectivity_variation(img_bgr, 0.8)
cv2.imwrite('aug_img.bmp', aug_img)