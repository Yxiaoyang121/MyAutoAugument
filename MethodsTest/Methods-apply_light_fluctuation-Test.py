import cv2
import numpy as np


from AugumentMethods import apply_light_fluctuation

img_bgr = cv2.imread('../ImageSourceTest/boli.bmp')
aug_img = apply_light_fluctuation(img_bgr, 1.3)
cv2.imwrite('aug_img.bmp', aug_img)