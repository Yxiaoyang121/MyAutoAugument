import cv2
import numpy as np


from AugumentMethods import apply_dust_interference

img_bgr = cv2.imread('../ImageSourceTest/boli.bmp')
aug_img = apply_dust_interference(img_bgr, 0,[0.01, 0.05])
cv2.imwrite('aug_img.bmp', aug_img)