import cv2
import matplotlib.pyplot as plt
import korallrev.tools as tools


im = cv2.imread('./templates/MT_1.jpg', 0)
#kontur = tools.lilla_contour(im)
print(cv2.mean(im))