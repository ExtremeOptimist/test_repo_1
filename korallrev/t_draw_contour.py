import cv2
import numpy as np
import matplotlib.pyplot as plt
# import template_matching_sicara as tmpl
import tools

SHAPE = (400, 400)

# im = cv2.imread('./korallrev/ref_images/mob_etter.jpg')
im = cv2.imread('./korallrev/ref_images/korallrev_for.png')
kontur = cv2.bilateralFilter(im, 10, 120, 120)
# plt.figure()
# plt.imshow(kontur)

kontur1 = tools.lilla_contour(im)  # Bare lilla områder
plt.figure()
plt.imshow(kontur1)

# Gav liten effekt
kontur2 = tools.lilla_contour(kontur)  # Bare lilla områder
plt.figure()
plt.imshow(kontur2)




plt.show()

