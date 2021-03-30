import cv2
import numpy as np
import matplotlib.pyplot as plt
# import template_matching_sicara as tmpl
import tools

SHAPE = (400, 400)

im = cv2.imread('./korallrev/ref_images/mob_for.jpg')
kontur = tools.lilla_contour(im)  # Bare lilla områder
roi_coords, unused = tools.biggest_contour_roi(kontur)  # Korallrev bounding box
roi = tools.crop_image(im, roi_coords)  # klippet ut
prep_img = cv2.resize(roi, SHAPE, interpolation=cv2.INTER_AREA)
plt.figure()
plt.imshow(prep_img)
# ==============================

# kmeans 
img = prep_img
Z = img.reshape((-1,3))
# convert to np.float32
Z = np.float32(Z)
# define criteria, number of clusters(K) and apply kmeans()
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
K = 3
ret,label,center=cv2.kmeans(Z,K,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)
# Now convert back into uint8, and make original image
center = np.uint8(center)
res = center[label.flatten()]
res2 = res.reshape((img.shape))
plt.figure()
plt.imshow(res2)


kontur2 = tools.lilla_contour(res2)
plt.figure()  # Vil tegne største kontur for seg selv
plt.imshow(kontur2)

unused, biggest_cont = tools.biggest_contour_roi(kontur2)
blank = np.zeros(SHAPE)
only_contour = cv2.drawContours(blank, [biggest_cont], 0, 255, -1)
plt.figure()  # Vil tegne største kontur for seg selv
plt.imshow(only_contour)

plt.show()