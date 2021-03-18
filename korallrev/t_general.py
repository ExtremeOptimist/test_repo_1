import cv2
import numpy as np
import matplotlib.pyplot as plt
# import template_matching_sicara as tmpl
import tools as tmpl

SHAPE = (400, 400)

im = cv2.imread('./korallrev/ref_images/mob_etter_2.jpg')
kontur = tmpl.lilla_contour(im)  # Bare lilla områder
roi_coords = tmpl.biggest_contour_roi(kontur)  # Korallrev bounding box
roi = tmpl.crop_image(im, roi_coords)  # klippet ut
prep_img = cv2.resize(roi, SHAPE, interpolation=cv2.INTER_AREA)
# plt.figure()
# plt.imshow(prep_img)

# kont2 = tmpl.lilla_contour(prep_img)
# plt.figure()
# plt.imshow(kont2)
# cv2.imwrite('templates/cuttbar.jpg', kont2)

tmp1 = cv2.imread('./korallrev/templates/1.jpg')
tmp1 = tmpl.Template(tmp1, 0.7)

kfor = tmpl.ObjectTotal(prep_img)
# kfor.show()
bfor = kfor.template_match([tmp1], filter_threshold=0.1)
kfor_bokses = kfor.draw_rectangles(bfor)
kfor.show(with_bokses=True)

# bokses1 = tmpl.visualize(prep_img, matches)
# plt.figure()
# plt.imshow(bokses1)

# matches2 = tmpl.non_max_suppression(matches, non_max_suppression_threshold=0.3)
# bokses1 = tmpl.visualize(prep_img, matches2)
# plt.figure()
# plt.imshow(bokses1)


plt.show()

