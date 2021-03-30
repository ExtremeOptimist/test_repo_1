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

# Kan jeg tegne enkel kontur på et tomt bilde? Blir stygt
# contours, hierarchy = cv2.findContours(kontur, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
# roi_coords, biggest_cont_ = tools.biggest_contour_roi(kontur)  # Korallrev bounding box
# blank = np.zeros((2574, 4576))
# contours = sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True)
# only_contour = cv2.drawContours(blank, [contours[6]], 0, 255, -1)
# plt.figure()
# plt.imshow(only_contour)

# roi = tools.crop_image(im, roi_coords)  # klippet ut
# prep_img = cv2.resize(roi, SHAPE, interpolation=cv2.INTER_AREA)
# plt.figure()
# plt.imshow(prep_img)

# kont2 = tmpl.lilla_contour(prep_img)
# plt.figure()
# plt.imshow(kont2)
# cv2.imwrite('templates/cuttbar.jpg', kont2)

# tmp1 = cv2.imread('./korallrev/templates/6.jpg')
# tmp1 = tools.Template(tmp1, 0.6)
# tmp1.show()

# kfor = tools.ObjectTotal(prep_img)
# kfor.show(kontur=True)
# bfor = kfor.template_match([tmp1], filter_threshold=0.1)
# kfor_bokses = kfor.draw_rectangles(bfor)
# kfor.show(with_bokses=True)

# bokses1 = tmpl.visualize(prep_img, matches)
# plt.figure()
# plt.imshow(bokses1)

# matches2 = tmpl.non_max_suppression(matches, non_max_suppression_threshold=0.3)
# bokses1 = tmpl.visualize(prep_img, matches2)
# plt.figure()
# plt.imshow(bokses1)


plt.show()

