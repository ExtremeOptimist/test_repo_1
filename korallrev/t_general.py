import cv2
import numpy as np
import matplotlib.pyplot as plt
# import template_matching_sicara as tmpl
import tools

SHAPE = (400, 400)

# Tilpass bilde før. (kan gjøres til funksjon)
# im = cv2.imread('./korallrev/ref_images/mob_etter.jpg')
im = cv2.imread('./korallrev/ref_images/korallrev_for.png')
kontur = tools.lilla_contour(im)  # Bare lilla områder
# plt.figure()
# plt.imshow(kontur)
roi_coords, biggest_cont_ = tools.biggest_contour_roi(kontur)  # Korallrev bounding box
# plt.figure()
# plt.imshow(only_contour)
roi = tools.crop_image(im, roi_coords)  # klippet ut
prep_img_for = cv2.resize(roi, SHAPE, interpolation=cv2.INTER_AREA)
# plt.figure()
# plt.imshow(prep_img)

# Tilpass bildet etter
# im = cv2.imread('./korallrev/ref_images/mob_etter.jpg')
im = cv2.imread('./korallrev/ref_images/korallrev_etter_innsyn_vink_10.png')
kontur = tools.lilla_contour(im)  # Bare lilla områder
# plt.figure()
# plt.imshow(kontur)
roi_coords, biggest_cont_ = tools.biggest_contour_roi(kontur)  # Korallrev bounding box
# plt.figure()
# plt.imshow(only_contour)
roi = tools.crop_image(im, roi_coords)  # klippet ut
prep_img_etr = cv2.resize(roi, SHAPE, interpolation=cv2.INTER_AREA)
# plt.figure()
# plt.imshow(prep_img)


# kont2 = tmpl.lilla_contour(prep_img)
# plt.figure()
# plt.imshow(kont2)
# cv2.imwrite('templates/cuttbar.jpg', kont2)

tmp1 = cv2.imread('./korallrev/templates/6.jpg')
tmp1 = tools.Template(tmp1, 0.5)
# tmp1.show()

kfor = tools.ObjectTotal(prep_img_for)
kfor.show(kontur=True)
bfor = kfor.template_match([tmp1], filter_threshold=0.1)
kfor_bokses = kfor.draw_rectangles(bfor)
kfor.show(with_bokses=True)

ketr = tools.ObjectTotal(prep_img_etr)
ketr.show(kontur=True)
betr = ketr.template_match([tmp1], filter_threshold=0.1)
ketr_bokses = ketr.draw_rectangles(betr)
ketr.show(with_bokses=True)

# bokses1 = tmpl.visualize(prep_img, matches)
# plt.figure()
# plt.imshow(bokses1)

# matches2 = tmpl.non_max_suppression(matches, non_max_suppression_threshold=0.3)
# bokses1 = tmpl.visualize(prep_img, matches2)
# plt.figure()
# plt.imshow(bokses1)


plt.show()

