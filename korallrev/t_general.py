import cv2
import matplotlib.pyplot as plt
import korallrev.tools as tools

SHAPE = (400, 400)

# Tilpass bilde før. (kan gjøres til funksjon)
# im = cv2.imread('./korallrev/ref_images/mob_etter.jpg')
im = cv2.imread('./ref_images/korallrev_for.png')
kontur = tools.lilla_contour(im)  # Bare lilla områder
# plt.figure()
# plt.imshow(im)
roi_coords, biggest_cont_ = tools.biggest_contour_roi(kontur)  # Korallrev bounding box
# plt.figure()
# plt.imshow(only_contour)
roi = tools.crop_image(im, roi_coords)  # klippet ut
prep_img_for = cv2.resize(roi, SHAPE, interpolation=cv2.INTER_AREA)
# plt.figure()
# plt.imshow(prep_img)

# Tilpass bildet etter
# im = cv2.imread('./korallrev/ref_images/mob_etter.jpg')
# im = cv2.imread('./korallrev/ref_images/korallrev_etter_innsyn_vink_10.png')
im = cv2.imread('./ref_images/bilde_mobil_2.jpg')
kontur = tools.lilla_contour(im)  # Bare lilla områder
# plt.figure()
# plt.imshow(kontur)
# Kommentert ut alt utenom plt.show
roi_coords, biggest_cont_2 = tools.biggest_contour_roi(kontur)  # Korallrev bounding box
# plt.figure()
# plt.imshow(only_contour)
roi = tools.crop_image(im, roi_coords)  # klippet ut
prep_img_etr = cv2.resize(roi, SHAPE, interpolation=cv2.INTER_AREA)
# plt.figure()
# plt.imshow(prep_img)


# kont2 = tools.lilla_contour(prep_img_etr)
# plt.figure()
# plt.imshow(kont2)
# cv2.imwrite('templates/cuttbar2.jpg', kont2)

templates = [
    tools.make_template('./templates/MT_1.jpg', 0.8),
    tools.make_template('./templates/MT_2.jpg', 0.8),
    tools.make_template('./templates/MT_3.jpg', 0.8)
]

# tmp1.show()

kfor = tools.ObjectTotal(prep_img_for)
# kfor.show(kontur=True)
bfor = kfor.template_match(templates, filter_threshold=0.1)
kfor_bokses = kfor.draw_rectangles(bfor)
plt.subplot(131)
kfor.show(with_bokses=True)

ketr = tools.ObjectTotal(prep_img_etr)
# ketr.show(kontur=True)
betr = ketr.template_match(templates, filter_threshold=0.1)
ketr_bokses = ketr.draw_rectangles(betr)
plt.subplot(132)
ketr.show(with_bokses=True)

diff_bokses = tools.find_non_overlap(bfor, betr)

kfor.set_before()
tools.check_white_parts(prep_img_for, prep_img_etr, diff_bokses)
for part in diff_bokses:
    part.set_color()

diff_etr = ketr.draw_rectangles(diff_bokses)

# plt.figure()
plt.subplot(133)
plt.imshow(diff_etr)

plt.show()
