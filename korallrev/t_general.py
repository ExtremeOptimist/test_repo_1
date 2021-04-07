import cv2
import matplotlib.pyplot as plt
import korallrev.tools as tools


# Tilpass bilde før. (kan gjøres til funksjon)
prep_img_for = tools.prep_image('./ref_images/mob_for.jpg', shape=(400, 400))
# plt.figure()
# plt.imshow(prep_img)

# Tilpass bildet etter
prep_img_etr = tools.prep_image('./ref_images/mob_etter.jpg', shape=(400, 400))
# plt.figure()
# plt.imshow(prep_img)

# Kode for å lagre et bilde man kan kutte ut maler fra
# unused, kont2 = tools.lilla_contour(prep_img_for)
# cv2.imwrite('templates/cuttbar4.jpg', kont2)
# plt.figure()
# plt.imshow(kont2)

# Kode for å sette opp malene
templates = [
    tools.make_template('./templates/EG_1.jpg', 0.95),
    tools.make_template('./templates/EG_2.jpg', 0.7),
    tools.make_template('./templates/EG_3.jpg', 0.7)
    # tools.make_template('./templates/EG_4.jpg', 0.95)
]

# Kode for å hente ut bokser før
kfor = tools.ObjectTotal(prep_img_for)
# kfor.show(kontur=True)
bfor = kfor.get_bokses(templates, filter_threshold=0.1)
kfor_bokses = kfor.draw_rectangles(bfor)
plt.subplot(131)
kfor.show(with_bokses=True)

# Kode for å hente ut bokser etter
ketr = tools.ObjectTotal(prep_img_etr)
# ketr.show(kontur=True)
betr = ketr.get_bokses(templates, filter_threshold=0.1)
ketr_bokses = ketr.draw_rectangles(betr)
plt.subplot(132)
ketr.show(with_bokses=True)

# Finner forskjeller
diff_bokses = tools.find_non_overlap(bfor, betr)
kfor.set_before()  # setter bokser fra bilde før til gamle.

# Sjekker for hvite deler, og setter riktig farge på bokser
tools.check_white_parts(prep_img_for, prep_img_etr, diff_bokses)
for part in diff_bokses:
    part.set_color()

# Tegner rektanglene med riktige farger på bilde etter
diff_etr = ketr.draw_rectangles(diff_bokses)

# plt.figure()
plt.subplot(133)
plt.imshow(diff_etr[:, :, ::-1])

# cv2.imwrite('resultat.jpg', diff_etr)

plt.show()
