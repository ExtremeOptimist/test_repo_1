"""Ferdig fil for kode som tillhører sjekking av korallrevets helse.
Klassebasert. Opprettet 20.Jan.2021 av Bjørnar Størksen Wiik.
Plan videre: Bruke non max supression utenfor klasse, og prøve og
bruke den både til å fjerne overflødige bokser, men og
bokser vi trenger for å finne forskjellr (tror man kan sette lav terskel)
Sist endret: 20.Jan.2021 av BSW, kl: 4:16
Sist endreet 27 jan morgen."""

import cv2
import numpy as np
import matplotlib.pyplot as plt


class ObjectTotal:
    def __init__(self, image_path):
        # Hva hvert korallrev trenger
        self.image = cv2.imread(image_path)
        # assert self.image is not None  # check if image is loaded
        self.image_contour = None
        self.image_copy = self.image.copy()
        self.parts = []  # liste med deler av korallrevet
        self.f_parts = []  # filtrert liste

    def set_contour(self):
        self.image_contour, self.image = extract_color(self.image)

    def set_before(self):
        for part_ in self.f_parts:
            part_.new = False
        return

    def template_match(self, templates, filter_threshold=0.1):
        # Tar inn en eller flere templates
        for template_ in templates:
            match = cv2.matchTemplate(self.image_contour, template_.image_contour, cv2.TM_CCOEFF_NORMED)
            above = np.where(match >= template.matching_threshold)
            for (x, y) in zip(above[1], above[0]):
                # Lager et del objekt for hver match over terskelen
                self.parts.append(ObjectPart(x, y, x+template_.width, y+template_.height,
                                             match[y, x], (0, 0, 0), True, False))
        filtered_parts = filter_overlap(self.parts, filter_threshold)
        for f_part_ in filtered_parts:
            self.f_parts.append(f_part_)
        return filtered_parts

    def draw_rectangles(self, parts=None):
        image = self.image.copy()
        if parts is None:
            parts = self.f_parts
        for part_ in parts:
            cv2.rectangle(
                image,
                (part_.ulx, part_.uly),
                (part_.brx, part_.bry),
                part_.col, 7)
        return image

    def resize_to_image(self, image_ref):
        img_scaled = cv2.resize(self.image_contour,
                                (image_ref.shape[1], image_ref.shape[0]), interpolation=cv2.INTER_AREA)
        img_org_scaled = cv2.resize(self.image,
                                (image_ref.shape[1], image_ref.shape[0]), interpolation=cv2.INTER_AREA)
        self.image_contour = img_scaled
        self.image = img_org_scaled
        return img_scaled

    def check_white_parts(self, list_of_parts):
        if list_of_parts is not []:
            for part_ in list_of_parts:
                roi = self.image[part_.uly:part_.bry, part_.ulx:part_.brx, :]
                sum_all = np.sum(roi[:, :, :])
                pixel_brightness = sum_all / (roi.shape[0] * roi.shape[1])
                if pixel_brightness > 450:
                    part_.wht = True
                else:
                    part_.wht = False
                print(f'Pixel whitenes = {pixel_brightness}\nIs white = {part_.wht}')
                # cv2.imshow('hvit', roi)
                # cv2.waitKey(0)
                # cv2.destroyAllWindows()
        return True


class Template:
    def __init__(self, image_path, matching_threshold):
        self.image = cv2.imread(image_path)
        self.matching_threshold = matching_threshold
        self.height, self.width, self.depth = self.image.shape
        self.image_contour = None

    def set_contour(self):
        self.image_contour, a = extract_color(self.image, crop=False)


class ObjectPart:
    __id = 0  # unik id for hver del, gjør det lettere å se hva som forsvinner/står igjen.
    # Hva hver del av korallrevet trenger

    def __init__(self, upper_left_x, upper_left_y,
                 bottom_right_x, bottom_right_y, matching_value, color, new, white):
        self.ulx = upper_left_x
        self.uly = upper_left_y
        self.brx = bottom_right_x
        self.bry = bottom_right_y
        self.mav = matching_value
        self.col = color
        self.new = new
        self.wht = white
        ObjectPart.__id += 1
        self._id = ObjectPart.__id

    def get_rectangle(self):
        # Gir ut liste med [x1, y1, x2, y2] koordinater
        # altså, hjørnene oppe venstre og nede høyre
        return [self.ulx, self.uly, self.brx, self.bry]

    def set_color(self):
        if self.new:  # new yes
            if self.wht:  # wht yes
                self.col = (255, 0, 0)  # bleaching, red
            else:  # white no
                self.col = (0, 255, 0)  # growth, green
        else:  # new no
            if self.wht:  #
                self.col = (0, 0, 255)  # recover, blue
            else:
                self.col = (255, 255, 0)  # death, yellow
        return True

    def __str__(self):
        return f'id er {self._id}'

    def __repr__(self):
        return self.__str__()


def extract_color(image, hue=(121, 177), crop = True):
    # Returnerer et gråskala bilde av contour til spesifisert farge (hsv) fra et et bilde
    # 145 til 170 har jeg funnet ut at stemmer godt med lilla for korallrevet
    blob_size = 5
    img_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    color_low = np.array([hue[0], 60, 50])
    color_high = np.array([hue[1], 200, 221])
    mask = cv2.inRange(img_hsv, color_low, color_high)  # lager maske
    res = cv2.bitwise_and(image, image, mask=mask)  # Bildet som representerer fargen i korallrevet
    # koden over, bitwise and brukes for å sortere ut de lilla pikslene som vi er interreserte i.
    res = cv2.cvtColor(res, cv2.COLOR_HSV2BGR)  # Gjør til BGR
    res = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)  # Gjør til gråskala
    kernel = np.ones((blob_size, blob_size), np.uint8)
    retur_img = cv2.morphologyEx(res, cv2.MORPH_CLOSE, kernel)
    ret, res = cv2.threshold(retur_img, 90, 250, cv2.THRESH_BINARY)
    contours, hir = cv2.findContours(res, 1, 2)
    biggest_cont = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(biggest_cont)
    if crop:
        img_cropped = image[y:y + h, x:x + w]
        contour_cropped = res[y:y + h, x:x + w]
    else:
        img_cropped = image
        contour_cropped = res
    """Midlertidig for test"""
    plt.subplot(121)
    plt.imshow(contour_cropped)
    plt.subplot(122)
    plt.imshow(img_cropped)
    plt.show()
    # --------------------
    return contour_cropped, img_cropped


def compute_iou(a, b):
    # Tar inn to rektangler,
    # a, b = [x1, y1, x2, y2]
    # og regner ut prosentvis overlapp
    # 0 = 0 % 0.5 = 50 % 1 = 100 %
    epsilon = 1e-5

    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])

    width = (x2 - x1)  # Area of intersection
    height = (y2 - y1)

    # handle case where there is NO overlap
    if (width < 0) or (height < 0):
        return 0.0
    area_overlap = width * height

    # COMBINED AREA
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    area_combined = area_a + area_b - area_overlap

    # RATIO OF AREA OF OVERLAP OVER COMBINED AREA
    iou = area_overlap / (area_combined + epsilon)
    return iou


def filter_overlap(list_of_parts, threshold):
    # Filtrerer bokser som overlapper over en terskel
    non_overlapping = []
    for obj in list_of_parts:
        overlap = False
        for f_obj in non_overlapping:
            iou = compute_iou(obj.get_rectangle(), f_obj.get_rectangle())  # se funskjon "compute iou"
            if iou > threshold:
                overlap = True
                break
        if not overlap:
            non_overlapping.append(obj)
    # print(f'Antall som ikke overlapper: {len(non_overlapping)}')
    return non_overlapping


def find_non_overlap(objects_before, objects_after, non_max_suppression_threshold=0.2):
    filtered_objects = []
    if len(objects_before) <= len(objects_after):
        o_list_1 = objects_after
        o_list_2 = objects_before
    else:
        o_list_1 = objects_before
        o_list_2 = objects_after
    combined = objects_before + objects_after
    for o1 in combined:  # Går gjennom alle objekter
        overlap_found = False  # Sier at man finner overlapp med en gang
        for o2 in o_list_1:  # sjekker med alle objekter i liste 2
            iou = compute_iou(o1.get_rectangle(), o2.get_rectangle())
            # Regner ut iou for object og objekt som ikke har overlapp
            if iou >= non_max_suppression_threshold:  # Hvis område er høyt betyr det at det er overlapp
                overlap_found = True
                break  # går ut av løkken hvis den finner overlapp
        if not overlap_found:  # Hvis ingen hadde område overlapp, vil øverste loop objekt gå til filtrert listen
            filtered_objects.append(o1)
    for o1 in combined:  # Går gjennom alle objekter
        overlap_found = False  # Sier at man finner overlapp med en gang
        for o2 in o_list_2:  # sjekker med alle objekter i liste 2
            iou = compute_iou(o1.get_rectangle(), o2.get_rectangle())
            # Regner ut iou for object og objekt som ikke har overlapp
            if iou >= non_max_suppression_threshold:  # Hvis område er høyt betyr det at det er overlapp
                overlap_found = True
                break  # går ut av løkken hvis den finner overlapp
        if not overlap_found:  # Hvis ingen hadde område overlapp, vil øverste loop objekt gå til filtrert listen
            filtered_objects.append(o1)
    print(f'antall differanser: {len(filtered_objects)}')
    return filtered_objects


# ============================================================================
if __name__ == '__main__':
    templater = []
    templater.append(Template('ut_2.png', 0.65))
    template.set_contour()
    # img1 = cv2.imread('mob_for.jpg.png')
    # img2 = cv2.imread('mob_etter.jpg.jpg')
    kfor = ObjectTotal('ref_images\mob_for.jpg')
    kfor.set_contour()  # setter contouren til formen av korallrevet
    # matcher templates og filtrere overlapp, returnerer en liste med deler.
    bokser_for = kfor.template_match(templater, 0.01)
    for_med_rektangler = kfor.draw_rectangles(bokser_for)  # returnerer objektbildet med valgte deler


    ketr = ObjectTotal('mob_etter.jpg')
    ketr.set_contour()  # ikke fornøyd med "set contour" som metode navn
    #cont_scal = ketr.resize_to_image(kfor.image)
    bokser_etter = ketr.template_match(templater, 0.01)
    etter_med_rektangler = ketr.draw_rectangles(bokser_etter)

    forskjeller = find_non_overlap(bokser_for, bokser_etter)
    kfor.check_white_parts(forskjeller)
    ketr.check_white_parts(forskjeller)
    for part in forskjeller:
        part.set_color()
    etter_med_forskjeller = ketr.draw_rectangles(forskjeller)


    # tester å tegne inn alle boksene som er funnet
    # etter_med_rektangler = ketr.draw_rectangles(bokser_for+bokser_etter)

    plt.subplot(121)
    plt.imshow(for_med_rektangler)
    plt.subplot(122)
    plt.imshow(etter_med_forskjeller)
    plt.show()