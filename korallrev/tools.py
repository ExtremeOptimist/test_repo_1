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


def lilla_contour(img):
    hue = (155, 179)
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # plt.figure()
    # plt.imshow(img_hsv)
    color_low = np.array([hue[0], 30, 50])
    color_high = np.array([hue[1], 220, 255])
    mask = cv2.inRange(img_hsv, color_low, color_high)
    res = cv2.bitwise_and(img, img, mask=mask)
    bare_h_lag = res[:, :, 0]
    ret, contour_med_stoy = cv2.threshold(bare_h_lag, 30, 250, cv2.THRESH_BINARY)
    return contour_med_stoy


def biggest_contour_roi(gray_img):
    contours, hierarchy = cv2.findContours(gray_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    biggest_cont = max(contours, key=cv2.contourArea)
    bounding_rect = cv2.boundingRect(biggest_cont)
    return bounding_rect


def crop_image(image, rectangle_coords):
    sx = int(rectangle_coords[0] - (0.05 * image.shape[0]))
    sy = rectangle_coords[1]
    wd = int(rectangle_coords[2] + (0.05 * image.shape[0]))
    he = rectangle_coords[3]
    return image[sy:sy + he, sx:sx + wd]


if __name__ == '__main__':
    templater = []
    templater.append(Template('./korallrev/templates/5.jpg', 0.65))
    # img1 = cv2.imread('mob_for.jpg.png')
    # img2 = cv2.imread('mob_etter.jpg.jpg')
    kfor = ObjectTotal('./korallrev/ref_images/korallrev_for.PNG')
    # kfor.set_contour()  # setter contouren til formen av korallrevet
    # matcher templates og filtrere overlapp, returnerer en liste med deler.
    bokser_for = kfor.template_match(templater, 0.01)
    for_med_rektangler = kfor.draw_rectangles(bokser_for)  # returnerer objektbildet med valgte deler

    plt.figure()
    plt.imshow(for_med_rektangler)
    plt.show()