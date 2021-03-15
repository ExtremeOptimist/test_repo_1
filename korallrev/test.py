import cv2
import matplotlib.pyplot as plt

im = cv2.imread('./korallrev/templates/ut_1.png')
plt.imshow(im)
plt.show()

# this works
# 'd:\\coding\\test_repo_1\\korallrev\\ut_2.png'

# This does not: (I want this)
# './ut_2.png'


# This does not: (unicode escape error backslash)
# '.\ut_2.png'
