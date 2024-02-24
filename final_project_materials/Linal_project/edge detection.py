import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

img = cv.imread('edge detection.jpg')
assert img is not None, "file could not be read, check with os.path.exists()"

edges = cv.Canny(img, 100, 50)

plt.figure(figsize=(10, 5))

plt.subplot(121), plt.imshow(img)
plt.title('Original'), plt.xticks([]), plt.yticks([])

plt.subplot(122), plt.imshow(edges)
plt.title('Edge detection'), plt.xticks([]), plt.yticks([])

plt.show()
