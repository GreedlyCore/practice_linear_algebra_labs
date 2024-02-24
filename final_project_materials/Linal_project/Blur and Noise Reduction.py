import cv2
import numpy as np
import matplotlib.pyplot as plt

original_image = cv2.imread('Blur and Noise Reduction.jpg')

M = 1
blur_kernel = np.ones((M, M)) * 1 / (M * M)
# apply the convolution to blur
blur_image = cv2.filter2D(src=original_image, ddepth=-1, kernel=blur_kernel)

median_image = cv2.medianBlur(blur_image, 29)

fig, ax = plt.subplots(1, 2, figsize=(12, 4))
ax[0].imshow(original_image)
ax[0].set_title('Original')
ax[0].axis('off')

ax[1].imshow(original_image)
ax[1].set_title('Blur and Noise Reduction')
ax[1].axis('off')

plt.show()


