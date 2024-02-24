from skimage import io, color
import numpy as np
import matplotlib.pyplot as plt
import math

def apply_vignette(image_path):

    img = io.imread(image_path)
    img = color.rgb2hsv(img)

    [rows, cols, depth] = img.shape
    v_center = [cols / 2, rows / 2, 0]

    def brightness(radius, image_width):
        return (100/radius) * math.exp(-radius**2 / (2 * (image_width / 2)**2))

    for y in range(rows):
        for x in range(cols):
            dist = np.linalg.norm(np.array([x, y, 0]) - v_center)
            if dist > 700:
                img[y, x, 2] *= brightness(dist, cols)

    img_vig = color.hsv2rgb(img)

    return img_vig

image_path =
vignette_image = apply_vignette(image_path)

original_image = io.imread(image_path)
fig, ax = plt.subplots(1, 2, figsize=(12, 6))
ax[0].imshow(original_image)
ax[0].set_title('Original')
ax[0].axis('off')

ax[1].imshow(vignette_image)
ax[1].set_title('Vignette Effect')
ax[1].axis('off')

plt.show()
