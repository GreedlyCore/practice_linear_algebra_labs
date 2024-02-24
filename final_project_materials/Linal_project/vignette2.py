# Change intensity of pixel colour, depending on the distance the pixel is from the centre of the image

from skimage import data
import numpy as np
import matplotlib.pyplot as plot
from skimage import color, img_as_float, img_as_ubyte
import math

# We can use the built in image of Chelsea the cat
cat = data.chelsea()

# Convert to HSV to be able to manipulate the image intensity(v)
cat_vig = color.rgb2hsv(cat.copy())

print(f'Data type of cat is: {cat_vig.dtype}')
print(f'Shape of cat is: {cat_vig.shape}')

# Get the dimension of the matrix (n-dimensional array of colour information)
[r, c, depth] = cat_vig.shape
v_center = [c / 2, r / 2, 0]

# Derive the pixel intensity from the distance from center
def brightness(radius, image_width):
    return (100/radius) * math.exp(-radius / image_width)


# Go through each pixel and calculate its distance from center
# feed this into the brightness function
# modify the intensity component (v) of [h,s,v] for that pixel
def version1(rows, cols, rgb_img, v_center):
    for y in range(rows):
        for x in range(cols):
            me = np.array([x, y, 0])
            dist = np.linalg.norm(v_center - me)
            if dist > 100:
                cat_vig[y][x] *= [1, 1, brightness(dist, cols)]
            # cat_vign[y][x][2] *= brightness(dist, cols)

# do it
version1(r,c, cat_vig, v_center)

# Convert back to RGB so we can show in imshow()
cat_vig = color.hsv2rgb(cat_vig)
fig, ax = plot.subplots(1, 2)
ax[0].imshow(cat)  # Original version
ax[1].imshow(cat_vig)  # vignette version
plot.show()