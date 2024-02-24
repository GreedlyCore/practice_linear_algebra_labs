import cv2
import numpy as np

import sys

print("Python version is ", sys.version)

print("opencv version is: ", cv2.__version__)
print("numpy version is: ", np.__version__)

img = cv2.imread('color-balance.png')
cv2.waitKey(0)
cv2.destroyAllWindows()

def GW_white_balance(img):
    img_LAB = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    avg_a = np.average(img_LAB[:, :, 1])
    avg_b = np.average(img_LAB[:, :, 2])
    img_LAB[:, :, 1] = img_LAB[:, :, 1] - ((avg_a - 128) * (img_LAB[:, :, 0] / 255.0) * 1.2)
    img_LAB[:, :, 2] = img_LAB[:, :, 2] - ((avg_b - 128) * (img_LAB[:, :, 0] / 255.0) * 1.2)
    balanced_image = cv2.cvtColor(img_LAB, cv2.COLOR_LAB2BGR)
    return balanced_image


image_gw_balanced = GW_white_balance(img)

cv2.waitKey(0)
cv2.destroyAllWindows()

def click_event(event, x, y, flags, params):
    # checking for left mouse clicks
    if event == cv2.EVENT_LBUTTONDOWN:
        # displaying the coordinates
        # on the Shell
        print(x, ' ', y)

        # displaying the coordinates
        # on the image window
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img, str(x) + ',' +
                    str(y), (x, y), font,
                    1, (255, 0, 0), 2)


    # checking for right mouse clicks
    if event == cv2.EVENT_RBUTTONDOWN:
        # displaying the coordinates
        # on the Shell
        print(x, ' ', y)

        # displaying the coordinates
        # on the image window
        font = cv2.FONT_HERSHEY_SIMPLEX
        b = img[y, x, 0]
        g = img[y, x, 1]
        r = img[y, x, 2]
        cv2.putText(img, str(b) + ',' +
                    str(g) + ',' + str(r),
                    (x, y), font, 1,
                    (255, 255, 0), 2)


# reading the image
img = cv2.imread('color-balance.png', 1)
clone = img.copy()

cv2.waitKey(0)
cv2.destroyAllWindows()

h_start, w_start, h_width, w_width = 174, 502, 10, 10

image = clone
image_patch = image[h_start:h_start + h_width,
              w_start:w_start + w_width]


image_normalized = image / image_patch.max(axis=(0, 1))
print(image_normalized.max())
image_balanced = image_normalized.clip(0, 1)

cv2.rectangle(clone, (w_start, h_start), (w_start + w_width, h_start + h_width), (0, 0, 255), 2)


cv2.waitKey(0)
cv2.destroyAllWindows()
image_balanced_8bit = (image_balanced * 255).astype(int)

cv2.imwrite("color-balanced-lake.png", image_balanced_8bit)

# Assuming both images are read correctly and are named 'img' and 'image_balanced'

# Convert image_balanced to 8-bit if it is not already
image_balanced_8bit = np.clip(image_balanced * 255, 0, 255).astype('uint8')

# Check if both images have the same dimensions and resize if necessary
if img.shape[:2] != image_balanced_8bit.shape[:2]:
    image_balanced_8bit = cv2.resize(image_balanced_8bit, (img.shape[1], img.shape[0]))

# Combine both images for side-by-side comparison
combined_image = np.hstack((img, image_balanced_8bit))

# Display the combined image
cv2.waitKey(0)
cv2.destroyAllWindows()

# Set a new width and height for display
display_width = 800  # for example
display_height = 600  # for example

# Read the original image
img = cv2.imread('color-balance.png')

# Resize the image for display if it's too large
if img.shape[1] > display_width or img.shape[0] > display_height:
    img = cv2.resize(img, (display_width, display_height))

# Perform your processing here...

# When displaying, ensure both images are the same size
if img.shape[:2] != image_balanced_8bit.shape[:2]:
    image_balanced_8bit = cv2.resize(image_balanced_8bit, (img.shape[1], img.shape[0]))

# Combine and display the images
combined_image = np.hstack((img, image_balanced_8bit))
cv2.imshow("Original vs Balanced", combined_image)
cv2.waitKey(0)
cv2.destroyAllWindows()