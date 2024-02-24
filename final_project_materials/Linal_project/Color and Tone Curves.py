import cv2
import numpy as np
import matplotlib.pyplot as plt


def apply_color_and_tone_curve(image, priority_channel):
    image = np.float32(image) / 255.0

    contrast_alpha = 1.3
    image = cv2.pow(image, contrast_alpha)

    factor = 1.2
    if priority_channel == 'R':
        image[..., 2] = np.clip(image[..., 2] * factor, 0, 1)
    elif priority_channel == 'G':
        image[..., 1] = np.clip(image[..., 1] * factor, 0, 1)
    elif priority_channel == 'B':
        image[..., 0] = np.clip(image[..., 0] * factor, 0, 1)


    saturation_alpha = 1.3
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv_image[:, :, 1] *= saturation_alpha
    image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)

    brightness_beta = 0.1
    image = cv2.add(image, brightness_beta)

    image = np.clip(image, 0.0, 1.0)

    image = np.uint8(image * 255)

    kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    image = cv2.filter2D(image, -1, kernel)

    return image


img = cv2.imread('D:\MATLAB2022\projects\practice_linear_algebra\practice_linear_algebra_labs\project_materials\Linal_project\Color and Tone Curves.jpg')
assert img is not None, "file could not be read, check with os.path.exists()"

img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

priority_channel = input("Please select the priority color channel (R, G, B): ").strip().upper()
while priority_channel not in ('R', 'G', 'B'):
    print("Invalid input. Please choose R, G, or B.")
    priority_channel = input("Please select the priority color channel (R, G, B): ").strip().upper()

img_processed = apply_color_and_tone_curve(img, priority_channel)
img_processed_rgb = cv2.cvtColor(img_processed, cv2.COLOR_BGR2RGB)

plt.figure(figsize=(12, 6))

plt.subplot(121)
plt.imshow(img_rgb)
plt.title('Original Image')
plt.axis('off')

plt.subplot(122)
plt.imshow(img_processed_rgb)
plt.title(f'Processed Image with {priority_channel} channel priority')
plt.axis('off')

plt.show()
