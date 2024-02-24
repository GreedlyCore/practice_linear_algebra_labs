import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# Function to apply color and tone enhancements
def apply_color_and_tone_curve(image, priority_channel):
    # Convert to float to prevent clipping values
    image = np.float32(image) / 255.0

    # Apply a simple contrast curve
    contrast_alpha = 1.3  # Contrast control (1.0-3.0)
    image = cv2.pow(image, contrast_alpha)

    # Priority channel enhancement
    # Increase the selected channel by a factor, capping it at 1.0
    factor = 1.2  # The amount by which to enhance the priority channel
    if priority_channel == 'R':
        image[..., 2] = np.clip(image[..., 2] * factor, 0, 1)
    elif priority_channel == 'G':
        image[..., 1] = np.clip(image[..., 1] * factor, 0, 1)
    elif priority_channel == 'B':
        image[..., 0] = np.clip(image[..., 0] * factor, 0, 1)

    # Apply a saturation curve
    saturation_alpha = 1.3  # Saturation control (1.0-2.0)
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv_image[:, :, 1] *= saturation_alpha
    image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)

    # Apply a simple brightness curve
    brightness_beta = 0.1  # Brightness control (-0.5 to 0.5)
    image = cv2.add(image, brightness_beta)

    # Clip the values to keep them between 0.0 and 1.0
    image = np.clip(image, 0.0, 1.0)

    # Convert back to 8-bit
    image = np.uint8(image * 255)

    # Apply sharpening
    kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    image = cv2.filter2D(image, -1, kernel)

    return image


# Load the image from the file
# print(os.path.exists('dog.png'))
img = cv2.imread('D:\MATLAB2022\projects\practice_linear_algebra\practice_linear_algebra_labs\project_materials\dog.png')
assert img is not None, "file could not be read, check with os.path.exists()"

# Convert BGR to RGB for matplotlib
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Prompt user to select the priority channel
priority_channel = input("Please select the priority color channel (R, G, B): ").strip().upper()
while priority_channel not in ('R', 'G', 'B'):
    print("Invalid input. Please choose R, G, or B.")
    priority_channel = input("Please select the priority color channel (R, G, B): ").strip().upper()

# Apply the color and tone adjustments with the selected priority channel
img_processed = apply_color_and_tone_curve(img, priority_channel)
img_processed_rgb = cv2.cvtColor(img_processed, cv2.COLOR_BGR2RGB)

# Display the original and processed images using matplotlib
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
