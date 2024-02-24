from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

image_path = 'Brightness and Contrast Adjustment.jpg'
image = Image.open(image_path)

if image.mode != 'RGBA':
    image = image.convert('RGBA')

# Функция для корректировки яркости и контрастности с использованием матричных операций для изображений RGBA
def adjust_brightness_contrast_matrix_rgba(input_img, brightness_factor, contrast_factor):
    img_arr = np.asarray(input_img, dtype=np.float32)

    img_rgb = img_arr[..., :3]
    img_alpha = img_arr[..., 3:]

    img_brightened = img_rgb * brightness_factor

    img_normalized = img_brightened / 255.0

    mean = np.mean(img_rgb)
    img_contrasted = ((img_normalized - 0.5) * contrast_factor + 0.5) * 255.0

    img_adjusted = np.concatenate((img_contrasted, img_alpha), axis=-1)

    img_adjusted = np.clip(img_adjusted, 0, 255).astype(np.uint8)
    return Image.fromarray(img_adjusted, 'RGBA')

brightness_factor = 4
contrast_factor = 2
adjusted_image = adjust_brightness_contrast_matrix_rgba(image, brightness_factor, contrast_factor)

fig, ax = plt.subplots(1, 2, figsize=(12, 5))

ax[0].imshow(image)
ax[0].set_title('Original')
ax[0].axis('off')

ax[1].imshow(adjusted_image)
ax[1].set_title('Brightness and Contrast Adjustment')
ax[1].axis('off')

plt.tight_layout()
plt.show()