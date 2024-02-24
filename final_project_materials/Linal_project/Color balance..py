import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

A = np.array([[[0, 0, 255], [255, 0, 0]],
              [[0, 255, 0], [255, 255, 0]]], dtype=np.uint8)
def Dim(S):
    return (S / 2).astype(np.uint8)

def Brighten(S):
    return np.clip(120 + (2 * S / 3), 0, 255).astype(np.uint8)

def Red(S):
    red_channel = S[:, :, 0]
    return np.stack([red_channel, np.zeros_like(red_channel), np.zeros_like(red_channel)], axis=-1)

def Green(S):
    green_channel = S[:, :, 1]
    return np.stack([np.zeros_like(green_channel), green_channel, np.zeros_like(green_channel)], axis=-1)

def Blue(S):
    blue_channel = S[:, :, 2]
    return np.stack([np.zeros_like(blue_channel), np.zeros_like(blue_channel), blue_channel], axis=-1)

def Rotate(S):
    return S[:, :, [2, 0, 1]]

def Reflect(S):
    return 255 - S

uploaded_image_path = 'Color Balance.jpg'
uploaded_image = Image.open(uploaded_image_path)
A_uploaded = np.array(uploaded_image)
B1_uploaded = Dim(A_uploaded)
B2_uploaded = Brighten(A_uploaded)
B3_uploaded = Red(A_uploaded)
B4_uploaded = Green(A_uploaded)
B5_uploaded = Blue(A_uploaded)
B6_uploaded = Rotate(A_uploaded)
B7_uploaded = Reflect(A_uploaded)

fig, axes = plt.subplots(2, 4, figsize=(12, 6))

# Original image
# Original image
axes[0, 0].imshow(A_uploaded)
axes[0, 0].set_title('Original')
axes[0, 0].axis('off')

# Dimmed image
axes[1, 0].imshow(B1_uploaded)
axes[1, 0].set_title('Dimmed')
axes[1, 0].axis('off')

# Brightened image
axes[1, 1].imshow(B2_uploaded)
axes[1, 1].set_title('Brightened')
axes[1, 1].axis('off')

# Red image
axes[0, 1].imshow(B3_uploaded)
axes[0, 1].set_title('Red Filter')
axes[0, 1].axis('off')

# Green image
axes[0, 2].imshow(B4_uploaded)
axes[0, 2].set_title('Green Filter')
axes[0, 2].axis('off')

# Blue image
axes[0, 3].imshow(B5_uploaded)
axes[0, 3].set_title('Blue Filter')
axes[0, 3].axis('off')

# Rotated colors image
axes[1, 2].imshow(B6_uploaded)
axes[1, 2].set_title('Rotated Colors')
axes[1, 2].axis('off')

# Reflected colors image
axes[1, 3].imshow(B7_uploaded)
axes[1, 3].set_title('Color Inversion')
axes[1, 3].axis('off')

# Adjust layout
plt.tight_layout()
plt.show()

