import cv2
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")

image = cv2.imread("D:\MATLAB2022\projects\practice_linear_algebra\practice_linear_algebra_labs\project_materials\Linal_project\Image Segmentation.jpg", 0)

flattened_image = image.reshape((image.shape[0] * image.shape[1],))

ret, thresh1 = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

plt.figure(figsize=(8, 5))
plt.subplot(121)
plt.title('Original Image')
plt.imshow(image, cmap="gray")
plt.axis("off")

plt.subplot(122)
plt.imshow(thresh1, cmap="binary")
plt.title('Image Segmentation')
plt.axis("off")
plt.show()