import numpy as np
import cv2

def sharpen_laplacian(image):
    height, width, _ = image.shape

    # выделение границ
    # kernel = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]])
    # тождественное
    # kernel = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]])
    # kernel = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]])
    # kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
    # шум ???
    # kernel = np.array([[-1.25, 0.85, 0.12], [0.62, -0.91, 1.03], [-0.37, 0.68, -1.14]])
    kernel = np.array([[0.1496, 0.2417, 0.1496], [0.2417, 0.3894, 0.2417], [0.1496, 0.2417, 0.1496]])
    new_image = np.zeros_like(image)

    for y in range(1, height-1):
        for x in range(1, width-1):
            new_pixel = np.sum(kernel * image[y-1:y+2, x-1:x+2])
            new_image[y, x] = np.clip(new_pixel, 0, 255)

    return new_image

def main():
    # image = cv2.imread("D:\MATLAB2022\projects\practice_linear_algebra\practice_linear_algebra_labs\project_materials\dog.jpg")
    image = cv2.imread("D:\MATLAB2022\projects\practice_linear_algebra\practice_linear_algebra_labs\project_materials\catty.png")

    sharpened_image = sharpen_laplacian(image)

    cv2.imshow("Original", image)
    cv2.imshow("Sharpened", sharpened_image)
    cv2.waitKey(0)

if __name__ == "__main__":
    main()
