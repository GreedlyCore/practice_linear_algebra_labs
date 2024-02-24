import cv2
from matplotlib import pyplot as plt


src1 = cv2.imread('D:\MATLAB2022\projects\practice_linear_algebra\practice_linear_algebra_labs\project_materials\dog.jpg')
src2 = cv2.imread('D:\MATLAB2022\projects\practice_linear_algebra\practice_linear_algebra_labs\project_materials\me2.png')

src2 = cv2.resize(src2, src1.shape[1::-1])
dst = cv2.addWeighted(src1, 0.5, src2, 0.5, 0)


cv2.imwrite('lol.jpg', dst)