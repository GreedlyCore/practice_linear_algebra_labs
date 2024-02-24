import numpy as np
import scipy as sp
import itertools as it
import matplotlib.pyplot as plt
from scipy import signal
from matplotlib.animation import FuncAnimation

plt.style.use('dark_background')  # comment out for "light" theme
plt.rcParams["font.size"] = 12

plt.rcParams["figure.figsize"] = (12, 7)


def RGB_convolve(im1, kern):
    im2 = np.empty_like(im1)
    for dim in range(im1.shape[-1]):  # loop over rgb channels
        im2[:, :, dim] = sp.signal.convolve2d(im_data[:, :, dim],
                                              kern,
                                              mode="same",
                                              boundary="symm")
    return im2

def RGB2RGBA(arr, fill_value=1):
    """Add an alpha channel to an RGB array"""
    if arr.shape[-1] >= 4:
        return arr
    arr2 = np.full(shape=(*arr.shape[:-1], 4),
                   fill_value=fill_value,
                   dtype=arr.dtype)
    arr2[:, :, :-1] = arr/255.
    return arr2

FNAME = "Sharpening.jpg"

T = 10  # seconds
FPS = 40
FTOTAL = T*FPS  # total number of frames

KERNELS = {"Edge Detection 3x3": np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]]),
           "Sharpen 3x3": np.array([[-0.1, -0.2, -0.1], [-0.2, 3, -0.2], [-0.1, -0.2, -0.1]])}

kernel_name = "Sharpen 3x3"

kernel = KERNELS[kernel_name]

im_data = RGB2RGBA(plt.imread(FNAME).astype(float))


im_filtered = RGB_convolve(im_data, kernel)
im_filtered[:, :, -1] = 1

im_display = np.copy(im_data)


fig, (axL, axR) = plt.subplots(ncols=2, constrained_layout=True)
fig.suptitle(kernel_name)
imL = axL.imshow(im_data, interpolation="none")
imR = axR.imshow(im_data, interpolation="none")
axR.set_xlim(axL.get_xlim()), axR.set_ylim(axL.get_ylim())
axL.axis('off'), axR.axis('off')
imR.set_clim([0, 1])

indices = list(it.product(
    range(im_filtered.shape[0]), range(im_filtered.shape[1])))

Ninc = int(len(indices) / (FTOTAL)) # increment

def init_plot():
    return (imR,)


def update_plot(frame):
    for i in range(frame, frame + Ninc):
        if i >= len(indices):
            break
        idx_x, idx_y = indices[i]
        im_display[idx_x, idx_y, :] = im_filtered[idx_x, idx_y, :]

    imR.set_data(im_display)
    return (imR,)


if __name__ == "__main__":

    ani = FuncAnimation(fig, func=update_plot, init_func=init_plot, interval=1000/FPS,
                        frames=range(0, len(indices), Ninc), repeat=False, blit=True)
    plt.show()