import numpy as np
import time

from skimage import transform, data
from matplotlib import pyplot as plt
from scipy import optimize
from contextlib import contextmanager

from lib.phasecorr import correlation
from lib.optimization import mse_rigid_registration_func_and_grad, sobel_grad_xy, RigidRegistrationObjective, centered_rigid_transform

@contextmanager
def timer(name: str = "Timer"):
    start_time = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start_time
        print(f"{name}: {elapsed:.6f} seconds")

def make_test_image(width=16):
    img = np.zeros((width, width))
    img[width // 4 : 3 * width // 4, width // 4 : 3 * width // 4] = 1
    return img

def pad_image(img, width):
    padded_img = np.zeros(np.array(img.shape) + 2*width)
    padded_img[width:-width, width:-width] = img 
    return padded_img

def display_obj(theta, dx, dy, s=None):
    x = np.array([theta, dx, dy])
    if s is not None:
        print(s)
    print("(theta, dx, dy):", np.rad2deg(theta), dx, dy)
    print("Value:", obj.value(x))
    print("Gradient:", obj.grad(x), "\n")

### Create reference image
ref_img = data.camera().astype('float') / 255 # convert to float
ref_img = (ref_img - ref_img.mean()) / ref_img.std() # normalize
ref_img = ref_img*np.outer(np.hanning(ref_img.shape[0]), np.hanning(ref_img.shape[1])) # hanning window
ref_img = pad_image(ref_img, 100)
# ref_img = make_test_image()

h, w = ref_img.shape

### Create misaligned image

theta, dx, dy = np.deg2rad(10), 10, 0

tform = centered_rigid_transform(center=(w/2,h/2), rotation=theta, translation=(dx, dy))
img = transform.warp(ref_img, tform.inverse) # warp takes the inverse transform as input

obj = RigidRegistrationObjective(ref_img, img)

display_obj(theta, dx, dy, "Ground truth")

correlation_img = correlation(img, ref_img)
theta = 0
dy, dx = np.unravel_index(np.argmax(correlation_img), correlation_img.shape)
# (dy, dx) in [0,h-1] x [0,w-1] -> [h//2+1-h, h//2] = [-h//2+1, h//2]
h, w = correlation_img.shape
dy = dy if dy <= h // 2 else dy - h
dx = dx if dx <= w // 2 else dx - w

display_obj(theta, dx, dy, "Coarse estimate")

### Register image
# Optimize
def callback(x):
    pass
    #display_obj(x[0], x[1], x[2])

# result = optimize.minimize(obj.value,
#                            x0=np.array([0.0, dx, dy]),
#                            method='trust-exact',
#                            jac=obj.grad,
#                            hess=obj.hess,
#                            callback=callback, options={'disp': True})
func = lambda x : mse_rigid_registration_func_and_grad(ref_img, img, x[0], x[1], x[2])

result = optimize.minimize(func,
                           x0=np.array([0.0, dx, dy]),
                           method='BFGS',
                           jac=True,
                           callback=callback, options={'disp': True})

theta, dx, dy = result.x # estimated parameters of the transform ref_img -> img

display_obj(theta, dx, dy, "Refined estimate")

tform = centered_rigid_transform(center=(w/2,h/2), rotation=theta, translation=(dx, dy))
registered_img = transform.warp(img, tform) # inverse of the inverse here, be careful
registered_img_grad_xy = sobel_grad_xy(registered_img) # (2,H,W)

fig, axes = plt.subplots(3,3)
axes[0,0].imshow(ref_img)
axes[0,1].imshow(img)
axes[0,2].imshow(registered_img)

# for i in range(3):
#     axes[1,i].imshow(out1[i])
#     axes[2,i].imshow(out2[i])
#     print(np.mean(out2[i]))

plt.show()