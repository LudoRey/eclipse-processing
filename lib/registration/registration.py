from typing import Tuple
import numpy as np
import cv2

from core.lib import filters, transform

def correlation(img1, img2):
    img1 = cv2.dft(img1, flags=cv2.DFT_COMPLEX_OUTPUT)
    img2 = cv2.dft(img2, flags=cv2.DFT_COMPLEX_OUTPUT)
    correlation_spectrum = cv2.mulSpectrums(img1, img2, 0, conjB=True)
    return cv2.idft(correlation_spectrum, flags=cv2.DFT_REAL_OUTPUT | cv2.DFT_SCALE)

def correlation_peak(img, ref_img) -> Tuple[int, int]:
    '''
    The highest peak of the correlation between img and ref_img minimizes the MSE w.r.t. integer translation img -> ref_img.
    We typically use it as an initial guess for the rigid registration.
    '''
    correlation_img = correlation(img, ref_img)
    ty, tx = np.unravel_index(np.argmax(correlation_img), correlation_img.shape)
    # Convert tx, ty to proper range
    h, w = img.shape[0:2]
    ty = ty if ty <= h // 2 else ty - h # ty in [0,h-1] -> [-h//2+1, h//2]
    tx = tx if tx <= w // 2 else tx - w
    return tx, ty

class DiscreteRigidRegistrationObjective:
    def __init__(self, ref_img, img, rotation_center):
        self.ref_img = ref_img 
        self.img = img
        self.rotation_center = rotation_center
        # Cache
        self.x = None
        self.value_at_x = None
    
    def value(self, x):
        # Cached computation
        if not np.array_equal(x, self.x) or self.value_at_x is None:
            theta, tx, ty = self.convert_x_to_params(x)
            h, w = self.img.shape[0:2]
            tform = transform.centered_rigid_transform(center=self.rotation_center, rotation=theta, translation=(tx, ty))
            registered_img = transform.warp(self.img, tform.params)
            # Compute objective value and update cache
            self.value_at_x = 1/2*np.mean((registered_img - self.ref_img)**2)
        return self.value_at_x
    
    def grad(self, x, perturbation=0.01):
        # Forward difference
        value_at_x = self.value(x)
        objective_grad = np.zeros(3)
        for i in range(3):
            perturbed_x = x.copy()
            perturbed_x[i] += perturbation
            objective_grad[i] = (self.value(perturbed_x) - value_at_x) / perturbation
        return objective_grad
    
    @staticmethod
    def convert_x_to_params(x):
        theta, tx, ty = x[0]/1800 * np.pi, x[1], x[2] # parameters of img -> registered_img 
        return theta, tx, ty
    
    @staticmethod
    def convert_params_to_x(theta, tx, ty):
        x = np.array([theta/np.pi * 1800, tx, ty])
        return x