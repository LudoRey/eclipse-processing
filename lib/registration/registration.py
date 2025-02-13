from typing import Tuple
import numpy as np
import cv2
import numba

from core.lib import filters, transform
from core.lib.utils import Timer

def correlation(img1, img2):
    img1 = cv2.dft(img1, flags=cv2.DFT_COMPLEX_OUTPUT)
    img2 = cv2.dft(img2, flags=cv2.DFT_COMPLEX_OUTPUT)
    correlation_spectrum = cv2.mulSpectrums(img1, img2, 0, conjB=True)
    return cv2.idft(correlation_spectrum, flags=cv2.DFT_REAL_OUTPUT | cv2.DFT_SCALE)

def correlation_peak(img1, img2) -> Tuple[int, int]:
    '''
    The highest peak of the correlation between img1 and img2 minimizes the MSE w.r.t. integer translation img2 -> img1.
    We typically use it as an initial guess for the rigid registration.
    '''
    correlation_img = correlation(img1, img2)
    ty, tx = np.unravel_index(np.argmax(correlation_img), correlation_img.shape)
    # Convert tx, ty to proper range
    h, w = img1.shape[0:2]
    ty = ty if ty <= h // 2 else ty - h # ty in [0,h-1] -> [-h//2+1, h//2]
    tx = tx if tx <= w // 2 else tx - w
    return tx, ty

class DiscreteRigidRegistrationObjective:
    def __init__(self, ref_img, img, rotation_center):
        '''
        To find the params of the optimal *inverse* transform, i.e. ref_img -> img
        '''
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
            tform = transform.centered_rigid_transform(center=self.rotation_center, rotation=theta, translation=(tx, ty))
            registered_img = transform.warp(self.img, tform.inverse.params)
            # Compute objective value and update cache
            self.value_at_x = 1/2*np.mean((registered_img - self.ref_img)**2)
        return self.value_at_x
    
    def grad(self, x, perturbation=0.1):
        # Forward difference
        value_at_x = self.value(x)
        objective_grad = np.zeros(3)
        for i in range(3):
            perturbed_x = x.copy()
            perturbed_x[i] -= perturbation
            objective_grad[i] = (-self.value(perturbed_x) + value_at_x) / perturbation
        return objective_grad
    
    @staticmethod
    def convert_x_to_params(x):
        theta, tx, ty = x[0]/1800 * np.pi, x[1], x[2] # parameters of img -> registered_img 
        return theta, tx, ty
    
    @staticmethod
    def convert_params_to_x(theta, tx, ty):
        x = np.array([theta/np.pi * 1800, tx, ty])
        return x
    
class RigidRegistrationObjective:
    def __init__(self, ref_img, img, rotation_center):
        self.ref_img = ref_img 
        self.img = img
        self.rotation_center = rotation_center

        # Precompute constants
        self.h, self.w = self.ref_img.shape
        self.x_grid, self.y_grid = np.meshgrid(np.arange(self.w) - self.rotation_center[0],
                                               np.arange(self.h) - self.rotation_center[1]) # each (H,W)
        
        self.img_grad_xy = np.zeros((*img.shape, 2))
        self.img_grad_xy[...,0] = cv2.Sobel(img, cv2.CV_64F, 1, 0)
        self.img_grad_xy[...,1] = cv2.Sobel(img, cv2.CV_64F, 0, 1)
        # self.img_hess_xy = np.zeros((*img.shape, 3)) # can store a symmetric (2,2) matrix in a (3,) vector
        # self.img_hess_xy[0] = sobel(self.img_grad_xy[0], axis=1) # d_xx
        # self.img_hess_xy[1] = sobel(self.img_grad_xy[0], axis=0) # d_xy (same as d_yx by commutativity)
        # self.img_hess_xy[2] = sobel(self.img_grad_xy[1], axis=0) # d_yy
    
    def value(self, x):
        # Get the transform "img -> registered_img" determined by theta, tx, ty
        # What we actually want to apply is its inverse "img -> registered_img"
        theta, tx, ty = self.convert_x_to_params(x) 
        tform = transform.centered_rigid_transform(center=self.rotation_center, rotation=theta, translation=(tx, ty)) 
        registered_img = transform.warp(self.img, tform.inverse.params) # is zero-padding fine here ?
        diff = registered_img - self.ref_img
        return 1/2*cv2.mean(diff*diff)[0]

    def grad(self, x):
        # Get the transform "img -> registered_img" determined by theta, tx, ty
        # What we actually want to apply is its inverse, i.e. "img -> registered_img"
        theta, tx, ty = self.convert_x_to_params(x) 
        tform = transform.centered_rigid_transform(center=self.rotation_center, rotation=theta, translation=(tx, ty)) 
        registered_img_grad_xy = transform.warp(self.img_grad_xy, tform.inverse.params) # (H,W,2)
    
        # We now compute the gradient of the registered image with respect to theta, tx, ty for all points (x,y)
        # which is given by registered_img_grad = tform_jac.T @ registered_img_grad_xy
        # Here, tform_jac is the (2,3) Jacobian matrix of the transform (w.r.t. its parameters theta, tx, ty) at each point (x,y)
        # a) tform_jac[:,0] = tform_jac_theta depends on (x,y), so we store it in (H,W) arrays
        tform_0_jac_theta = cv2.addWeighted(self.x_grid, -np.sin(theta), # (H,W)
                                             self.y_grid, -np.cos(theta),
                                             0.0)
        tform_1_jac_theta = cv2.addWeighted(self.x_grid, np.cos(theta), # (H,W)
                                             self.y_grid, -np.sin(theta),
                                             0.0)
        # registered_img_grad_theta = tform_jac_theta.T @ registered_img_grad_xy, i.e. a dot product (for every point)
        registered_img_grad_theta = cv2.multiply(tform_0_jac_theta, registered_img_grad_xy[...,0]) + \
                                    cv2.multiply(tform_1_jac_theta, registered_img_grad_xy[...,1])
        # b) Since tform_jac[:,1:] = eye(2) for all (x,y), we have registered_img_grad_txty = registered_img_grad_xy
        registered_img_grad_tx = registered_img_grad_xy[...,0]
        registered_img_grad_ty = registered_img_grad_xy[...,1]

        # We are now ready to compute the gradient of the objective function (MSE) with respect to theta, tx, ty
        # Simple application of the chain rule here
        objective_grad = np.zeros(3)
        objective_grad[0] = -cv2.mean(self.ref_img*registered_img_grad_theta)[0]
        objective_grad[1] = -cv2.mean(self.ref_img*registered_img_grad_tx)[0]
        objective_grad[2] = -cv2.mean(self.ref_img*registered_img_grad_ty)[0]
        # Compute objective gradient with respect to x (account for scaling)
        objective_grad[0] *= np.pi / 1800 # dx / dthetha
        return objective_grad
    
    def grad_numba(self, x):
        # Get the transform "img -> registered_img" determined by theta, tx, ty
        # What we actually want to apply is its inverse, i.e. "img -> registered_img"
        theta, tx, ty = self.convert_x_to_params(x) 
        tform = transform.centered_rigid_transform(center=self.rotation_center, rotation=theta, translation=(tx, ty)) 
        registered_img_grad_xy = transform.warp(self.img_grad_xy, tform.inverse.params) # (H,W,2)
        return self._grad_numba(self.ref_img, registered_img_grad_xy, self.x_grid, self.y_grid, theta)
    
    @staticmethod
    @numba.njit()
    def _grad_numba(ref_img, registered_grad_xy, x_grid, y_grid, theta):
        d_theta_0 = - x_grid*np.sin(theta) - y_grid*np.cos(theta)
        d_theta_1 = x_grid*np.cos(theta) - y_grid*np.sin(theta)

        registered_img_grad_theta = d_theta_0*registered_grad_xy[...,0] + \
                                    d_theta_1*registered_grad_xy[...,1] # (H,W)

        objective_grad = np.zeros(3)
        objective_grad[0] = -np.mean(ref_img*registered_img_grad_theta)
        objective_grad[1] = -np.mean(ref_img*registered_grad_xy[...,0])
        objective_grad[2] = -np.mean(ref_img*registered_grad_xy[...,1])
        # Compute objective gradient with respect to x (account for scaling)
        objective_grad[0] *= np.pi / 1800 # dx / dthetha
        return objective_grad
    
    # def hess(self, x):
    #     # Cached computation of the gradient of the registered image with respect to the spatial coords (x,y)
    #     registered_img_grad_xy = self.get_registered_img_grad_xy(x) # (2,H,W)
    #     # Unpack
    #     theta, tx, ty = x[0], x[1], x[2] # parameters of registered_img -> img (we call it the *inverse* transform)
    #     h, w = self.ref_img.shape

    #     # We first compute the Hessian of the registered image with respect to theta, tx, ty for all points
    #     # img_hess = tform_jac^T @ img_hess_xy @ tform_jac + tform_hess @ img_grad_xy
    #     # By symmetry, only need to store the (flattened) upper triangular part, i.e.
    #     # the second order derivatives in the order: theta^2, theta*tx, theta*ty, tx**2, tx*ty, ty**2
    #     registered_img_hess = np.zeros((6,h,w)) 

    #     # 1) tform_jac^T @ img_hess_xy @ tform_jac
    #     # tform_jac is the (2,3) Jacobian matrix of the transform (w.r.t. theta, tx, ty)
    #     # where tform_jac[:,1:] = eye(2) and tform_jac[:,0] depends on the spatial coords x,y
    #     # a) Hessian with respect to tx, ty is simply registered_img_hess_xy (the Hessian with respect to the spatial coords x, y)
    #     registered_img_hess_xy = registered_img_hess[3:6] # only a view; share the same memory
    #     inv_transform = centered_rigid_transform(center=(w/2,h/2), rotation=theta, translation=(tx, ty))
    #     registered_img_hess_xy[:] = transform.warp(self.img_hess_xy.transpose((1,2,0)), inv_transform).transpose((2,0,1)) # (2,H,W)
    #     # b) Cross terms theta^2, theta*tx, theta*ty involve tform_jac[:,0] = d(tform)/d(theta)
    #     x, y = np.meshgrid(np.arange(w) - w/2, np.arange(h) - h/2) # each (H,W)
    #     d_theta = rotation_first_derivative(theta, x, y) # (2,H,W)
    #     registered_img_hess[0] = d_theta[0]**2*registered_img_hess_xy[0] + 2*d_theta[0]*d_theta[1]*registered_img_hess_xy[1] + d_theta[1]**2*registered_img_hess_xy[1]
    #     registered_img_hess[1] = np.sum(d_theta*registered_img_hess_xy[:-1], axis=0)
    #     registered_img_hess[2] = np.sum(d_theta*registered_img_hess_xy[1:], axis=0)

    #     # 2) tform_hess @ img_grad_xy
    #     # where tform_hess is the (2,3,3) Hessian tensor of the transform (w.r.t. theta, tx, ty)
    #     # tform_hess[:,0,0] = d^2(tform)/d(theta)^2 is the only non-zero component
    #     d_theta = rotation_second_derivative(theta, x, y) # (2,H,W), reuse the same memory 
    #     registered_img_hess[0] += np.sum(d_theta*registered_img_grad_xy, axis=0)

    #     # Compute objective Hessian
    #     objective_hess = np.mean(-self.ref_img.reshape(1,h,w)*registered_img_hess, axis=(1,2)) # (6,)
    #     return np.array([[objective_hess[0], objective_hess[1], objective_hess[2]],
    #                      [objective_hess[1], objective_hess[3], objective_hess[4]],
    #                      [objective_hess[2], objective_hess[4], objective_hess[5]]])
    
    @staticmethod
    def convert_x_to_params(x):
        theta, tx, ty = x[0]/1800 * np.pi, x[1], x[2] # parameters of registered_img -> img (we call it the *inverse* transform)
        return theta, tx, ty
    
    @staticmethod
    def convert_params_to_x(theta, tx, ty):
        x = np.array([theta/np.pi * 1800, tx, ty])
        return x
    
