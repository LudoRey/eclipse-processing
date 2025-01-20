from scipy.ndimage import sobel
from skimage import transform
import numpy as np

def centered_rigid_transform(center, rotation, translation):
    '''Rotate first around center, then translate'''
    t_uncenter = transform.AffineTransform(translation=center)
    t_center = t_uncenter.inverse
    t_rotate = transform.AffineTransform(rotation=rotation)
    t_translate = transform.AffineTransform(translation=translation)
    return t_center + t_rotate + t_uncenter + t_translate

def sobel_grad_xy(img):
    'Computes the gradient of the image using Sobel filters. Returns a (2,H,W) ndarray.'
    return np.stack([sobel(img, axis=1), sobel(img, axis=0)])

def rotation_first_derivative(theta, x, y):
    '''
    Computes the first derivative (jacobian) of a 2D rotation with respect to the *angle*. 
    Batch computation is supported, e.g. x, y can be the output of np.meshgrid
    Returns an array of shape (2, *x.shape)
    '''
    d_theta = np.stack([-x*np.sin(theta) - y*np.cos(theta),
                         x*np.cos(theta) - y*np.sin(theta)])
    return d_theta

def rotation_second_derivative(theta, x, y):
    d2_theta = np.stack([-x*np.cos(theta) + y*np.sin(theta),
                         -x*np.sin(theta) - y*np.cos(theta)])
    return d2_theta

class RigidRegistrationObjective:
    def __init__(self, ref_img, img):
        # Precompute constants
        self.ref_img = ref_img 
        self.img = img
        h, w = img.shape
        self.img_grad_xy = sobel_grad_xy(img)
        self.img_hess_xy = np.zeros((3,h,w))
        self.img_hess_xy[0] = sobel(self.img_grad_xy[0], axis=1) # d_xx
        self.img_hess_xy[1] = sobel(self.img_grad_xy[0], axis=0) # d_xy (same as d_yx by commutativity)
        self.img_hess_xy[2] = sobel(self.img_grad_xy[1], axis=0) # d_yy
        # Cache
        self.x = None
        self.registered_img = None
        self.registered_img_grad_xy = None
        
    def get_registered_img(self, x):
        'Uses cache. Computes the registered image.'
        if not np.array_equal(x, self.x) or self.registered_img is None:
            theta, tx, ty = self.convert_x_to_params(x) # parameters of registered_img -> img (we call it the *inverse* transform)
            h, w = self.img.shape[0:2]
            inv_transform = centered_rigid_transform(center=(w/2,h/2), rotation=theta, translation=(tx, ty))
            # Update cache
            self.x = x
            self.registered_img = transform.warp(self.img, inv_transform) # Need to ensure that zero-padding is fine here !
            self.registered_img_grad_xy = None
        return self.registered_img
    
    def get_registered_img_grad_xy(self, x):
        'Uses cache. Computes gradient of the registered image with respect to the spatial coords (x,y)'
        if not np.array_equal(x, self.x) or self.registered_img_grad_xy is None:
            registered_img = self.get_registered_img(x) # updates self.x and self.registered_img
            theta, tx, ty = self.convert_x_to_params(x) # parameters of registered_img -> img (we call it the *inverse* transform)
            h, w = self.img.shape[0:2]
            inv_transform = centered_rigid_transform(center=(w/2,h/2), rotation=theta, translation=(tx, ty))
            # Update cache
            self.registered_img_grad_xy = transform.warp(self.img_grad_xy.transpose((1,2,0)), inv_transform).transpose((2,0,1)) # (2,H,W)
        return self.registered_img_grad_xy
    
    def convert_x_to_params(self, x):
        theta, tx, ty = x[0]/1800 * np.pi, x[1], x[2] # parameters of registered_img -> img (we call it the *inverse* transform)
        return theta, tx, ty
    
    def convert_params_to_x(self, theta, tx, ty):
        x = np.array([theta/np.pi * 1800, tx, ty])
        return x

    def value(self, x):
        # Cached computation of the registered image
        registered_img = self.get_registered_img(x)
        # Compute objective value
        objective_value = 1/2*np.mean((registered_img - self.ref_img)**2)
        return objective_value

    def grad(self, x):
        # Cached computation of the gradient of the registered image with respect to the spatial coords (x,y)
        registered_img_grad_xy = self.get_registered_img_grad_xy(x) # (2,H,W)

        theta, tx, ty = self.convert_x_to_params(x)
        h, w = self.ref_img.shape
        
        # We first compute the gradient of the registered image with respect to theta, tx, ty for all points
        # img_grad = tform_jac^T @ img_grad_xy
        registered_img_grad = np.zeros((3,h,w)) # (3,H,W)

        # tform_jac if the (2,3) Jacobian matrix of the transform (w.r.t. theta, tx, ty)
        # where tform_jac[:,1:] = eye(2) and tform_jac[:,0] depends on the spatial coords x,y
        # a) Gradient with respect to tx, ty is simply registered_img_grad_xy (the gradient with respect to the spatial coords x, y)
        registered_img_grad[1:] = registered_img_grad_xy
        # b) d(img)/d(theta) involves tform_jac[:,0] = d(tform)/d(theta)
        x, y = np.meshgrid(np.arange(w) - w/2, np.arange(h) - h/2) # each (H,W)
        d_theta = rotation_first_derivative(theta, x, y) # (2,H,W)
        registered_img_grad[0] = np.sum(d_theta*registered_img_grad_xy, axis=0)

        # Compute objective gradient with respect to theta, tx, ty
        objective_grad = np.mean(-self.ref_img.reshape(1,h,w)*registered_img_grad, axis=(1,2)) # (3,)
        # Compute objective gradient with respect to x (account for scaling)
        objective_grad[0] *= np.pi / 1800
        return objective_grad
    
    def hess(self, x):
        # Cached computation of the gradient of the registered image with respect to the spatial coords (x,y)
        registered_img_grad_xy = self.get_registered_img_grad_xy(x) # (2,H,W)
        # Unpack
        theta, tx, ty = x[0], x[1], x[2] # parameters of registered_img -> img (we call it the *inverse* transform)
        h, w = self.ref_img.shape

        # We first compute the Hessian of the registered image with respect to theta, tx, ty for all points
        # img_hess = tform_jac^T @ img_hess_xy @ tform_jac + tform_hess @ img_grad_xy
        # By symmetry, only need to store the (flattened) upper triangular part, i.e.
        # the second order derivatives in the order: theta^2, theta*tx, theta*ty, tx**2, tx*ty, ty**2
        registered_img_hess = np.zeros((6,h,w)) 

        # 1) tform_jac^T @ img_hess_xy @ tform_jac
        # tform_jac is the (2,3) Jacobian matrix of the transform (w.r.t. theta, tx, ty)
        # where tform_jac[:,1:] = eye(2) and tform_jac[:,0] depends on the spatial coords x,y
        # a) Hessian with respect to tx, ty is simply registered_img_hess_xy (the Hessian with respect to the spatial coords x, y)
        registered_img_hess_xy = registered_img_hess[3:6] # only a view; share the same memory
        inv_transform = centered_rigid_transform(center=(w/2,h/2), rotation=theta, translation=(tx, ty))
        registered_img_hess_xy[:] = transform.warp(self.img_hess_xy.transpose((1,2,0)), inv_transform).transpose((2,0,1)) # (2,H,W)
        # b) Cross terms theta^2, theta*tx, theta*ty involve tform_jac[:,0] = d(tform)/d(theta)
        x, y = np.meshgrid(np.arange(w) - w/2, np.arange(h) - h/2) # each (H,W)
        d_theta = rotation_first_derivative(theta, x, y) # (2,H,W)
        registered_img_hess[0] = d_theta[0]**2*registered_img_hess_xy[0] + 2*d_theta[0]*d_theta[1]*registered_img_hess_xy[1] + d_theta[1]**2*registered_img_hess_xy[1]
        registered_img_hess[1] = np.sum(d_theta*registered_img_hess_xy[:-1], axis=0)
        registered_img_hess[2] = np.sum(d_theta*registered_img_hess_xy[1:], axis=0)

        # 2) tform_hess @ img_grad_xy
        # where tform_hess is the (2,3,3) Hessian tensor of the transform (w.r.t. theta, tx, ty)
        # tform_hess[:,0,0] = d^2(tform)/d(theta)^2 is the only non-zero component
        d_theta = rotation_second_derivative(theta, x, y) # (2,H,W), reuse the same memory 
        registered_img_hess[0] += np.sum(d_theta*registered_img_grad_xy, axis=0)

        # Compute objective Hessian
        objective_hess = np.mean(-self.ref_img.reshape(1,h,w)*registered_img_hess, axis=(1,2)) # (6,)
        return np.array([[objective_hess[0], objective_hess[1], objective_hess[2]],
                         [objective_hess[1], objective_hess[3], objective_hess[4]],
                         [objective_hess[2], objective_hess[4], objective_hess[5]]])