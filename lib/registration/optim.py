import numpy as np
import skimage as sk

from typing import Callable
from core.lib.registration.transform import centered_rigid_transform

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
            inv_transform = centered_rigid_transform(center=self.rotation_center, rotation=theta, translation=(tx, ty))
            registered_img = sk.transform.warp(self.img, inv_transform) # Need to ensure that zero-padding is fine here !
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
    
    def convert_x_to_params(self, x):
        theta, tx, ty = x[0]/1800 * np.pi, x[1], x[2] # parameters of registered_img -> img (we call it the *inverse* transform)
        return theta, tx, ty
    
    def convert_params_to_x(self, theta, tx, ty):
        x = np.array([theta/np.pi * 1800, tx, ty])
        return x

def line_search_gradient_descent(x0: np.ndarray, func: Callable, grad: Callable, c=0.5, delta_initial=0.1, delta_final=1e-4):
    '''
    Gradient descent with Armijo line search. Uses an adaptive alpha_max scheme.
    An important variable here describes how much we move at each step : delta(alpha) := max|alpha*grad(x)|

    Parameters
    ----------
    - x0 : initial guess
    - func, grad : callables that return the function value and its gradient respectively.
    - c : positive value to ensure sufficient decrease
    - delta_initial : propose to initially move by delta_initial, i.e. we set alpha_max s.t. delta(alpha_max) = delta_initial
    - delta_final : stop the optimization loop when we move by less than delta_final 
    '''
    stopping_flag = False
    x = x0
    iter = 0
    f = func(x0)
    g = grad(x0)
    # Determines alpha_max
    alpha_max = delta_initial/np.max(np.abs(g))
    while not stopping_flag:
        # Armijo line search
        alpha = alpha_max
        while not (f_next:= func(x - alpha*g)) <= f - c*alpha*np.dot(g, g):
            alpha /= 2
            # Stopping criterion : if not moving by much
            if alpha*np.max(np.abs(g)) <= delta_final:
                stopping_flag = True
                alpha = 0
        # Accept step
        x = x - alpha*g
        f = f_next
        g = grad(x)
        # Update alpha max (double or halve)
        if alpha == alpha_max:
            alpha_max *= 2
        else:
            alpha_max /= 2
        # Display info
        print(f"Iteration {iter}:")
        print(f"Value : {f:.3e}")
        print(f"x : {x}")
        print(f"Gradient : {g}")
        print(f"alpha : {alpha} \n")
        iter += 1
    return x
