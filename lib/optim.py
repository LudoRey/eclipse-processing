import numpy as np
from typing import Callable

def line_search_gradient_descent(x0: np.ndarray, func: Callable, grad: Callable,
                                 c=0.5, delta_initial=0.1, delta_final=1e-4,
                                 callback=lambda x: None):
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
    callback(iter, x, f, g, None)
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
        # Callback
        callback(iter, x, f, g, alpha)
        iter += 1
    return x
