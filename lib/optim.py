import numpy as np
from typing import Callable
from core.lib.utils import Timer

def line_search_gradient_descent(x0: np.ndarray, func: Callable, grad: Callable,
                                 c=0.1, delta_max=1, delta_min=1e-3, max_iter=20,
                                 callback=lambda x: None):
    '''
    Gradient descent with Armijo line search. Two-way backtracking (see https://en.wikipedia.org/wiki/Backtracking_line_search).
    An important variable here describes how much we move at each step : delta(alpha) := max|alpha*grad(x)|

    Parameters
    ----------
    - x0 : initial guess
    - func, grad : callables that return the function value and its gradient respectively.
    - c : positive value to ensure sufficient decrease.
    - delta_max : ensures the line search does not make us move by more than delta_max
    - delta_min : stop the optimization loop when we move by less than delta_min
    '''
    x = x0
    f = func(x0)
    g = grad(x0)
    callback(0, x, f, g, None)
    # Set alpha
    alpha_initial = delta_max/np.max(np.abs(g)) # try to move as much as possible
    # Early termination flag
    converged = False
    for iter in range(1, max_iter+1):
        # Armijo condition : decrease must be at least proportional to t
        t = c*np.dot(g,g)
        # 2-way line search
        alpha = alpha_initial
        f_next = func(x - alpha*g) # proposed value
        accepted = False
        if f - f_next >= alpha*t: # Armijo's condition is initially satisfied: increase the step size until it isn't
            while not accepted:
                alpha *= 2
                if alpha*np.max(np.abs(g)) > delta_max: # Alpha is too big; stop here
                    alpha /= 2 # Compensate for overshoot
                    accepted = True
                else:
                    f_temp = f_next
                    f_next = func(x - alpha*g)
                    if f - f_next < alpha*t: # Armijo's condition is not satisfied anymore; stop here
                        alpha /= 2 # Compensate for overshoot
                        f_next = f_temp
                        accepted = True
        else: # Armijo's condition is *not* satisfied; reduce step size until it is
            while not (accepted or converged):
                alpha /= 2
                if alpha*np.max(np.abs(g)) <= delta_min: # Stopping criterion : if not moving by much
                    converged = True
                else:
                    f_next = func(x - alpha*g)
                    if f - f_next >= alpha*t: # Armijo's condition is now satisfied: stop here
                        accepted = True

        if converged:
            print("Convergence reached. Optimization was terminated early.")
            return x
        # Accept step
        x = x - alpha*g
        delta = alpha*np.max(np.abs(g))
        f = f_next
        with Timer():
            g = grad(x)
        alpha_initial = alpha
        # Callback
        callback(iter, x, f, g, delta)
    print("Maximum number of iterations reached. Optimization terminated.")
    return x
