import tracemalloc
import numpy as np
from lib.utils import Timer
from lib.optimization import rotation_first_derivative, RigidRegistrationObjective

# Start tracing memory allocations
tracemalloc.start()

# Your script logic
h, w = 4000, 6000
ref_img = np.zeros((h,w))
img = np.zeros((h,w))

obj = RigidRegistrationObjective(ref_img, img)

x = np.zeros(3)
with Timer():
    g = obj.grad(x)
    h = obj.hess(x)


# Get memory usage
current, peak = tracemalloc.get_traced_memory()
print(f"Current memory usage: {current / 1024**2:.2f} MB")
print(f"Peak memory usage: {peak / 1024**2:.2f} MB")

# Stop the tracing
tracemalloc.stop()