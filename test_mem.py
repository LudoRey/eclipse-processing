import tracemalloc
import numpy as np
from scipy.ndimage import sobel
from lib.utils import Timer

# Start tracing memory allocations
# tracemalloc.start()

# Your script logic

class Foo:
    def __init__(self, a):
        self.a = a

a = 0
foo = Foo(a)
a = input("Type any number:")
print(foo.a)

a = 0
foo_factory = lambda a : Foo(a)
a = input("Type any number:")
print(foo_factory(a).a)


# # Get memory usage
# current, peak = tracemalloc.get_traced_memory()
# print(f"Current memory usage: {current / 1024**2:.2f} MB")
# print(f"Peak memory usage: {peak / 1024**2:.2f} MB")

# # Stop the tracing
# tracemalloc.stop()