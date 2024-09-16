import time
import io
import sys

class Timer:
    def __enter__(self):
        self.start_time = time.perf_counter() 
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.perf_counter() 
        elapsed_time = self.end_time - self.start_time
        print(f"Elapsed time: {elapsed_time:.2f} seconds")

class ColorTerminalStream(io.TextIOWrapper):
    def __init__(self, **kwargs):
        super().__init__(buffer=sys.__stdout__.buffer, **kwargs)
        self.ansi_dict = {
            "default": 0,
            "red": 31,
            "green": 32,
            "yellow": 33,
            "blue": 34,
            "magenta": 35,
            "cyan": 36,
            "white": 37
        }

    def write_color(self, text, color):
        self.write(f"\033[{self.ansi_dict[color]}m{text}\033[0m")

#color_stream = ColorTerminalStream()

def cprint(*values, color=None, sep=" ", end="\n", stream=None, flush=True):
    if stream is None:
        stream = sys.stdout
    for i, value in enumerate(values):
        if i != 0:
            stream.write(sep)
        if hasattr(stream, "write_color") and color is not None:
            stream.write_color(value, color)
        else:
            stream.write(value)
    stream.write(end)
    if flush:
        stream.flush()