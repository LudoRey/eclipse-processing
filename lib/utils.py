import time
import io
import sys

class Timer:
    def __init__(self, text=None):
        self.text = text 

    def __enter__(self):
        self.start_time = time.perf_counter() 
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.perf_counter() 
        elapsed_time = self.end_time - self.start_time
        text = "Elapsed time" if self.text is None else self.text
        print(f"{text}: {elapsed_time:.2f} seconds")

class ColorTerminalStream(io.TextIOWrapper):
    def __init__(self, **kwargs):
        super().__init__(buffer=sys.__stdout__.buffer, line_buffering=True, **kwargs)
        self.ansi_dict = {
            "default": 0,
            "red": 91,
            "green": 92,
            "yellow": 93,
            "blue": 94,
            "magenta": 95,
            "cyan": 96,
            "white": 97
        }

    def write_color(self, text, color):
        self.write(f"\033[{self.ansi_dict[color]}m{text}\033[0m")

def cprint(*values, color=None, sep=" ", end="\n", stream=None, flush=False):
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