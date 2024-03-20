#!/usr/bin/python

from importlib.metadata import version
import code

try:
    import readline
except ImportError:
    pass
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from polylx import *

try:
    ver = version("polylx")
except ImportError:
    ver = None


def main():
    banner = "+----------------------------------------------------------+\n"
    banner += "    PolyLX toolbox"
    if ver is not None:
        banner += f" {ver}"
    banner += " - https://polylx.readthedocs.io\n"
    banner += "+----------------------------------------------------------+"
    vars = globals().copy()
    vars.update(locals())
    shell = code.InteractiveConsole(vars)
    shell.interact(banner=banner)


if __name__ == "__main__":
    main()
