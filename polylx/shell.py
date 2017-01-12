#!/usr/bin/python

import pkg_resources
import readline
import code
from pylab import *
from polylx import *

def main():
    banner = '+----------------------------------------------------------+\n'
    banner += '    PolyLX toolbox '
    banner += pkg_resources.require('polylx')[0].version
    banner += ' - https://polylx.readthedocs.io\n'
    banner += '+----------------------------------------------------------+'
    vars = globals().copy()
    vars.update(locals())
    shell = code.InteractiveConsole(vars)
    shell.interact(banner=banner)

if __name__ == "__main__":
    main()
