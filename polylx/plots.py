import numpy as np
import matplotlib.pyplot as plt


def surfor_plot(ob, averaged=True):
    if averaged:
        res = [ob[name].surfor().mean(axis=0) for name in ob.names]
    else:
        res = [ob[name].surfor().sum(axis=0) for name in ob.names]
    plt.plot(np.transpose(res))
    plt.legend(ob.names)
    plt.show()

def paror_plot(ob, averaged=True):
    if averaged:
        res = [ob[name].paror().mean(axis=0) for name in ob.names]
    else:
        res = [ob[name].paror().sum(axis=0) for name in ob.names]
    plt.plot(np.transpose(res))
    plt.legend(ob.names)
    plt.show()