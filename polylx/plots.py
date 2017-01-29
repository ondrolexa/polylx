import numpy as np
import matplotlib.pyplot as plt


def surfor_plot(ob):
    res = [ob[name].surfor().mean(axis=0) for name in ob.names]
    plt.plot(np.transpose(res))
    plt.legend(ob.names)
    plt.show()
