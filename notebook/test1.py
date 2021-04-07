from memory_profiler import profile
import numpy as np
from K_DeZero.test import Variable, add, square, Config

@profile
def hoge():
    x = Variable(np.random.randn(10000))
    y = square(square(square(x)))
    for i in range(10):
        x = Variable(np.random.randn(10000))
        y = square(square(square(x)))

    Config.enable_backprop = True
    x = Variable(np.ones((100, 100, 100)))
    y = square(square(square(x)))
    y.backward()

    Config.enable_backprop = False
    x = Variable(np.ones((100, 100, 100)))
    y = square(square(square(x)))

hoge()