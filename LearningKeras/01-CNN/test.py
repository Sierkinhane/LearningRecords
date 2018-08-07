import random
import numpy as np
import sys

a = random.choice((1, 2))
print(a)

# weights initialing
w = np.random.normal(0, 1, (4, 3))*np.sqrt(1/4)