'''
Python script to demonstrate linear algebra with constraints
'''

import numpy as np
from scipy import optimize

np.random.seed(19)

def func_tominimise(x):
    y = np.dot(A,x) - b
    return np.dot(y,y)

#dict of constraints - elems of x must sum to one
cons = ({"type": "eq", "fun": lambda x: x.sum() - 1})

A = np.random.rand(3,3)
b = np.random.rand(3)
res = optimize.minimize(func_tominimise, x0=[0.,0.,0.], method="SLSQP", \
                        constraints=cons)

print res
print "result:", res.x, "sum:", res.x.sum()

# sequence of dicts for constraints - first element of x must be +1
# AND second elem must be >= 0
cons2 = ({"type": "eq", "fun": lambda x: x[0] - 1}, \
         {"type": "ineq", "fun": lambda x: x[1]})
         
res2 = optimize.minimize(func_tominimise, x0=[0.,0.,0.], method="SLSQP", \
                         constraints=cons2)

print res2
print "result:", res2.x

