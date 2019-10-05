import pyemma.msm as msm
import numpy as np
from scipy.linalg import expm

# Example generator (transition rate) matrix
K = np.array([[-2.90164106e-02,2.47438983e-02,1.88788911e-03,4.51348128e-05,2.33948836e-03],
              [8.75487503e-02,-1.06954102e-01,0.00000000e+00,1.94053514e-02,0.00000000e+00],
              [9.75896673e-03,0.00000000e+00,-3.56068076e-02,1.53539843e-02,1.04938566e-02],
              [4.98873182e-05,6.06203157e-03,3.28301032e-03,-9.39492921e-03,0.00000000e+00],
              [1.35496766e-02,0.00000000e+00,1.17575223e-02,0.00000000e+00,-2.53071990e-02]])

tau = 1.E-2
T = expm(K*tau) # NB TPT results should be invariant with tau

my_msm = msm.markov_model(T)
A = [0]
B = [4]
my_msm_tpt = msm.tpt(my_msm,A,B)

print("Stationary distribution:")
print(my_msm_tpt.mu)

print("Forward committor:")
print(my_msm_tpt.committor)
print("Backward committor:")
print(my_msm_tpt.backward_committor)
