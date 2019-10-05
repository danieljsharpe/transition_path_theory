import pyemma.msm as msm
import numpy as np

# TOY MODEL CONSISTING OF 5 NODES

# transition matrix
P = np.array([[0.8,  0.15, 0.05,  0.0,  0.0],
              [0.1,  0.75, 0.05, 0.05, 0.05],
              [0.05,  0.1,  0.8,  0.0,  0.05],
              [0.0,  0.2, 0.0,  0.8,  0.0],
              [0.0,  0.02, 0.02, 0.0,  0.96]])
M = msm.markov_model(P)
A = [0]
B = [4]
I = [1,2,3]
tpt = msm.tpt(M,A,B)

print("Stationary distribution:")
print(tpt.mu)
# print("Rate matrix:")
# print(tpt.M)
print("Forward committor:")
print(tpt.committor)
print("Backward committor:")
print(tpt.backward_committor)
print("Gross flux:")
print(tpt.gross_flux)
print("Net flux:")
print(tpt.net_flux)
print("Total flux:")
print(tpt.flux[A,:][:,B].sum() + tpt.flux[A,:][:,I].sum())
print("Rate constant:")
print(tpt.rate)
print("MFPT:")
print(tpt.mfpt)

'''
#compare to rate matrix
L = np.array([[
M2 = msm.markov_model(L)
tpt2 = msm.tpt(M2,A,B,rate_matrix=True)
'''
