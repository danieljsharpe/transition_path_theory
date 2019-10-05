'''Python script containing some test examples for the TPT code'''

from tpt import Tpt_Markov
import erdos_renyi
import numpy as np
from math import floor
from copy import deepcopy
import matplotlib.pyplot as plt

''' For a monodirected graph (in the form of node lists from lower to higher indices)
    return the corresponding explicit bidirected graph '''
def get_bidir_graph(G):
    G_bd = deepcopy(G)
    for v in G.iterkeys():
        for u in G[v].iterkeys():
            G_bd[u][v] = G[v][u]
    return G_bd


def tpt_examples():

    np.random.seed(19)
    # EXAMPLE 0 - A VERY SIMPLE EXAMPLE TO TEST IMPLEMENTATION (COMPARISON WITH PyEMMA)
    G0 = {0: {1: 1.5E-1, 2: 5.0E-2}, 1: {0: 1.0E-01, 2: 5.0E-02, 3: 5.0E-02, 4: 5.0E-02},
          2: {0: 5.0E-2, 1: 1.0E-1, 4: 5.0E-2}, 3: {1: 2.E-01}, 4: {1: 2.E-02, 2: 2.E-02} }
    A = set([0])
    B = set([4])
    tpt_markov0 = Tpt_Markov(G0, A, B, weights_as_t_mtx=True)

    '''
    # EXAMPLE 1 - ERDOS-RENYI RANDOM GRAPH
    erdos_renyi1 = erdos_renyi.Erdos_Renyi(40,0.10,bidirected=True) # a random graph
    G = {idx: edges.unpack_linked_list() for idx, edges in erdos_renyi1.G.iteritems()}
    A = set([0, 3])
    B = set([20, 38])
    tpt_markov1 = Tpt_Markov(G, A, B)
    '''

    '''
    # EXAMPLE 2 - "STRUCTURED" NETWORK
    # maybe use this as an example of holding times?
    G2 = {1: {2: 11, 3: 4, 7: 6, 4: 2, 5: 5}, 2: {6: 6, 3: 3}, 3: {6: 4, 7: 2, 4: 1}, \
         4: {5: 1, 7: 3, 10: 5, 8: 5, 9: 6}, 5: {9: 4}, 6: {7: 1, 12: 5, 10: 3}, \
         7: {10: 3}, 8: {12: 7, 13: 5, 11: 5, 9: 3}, 9: {11: 5, 14: 5}, \
         10: {12: 2, 13: 4, 11: 6}, 11: {13: 4, 16: 1, 14: 2}, 12: {13: 4, 15: 5}, \
         13: {15: 2, 16: 3, 17: 4}, 14: {16: 2, 17: 8}, 15: {16: 4, 17: 6}, \
         16: {17: 2}, 17: {}}
    G3 = {}
    np.random.seed(17)
    for v in G2.iterkeys():
        for u in G2[v].iterkeys():
            try:
                G3[v-1][u-1] = np.random.rand()
            except KeyError:
                G3[v-1] = {}
                G3[v-1][u-1] = np.random.rand()
            try:
                G3[u-1][v-1] = np.random.rand()
            except KeyError:
                G3[u-1] = {}
                G3[u-1][v-1] = np.random.rand()
    A = set([0])
    B = set([16])
    tpt_markov2 = Tpt_Markov(G3, A, B)
    '''

    '''
    # EXAMPLE 3 - GRID
    T = 1. # reduced temperature
    grid_size = 15
    high_en_prob = 0.9 # probability of a node being given a high energy
    # cardinal_nbrs = [[-1,0],[+1,0],[0,-1],[0,+1]]
    L = np.zeros([grid_size*grid_size]*2)
    E = np.zeros((grid_size,grid_size)) # energy
    a = np.zeros((grid_size,grid_size)) # no. of neighbours
    p = np.zeros((grid_size,grid_size)) # occupation probabilities
    n_nodes = grid_size*grid_size
    G_grid = {i: {} for i in range(n_nodes)}
    # give random energies (could equally well model a 2D PES)
    for i in range(grid_size):
        for j in range(grid_size):
            if np.random.rand() < high_en_prob: # node is given high energy
                E[i,j] = 5. + (5.*np.random.rand())
            else: # node is given low energy, energy more likely to be lower for higher j
                try: E[i,j] = (1./float(j))**np.random.rand()
                except ZeroDivisionError: E[i,j] = 1.**np.random.rand()
    # note that we have non-periodicity (i.e. we are not pacman-ing!)
    for i in range(grid_size):
        for j in range(grid_size):
            node_idx = (grid_size*i) + j # index of node corresponding to indices i, j
            a_ij = 0
            # disclude 0-19 (first/bottom row)
            if not node_idx < grid_size: # neighbour below exists
                a_ij += 1
                G_grid[node_idx-grid_size][node_idx] = 1.
            # disclude (20*20)-1, (20*20)-2, ... (20*20)-21 (last/top row)
            if not node_idx > n_nodes - (grid_size+1): # neighbour above exists
                a_ij += 1
            # disclude 19, 38...
            if not (node_idx+1) % grid_size == 0: # neighbour to right exists
                a_ij += 1
            # disclude 0, 20...
            if not (node_idx) % grid_size == 0 : # neighbour to left exists
                a_ij += 1
                G_grid[node_idx-1][node_idx] = 1.
            a[i,j] = a_ij
        for j in range(grid_size): p[i,j] = a[i,j] / np.sum(a[i,:])
    print "a is:\n", a
    for i in range(n_nodes):
        i_k, i_l = i%grid_size, int(floor(i/grid_size)) # row / col indices for node i
        for j in range(n_nodes):
            j_k, j_l = j%grid_size, int(floor(j/grid_size)) # row / col indices for node j
            # Note that e.g. occ prob of node i is: p_i = p[i_k,i_l] (same for E & a matrices)
            if i != j: L[i,j] = p[i_k,i_l]*min((np.exp(-E[j_k,j_l]/T)*p[j_k,j_l]) \
                                             / (np.exp(-E[i_k,i_l]/T)*p[i_k,i_l]),1.) # ?
        L[i,i] = -(np.sum(L[i,:]))# - L[i,j]) # needs to be outside of loop?
    #print "L:\n", L
    #for i in range(n_nodes):
    #    print np.sum(L[i,:]) # rows should sum to zero
    A = set([i for i in range(grid_size)]) # bottom row is set A
    B = set([i for i in range(n_nodes-1,n_nodes-1-grid_size,-1)]) # top row is set B
    print "set A:", A, "set B:", B
    #print "Graph is:\n", G_grid
    G_grid_bd = get_bidir_graph(G_grid)
    tpt_markov3 = Tpt_Markov(G_grid_bd,A,B,L=L)
    # reshape the committor list, the pdf of reactive trajectories and the probability current list and plot them
    q_f = tpt_markov3.q_f.reshape((grid_size,grid_size))
    m = tpt_markov3.m.reshape((grid_size,grid_size))
    print np.shape(tpt_markov3.q_f), np.shape(tpt_markov3.m), np.shape(tpt_markov3.f)
    plt1 = plt.imshow(np.log(q_f),interpolation="nearest",cmap="cool")
    plt.colorbar(plt1)
    plt.show()
    plt2 = plt.imshow(m,interpolation="nearest",cmap="viridis")
    plt.colorbar(plt2)
    plt.show()
    plt3 = plt.imshow(tpt_markov3.f,interpolation="nearest",cmap="viridis")
    plt.colorbar(plt3)
    plt.show()
    '''

    '''
    a_ij = a+ji: = 1 (if i & j are carindal neighbours), = 0 (otherwise)
    p_ij = a_ij / ( sum_j a_ij )
    L_ij = p_ij * min ( (exp(-E_j/T)*p_ji) / (exp(-E_i/T)*p_ij) , 1  )    ( i =/= j )
    L_ii = 1 - ( sum_j=/=i L_ij )
    '''
    # EXAMPLE 4 - TOY MODEL FOR STRUCTURAL ELEMENTS OF PROTEIN


    # EXAMPLE 5 - COARSE-GRAINING A MODEL FREE ENERGY SURFACE
