'''
Python script for transition path theory (TPT) analysis of a graph modelled as a continuous-time Markov chain
'''

import numpy as np
import math
from collections import deque
from scipy import optimize
from copy import deepcopy
import erdos_renyi

class Tpt_Markov(object):

    def __init__(self, G, A, B, weights_as_t_mtx=False, lambda_times=None, randseed=17):
        np.random.seed(randseed)
        self.G = G # graph G (representing the state space S). Bidirectional edge weights are
                   # the elements l_ij of the generator matrix
                   # the graph is in adjacency list representation (as a dict of dicts)
        self.A, self.B = A, B
        self.n_nodes = len(self.G)
        print "Graph is:\n", self.G, "\nNumber of nodes:", self.n_nodes, "\n"
        print "Set A is:    ", self.A, "  Set B is:    ", self.B, "\n"
        # check that A and B sets satisfy requisite criteria
        Tpt_Markov.check_sets(self.G, deepcopy(self.A), deepcopy(self.B))
        self.idcs_map = {key: i for (i, key) in enumerate(self.G.iterkeys())} # map indices of nodes on graph onto indices for arrays
        self.nodelist = sorted([node for node in self.G.iterkeys()]) # reverse of self.idcs_map
        self.L = self.construct_generator() # infinitesimal generator matrix. Rows should sum to zero
        self.pi = Tpt_Markov.solve_homog(self.L.T) # (invariant) stationary distribution, satisfies: np.dot(np.tranpose(pi),L) = 0
        self.pi = self.pi / np.sum(self.pi) # normalised stationary distribution
        print "stationary distribution:\n", self.pi
        self.L_rev = self.construct_reverse_generator() # time-reverse infinitesimal generator matrix. Again, rows should sum to zero
        self.q_f = np.zeros(self.n_nodes) # forward committor  (A -> B)
        self.q_b = np.zeros(self.n_nodes) # backward committor (B -> A)
        self.q_f = self.solve_dirichlet(0)
        print "forward committor:\n", self.q_f
        self.q_b = self.solve_dirichlet(1)
        print "backward committor:\n", self.q_b
        print "\nDetailed balance condition:", self.check_detailed_balance()
        self.m = self.calc_reactive_trajectory_pdf()
        self.f = self.calc_probability_current()
        self.f_eff, self.G_f_eff = self.calc_effective_current()
        self.k_AB = self.calc_rate_const()
        print "\nRate constant k_AB is:", self.k_AB
        print self.G_f_eff
        self.bottleneck_nodes = self.find_bottleneck()
        print "\nBottleneck is:", self.bottleneck_nodes

    @staticmethod
    def check_sets(G, A, B):
        # check that A and B sets are not directly connected
        for node in A:
             for conn_node in G[node].iterkeys():
                 assert conn_node not in B, "Error in input: A and B sets are directly connected via nodes %i & %i" \
                                            % (node, conn_node)
        Tpt_Markov.check_set_connected(G, A)
        Tpt_Markov.check_set_connected(G, B)

    # check that nodes within sets A and B are internally connected, respectively
    @staticmethod
    def check_set_connected(G, A):
        bfs_tree = Tpt_Markov.bfs_node(G, A.pop(), A)
        for node in A:
            assert node in bfs_tree.iterkeys(), ("Error in input: set not internally connected, node %i lacks connection."
                                                 " BFS tree is:\n%s" % (node, str(bfs_tree)))

    # breadth-first search to check that members of a set are *directly* connected
    # returns a breadth-first tree as a dict of parents
    @staticmethod
    def bfs_node(G, root, A=None):
        parents = {root: None}
        visit_queue = deque([root])
        while visit_queue:
            curr_node = visit_queue.popleft()
            try:
                for conn_node in G[curr_node]:
                    if A is not None:
                        if conn_node not in A: continue
                    if conn_node in parents: continue
                    parents[conn_node] = curr_node
                    visit_queue.append(conn_node)
            except KeyError:
                continue
        return parents

    # function to solve the discrete Dirichlet problem satisfied by the committors
    def solve_dirichlet(self, direction):
        q = np.zeros(self.n_nodes)
        if direction == 0: # forward committor
            zero_set, one_set = self.A, self.B
            gen_mtx = deepcopy(self.L)
        elif direction == 1: # backward committor
            zero_set, one_set = self.B, self.A
            gen_mtx = deepcopy(self.L_rev)
        # nodes of 'starting' set (A for forward direction, B for backward direction) have committor equal to zero
        q_constraints_zero = [{"type": "eq", "fun": Tpt_Markov.zero_constraint_func(self.idcs_map[key])} \
                               for i, key in enumerate(self.G.iterkeys()) if key in zero_set]
        # nodes of 'finish' set (B for forward direction, A for backward direction) have committor equal to unity
        q_constraints_one = [{"type": "eq", "fun": Tpt_Markov.one_constraint_func(self.idcs_map[key])} \
                              for i, key in enumerate(self.G.iterkeys()) if key in one_set]
        q_constraints = q_constraints_zero + q_constraints_one
        # committor satisfies: 0 <= q_i <= 1 for all i
        q_constraints.append({"type": "ineq", "fun": lambda x: x[:]})
        q_constraints.append({"type": "ineq", "fun": lambda x: -x+1})
        for i in range(self.n_nodes):
            if i not in self.A and i not in self.B:
                # for all nodes i not in sets A or B, dot product of forward committor with corresponding row
                # of generator matrix must equal zero
                q_constraints.append({"type": "eq", "fun": Tpt_Markov.not_ab_constraint_func(self.idcs_map[i], \
                                      gen_mtx)})
        optimisation_func = Tpt_Markov.get_optimisation_func(gen_mtx)
        q_res = optimize.minimize(optimisation_func, x0=np.random.rand(self.n_nodes),
                    method="SLSQP", constraints=q_constraints, tol=1.0E-04, options={"maxiter": 200})
        if q_res.success: return q_res.x
        else: raise RuntimeError("Fatal: could not solve the discrete Dirichlet problem")

    # function for scipy.optimize that constrains element i of committor (solution) array to be zero
    @classmethod
    def zero_constraint_func(cls, i):
        return lambda x: x[i]

    # function for scipy.optimize that constrains element i of committor (solution) array to be unity
    @classmethod
    def one_constraint_func(cls, i):
        return lambda x: x[i] - 1

    # function for scipy.optimize that constrains dot prod of row i of generator matrix with committor array to be zero
    @classmethod
    def not_ab_constraint_func(cls, i, gen_mtx):
        return lambda x: np.dot(gen_mtx[i,:],x)

    # creates instance of class containing function to solve the constrained linear equation to find the committors
    @classmethod
    def get_optimisation_func(cls, gen_mtx):
        constrained_lin_eqn1 = Tpt_Markov.Constrained_Lin_Eqn(gen_mtx)
        return constrained_lin_eqn1.constrained_lin_eqn

    # class instantiated to create a function used to solve the discrete Dirichlet problems to find the committors
    class Constrained_Lin_Eqn():

        def __init__(self, gen_mtx):
            self.gen_mtx = gen_mtx

        # solve linear equation Ax = 0 in constraint space
        def constrained_lin_eqn(self, x):
            y = np.dot(self.gen_mtx,x)
            return np.dot(y,y)

    # check if the Markov jump process is 'reversible' (if so, L and L_rev are equivalent)
    def check_detailed_balance(self):
        for i in range(self.n_nodes):
            for j in range(self.n_nodes):
                if ((self.pi[i]*self.L[i,j]) - (self.pi[j]*self.L[j,i])) > 1.0E-04:
                    return False
        return True

    # solve homogeneous linear equation Ux = 0, used to find the stationary distribution
    @staticmethod
    def solve_homog(U):
        e_vals, e_vecs = np.linalg.eig(np.dot(U.T,U))
        return e_vecs[:,np.argmin(e_vals)]

    # calculate magnitude of vector
    @staticmethod
    def magn(x):
        return math.sqrt(sum(i**2 for i in x))

    # construct the infinitesimal generator matrix L based on bidirected graph G whose edge
    # weights are off-diagonal elements l_ij of L
    def construct_generator(self):
        L = np.zeros((self.n_nodes,self.n_nodes))
        for i, key1 in enumerate(self.G.iterkeys()):
            for key2 in self.G[key1].iterkeys():
                L[i,self.idcs_map[key2]] = self.G[key1][key2]
            L[i,i] = -np.sum(L[i,:])
        return L

    # construct the time-reverse infinitesimal generator matrix L_rev from knowledge of L and the corresponding
    # stationary distribution
    def construct_reverse_generator(self):
        L_rev = np.zeros((self.n_nodes,self.n_nodes))
        for i in range(self.n_nodes):
            for j in range(self.n_nodes):
                L_rev[i,j] = (self.pi[j]/self.pi[i])*self.L[j,i]
        return L_rev

    # calculate the probability distribution function of reactive trajectories
    # m_i is the eq probability to observe a reactive (i.e. reaches B before A) trajectory at state i
    def calc_reactive_trajectory_pdf(self):
        m = [self.pi[i]*self.q_f[i]*self.q_b[i] for i in range(self.n_nodes)]
        return np.array(m,dtype=float)

    # calculate the discrete probability current of reactive trajectories
    # f[i,j] is the avg current of reactive trajectories flowing from state i to state j per unit time
    def calc_probability_current(self):
        f = np.zeros((self.n_nodes,self.n_nodes))
        for i in range(self.n_nodes):
            for j in range(self.n_nodes):
                if i == j: continue
                else: f[i,j] = self.pi[i]*self.q_b[i]*self.L[i,j]*self.q_f[j]
        return f

    # calculate the effective current: f_eff[i,j] is the net avg no. of reactive trajectories per unit time
    # making a transition between states i -> j in the course of the overall process A -> B
    # note that f_eff[i,j] = 0 if states i and j are not connected on the input graph G, or if both states
    # i and j are contained within either set A or B (trajectories must start from A and end in B, cannot
    # have a flux internally within A or within B)
    def calc_effective_current(self):
        f_eff = np.zeros((self.n_nodes,self.n_nodes))
        G_f_eff = {} # directed graph where edge weights are the non-zero elements of f_eff
#        for idx1 in self.G.iterkeys():
#            G_f_eff[idx1] = {idx2: 0.0 for idx2 in self.G[idx1].iterkeys()}
        for i in range(self.n_nodes):
            for j in range(self.n_nodes):
                f_eff[i,j] = max(self.f[i,j]-self.f[j,i],0.0)
                if f_eff[i,j] != 0.0:
                    while True:
                        try:
                            G_f_eff[self.nodelist[i]][self.nodelist[j]] = f_eff[i,j]
                            break
                        except KeyError:
                            G_f_eff[self.nodelist[i]] = {}
                            continue
        return f_eff, G_f_eff

    # calculate the rate const k_AB for the overall A -> B transition
    def calc_rate_const(self):
        k_AB = 0.0
        for i in range(self.n_nodes):
            for j in range(self.n_nodes):
                if i in self.A: k_AB += self.f_eff[i,j]
        return k_AB

    # function to find the pair of nodes that define the dynamical bottleneck of a reaction
    # pathway on the graph G - that is, the edge with minimal effective current f_eff[i,j]
    def find_bottleneck(self):
        sorted_edges = []
        for idx1 in self.G_f_eff.iterkeys():
            for idx2 in self.G_f_eff[idx1].iterkeys():
                sorted_edges.append((self.G_f_eff[idx1][idx2],idx1,idx2))
        sorted_edges = sorted(sorted_edges, key = lambda x: x[0])
        l = 1
        r = len(sorted_edges)
        nit = 0
        print "sorted edges:\n", sorted_edges
        while r - l > 1:
            m = int(math.floor((r-l)/2))
            print "m = ", m
            subset_edges = sorted_edges[m:]
            print "\nsubset of edges:\n", subset_edges
            # check if there is a connected pathway from A to B on the graph that includes
            # only the edges in subset_edges
            subset_graph = {}
            for edge in subset_edges:
                while True:
                    try:
                        subset_graph[edge[1]][edge[2]] = edge[0]
                        break
                    except KeyError:
                        subset_graph[edge[1]] = {}
                        continue
#            print "\nsubset_graph:\n", subset_graph
            parents = Tpt_Markov.bfs_node(subset_graph,next(iter(subset_graph)))
#            print "\nparents:\n", parents
            nodes_in_bfs_tree = [node for node in parents.iterkeys()] + [node for node in parents.iteritems()]
            pathway = False
            for A_node in self.A:
                for B_node in self.B:
                    if A_node in nodes_in_bfs_tree and B_node in nodes_in_bfs_tree:
                        l = m # there exists a pathway connecting A and B on the subset graph
                        pathway = True
                        break
                if pathway: break
            if not pathway: r = m # there does not exist a pathway connecting A and B on the subset graph
            print "r = ", r, "l =", l
            nit += 1
            if nit == 10: break
        return (sorted_edges[l][1],sorted_edges[l][2])

    def find_representative_pathway(self):
        return

if __name__ == "__main__":
    # EXAMPLE 1
    erdos_renyi1 = erdos_renyi.Erdos_Renyi(40,0.10,bidirected=True) # a random graph
    G = {idx: edges.unpack_linked_list() for idx, edges in erdos_renyi1.G.iteritems()}
    A = set([0, 3])
    B = set([20, 38])
    Tpt_markov1 = Tpt_Markov(G, A, B)
