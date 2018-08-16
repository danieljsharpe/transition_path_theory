'''
Python class for generating a random graph according to the Erdos-Renyi model
'''

import numpy as np

# linked list
class Node:
    def __init__(self, node, cost=None):
        self.node = node
        self.cost = cost
        self.next_node = None

    def get_node(self):
        return self.node

    def get_next_node(self):
        return self.next_node

    def set_next_node(self, new_next_node):
        self.next_node = new_next_node

#    def __repr__(self):
#        return "Index: " + repr(self.node) + " cost: " + repr(self.cost)

class LinkedList(object):
    def __init__(self, head=None, tail=None):
        self.head = head
        self.tail = tail

    def push_node(self, node, cost):
        new_node = Node(node, cost)
        new_node.set_next_node(self.head)
        self.head = new_node

    def pop_node(self):
        return

    # unpack linked list into a dictionary
    def unpack_linked_list(self):
        linked_list = {}
        node = self.head
        while node != None:
            linked_list[node.node] = node.cost
            node = node.next_node
        return linked_list

class Erdos_Renyi(object):

    def __init__(self, n_nodes, p=0.1, weight_distrib="uniform", max_weight=10.0, mean_weight=None, \
                 stddev_weight=None, loop=False, bidirected=False, prune=True, seed=1):
        self.n_nodes = n_nodes # (target) number of nodes in the graph
        self.p = p # probability of adding an edge between a pair of nodes
        self.max_weight = max_weight
        self.loop = loop # allow internal (self)-loops Y/N
        self.bidirected = bidirected # different edge weights for reverse directions Y/N
        np.random.seed(seed)
        if weight_distrib == "uniform":
            self.random_cost = np.random.uniform
            self.random_cost_arg1 = 0.0
            self.random_cost_arg2 = max_weight
        elif weight_distrib == "gaussian":
            self.random_cost = np.random.normal
            self.random_cost_arg1 = mean_weight
            self.random_cost_arg2 = stddev_weight
        self.G = Erdos_Renyi.initialise_graph(self.n_nodes)
        self.add_edges()
        if prune: # retain only nodes that are part of the largest tree in the forest
            forest = self.find_trees()
            main_tree = max(forest, key=len)  # largest tree in forest
            del_keys = [key for key in self.G.iterkeys() if key not in main_tree]
            for del_key in del_keys: self.G.pop(del_key)

    @staticmethod
    def initialise_graph(n_nodes):
        G = {i: LinkedList() for i in range(n_nodes)}
        return G

    def add_edges(self):
        for i in range(len(self.G)):
            for j in range(i,len(self.G)):
                if not self.loop and i == j: continue
                if np.random.random() < self.p:
                    edge_cost = self.random_cost(self.random_cost_arg1,self.random_cost_arg2)
                    self.G[i].push_node(j, edge_cost)
                    if not self.bidirected:
                        self.G[j].push_node(i, edge_cost)
                    else:
                        edge_cost_rev = self.random_cost(self.random_cost_arg1,self.random_cost_arg2)
                        self.G[j].push_node(i, edge_cost_rev)
        return

    # discover all neighbours of a single node
    def dfs_node(self, node, visit_queue, not_visited):
        nbr_node = self.G[node].head
        try:
            nbr_node_idx = nbr_node.node
        except AttributeError:
            return None
        while True:
            if nbr_node_idx in not_visited and nbr_node_idx not in visit_queue:
                visit_queue.append(nbr_node_idx)
            if nbr_node.next_node is None: break
            nbr_node = nbr_node.next_node
            nbr_node_idx = nbr_node.node
        return visit_queue

    # find trees (connected components) using depth-first search
    def find_trees(self):
        trees = []
        not_visited = set([i for i in range(len(self.G))])
        while bool(not_visited):
            tree = []
            visit_queue = []
            root = np.random.choice(tuple(not_visited)) # choose start node from nodes not yet visited
            visit_queue.append(root)
            while visit_queue:
                curr_node = visit_queue.pop()
                tree.append(curr_node)
                not_visited.remove(curr_node)
                visit_queue = self.dfs_node(curr_node, visit_queue, not_visited)
            trees.append(tree)
        return trees

# Example usage
if __name__ == "__main__":
    print "\nExample usage of Linked_List and Node classes:\n"
    G = []
    G.append(LinkedList())
    G[0].push_node(1,3)
    print G[0].head, G[0].tail
    print G[0].head.node, G[0].head.cost, G[0].head.next_node
    G[0].push_node(8,4)
    print G[0].head.node, G[0].head.cost, G[0].head.next_node, G[0].head.next_node.node
    print G[0].unpack_linked_list()
    print "\nExample usage of Erdos_Renyi class:\n"
    erdos_renyi1 = Erdos_Renyi(10,0.15,bidirected=True)
    G_as_dicts = {idx: edges.unpack_linked_list() for idx, edges in erdos_renyi1.G.iteritems()}
    print G_as_dicts
