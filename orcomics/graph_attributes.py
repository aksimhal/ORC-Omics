"""
    Functions for computing graph attributes
"""

from functools import lru_cache
import networkx as nx
import numpy as np

from orcomics.util import logger

EPS = 1e-7
cache_maxsize = 1000000


@lru_cache(cache_maxsize)
def pij(G, source, target, n_weight="weight", EPS=1e-7):
    """ Compute the 1-step Markov transition probability of going from source to target node in G
    Note: not a lazy walk (i.e. alpha=0)"""

    if G.nodes[source][n_weight] <= EPS:
        logger.warning(f"Node {source} with weight < EPS does not interact with any other nodes - setting pij = 0.")
        return 0.0
    if target not in G.neighbors(source):
        return 0.0
    if G.nodes[target][n_weight] <= EPS:
        logger.warning(f"Node {source} does not interact with {target} with weight < EPS - setting pij = 0.")
        return 0.0
    w = [G.nodes[nbr][n_weight] for nbr in G.neighbors(source)]    
    if sum(w) > EPS:  # ensure no dividing by zero
        return G.nodes[target][n_weight]/sum(w)
    else:  # ensure no dividing by zero
        raise ValueError(f"Net weight for neighbors of {source} is too small to compute interaction probability with {target}.")


def compute_edge_weights(G: nx.Graph, n_weight="weight", e_weight="weight", e_normalized=False, e_sqrt=False, e_wprob=False):
    """ compute edge weights from given nodal weights 
    e_normalized = True AND e_sqrt = True : w_ij = 1/sqrt([(p_ij + p_ji)/2])
    e_normalized = True AND e_sqrt = False : w_ij = 1/[(p_ij + p_ji)/2]
    e_normalized = False AND e_sqrt = True : w_ij = 1/sqrt(w_i * w_j)
    e_normalized = False AND e_sqrt = False : w_ij = (1/w_i)*(1/w_j)    
    NOTE: w_ij = INF if w_i=0 or w_j=0
    e_wprob = True (then e_normalizezd and e_sqrt are ignored) : w_ij = 1 / (p_ij + p_ji - (p_ij * p_ji))
    """
    assert ~(not nx.get_node_attributes(G, n_weight)), "Node weight not detected in graph."
    
    # compute edge weight
    weights = {}
    if e_normalized or e_wprob:  # normalized
        for i, j in G.edges():
            wij = pij(G, i, j, n_weight)
            wji = pij(G, j, i, n_weight)
            if e_wprob:
                w = wij + wji - (wij * wji)
            else: # normalized
                w = (wij+wji)/2  # d(i,j) = 1/sqrt(w_ij)
                if e_sqrt:  # d(i,j) = 1/sqrt(w_ij)
                    w = np.sqrt(w)
            weights[(i, j)] = 1/w if min([w, wij, wji]) > EPS else np.inf
    else:  # not normalized
        for i, j in G.edges():
            w = G.nodes[i][n_weight]*G.nodes[j][n_weight]  # w = (1/w_i)*(1/w_j)
            if e_sqrt:  # w_ij = 1/sqrt(w_i * w_j)"
                w = np.sqrt(w)
            weights[(i, j)] = 1/w if w > EPS else np.inf
    nx.set_edge_attributes(G, weights, name=e_weight)
    return G
    
