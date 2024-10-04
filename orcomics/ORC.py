# ORC.py

"""
Description
-----------
Code by Anish Simhal & Rena Elkin

A class to compute Ollivier-Ricci curvature (ORC)
based off of nodal weights in a given Networkx graph.


Credit
------
Code based off of GraphRicciCurvature Python package written by Chien-Chun Ni.
        Original code can be found at https://github.com/saibalmars/GraphRicciCurvature.


"""
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import cvxpy as cvx
import heapq
import itertools
from functools import lru_cache
import math
import multiprocessing as mp
import networkx as nx
import numpy as np
import ot
import time

import pandas as pd
import scipy.sparse.csgraph as scg

from orcomics.util import logger, set_verbose
from orcomics.graph_attributes import pij, compute_edge_weights

EPS = 1e-7

###############################
# ~~ Shared global variables ~~
###############################

_G = None
_alpha = 0.0
_pdistr = "mass_action"
_gdist = "hop"
_nbr_maxk = 500
_C = None
_n_weight = "weight"
_e_weight = "weight"
_e_normalized = False
_e_wdeg = False
_e_wpro = False
_e_sqrt = False
_proc = mp.cpu_count()
_cache_maxsize = 1000000
_EPS = EPS
_edge_list = None
_power = 1
_contraction = 'ED'

######################
# ~~ Local methods ~~
######################


def _compute_alt_edge_weights(G: nx.Graph, n_weight="weight", e_weight="weight", c_weight=None):
    """ compute edge weights from given nodal weights as function of degree

    - weighted degree: deg(x) = c_weight(x)*sum_{z~x}n_weight(z)                                                                   
    - e_weight(u,v) := max{EPS,min{1/sqrt(deg(u)), 1/sqrt(deg(v)), 1}}                                                             
                                                                                                                                   
    Parameters                                                                                                                     
    ----------                                                                                                                     
    G : NetworkX graph                                                                                                             
        Given graph                                                                                                                
    n_weight : str                                                                                                                 
        Node weight used to compute weighted degree and resulting edge weight (Default = "weight").                                
    e_weight : str                                                                                                                 
        Name of edge attribute in G to assign resulting edge weight (Default = "weight").                                          
    c_weight : str                                                                                                                 
        Name of node attribute in G to weight the contraction for the weighted degree.                                             
        If not provided, a default value of 1 (i.e., no weight) is used.                                                           

    """
    weighted_degrees = {x: sum([G.nodes[z][n_weight] for z in G.neighbors(x)]) for x in G.nodes()}
    if c_weight:
        weighted_degrees = {x: G.nodes[x][c_weight] * weighted_degrees[x] for x in weighted_degrees}
    assert EPS < min(weighted_degrees.values()) <= max(
        weighted_degrees.values()) < 1 / EPS, "Weighted degree outside of feasible range detected."
    nx.set_edge_weight(G, {(i, j): max(EPS, min(1 / np.sqrt(weighted_degrees[i]),
                                                1 / np.sqrt(weighted_degrees[j]), 1)) for i, j in G.edges()},
                       name=e_weight)


def _get_graph_distance(G: nx.Graph, gdist='hop', e_weight='weight'):
    """ compute scg of graph distances between every two nodes """
    assert gdist in ['hop', 'whop'], "Unrecognized gdist. Options = ['hop','whop']."
    weight = None if gdist == 'hop' else e_weight
    unweighted = True if gdist == 'hop' else False
    logger.info(f"Computing {gdist} distances...")
    nodelist = list(G.nodes())
    t0 = time.time()
    
    adj = nx.to_numpy_array(G, nodelist=nodelist, weight=e_weight) 
    if gdist == 'hop': 
        adj = adj > 0 
        adj = adj * 1

    C = scg.dijkstra(adj, directed=True, unweighted=unweighted)
    # C = scg.dijkstra(nx.adjacency_matrix(G, weight=weight, nodelist=nodelist),
    #                  directed=True, unweighted=unweighted)
    logger.info("{:.8f} secs computing Graph {} distances.".format(time.time() - t0, gdist))
    return pd.DataFrame(data=C, index=nodelist, columns=nodelist)
        
@lru_cache(_cache_maxsize)
def _node_distribution_1step(G, node, alpha=0.0, n_weight="weight", e_weight="weight",
                             pdistr="mass_action", power=1, EPS=1e-7):
    """ Compute probability distribution of a one step Markov walk from given node.
    
    Note: returns uniform distribution when neighboring weights are too small to normalize.
    Note: if e_sqrt is true, and want to use 'edge_based' or 'edge_simple' then 'power' should be set to 2 
    """
    logger.debug(
        f"1-step node distribution selected options: pdistr = {pdistr}, alpha={alpha}, power = {power}, n-weight = {n_weight}, e_weight = {e_weight}")
    assert pdistr in ['mass_action', 'edge_based', 'combo',
                      'edge_simple'], "Unrecognized pdistr, options = ['mass_action', 'edge_based', 'combo', 'edge_simple']."
    if type(alpha) is str:
        assert alpha in ['ED', 'EDnot', 'weight'], "Unrecognized alpha, options = ['ED','EDnot','weight', float]."
        if alpha == 'weight':
            assert nx.get_node_attributes(G, n_weight), "Node weight not detected for node-specific alpha."
            alpha_weight = sum(G.nodes[ww][n_weight] for ww in G.nodes())
            al = G.nodes[node][n_weight] / alpha_weight
        else:  # alpha == 'ED','EDnot'
            assert nx.get_node_attributes(G, 'ED'), "ED not detected for node-specific alpha."
            if alpha == 'ED':
                al = G.nodes[node]['ED']
            else:  # 'EDnot'
                al = 1 - G.nodes[node]['ED']
    else:  # float
        al = alpha
    nbrs = list(G.neighbors(node))
    paired_weight_node = []
    for nbr in nbrs:
        if pdistr == "mass_action":
            w = G.nodes[nbr][n_weight]
        elif pdistr == 'edge_simple':
            w = 1 / (G.edges[(node, nbr)][e_weight] ** power)
        elif pdistr == 'edge_based':
            w = math.e ** (-G.edges[(node, nbr)][e_weight] ** power)
        else:  # 'combo'
            w = G.nodes[nbr][n_weight] * (math.e ** (-G.edges[(node, nbr)][e_weight] ** power))
        heapq.heappush(paired_weight_node, (w, nbr))
    weighted_nodal_degree = sum([x[0] for x in paired_weight_node])
    if len(nbrs) == 0:  # i.e. no neighbors so all mass stays at node
        return [1.0], [node]
    elif weighted_nodal_degree > EPS:  # ensure no dividing by zero
        dist = [(1.0 - al) * w / weighted_nodal_degree for w, _ in paired_weight_node]
    else:  # uniform distribution when neighboring weights are too small to normalize
        logger.warning("{}: using uniform distribution because weighted nodal degree too small:".format(node),
                       paired_weight_node)
        dist = [(1.0 - al) / len(nbrs)] * len(nbrs)
        # return None, None
    nbr = [x[1] for x in paired_weight_node]
    return dist + [al], nbr + [node]


def _local_dist_array(A, N0, N1):
    """ Given dataframe of shortest distances "A,"  
        return the corresponding distance matrix between nodes in "N0" and nodes in "N1".
        The node names in N0 and N1 should be keys in A. 
    """
    D = np.ascontiguousarray(A.loc[N0, N1].values, dtype=np.float64)
    return D


def _optimal_transportation_distance(x, y, d, solvr=None):
    """ Compute the optimal transportation distance (OTD) of the given density distributions by CVXPY. 
    Parameters
    ----------
    x : (m,) numpy.ndarray  
        Source's density distributions, includes source and source's neighbors.
    y : (n,) numpy.ndarray
        Target's density distributions, includes source and source's neighbors.
    d : (m, n) numpy.ndarray
        Shortest path matrix. 

    Returns
    -------
    m : float
        Optimal transportation distance. 
    """
    rho = cvx.Variable((len(y), len(x)))  # the transportation plan rho
    # objective function d(x,y) * rho * x, need to do element-wise multiply here
    obj = cvx.Minimize(cvx.sum(cvx.multiply(np.multiply(d.T, x.T), rho)))
    # \sigma_i rho_{ij}=[1,1,...,1]
    source_sum = cvx.sum(rho, axis=0, keepdims=True)
    # constrains = [rho * x == y, source_sum == np.ones((1, (len(x)))), 0 <= rho, rho <= 1]
    constrains = [rho @ x == y, source_sum == np.ones((1, (len(x)))), 0 <= rho, rho <= 1]
    prob = cvx.Problem(obj, constrains)
    if solvr is None:
        m = prob.solve()
    else:
        m = prob.solve(solver=solvr)
        # m = prob.solve(solver='ECOS', abstol=1e-6,verbose=True)
        # m = prob.solve(solver='OSQP', max_iter=100000,verbose=False)
    return m


def _compute_wasserstein_edge(source, target):
    """ compute Wasserstein distance for given (possible fictitious) edge.

    Parameters
    ----------
    source : str
        Reference name of source node in Networkx graph _G
    target : str
        Reference name of target node in Networkx graph _G

    Returns:
    ________
    output : dict[(str,str), float]
        The Wasserstein distance of given edge in dict format (e.g. {(source,target): Wasserstein})
    """
    if source == target:
        return {(source, target): 0.}

    # if the weight of either node is too small, return np.nan instead
    if (_G.nodes[source][_n_weight] < _EPS) or (_G.nodes[target][_n_weight] < _EPS):
        logger.warning(
            "Zero weight node detected for edge ({:s},{:s}), return Ricci Curvature as np.nan instead.".format(source,
                                                                                                               target))
        return {(source, target): np.nan}

    assert _pdistr in ['mass_action', 'edge_based', 'combo',
                       'edge_simple'], "Unrecognized pdistr, options = ['mass_action', 'edge_based', 'combo', 'edge_simple']."

    if type(_alpha) is str:
        assert _alpha in ['ED', 'EDnot', 'weight'], "Unrecognized alpha, options = ['ED','EDnot','weight', float]."
    
    p0, nbrs0 = _node_distribution_1step(_G, source, alpha=_alpha, n_weight=_n_weight,
                                         e_weight=_e_weight, pdistr=_pdistr, power=_power, EPS=_EPS)
    p1, nbrs1 = _node_distribution_1step(_G, target, alpha=_alpha, n_weight=_n_weight,
                                         e_weight=_e_weight, pdistr=_pdistr, power=_power, EPS=_EPS)
    

    D = _local_dist_array(_C, nbrs0, nbrs1)
    x = np.asarray(p0, dtype=np.float64)
    x /= x.sum()
    y = np.asarray(p1, dtype=np.float64)
    y /= y.sum()
    try:
        omtd, lg = ot.emd2(x.astype(np.float64), y.astype(np.float64), D.astype(np.float64), log=True,
                           numItermax=10000000) 
        if lg['warning'] is not None:
            logger.warning(
                'POT library failed: warning = {}, retry with explicit computation OTD(p_{},p_{})'.format(
                    lg['warning'],
                    source, target))
            del (omtd)
            omtd = _optimal_transportation_distance(x, y, D)
            logger.trace("OTD(p-{},p-{}) = {:.5e}".format(source, target, omtd))
    except cvx.error.SolverError:
        logger.warning("OTD(p-{},p-{}) failed, retry with SCS solver".format(source, target))
        omtd = _optimal_transportation_distance(x, y, D, solvr='SCS')
        logger.trace("OTD-SCS(p-{},p-{}) = {:.5e}".format(source, target, omtd))
    return {(source, target): omtd}


def _wrap_compute_wasserstein_edge(stuff):
    """Wrapper for args in multiprocessing."""
    return _compute_wasserstein_edge(*stuff)


def _compute_wasserstein_edges(G, n_weight="weight", e_weight="weight",
                               e_normalized=False, e_sqrt=False, e_wdeg=False, e_wprob=False,
                               pdistr="mass_action", gdist="hop", alpha=0.0, 
                               power=1, edge_list=None, C=None, 
                               EPS=1e-7, nbr_maxk=500,
                               proc=mp.cpu_count(), chunksize=None, cache_maxsize=1000000, return_C=False):
    """ Compute Wasserstein distance on all edges in edge_list

    Parameters
    ----------
    G : NetworkX graph
            A given (undirected, connected) NetworkX graph.
    n_weight : str
        Name of nodal feature used for constructing distance/distributions (Default = "weight")
        Note: if attribute "n_weight" does not exist, default value of 1 is assigned to all nodes
    e_weight : str
        (Optional) Name of edge feature used to compute distance when gdist='whop' (Default = "weight")
        Note: When gdist='hop', edge weights are all treated as 1.
        If attribute "e_weight" does not exist and gdist='whop', edge weight is computed as w_ij = (1/w_i)*(1/w_j)
        where w_i is the node weight specified by 'n_weight'.
    e_normalized : logical
        Indicate if normalized edge weight should be used when gdist="whop" (Default = False).
        Note: Ignored if e_wdeg is True.
    e_sqrt : logical
        Indicate if sqrt should be taken of inferred edge weights (Default = False).
        Note: Ignored if e_wdeg is True .
    e_wdeg : logical
        Indicate if edge weight should be inferred from weighted degree contracted with the invariant measure.
        (Default = False)
    e_wprob : logical
        Indicate if edge weight should be inferred from probability of transition occuring in at least one direction.
        (Default = False)
    pdistr : {'mass_action', 'edge_based', 'combo', 'edge_simple'}
        Specify if nodal probability distribution should be derived from the mass action principle of edge based.
        (Default = 'mass_action').

        options

        - 'mass_action' : node-based on mass action principle --> m_ij := w_j (Default).
        - 'edge_based' : edge-based on distance --> m_ij := exp(-d(i,j)^power)
        - 'combo' : combine node and edge attributes --> m_ij := w_j * exp(-d(i,j)^power)
        - 'edge_simple' : simple edge-based on distance --> m_ij := 1/(d(i,j)^power)
    gdist : {'hop','whop'}
        Graph distance (Default = 'hop')

        options:

        - 'hop' : hop-distance (Default)
        - 'whop' : weighted hop-distance
    alpha : {float: (0.0 <= alpha <= 1.0), str: ['ED','EDnot','weight']}
        Idle probability (Default = 0.0)

        options:

        - float : alpha is a constant in the range [0,1]
        - 'ED' : alpha is node-dependent prescribed by the ED (alpha_i = pi_i)
        - 'EDnot' : alpha is node-dependent prescribed by the probabilistic negation of the ED (alpha_i = 1 - pi_i)
        - 'weight' : alpha is node-dependent prescribed by the given node weight 'n_weight'
    power : float
        Edge weight power for edge-based distribution. (Default value = 1)
    edge_list : {list, None, "all"}
        Specify particular set of edges to compute curvature on. (Default = None).
        If edge_list is None, curvature is computed on all edges in the given graph.
        If edge_list is "all," curvature is computed between every two nodes in the given graph.

        Note: Edges may specify any node pairing, meaning the two nodes may define a "real" edge
        that is in the graph or a "fictitious" edge that does not appear in the given graph.
    C : dict
        Dictionary of graph distance (gdist) between every two nodes

        Note: if not specified, C is computed according to the specified 'gdist'
    EPS : float
        Threshold for approximately 0
    nbr_maxk : int
        Only take the top k edge weight neighbors for density distribution.
        Smaller k run faster but the result is less accurate. (Default value = 500).
    proc : int
        Number of processor used for multiprocessing. (Default value = cpu_count()).
    chunksize : int
        Chunk size for multiprocessing, set None for auto decide. (Default value = None).
    cache_maxsize : int
        Max size for LRU cache for pairwise shortest path computation.
        Set this to `None` for unlimited cache. (Default value = 1000000).

    Returns
    -------
    dict "output" with curvature for fictitious edges {(v1,v2): "ORC"}
    """

    logger.info("Preparing to compute edge Wasserstein ditances")
    logger.info("Number of nodes: %d" % G.number_of_nodes())
    logger.info("Number of edges: %d" % G.number_of_edges())

    assert gdist in ['hop', 'whop'], "Unrecognized gdist. Options = ['hop','whop']."

    self_loop_edges = list(nx.selfloop_edges(G))
    if self_loop_edges:
        logger.info('Removing {:d} self-loop edges.'.format(len(self_loop_edges)))
        G.remove_edges_from(self_loop_edges)
        logger.info("Updated number of nodes: %d" % G.number_of_nodes())
        logger.info("Updated number of edges: %d" % G.number_of_edges())

    if not nx.get_node_attributes(G, n_weight):
        logger.info("Node weight not detected, set default node weight values = 1.0")
        nx.set_node_attributes(G, 1.0, name=n_weight)

    if not nx.get_edge_attributes(G, e_weight):
        if gdist == "hop":
            logger.info("Edge weight not detected, setting default edge weight values 1.0")
            nx.set_edge_attributes(G, 1.0, name=e_weight)
        else:  # "whop"
            """ computing edge weights from given nodal weights """
            if e_wdeg:
                logger.info(
                    'Edge weight not detected, setting invariant measure contracted weighted degree based edge weights')
                if not nx.get_node_attributes(G, 'ED'):
                    logger.info("Computing invar_distr...")
                    t1 = time.time()
                    _equilib_dist(G, n_weight=n_weight)
                    logger.info("invar_distr computation computed in {:.5e} seconds.".format(time.time()-t1))
                _compute_alt_edge_weights(G, n_weight=n_weight, e_weight=e_weight, c_weight="ED")
            else:
                """
                e_normalized = True AND e_sqrt = True : w_ij = 1/sqrt([(p_ij + p_ji)/2])
                e_normalized = True AND e_sqrt = False : w_ij = 1/[(p_ij + p_ji)/2]
                e_normalized = False AND e_sqrt = True : w_ij = 1/sqrt(w_i * w_j)
                e_normalized = False AND e_sqrt = False : w_ij = (1/w_i)*(1/w_j) 
                NOTE: w_ij = INF if w_i=0 or w_j=0
                """
                if e_wprob:
                    logger.info(
                        "Edge weight not detected, setting probability of transition in at least one direction based edge weights")
                elif e_normalized:
                    if e_sqrt:
                        logger.info(
                            'Edge weight not detected, setting normalized root edge weight values sqrt(2/(p_ij + p_ji))')
                    else:
                        logger.info("Edge weight not detected, setting normalized edge weight values 2/(p_ij + p_ji)")
                else:
                    if e_sqrt:
                        logger.info("Edge weight not detected, setting root edge weight values 1/sqrt(w_ij)")
                    else:
                        logger.info("Edge weight not detected, setting edge weight values (1/w_ij)")
                G = compute_edge_weights(G, n_weight=n_weight, e_weight=e_weight, e_normalized=e_normalized,
                                         e_sqrt=e_sqrt, e_wprob=e_wprob)

    global _G
    global _alpha
    global _pdistr
    global _gdist
    global _nbr_maxk
    global _C
    global _n_weight
    global _e_weight
    global _e_normalized
    global _e_wdeg
    global _e_wprob
    global _e_sqrt
    global _power
    global _proc
    global _cache_maxsize
    global _EPS
    global _edge_list

    _G = G.copy()
    _alpha = alpha
    _pdistr = pdistr
    _gdist = gdist
    _nbr_maxk = nbr_maxk
    _C = C
    _n_weight = n_weight
    _e_weight = e_weight
    _e_normalized = e_normalized
    _e_sqrt = e_sqrt
    _e_wdeg = e_wdeg
    _e_wprob = e_wprob
    _power = power
    _proc = proc
    _cache_maxsize = cache_maxsize
    _EPS = EPS
    _edge_list = edge_list

    # --- compute ED ---
    if (_alpha in ['ED', 'EDnot']) and (not nx.get_node_attributes(_G, 'ED')):
        logger.info("Computing ED...")
        t1 = time.time()
        _equilib_dist(_G, n_weight=n_weight)
        logger.info("ED computation computed in {:.5e} seconds.".format(time.time() - t1))

    if _C is None:
        if _gdist not in ['hop', 'whop']:
            logger.warning('Unrecognized gdist, options: ["hop","whop"], use "hop" instead')
            _gdist = 'hop'
        _C = _get_graph_distance(_G, gdist=_gdist, e_weight=_e_weight)

    if not _edge_list:
        args = list(_G.edges())
        nargs = len(args)
        logger.info('Computing curvature on {:d} edges'.format(nargs))
    elif type(_edge_list) == str:
        N = _G.number_of_nodes()
        nargs = int(0.5 * N * (N - 1))
        if nargs > 25000:
            logger.info(
                'Over 25,000 fictitious edges in network, reducing to {} true edges'.format(_G.number_of_edges()))
            args = list(_G.edges())
        else:
            args = itertools.combinations(_G.nodes(), 2)
            logger.info('Computing curvature on all {:d} pairs'.format(nargs))
    else:
        args = _edge_list
        nargs = len(_edge_list)
        logger.info('Computing curvature on {:d} selected pairs'.format(nargs))

    t0 = time.time()
    with mp.get_context('fork').Pool(processes=_proc) as pool:
        chunksize, extra = divmod(nargs, _proc * 4)
        if extra:
            chunksize += 1
        result = pool.imap_unordered(_wrap_compute_wasserstein_edge, args, chunksize=chunksize)
        pool.close()
        pool.join()

    output = {}
    for K in result:
        for k in list(K.keys()):
            output[(k[0], k[1])] = K[k]

    logger.info("{} seconds for edge wasserstein distance computation".format(time.time() - t0))
    if return_C:
        return output, _C

    return output


def _compute_curvature_edge(source, target):
    """ compute ORC for given (possible fictitious) edge.

    Parameters
    ----------
    source : str
        Reference name of source node in Networkx graph _G
    target : str
        Reference name of target node in Networkx graph _G
    
    Returns:
    ________
    output : dict[(str,str), float]
        The ORC of given edge in dict format (e.g. {(source,target): ORC})

    """
    assert source != target, "Self loop is not allowed."

    # if the weight of either node is too small, return np.nan instead
    if (_G.nodes[source][_n_weight] < _EPS) or (_G.nodes[target][_n_weight] < _EPS):
        
        if isinstance(source, str): 
            logger.warning(
                "Zero weight node detected for edge ({:s},{:s}), return Ricci Curvature as np.nan instead.".format(source,
                                                                                                               target))
        else: 
            logger.warning(
                "Zero weight node detected for edge ({:f},{:f}), return Ricci Curvature as np.nan instead.".format(source,
                                                                                                               target))
        return {(source, target): np.nan}

    omtd = _compute_wasserstein_edge(source, target)

    ORC = 1 - (omtd[(source, target)] / _C[source][target])
    return {(source, target): ORC}


def _wrap_compute_curvature_edge(stuff):
    """Wrapper for args in multiprocessing."""
    return _compute_curvature_edge(*stuff)


def _compute_curvature_edges(G, n_weight="weight", e_weight="weight",
                             e_normalized=False, e_sqrt=False, e_wdeg=False, e_wprob=False,
                             pdistr="mass_action", gdist="hop", alpha=0.0, 
                             power=1, edge_list=None, C=None, EPS=1e-7, nbr_maxk=500,
                             proc=mp.cpu_count(), chunksize=None, cache_maxsize=1000000):
    """ Compute curvature on all edges in edge_list

    Parameters
    ----------
    G : NetworkX graph
            A given (undirected, connected) NetworkX graph.
    n_weight : str
        Name of nodal feature used for constructing distance/distributions (Default = "weight")
        Note: if attribute "n_weight" does not exist, default value of 1 is assigned to all nodes
    e_weight : str
        (Optional) Name of edge feature used to compute distance when gdist='whop' (Default = "weight")
        Note: When gdist='hop', edge weights are all treated as 1.
        If attribute "e_weight" does not exist and gdist='whop', edge weight is computed as w_ij = (1/w_i)*(1/w_j) 
        where w_i is the node weight specified by 'n_weight'.
    e_normalized : logical
        Indicate if normalized edge weight should be used when gdist="whop" (Default = False).    
        Note: Ignored if e_wdeg is True.
    e_sqrt : logical
        Indicate if sqrt should be taken of inferred edge weights (Default = False).
        Note: Ignored if e_wdeg is True .
    e_wdeg : logical 
        Indicate if edge weight should be inferred from weighted degree contracted with the invariant measure.
        (Default = False)
    e_wprob : logical
        Indicate if edge weight should be inferred from probability of transition in at least one direction.
        (Default = False)
    pdistr : {'mass_action', 'edge_based', 'combo', 'edge_simple'}
        Specify if nodal probability distribution should be derived from the mass action principle of edge based.
        (Default = 'mass_action').

        options:

        - 'mass_action' : node-based on mass action principle --> m_ij := w_j (Default).
        - 'edge_based' : edge-based on distance --> m_ij := exp(-d(i,j)^power)
        - 'combo' : combine node and edge attributes --> m_ij := w_j * exp(-d(i,j)^power)
        - 'edge_simple' : simple edge-based on distance --> m_ij := 1/(d(i,j)^power)

    gdist : {'hop','whop'}
        Graph distance (Default = 'hop')

        options:

        - 'hop' : hop-distance (Default)
        - 'whop' : weighted hop-distance
    alpha : {float: (0.0 <= alpha <= 1.0), str: ['ED','EDnot','weight']}
        Idle probability (Default = 0.0)

        options:

        - float : alpha is a constant in the range [0,1]
        - 'ED' : alpha is node-dependent prescribed by the ED (alpha_i = pi_i)
        - 'EDnot' : alpha is node-dependent prescribed by the probabilistic negation of the ED (alpha_i = 1 - pi_i)
        - 'weight' : alpha is node-dependent prescribed by the given node weight 'n_weight'
    power : float
        Edge weight power for edge-based distribution. (Default value = 1) 
    edge_list : {list, None, "all"}
        Specify particular set of edges to compute curvature on. (Default = None).
        If edge_list is None, curvature is computed on all edges in the given graph. 
        If edge_list is "all," curvature is computed between every two nodes in the given graph.

        Note: Edges may specify any node pairing, meaning the two nodes may define a "real" edge
        that is in the graph or a "fictitious" edge that does not appear in the given graph.
    C : dict
        Dictionary of graph distance (gdist) between every two nodes

        Note: if not specified, C is computed according to the specified 'gdist'
    EPS : float 
        Threshold for approximately 0
    nbr_maxk : int
        Only take the top k edge weight neighbors for density distribution.
        Smaller k run faster but the result is less accurate. (Default value = 500).
    proc : int
        Number of processor used for multiprocessing. (Default value = cpu_count()).
    chunksize : int
        Chunk size for multiprocessing, set None for auto decide. (Default value = None).
    cache_maxsize : int
        Max size for LRU cache for pairwise shortest path computation.
        Set this to `None` for unlimited cache. (Default value = 1000000).

    Returns
    -------
    dict "output" with curvature for fictitious edges {(v1,v2): "ORC"}
    """

    logger.info("Preparing to compute edge curvatures")
    logger.info("Number of nodes: %d" % G.number_of_nodes())
    logger.info("Number of edges: %d" % G.number_of_edges())

    assert gdist in ['hop', 'whop'], "Unrecognized gdist. Options = ['hop','whop']."

    self_loop_edges = list(nx.selfloop_edges(G))
    if self_loop_edges:
        logger.info('Removing {:d} self-loop edges.'.format(len(self_loop_edges)))
        G.remove_edges_from(self_loop_edges)
        logger.info("Updated number of nodes: %d" % G.number_of_nodes())
        logger.info("Updated number of edges: %d" % G.number_of_edges())

    if not nx.get_node_attributes(G, n_weight):
        logger.info("Node weight not detected, set default node weight values = 1.0")
        nx.set_node_attributes(G, 1.0, name=n_weight)

    if not nx.get_edge_attributes(G, e_weight):
        if gdist == "hop":
            logger.info("Edge weight not detected, setting default edge weight values 1.0")
            nx.set_edge_attributes(G, 1.0, name=e_weight)
        else:  # "whop"
            """ computing edge weights from given nodal weights """

            if e_wdeg:
                logger.info(
                    'Edge weight not detected, setting invariant measure contracted weighted degree based edge weights')
                if not nx.get_node_attributes(G, 'ED'):
                    logger.info("Computing invar_distr...")
                    t1 = time.time()
                    _equilib_dist(G, n_weight=n_weight)
                    logger.info("invar_distr computation computed in {:.5e} seconds.".format(time.time()-t1))
                _compute_alt_edge_weights(G, n_weight=n_weight, e_weight=e_weight, c_weight="ED")
            else:
                """
                e_normalized = True AND e_sqrt = True : w_ij = 1/sqrt([(p_ij + p_ji)/2])
                e_normalized = True AND e_sqrt = False : w_ij = 1/[(p_ij + p_ji)/2]
                e_normalized = False AND e_sqrt = True : w_ij = 1/sqrt(w_i * w_j)
                e_normalized = False AND e_sqrt = False : w_ij = (1/w_i)*(1/w_j) 
                NOTE: w_ij = INF if w_i=0 or w_j=0
                """
                if e_wprob:
                    logger.info(
                        "Edge weight not detected, setting probability of transition in at least one direction based edge weights")
                elif e_normalized:
                    if e_sqrt:
                        logger.info(
                            'Edge weight not detected, setting normalized root edge weight values sqrt(2/(p_ij + p_ji))')
                    else:
                        logger.info("Edge weight not detected, setting normalized edge weight values 2/(p_ij + p_ji)")
                else:
                    if e_sqrt:
                        logger.info("Edge weight not detected, setting root edge weight values 1/sqrt(w_ij)")
                    else:
                        logger.info("Edge weight not detected, setting edge weight values (1/w_ij)")
                G = compute_edge_weights(G, n_weight=n_weight, e_weight=e_weight, e_normalized=e_normalized,
                                         e_sqrt=e_sqrt, e_wprob=e_wprob)

    global _G
    global _alpha
    global _pdistr
    global _gdist
    global _nbr_maxk
    global _C
    global _n_weight
    global _e_weight
    global _e_normalized
    global _e_sqrt
    global _e_wdeg
    global _power
    global _proc
    global _cache_maxsize
    global _EPS
    global _edge_list

    _G = G.copy()
    _alpha = alpha
    _pdistr = pdistr
    _gdist = gdist
    _nbr_maxk = nbr_maxk
    _C = C
    _n_weight = n_weight
    _e_weight = e_weight
    _e_normalized = e_normalized
    _e_wdeg = e_wdeg
    _e_wprob = e_wprob
    _e_sqrt = e_sqrt
    _power = power
    _proc = proc
    _cache_maxsize = cache_maxsize
    _EPS = EPS
    _edge_list = edge_list

    # --- compute ED ---
    if (_alpha in ['ED', 'EDnot']) and (not nx.get_node_attributes(_G, 'ED')):
        logger.info("Computing ED...")
        t1 = time.time()
        _equilib_dist(_G, n_weight=n_weight)
        logger.info("ED computation computed in {:.5e} seconds.".format(time.time() - t1))

    if _C is None:
        if _gdist not in ['hop', 'whop']:
            logger.warning('Unrecognized gdist, options: ["hop","whop"], use "hop" instead')
            _gdist = 'hop'
        _C = _get_graph_distance(_G, gdist=_gdist, e_weight=_e_weight)

    if not _edge_list:
        args = list(_G.edges())
        nargs = len(args)
        logger.info('Computing curvature on {:d} edges'.format(nargs))
    elif type(_edge_list) == str:
        N = _G.number_of_nodes()
        nargs = int(0.5 * N * (N - 1))
        if nargs > 25000:
            logger.info(
                'Over 25,000 fictitious edges in network, reducing to {} true edges'.format(_G.number_of_edges()))
            args = list(_G.edges())
        else:
            args = itertools.combinations(_G.nodes(), 2)
            logger.info('Computing curvature on all {:d} pairs'.format(nargs))
    else:
        args = _edge_list
        nargs = len(_edge_list)
        logger.info('Computing curvature on {:d} selected pairs'.format(nargs))

    t0 = time.time()
    with mp.get_context('fork').Pool(processes=_proc) as pool:
        chunksize, extra = divmod(nargs, _proc * 4)
        if extra:
            chunksize += 1
        result = pool.imap_unordered(_wrap_compute_curvature_edge, args, chunksize=chunksize)
        pool.close()
        pool.join()

    output = {}
    for K in result:
        for k in list(K.keys()):
            output[(k[0], k[1])] = K[k]

    logger.info("{} seconds for edge curvature computation".format(time.time() - t0))

    return output


def _equilib_dist(G: nx.Graph, n_weight="weight"):
    """ compute ED (returned as nodal attribute "ED") """
   

    if not nx.is_connected(G):
        logger.warning("Graph is not connected -- equilibrium distribution not computed")
        return G
    else:
        ed = list()
        for n in G.nodes():
            if G.degree(n) != 0:
                n_ed = G.nodes[n][n_weight] * sum([G.nodes[nbr][n_weight] for nbr in G.neighbors(n)])
                G.nodes[n]["ED"] = n_ed
                ed.append(n_ed)
        sum_ed = sum(ed)
        if sum_ed > 1e-7:
            for n in G.nodes():
                G.nodes[n]["ED"] /= sum_ed
            logger.trace("ED-magnitude = {}".format(sum_ed))
        else:
            logger.warning('Equilibrium distribution normalization too small, normalization not applied')
    return G


def _SORCgraph(G: nx.Graph, contraction='ED', label="SORC",transition_opts={'n_weight':'weight'}):
    """ compute SORC for all nodes in graph G """
    assert nx.get_edge_attributes(G, 'ORC'), "ORC not detected in graph, required for SORC computation."
    assert contraction in ['ED', 'distance','transition','sum'], "Unrecognized contraction, options = ['ED','distance','transition','sum']."
    if (contraction == 'ED'):
        assert nx.get_node_attributes(G, 'ED'), "ED not detected for node-specific alpha."

    logger.info(f"Contracting scalar curvature by {contraction}")

    for n in G.nodes():
        orc_sum = 0
        if (G.degree(n) != 0):
            for nbr in G.neighbors(n):
                if ("ORC" in G.edges[(n, nbr)]) and (_C[n][nbr] < np.inf):
                    kx = G.edges[(n, nbr)]["ORC"]
                    if contraction == 'distance':
                        kx *= _C[n][nbr]
                    elif contraction == 'transition':
                        kx *= pij(G,n,nbr,**transition_opts)
                    orc_sum += kx
                else:
                    logger.warning("Node {} missing ORC".format(nbr))
            if contraction == 'ED':
                if 'ED' in G.nodes[n]:
                    G.nodes[n][label] = orc_sum * G.nodes[n]["ED"]
                else:
                    logger.warning("Node {} missing ED, SORC returned as nan".format(n))
                    G.nodes[n][label] = np.nan
            else:  # 'distance' or 'transition' or 'sum'
                G.nodes[n][label] = orc_sum
        else:
            logger.warning("Node {} found with no neighbors. SORC set to 0.0".format(n))
            G.nodes[n][label] = 0.0
    return G


def _SORCavg_graph(G: nx.Graph):
    """ Compute averadge incident edge ORC as ORCavg for each node in graph """
    for n in G.nodes():
        orc_sum = 0
        if G.degree(n) != 0:
            nbr_cnt = 0
            for nbr in G.neighbors(n):
                if ("ORC" in G.edges[(n, nbr)]) and (_C[n][nbr] < np.inf):
                    nbr_cnt += 1
                    orc_sum += G.edges[(n, nbr)]["ORC"]
            G.nodes[n]["ORCavg"] = orc_sum / nbr_cnt # G.degree(n)
            logger.trace("ORCavg({}) = {:.5e}".format(n, G.nodes[n]["ORCavg"]))
        else:
            G.nodes[n]["ORCavg"] = 0.0
            logger.trace("Node {} found with no neighbors. ORCavg set to 0.0".format(n))
    return G


def _compute_curvature(G: nx.Graph, n_weight="weight", C=None, gdist="hop", contraction='ED', **kwargs):
    """ Compute edge and nodal curvatures. 
    Parameters
    ----------
    G : NetworkX graph
        A given (undirected, connected) NetworkX graph.
    n_weight : str
        Name of nodal feature used for constructing distance/distributions (Default = "weight")
    C : dict
            Dictionary of graph distance (gdist) between every two nodes

            Note: if not specified, C is computed according to the specified 'gdist'
    contraction : str {'ED', 'distance'}
            Specify contraction for computing scalar curvature (Default value = 'ED'). 

            options:

            - 'ED' : k_x = pi_x * sum_{y ~ x} k(x,y)
            - 'distance' : k_x = sum_{y ~ x} d(x,y)*k(x,y)
    **kwargs
        Additional keyword arguments passed to `_compute_curvature_edges`.

    Returns
    -------
    G : NetworkX graph
        A graph with "Wass" and "ORC" on edges, "ED," "SORC," and "ORCavg" on nodes
    # EK : dict
    #    A dictionary with ORC of fictitious edges {(v1,v2): "ORC"} 
    """
    assert contraction in ['ED', 'distance'], "Unrecognized contraction, options = ['ED','distance']."

    t0 = time.time()
    wasserstein_distances, C = _compute_wasserstein_edges(G, n_weight=n_weight, C=C, gdist=gdist, return_C=True, **kwargs)
    nx.set_edge_attributes(G, wasserstein_distances, name="Wass")
    orc = {ee: np.nan if np.isnan(wd) else 1. - wd / C.loc[ee[0], ee[1]] for ee, wd in wasserstein_distances.items()}

    nx.set_edge_attributes(G, orc, name="ORC")

    # --- compute ED ---
    if not nx.get_node_attributes(G, 'ED'):
        logger.info("Computing ED...")
        t1 = time.time()
        _equilib_dist(G, n_weight=n_weight)
        logger.info("ED computation computed in {:.5e} seconds.".format(time.time() - t1))
    # --- compute SORC ---
    t2 = time.time()
    _SORCgraph(G, contraction=contraction)
    logger.info("SORC computation computed in {:.5e} seconds.".format(time.time() - t2))
    # --- compute SORCavg ---
    t3 = time.time()
    _SORCavg_graph(G)
    logger.info("SORCavg computation computed in {:.5e} seconds.".format(time.time() - t3))
    logger.info("Total curvature compuation computed in {:.5e} seconds.".format(time.time() - t0))

    return G


def orc_omics(G: nx.Graph): 
    """
    Run ORCO. Requires a networkX object G to be passed with node weights labeled as 'weight.'
              Returns ORCO results — edges and their corresponding ORC values. 
    """
    
    G = compute_edge_weights(G, e_normalized=True, e_sqrt=True)
    orc = ORC(G, verbose="ERROR") 
    edge_curvatures = orc.compute_curvature_edges()

    return edge_curvatures


########################
# ~~ Class definition ~~
########################

class ORC:
    """ A class to compute Ollivier-Ricci curvature (ORC) and Ricci flow (RF)
    based off of (non-negative) nodal weights in a given connected Networkx graph. 
    Nodal curvature is defined as the sum of adjacent edge curvatures weighted by 
    the corresponding component of the associated invariant distribution. 

    Notes
    -----
    - Currently only tested for undirected, connected graphs with non-negative weights.

    """

    def __init__(self, G: nx.Graph, n_weight="weight", e_weight="weight",
                 e_normalized=True, e_sqrt=True, e_wdeg=False, e_wprob=False,
                 pdistr="mass_action", gdist="whop", alpha=0.0, 
                 power=1, contraction='ED', edge_list=None, C=None, 
                 EPS=1e-7, nbr_maxk=500, verbose="INFO",
                 proc=mp.cpu_count(), chunksize=None, cache_maxsize=1000000):
        """ Initialize a container to compute Ollivier-Ricci curvature and Ricci flow

        Parameters
        ----------
        G : NetworkX graph
            A given (undirected, connected) NetworkX graph.
        n_weight : str
            Name of nodal feature used for constructing distance/distributions (Default = "weight")
            Note: if attribute "n_weight" does not exist, default value of 1 is assigned to all nodes
        e_weight : str
            (Optional) Name of edge feature used to compute distance when gdist='whop' (Default = "weight")
            Note: When gdist='hop', edge weights are all treated as 1.
            If attribute "e_weight" does not exist and gdist='whop', edge weight is computed as w_ij = (1/w_i)*(1/w_j) 
            where w_i is the node weight specified by 'n_weight'.
        e_normalized : logical
            Indicate if normalized edge weight should be used when gdist="whop" (Default = True).    
            Note: Ignored if e_wdeg is True.
        e_sqrt : logical                                                              
            Indicate if sqrt should be taken of inferred edge weights (Default = True).  
            Note: Ignored if e_wdeg is True.
        e_wdeg : logical 
            Indicate if edge weight should be inferred from weighted degree contracted with the invariant measure.
            (Default = False)
        e_wprob : logical
            Indicate if edge weight should be inferred from probability of transition in either direction where
            w_ij = p_ij + p_ji - (p_ij * p_ji)
        pdistr : {'mass_action', 'edge_based', 'combo', 'edge_simple'}
            Specify if nodal probability distribution should be derived from the mass action principle or edge based.
            (Default = 'mass_action').

            options:

            - 'mass_action' : node-based on mass action principle --> :math:`m_ij := w_j` (Default).
            - 'edge_based' : edge-based on distance --> :math:`m_ij := \exp(-d(i,j)^2)`
            - 'combo' : combine node and edge attributes --> :math:`m_ij := w_j * \exp(-d(i,j)^power)`
            - 'edge_simple' : simple edge-based on distance --> :math:`m_ij := 1/(d(i,j)^power)`
        gdist : {'hop','whop'}
            Graph distance (Default = 'whop')

            options:

            - 'hop' : hop-distance (Default)
            - 'whop' : weighted hop-distance
        alpha : {float: 0.0 <= alpha <= 1.0, str: ['ED','EDnot','weight']}
            Idle probability (Default = 0.0)

            options

            - float : alpha is a constant in the range [0,1]
            - 'ED' : alpha is node-dependent prescribed by the ED (alpha_i = pi_i)
            - 'EDnot' : alpha is node-dependent prescribed by the probabilistic negation of the ED (alpha_i = 1 - pi_i)
            - 'weight' : alpha is node-dependent prescribed by the given node weight 'n_weight'
        power : float
            Edge weight power for edge-based distribution. (Default value = 1)
        contraction : str {'ED', 'distance'}
            Specify contraction for computing scalar curvature (Default value = 'ED'). 

            options:

            - 'ED' : k_x = pi_x * sum_{y ~ x} k(x,y)
            - 'distance' : k_x = sum_{y ~ x} d(x,y)*k(x,y)
        edge_list : {list, None, "all"}
            Specify particular set of edges to compute curvature on. (Default = None).
            If edge_list is None, curvature is computed on all edges in the given graph. 
            If edge_list is "all," curvature is computed between every two nodes in the given graph.
            
            Note: Edges may specify any node pairing, meaning the two nodes may define a "real" edge
            that is in the graph or a "fictitious" edge that does not appear in the given graph.
        C : dict
            Dictionary of graph distance (gdist) between every two nodes

            Note: if not specified, C is computed according to the specified 'gdist'
        EPS : float 
            Threshold for approximately 0
        nbr_maxk : int
            Only take the top k edge weight neighbors for density distribution.
            Smaller k run faster but the result is less accurate. (Default value = 500).
        verbose : {"INFO", "TRACE","DEBUG","ERROR"}
            Verbose level. (Default value = "INFO")
                - "INFO": show only iteration process log.
                - "TRACE": show detailed iteration process log.
                - "DEBUG": show all output logs.
                - "ERROR": only show log if error happened.
        proc : int
            Number of processor used for multiprocessing. (Default value = cpu_count()).
        chunksize : int
            Chunk size for multiprocessing, set None for auto decide. (Default value = None).
        cache_maxsize : int
            Max size for LRU cache for pairwise shortest path computation.
            Set this to `None` for unlimited cache. (Default value = 1000000).

        Methods
        -------

        """
        assert not (e_wdeg and e_wprob), "e_wdeg and e_wprob cannot both be True."
        
        self.G = G.copy()
        self.n_weight = n_weight
        self.e_weight = e_weight
        self.e_normalized = e_normalized
        self.e_sqrt = e_sqrt
        self.e_wdeg = e_wdeg
        self.e_wprob = e_wprob
        self.pdistr = pdistr
        self.gdist = gdist
        self.alpha = alpha
        self.contraction = contraction
        self.power = power
        self.edge_list = edge_list
        self.C = C
        self.EPS = EPS
        self.nbr_maxk = nbr_maxk
        self.proc = proc
        self.chunksize = chunksize
        self.cache_maxsize = cache_maxsize
        self.set_verbose(verbose)
        self.verbose = verbose

        self.EK = {}

        self_loop_edges = list(nx.selfloop_edges(self.G))
        if self_loop_edges:
            logger.info('Removing {:d} self-loop edges.'.format(len(self_loop_edges)))
            self.G.remove_edges_from(self_loop_edges)

        # assert nx.is_connected(self.G),"Graph must be connected."
        if not nx.is_connected(self.G):
            logger.info("Graph is not connected, reduce to largest connected component instead")
            self.G = nx.Graph(self.G.subgraph(max(nx.connected_components(self.G), key=len)))

        if not nx.get_node_attributes(self.G, n_weight):
            logger.info("Node weight not detected, set default node weight values = 1.0")
            n_weight = "weight"
            self.n_weight = n_weight
            nx.set_node_attributes(self.G, 1.0, name=n_weight)

        assert min(nx.get_node_attributes(self.G, n_weight).values()) >= 0, "Negative node weight detected."

        if not nx.get_edge_attributes(self.G, e_weight):
            """ computing edge weights from given nodal weights: """
            if e_wdeg:
                logger.info(
                    "Edge weight not detected, setting invariant measure contracted weighted degree based edge weights")
                if not nx.get_node_attributes(G, 'ED'):
                    logger.info("Computing invar_distr...")
                    t1 = time.time()
                    _equilib_dist(self.G, n_weight=n_weight)
                    logger.info("invar_distr computation computed in {:.5e} seconds.".format(time.time() - t1))
                e_weight = "weight"
                self.e_weight = e_weight
                self.G = _compute_alt_edge_weights(self.G, n_weight=n_weight, e_weight=e_weight, c_weight="ED")                            
            else:
                """
                e_normalized = True AND e_sqrt = True : w_ij = 1/sqrt([(p_ij + p_ji)/2])
                e_normalized = True AND e_sqrt = False : w_ij = 1/[(p_ij + p_ji)/2]
                e_normalized = False AND e_sqrt = True : w_ij = 1/sqrt(w_i * w_j)
                e_normalized = False AND e_sqrt = False : w_ij = (1/w_i)*(1/w_j) 
                NOTE: w_ij = INF if w_i=0 or w_j=0
                e_wprob = True (ignores e_normalized and e_sqrt) : w_ij = 1 / (p_ij + p_ji - (p_ij * p_ji))
                """
                if e_wprob:
                    logger.info(
                        "Edge weight not detected, setting probability of transition in at least one direction based edge weights")
                elif e_normalized:
                    if e_sqrt:
                        logger.info(
                            "Edge weight not detected, setting normalized root edge weight values sqrt(2/(p_ij + p_ji))")
                    else:
                        logger.info("Edge weight not detected, setting normalized edge weight values 2/(p_ij + p_ji)")
                else:
                    if e_sqrt:
                        logger.info("Edge weight not detected, setting root edge weight values 1/sqrt(w_ij)")
                    else:
                        logger.info("Edge weight not detected, setting edge weight values (1/w_ij)")
                e_weight = "weight"
                self.e_weight = e_weight
                self.G = compute_edge_weights(self.G, n_weight=n_weight, e_weight=e_weight, e_normalized=e_normalized,
                                              e_sqrt=e_sqrt, e_wprob=e_wprob)

    def set_verbose(self, verbose):
        """Set the verbose level for this process.
        Parameters
        ----------
        verbose : {"INFO", "TRACE","DEBUG","ERROR"}
            Verbose level. (Default value = "ERROR")
                - "INFO": show only iteration process log.
                - "TRACE": show detailed iteration process log.
                - "DEBUG": show all output logs.
                - "ERROR": only show log if error happened.
        """
        self.verbose = verbose
        set_verbose(verbose)

    def info(self):
        """ Return a summary of information for the object ORC
        The summary includes the current parameter settings.
        
        Parameters
        ----------
        K: ORC object
            An ORC object for computing curvatures

        Returns
        _______
        info : str
            A string containing the short summary
        """
        Ginfo = "\n".join(nx.info(self.G).split(sep="\n", maxsplit=3)[2:])
        Pinfo = "\n".join(("n_weight : {:s}".format(self.n_weight),
                           "e_weight : {:s}".format(self.e_weight),
                           "e_normalized : {}".format(self.e_normalized),
                           "e_sqrt : {}".format(self.e_sqrt),
                           "e_wdeg : {}".format(self.e_wdeg),
                           "e_wprob : {}".format(self.e_wprob),
                           "pdistr : {:s}".format(self.pdistr),
                           "gdist : {:s}".format(self.gdist),
                           "alpha : {}".format(self.alpha if type(self.alpha) is str else np.round(self.alpha, 4)),
                           "contraction : {:s}".format(self.contraction),
                           "power : {:d}".format(self.power),
                           "EPS : {:.4f} (fixed)".format(EPS),
                           "nbr_maxk : {:d}".format(self.nbr_maxk),
                           "verbose : {:s}".format(self.verbose),
                           "proc : {:d}".format(self.proc),
                           "chunksize : {}".format("None" if self.chunksize is None else self.chunksize),
                           "cache_maxsize : {}".format(self.cache_maxsize),
                           "{}".format("edge_list : None" if self.edge_list is None else
                                       "Number of edges in edge_list : {}".format(self.edge_list.shape[0])),
                           "{}".format("C : None" if self.C is None else "C : dict")))

        return "\n".join((Ginfo, Pinfo))

    def get_graph_distance(self):
        """ compute scg of graph distances between every two nodes """
        return _get_graph_distance(self.G, gdist=self.gdist, e_weight=self.e_weight)

    def equilib_dist(self):
        """ Compute the equilibrium distributions based off of the node weighting stored as node attribute 'ED' """
        self.G = _equilib_dist(self.G, n_weight=self.n_weight)

    def compute_wasserstein_edges(self, edge_list=None):
        """ Compute wasserstein distance for edges in edge_list

        Parameters
        ----------
        edge_list : list
            Specify particular set of edges to compute curvature on. (Default = None).
            If edge_list is None, curvature is computed on all edges in the given graph.
            If edge_list is "all," curvature is computed between every two nodes in the given graph.
            Note: Edges may specify any node pairing, meaning the two nodes may define a "real" edge
            that is in the graph or a "fictitious" edge that does not appear in the given graph.

        Returns
        -------
        EK : dict
            A dictionary with Wasserstein distances of (possibly fictitious) edges {(v1,v2): "Wasserstein"}

        Example
        -------
        To compute the Wasserstein distance between every two nodes in karate club graph:
        >>> G = nx.karate_club_graph()
        >>> nx.set_node_attributes(G, 1.0, name="weight")
        >>> orc = ORC(G, gdist="whop", e_normalized=True, n_weight="weight", verbose="INFO")
        >>> wasserstein_distances = orc.compute_wasserstein_edges(edge_list="all")
        >>> wasserstein_distances[0,9]
        4.3125
        """

        return _compute_wasserstein_edges(G=self.G, n_weight=self.n_weight, e_weight=self.e_weight,
                                          e_wdeg=self.e_wdeg, e_wprob=self.e_wprob, 
                                          e_normalized=self.e_normalized, e_sqrt=self.e_sqrt, pdistr=self.pdistr,                                          
                                          alpha=self.alpha, gdist=self.gdist,
                                          power=self.power, edge_list=edge_list, C=self.C, 
                                          nbr_maxk=self.nbr_maxk, EPS=EPS, proc=self.proc,
                                          chunksize=self.chunksize, cache_maxsize=self.cache_maxsize)


    def compute_curvature_edges(self, edge_list=None):
        """ Compute curvature for edges in edge_list

        Parameters
        ----------
        edge_list : list
            Specify particular set of edges to compute curvature on. (Default = None).
            If edge_list is None, curvature is computed on all edges in the given graph. 
            If edge_list is "all," curvature is computed between every two nodes in the given graph.

            Note: Edges may specify any node pairing, meaning the two nodes may define a "real" edge
            that is in the graph or a "fictitious" edge that does not appear in the given graph.

        Returns
        -------
        EK : dict
            A dictionary with ORC of (possibly fictitious) edges {(v1,v2): "ORC"} 

        Example
        -------
        To compute the Ollivier-Ricci curvature between every two nodes in karate club graph:
            >>> G = nx.karate_club_graph()
            >>> nx.set_node_attributes(G, 1.0, name="weight")
            >>> orc = ORC(G, gdist="whop", e_normalized=True, n_weight="weight", verbose="INFO")
            >>> curvatures = orc.compute_curvature_edges(edge_list="all")
            >>> curvatures[0,9]
            0.26477447972329937
        """

        return _compute_curvature_edges(G=self.G, n_weight=self.n_weight, e_weight=self.e_weight,
                                        e_wdeg=self.e_wdeg, e_wprob=self.e_wprob, 
                                        e_normalized=self.e_normalized, e_sqrt=self.e_sqrt, pdistr=self.pdistr,
                                        
                                        alpha=self.alpha, gdist=self.gdist,
                                        power=self.power, edge_list=edge_list, C=self.C, 
                                        nbr_maxk=self.nbr_maxk, EPS=EPS, proc=self.proc,
                                        chunksize=self.chunksize, cache_maxsize=self.cache_maxsize)

    def compute_scalar_curvature(self, contraction='ED', label="SORC"):
        """ Compute scalar curvature for all nodes in graph (edge curvature must be computed first)

        Parameters
        ----------
        contraction : str {'ED', 'distance'}
            Specify contraction for computing scalar curvature (Default value = 'ED'). 

            options:

            - 'ED' : k_x = pi_x * sum_{y ~ x} k(x,y)
            - 'distance' : k_x = sum_{y ~ x} d(x,y)*k(x,y)
        label : str
            Node attribute that scalar curvature is saved to. (Default value = "SORC")

        Returns
        -------
        G : NetworkX graph
            A graph with scalar curvature as <label> attribute on nodes

        Example
        -------
        To compute the scalar curvature contracted by the distance as nodal attribute "dSORC":
            >>> G = nx.karate_club_graph()
            >>> nx.set_node_attributes(G, 1.0, name="weight")
            >>> orc = ORC(G, gdist="whop", e_normalized=True, n_weight="weight", verbose="INFO")
            >>> orc.compute_curvature()
            >>> orc.compute_scalar_curvature(contraction='distance',label="dSORC")
            >>> orc.G.nodes[0]
            {'club': 'Mr. Hi', 'weight': 1.0, 'ED': 0.10256410256410256, 'SORC': -0.4041629431232461,
            'ORCavg': -0.24628679346572807, 'dSORC': -6.6183923208398845}
        """
        self.G = _SORCgraph(G=self.G, contraction=contraction, label=label)
        return self.G

    def compute_curvature(self):
        """ Compute edge and nodal curvatures. 

        Returns
        -------
        G : NetworkX graph
            A graph with "ORC" on edges, "ED" and "SORC" on nodes

        Example
        -------
        To compute the Ollivier-Ricci curvature for karate club graph:
            >>> G = nx.karate_club_graph()
            >>> nx.set_node_attributes(G, 1.0, name="weight")
            >>> orc = ORC(G, gdist="whop", e_normalized=True, n_weight="weight", verbose="INFO")
            >>> orc.compute_curvature()
            >>> orc.G.edges[0,1]
            {'weight': 11.52, 'ORC': 0.4098512529947098}

        """

        self.G = _compute_curvature(G=self.G, n_weight=self.n_weight, e_weight=self.e_weight,
                                    e_wdeg=self.e_wdeg, e_wprob=self.e_wprob,
                                    e_normalized=self.e_normalized, e_sqrt=self.e_sqrt, 
                                    gdist=self.gdist,
                                    pdistr=self.pdistr, alpha=self.alpha, 
                                    power=self.power, contraction=self.contraction,
                                    edge_list=None, C=self.C, 
                                    nbr_maxk=self.nbr_maxk, EPS=EPS, proc=self.proc,
                                    chunksize=self.chunksize, cache_maxsize=self.cache_maxsize)
        return self.G

    
