import logging
import networkx as nx
import numpy as np
from functools import partial, partialmethod
import sys

logging.TRACE = logging.DEBUG + 5
logging.addLevelName(logging.TRACE, 'TRACE')
logging.Logger.trace = partialmethod(logging.Logger.log, logging.TRACE)
logging.trace = partial(logging.log, logging.TRACE)

logging.StreamHandler(stream=sys.stdout)
logger = logging.getLogger("ORCO")

handler = logging.StreamHandler()
formatter = logging.Formatter(
    '%(asctime)s %(name)-12s %(levelname)-8s %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

def set_verbose(verbose="INFO"): # "ERROR"):
    """ Set up the verbose level of ORCO.
    
    Parameters  
    ----------   
    verbose: {"INFO","TRACE","DEBUG","ERROR"} 
        Verbose level. (Default = "ERROR")
            - "INFO": show only iteration process log. 
            - "TRACE": show detailed iteration process log.
            - "DEBUG": show all output logs. 
            - "ERROR": only show log if error happened. 
    """
    if verbose == "INFO":
        logger.setLevel(logging.INFO)
    elif verbose == "TRACE":
        logger.setLevel(logging.TRACE)
    elif verbose == "DEBUG":
        logger.setLevel(logging.DEBUG)
    elif verbose == "ERROR":
        logger.setLevel(logging.ERROR)
    else:
        print("Unrecognized verbose level, options: ['INFO','DEBUG','ERROR'], use 'ERROR' instead")
        logger.setLevel(logging.ERROR)
        

def cut_graph_by_cutoff(G_origin, cutoff, e_weight="weight"):
    """Remove graph's edges with "weight" greater than "cutoff".

    Parameters
    ----------
    G_origin : NetworkX graph
        A graph with ``weight`` as Ricci flow metric to cut.
    cutoff : float
        A threshold to remove all edges with "weight" greater than it.
    e_weight : str
        The edge weight used as Ricci flow metric. (Default value = "weight")

    Returns
    -------
    G: NetworkX graph
        A graph with edges cut by given cutoff value.
    """
    assert nx.get_edge_attributes(G_origin, e_weight), "No edge weight detected, abort."

    G = G_origin.copy()
    edge_trim_list = []
    for ee in G.edges():
        if G.edges[ee][e_weight] > cutoff:
            edge_trim_list.append(ee)
    G.remove_edges_from(edge_trim_list)
    return G
