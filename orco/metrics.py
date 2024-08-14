"""
    Functions for comparing distributions
"""

from functools import lru_cache
import numpy as np
import cvxpy as cvx
import ot
import heapq

from scipy.optimize import linprog
from scipy import sparse
import scipy

from orco.util import logger

linprog_status = {0:'Optimization proceeding nominally.',
                  1:'Iteration limit reached.',
                  2:'Problem appears to be infeasible.',
                  3:'Problem appears to be unbounded.',
                  4:'Numerical difficulties encountered.'}


def OTD_linprog_unbalanced(x, y, d, gamma=0.1, log=False):
    """
    INCORRECT!
    Compute the unbalanced optimal transportation distance (OTD) 
    of the given density distributions by linprog.

    Parameters:
    ------------
    x : (m,) numpy.ndarray
        Source's density distributions, includes source and source's neighbors.
    y : (n,) numpy.ndarray
        Target's density distributions, includes source and source's neighbors.
    d : (m, n) numpy.ndarray
        Shortest path matrix.
    gamma : float
        Value to use for unbalanced cost of source/sink
    log : boolean, optional (default=False)
        If True, returns a dictionary containing the cost and optimization information.
        Otherwise returns only the optimal transportation cost.        

    Returns:
    --------
    m : float 
        Optimal transportation distance. 
    msg : dict
        If log is True, a Dictionary with information about the 
        optimization results and convergence.

        msg fields:
        ----------
        x : 1-D array
                The values of the decision variables that minimizes the
                objective function while satisfying the constraints.
        fun : float
                The optimal value of the objective function ``c @ x``.
        success : bool
                ``True`` when the algorithm succeeds in finding an optimal
                solution.
        status : int
                An integer representing the exit status of the algorithm.
    
                ``0`` : Optimization terminated successfully.
                ``1`` : Iteration limit reached.
                ``2`` : Problem appears to be infeasible.
                ``3`` : Problem appears to be unbounded.
                ``4`` : Numerical difficulties encountered.
        
        nit : int
                The total number of iterations performed in all phases.
        message : str
                A string descriptor of the exit status of the algorithm.
    
    """
    n_x, n_y = x.shape[0], y.shape[0]
    assert n_x == d.shape[0], "Distance matrix does not match the size of x."
    assert n_y == d.shape[1], "Distance matrix does not match the size of y."    
    A = sparse.hstack((sparse.eye(n_x), np.zeros((n_x, 1))))
    for j in range(n_y):
        A = sparse.hstack((A, sparse.eye(n_x), np.zeros((n_x, 1))))

    A_eq = sparse.vstack((A, sparse.hstack((sparse.block_diag([np.ones((1, n_y+1))]*n_x), np.zeros((n_x, n_y+1))))))

    b_eq = np.concatenate((x, y))

    C = np.vstack((np.hstack((d, gamma*np.ones((d.shape[0], 1)))), gamma*np.ones((1, d.shape[1]+1))))
    C[-1, -1] = 0.
    
    C = C.flatten(order='F')

    # positivity constraint
    lb = [(0, None)]*len(C)
    w = linprog(C, A_eq=A_eq, b_eq=b_eq, bounds=lb)
    # xx = np.reshape(res.x,(n_x+1,n_y+1))
    if not w.success:
        logger.error(f"linprog algorithm was not successfully completed. \Status message:: ({w.status}) {linprog_status.get(w.status)}")
        w_distance = np.nan
    else:
        w_distance = w.fun

    if log:
        msg = {'x': w.x,
               'fun': w.fun,
               'success': w.success,
               'status': w.status,
               'nit': w.nit,
               'message': w.message}
        return w_distance, msg
        
    return w_distance
    
    
def OTD_linprog(x, y, d, log=False):
    """Compute the optimal transportation distance (OTD) 
    of the given density distributions by linprog.

    Parameters:
    ------------
    x : (m,) numpy.ndarray
        Source's density distributions, includes source and source's neighbors.
    y : (n,) numpy.ndarray
        Target's density distributions, includes source and source's neighbors.
    d : (m, n) numpy.ndarray
        Shortest path matrix.
    log : boolean, optional (default=False)
        If True, returns a dictionary containing the cost and optimization information.
        Otherwise returns only the optimal transportation cost.

    Returns:
    --------
    m : float 
        Optimal transportation distance. 
    msg : dict
        If log is True, a Dictionary with information about the 
        optimization results and convergence.
        msg fields:
        ----------
        x : 1-D array
                The values of the decision variables that minimizes the
                objective function while satisfying the constraints.
        fun : float
                The optimal value of the objective function ``c @ x``.
        success : bool
                ``True`` when the algorithm succeeds in finding an optimal
                solution.
        status : int
                An integer representing the exit status of the algorithm.
    
                ``0`` : Optimization terminated successfully.
                ``1`` : Iteration limit reached.
                ``2`` : Problem appears to be infeasible.
                ``3`` : Problem appears to be unbounded.
                ``4`` : Numerical difficulties encountered.
        
        nit : int
                The total number of iterations performed in all phases.
        message : str
                A string descriptor of the exit status of the algorithm.
    """
    n_x, n_y = x.shape[0], y.shape[0]
    assert n_x == d.shape[0], "Distance matrix does not match the size of x."
    assert n_y == d.shape[1], "Distance matrix does not match the size of y."

    """
    A = np.eye(n_x)
    for j in range(n_y-1):
        A = np.hstack((A,np.eye(n_x)))

    A_eq = np.vstack((A,scipy.linalg.block_diag(*[np.ones((1,n_x))]*n_y)))        
    """
    # 
    A = sparse.eye(n_x)
    for j in range(n_y-1):
        A = sparse.hstack((A, sparse.eye(n_x)))

    A_eq = sparse.vstack((A, scipy.linalg.block_diag(*[np.ones((1, n_x))]*n_y)))
    #
    b_eq = np.concatenate((x, y))

    C = d.flatten(order='F')                   

    # positivity constraint
    lb = (0, None)  # [(0,None)]*len(C)

    w = linprog(C, A_eq=A_eq, b_eq=b_eq, bounds=lb)
    # xx = np.reshape(res.x,(n_x+1,n_y+1))
    
    if not w.success:
        logger.error(f"linprog algorithm was not successfully completed. \Status message:: ({w.status}) {linprog_status.get(w.status)}")
        w_distance = np.nan
    else:
        w_distance = w.fun

    if log:
        msg = {'x': w.x,
               'fun': w.fun,
               'success': w.success,
               'status': w.status,
               'nit': w.nit,
               'message': w.message}
        return w_distance, msg
        
    return w_distance


def OTD_KRduality_linprog(x, y, d, log=False):
    """Compute the optimal transportation distance (OTD) 
    using Kantorovich-Rubenstein Duality
    of the given density distributions by linprog.

    Parameters:
    ------------
    x : (m,) numpy.ndarray
        Source's density distributions, includes source and source's neighbors.
    y : (n,) numpy.ndarray
        Target's density distributions, includes source and source's neighbors.
    d : (m, n) numpy.ndarray
        Shortest path matrix.
    log : boolean, optional (default=False)
        If True, returns a dictionary containing the cost and optimization information.
        Otherwise returns only the optimal transportation cost.

    Returns:
    --------
    m : float 
        Optimal transportation distance. 
    msg : dict
        If log is True, a Dictionary with information about the 
        optimization results and convergence.
        msg fields:
        ----------
        x : 1-D array
                The values of the decision variables that minimizes the
                objective function while satisfying the constraints.
        fun : float
                The optimal value of the objective function ``c @ x``.
        success : bool
                ``True`` when the algorithm succeeds in finding an optimal
                solution.
        status : int
                An integer representing the exit status of the algorithm.
    
                ``0`` : Optimization terminated successfully.
                ``1`` : Iteration limit reached.
                ``2`` : Problem appears to be infeasible.
                ``3`` : Problem appears to be unbounded.
                ``4`` : Numerical difficulties encountered.
        
        nit : int
                The total number of iterations performed in all phases.
        message : str
                A string descriptor of the exit status of the algorithm.
    """
    assert len(y) == len(x), 'OMT KRduality must have distributions of the same size'
    p, n = x-y, len(x)
    
    assert d.shape == (n, n), "Distance matrix does not match the size of x"
    A_ub = np.vstack((createA_KR(n), -createA_KR(n)))
    A_ub[np.isclose(A_ub, 0)] = 0.0
    
    b_ub = np.hstack([d[np.where(np.triu(np.ones(d.shape, dtype=bool), 1))]]*2)  # upper triangular d values - row-wise

    bounds = (None, None)  # [(0,None)]*len(C)

    w = linprog(-p, A_ub=A_ub, b_ub=b_ub, bounds=bounds)
    
    if not w.success:
        logger.error(f"linprog algorithm was not successfully completed. \Status message:: ({w.status}) {linprog_status.get(w.status)}")
        w_distance = np.nan
    else:
        w_distance = -w.fun

    if log:
        msg = {'x': w.x,
               'fun': w.fun,
               'success': w.success,
               'status': w.status,
               'nit': w.nit,
               'message': w.message}
        return w_distance, msg
        
    return w_distance


def OTD_KRduality2(x, y, d):
    """Compute the optimal transportation distance (OTD) using Kantorovich-Rubenstein Duality
    of the given density distributions by CVXPY.
    
    *** Recommended not ot use until verified, use OTD_KRduality instead ***

    Parameters:
    ------------
    x : (m,) numpy.ndarray
        Source's density distributions, includes source and source's neighbors.
    y : (n,) numpy.ndarray
        Target's density distributions, includes source and source's neighbors.
    d : (m, n) numpy.ndarray
        Shortest path matrix.

    Returns:
    --------
    m : float 
        Optimal transportation distance. 
    """
    assert len(y) == len(x), 'OMT KRduality must have distributions of the same size'
    p = x-y
    n = len(x)
    B = np.hstack((-np.ones((n-1, 1)), np.eye(n-1)))
    A,BB = B.copy(), B.copy()
    for cn in range(n-1):
        BB[:, [cn, cn+1]] = BB[:, [cn+1, cn]]
        # B = np.vstack((B,np.roll(B, -1, axis=1)))
        A = np.vstack((A, BB))
    b = d[np.where(~np.eye(d.shape[0], dtype=bool))]  # off diagonal d values - row-wise
    u = cvx.Variable(n)
    obj = cvx.Maximize(u.T @ p)
    constraints = [A @ u <= b]
    prob = cvx.Problem(obj, constraints)
    m = prob.solve()
    # m = prob.solve(solver='ECOS', abstol=1e-6,verbose=True)
    # m = prob.solve(solver='OSQP', max_iter=100000,verbose=False)
    return m


def createA_KR(n):
    """ create A_eq for KR-duality optimization """
    A = np.hstack((np.ones((n-1, 1)), np.diag(-np.ones(n-1))))
    for cn in range(n-1):
        A = np.vstack((A, np.hstack((np.zeros((n-cn-2, cn+1)), np.ones((n-cn-2, 1)), np.diag(-np.ones(n-cn-2))))))
    return A


def OTD_KRduality(x, y, d):
    """Compute the optimal transportation distance (OTD) using Kantorovich-Rubenstein Duality
    of the given density distributions by CVXPY. (*works better than other KRduality methods)

    Parameters:
    ------------
    x : (m,) numpy.ndarray
        Source's density distributions, includes source and source's neighbors.
    y : (n,) numpy.ndarray
        Target's density distributions, includes source and source's neighbors.
    d : (m, n) numpy.ndarray
        Shortest path matrix.

    Returns:
    --------
    m : float 
        Optimal transportation distance. 
    """
    assert len(y) == len(x), 'OMT KRduality must have distributions of the same size'
    p, n = x-y, len(x)
    
    A = createA_KR(n)
    b = d[np.where(np.triu(np.ones(d.shape, dtype=bool), 1))] # upper triangular d values - row-wise
    
    u = cvx.Variable(n)
    obj = cvx.Maximize(u.T @ p)
    constraints = [A @ u <= b, A @ u >= -b]
    # constraints = [A @ u <= b] 
    prob = cvx.Problem(obj, constraints)
    m = prob.solve()
    return m


def optimal_transportation_distance(x, y, d, solvr=None):
    """ Compute the optimal transportation distance (OTD) of the given density distributions by CVXPY. 
    Parameters:
    -----------
    x : (m,) numpy.ndarray  
        Source's density distributions, includes source and source's neighbors.
    y : (n,) numpy.ndarray
        Target's density distributions, includes source and source's neighbors.
    d : (m, n) numpy.ndarray
        Shortest path matrix. 

    Returns:
    ---------
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


def OTD(x, y, d, solvr=None):
    """ Compute the optimal transportation distance (OTD) of the given density distributions 
    trying first with POT package and then by CVXPY. 

    Parameters:
    -----------
    x : (m,) numpy.ndarray  
        Source's density distributions, includes source and source's neighbors.
    y : (n,) numpy.ndarray
        Target's density distributions, includes source and source's neighbors.
    d : (m, n) numpy.ndarray
        Shortest path matrix. 

    Returns:
    ---------
    m : float
        Optimal transportation distance. 
    """
    try:
        wasserstein_distance, lg = ot.emd2(x, y, d, log=True)
        if lg['warning'] is not None:
            logger.info(f"POT library failed: warning = {lg['warning']}, retry with explicit computation")
            wasserstein_distance = optimal_transportation_distance(x, y, d)
    except cvx.error.SolverError:
        logger.info("OTD failed, retry with SCS solver")
        wasserstein_distance = optimal_transportation_distance(x, y, d, solvr='SCS')
    return wasserstein_distance


def dist_cvx(drho, D):
    """ Dual of the dual OMT problem with flux.

    Parameters
    ----------
    drho : numpy array
        Difference between measures (rho0 - rho1) with m elements.
    D : numy array
        Signed incidence matrix of size m x n, where n is the number of edges

    Returns
    -------
    emd : float
        EMD solution.
    """
    
    # Create two scalar optimization variables.  
    m = D.shape[1]
    u = cvx.Variable((m))
    # Form objective.
    objective = cvx.Minimize(cvx.sum(cvx.abs(u)))
    
    # Create two constraints.
    constraints = [drho-D@u == 0]
    
    # Form and solve problem.
    problem = cvx.Problem(objective, constraints)
    problem.solve()  # solver='OSQP') # 'SCS') # verbose=True)

    return problem.value
