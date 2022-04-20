"""
Illustrates and plots the selection of aggregates in AMG based on smoothed aggregation
"""

from __future__ import print_function
import sys

# Import necessary libraries
import numpy as np
import pyamg
import os
from mpi4py import MPI
from tacs import TACS, elements, constitutive, functions
from tmr import TMR, TopOptUtils
import matplotlib.pyplot as plt
import scipy.sparse as sp
import scipy.io as sio
import matplotlib as mplt
from scipy.sparse.linalg import cg
from scipy.io import savemat
import time
from generate_A import generateA
import scipy.sparse.linalg as splinalg


def _parse_options(options, default_options, default_solver):
    if options is None:
        options = default_options[default_solver]
    elif isinstance(options, str):
        options = default_options[options]
    else:
        assert 'type' in options and options['type'] in default_options \
            and options.keys() <= default_options[options['type']].keys()
        user_options = options
        options = default_options[user_options['type']]
        options.update(user_options)

    return options

def solver_options(
        tol=1e-9,
        maxiter=400,
        verb=False,
        rs_strength=('classical', {
            'theta': 0.25
        }),
        rs_CF='RS',  # 'RS' or 'ML'
        rs_presmoother=(
            'gauss_seidel', {
                'sweep': 'symmetric'
            }
        ),  # presmoothing function taking arguments (A,x,b), 'jacobi', 'gauss_seidel', 'block_gauss_seidel'
        rs_postsmoother=(
            'gauss_seidel', {
                'sweep': 'symmetric'
            }
        ),  # postsmoothing function taking arguments (A,x,b), 'jacobi', 'gauss_seidel', 'block_gauss_seidel'
        rs_max_levels=10,
        rs_max_coarse=500,
        rs_coarse_solver='pinv2',  # pinv2, pinv, or lu
        rs_cycle='V',  #  V, W, F, or G
        rs_accel='gmres',  # 'cg' or 'gmres'
        rs_tol=1e-12,
        rs_maxiter=100,
        sa_symmetry='hermitian',  # 'nonsymmetric', 'hermitian', 'antisymmetric'
        sa_strength='symmetric',  # the strength of connection: 'symmetric', 'classical', 'evolution'
        sa_aggregate='standard',  #  standard, naive, distance, or cg
        sa_smooth=(
            'jacobi', {
                'omega': 4.0 / 3.0
            }
        ),  # prolongation smoothing: 'jacobi', 'gauss_seidel', 'block_gauss_seidel'
        sa_presmoother=('block_gauss_seidel', {
            'sweep': 'symmetric'
        }),  # 'jacobi', 'gauss_seidel', 'block_gauss_seidel'
        sa_postsmoother=('block_gauss_seidel', {
            'sweep': 'symmetric'
        }),  # 'jacobi', 'gauss_seidel', 'block_gauss_seidel'
        sa_improve_candidates=(('block_gauss_seidel', {
            'sweep': 'symmetric',
            'iterations': 4
        }), None),  # 'jacobi', 'gauss_seidel', 'block_gauss_seidel'
        sa_max_levels=10,
        sa_max_coarse=500,
        sa_diagonal_dominance=False,  # True or False
        sa_coarse_solver='cholesky',  # ‘splu’, ‘lu’, ‘cholesky, ‘pinv’, ‘gauss_seidel’, ‘pinv2’
        sa_cycle='V',  #  V, W, F, or G
        sa_accel='gmres',  # 'cg' or 'gmres'
        sa_tol=1e-12,
        sa_maxiter=100,
        sa_keep=True,  # True or False
        asa_num_candidates=1,  # number of candidates to be considered for aggregation
        asa_symmetry='hermitian',  # 'nonsymmetric', 'hermitian', 'antisymmetric',
        asa_strength='evolution',  # the strength of connection: 'symmetric', 'classical', 'evolution'
        asa_aggregate='standard',  #  standard, naive, distance, or cg
        asa_smooth=(
            'jacobi', {
                'omega': 4.0 / 3.0,
                'degree': 2
            }
        ),  # prolongation smoothing: 'jacobi', 'gauss_seidel', 'block_gauss_seidel'
        asa_max_levels=10,  # maximum number of levels
        asa_max_coarse=5,  # maximum number of elements in the coarse grid
        asa_keep=True,  # keep the coarse grid or not
        asa_epsilon=1e-12,  # tolerance for the coarse grid
        asa_coarse_solver='pinv2',  #  pinv2, pinv, or cholesky
        asa_cycle='V',  #  V, W, F, or G
        asa_accel='gmres',  # 'cg' or 'gmres'
        asa_tol=1e-12,
        asa_maxiter=100):

    """Returns available solvers with default |solver_options| for the PyAMG backend.

    Parameters
    ----------
    tol
        Tolerance for PyAMG blackbox solver.maxiter
        Maximum iterations for PyAMG blackbox solver.verb
        Verbosity flag for PyAMG blackbox solver.rs_strength
        Parameter for PyAMG Ruge-Stuben solver.rs_CF
        Parameter for PyAMG Ruge-Stuben solver.rs_presmoother
        Parameter for PyAMG Ruge-Stuben solver.rs_postsmoother
        Parameter for PyAMG Ruge-Stuben solver.rs_max_levels
        Parameter for PyAMG Ruge-Stuben solver.rs_max_coarse
        Parameter for PyAMG Ruge-Stuben solver.rs_coarse_solver
        Parameter for PyAMG Ruge-Stuben solver.rs_cycle
        Parameter for PyAMG Ruge-Stuben solver.rs_accel
        Parameter for PyAMG Ruge-Stuben solver.rs_tol
        Parameter for PyAMG Ruge-Stuben solver.rs_maxiter
        Parameter for PyAMG Ruge-Stuben solver.sa_symmetry
        Parameter for PyAMG Smoothed-Aggregation solver.sa_strength
        Parameter for PyAMG Smoothed-Aggregation solver.sa_aggregate
        Parameter for PyAMG Smoothed-Aggregation solver.sa_smooth
        Parameter for PyAMG Smoothed-Aggregation solver.sa_presmoother
        Parameter for PyAMG Smoothed-Aggregation solver.sa_postsmoother
        Parameter for PyAMG Smoothed-Aggregation solver.sa_improve_candidates
        Parameter for PyAMG Smoothed-Aggregation solver.sa_max_levels
        Parameter for PyAMG Smoothed-Aggregation solver.sa_max_coarse
        Parameter for PyAMG Smoothed-Aggregation solver.sa_diagonal_dominance
        Parameter for PyAMG Smoothed-Aggregation solver.sa_coarse_solver
        Parameter for PyAMG Smoothed-Aggregation solver.sa_cycle
        Parameter for PyAMG Smoothed-Aggregation solver.sa_accel
        Parameter for PyAMG Smoothed-Aggregation solver.sa_tol
        Parameter for PyAMG Smoothed-Aggregation solver.sa_maxiter
        Parameter for PyAMG Smoothed-Aggregation solver.sa_keep
        Parameter for PyAMG Smoothed-Aggregation solver.
        Parameter for PyAMG Adaptive-Smoothed-Aggregation solver.asa_num_candidates
        Parameter for PyAMG Adaptive-Smoothed-Aggregation solver.asa_symmetry
        Parameter for PyAMG Adaptive-Smoothed-Aggregation solver.asa_strength
        Parameter for PyAMG Adaptive-Smoothed-Aggregation solver.asa_aggregate
        Parameter for PyAMG Adaptive-Smoothed-Aggregation solver.asa_smooth
        Parameter for PyAMG Adaptive-Smoothed-Aggregation solver.asa_max_levels
        Parameter for PyAMG Adaptive-Smoothed-Aggregation solver.asa_max_coarse
        Parameter for PyAMG Adaptive-Smoothed-Aggregation solver.asa_keep
        Parameter for PyAMG Adaptive-Smoothed-Aggregation solver.asa_epsilon
        Parameter for PyAMG Adaptive-Smoothed-Aggregation solver.asa_tol
        Parameter for PyAMG Adaptive-Smoothed-Aggregation solver.asa_maxiter
        Parameter for PyAMG Adaptive-Smoothed-Aggregation solver.asa_coarse_solver
        Parameter for PyAMG Adaptive-Smoothed-Aggregation solver.asa_cycle
        Parameter for PyAMG Adaptive-Smoothed-Aggregation solver.asa_accel

    Returns
    -------
    A dict of available solvers with default |solver_options|.
    """

    return {
        'direct': {
            'type': 'direct'
        },
        'pyamg_solve': {
            'type': 'pyamg_solve',
            'tol': tol,
            'maxiter': maxiter,
            'verb': verb
        },
        'pyamg_rs': {
            'type': 'pyamg_rs',
            'strength': rs_strength,
            'CF': rs_CF,
            'presmoother': rs_presmoother,
            'postsmoother': rs_postsmoother,
            'max_levels': rs_max_levels,
            'max_coarse': rs_max_coarse,
            'coarse_solver': rs_coarse_solver,
            'cycle': rs_cycle,
            'accel': rs_accel,
            'tol': rs_tol,
            'maxiter': rs_maxiter
        },
        'pyamg_sa': {
            'type': 'pyamg_sa',
            'symmetry': sa_symmetry,
            'strength': sa_strength,
            'aggregate': sa_aggregate,
            'smooth': sa_smooth,
            'presmoother': sa_presmoother,
            'postsmoother': sa_postsmoother,
            'improve_candidates': sa_improve_candidates,
            'max_levels': sa_max_levels,
            'max_coarse': sa_max_coarse,
            'diagonal_dominance': sa_diagonal_dominance,
            'coarse_solver': sa_coarse_solver,
            'cycle': sa_cycle,
            'accel': sa_accel,
            'tol': sa_tol,
            'maxiter': sa_maxiter,
            'keep': sa_keep
        },
        'pyamg_asa': {
            'type': 'pyamg_asa',
            'num_candidates': asa_num_candidates,
            'symmetry': asa_symmetry,
            'strength': asa_strength,
            'aggregate': asa_aggregate,
            'smooth': asa_smooth,
            'max_levels': asa_max_levels,
            'max_coarse': asa_max_coarse,
            'keep': asa_keep,
            'epsilon': asa_epsilon,
            'tol': asa_tol,
            'maxiter': asa_maxiter,
            'coarse_solver': asa_coarse_solver,
            'cycle': asa_cycle,
            'accel': asa_accel
        }
    }

def solveEqSystem(A, options=None, default_solver='pyamg_sa'):
    """Solve linear equation system.

    Parameters
    ----------
    A
        A matrix.
    options
        The |solver_options| to use (see :func:`solver_options`).
    least_squares
        If True, the system is solved with a least-squares solver for check.
    default_solver
        Default solver to use (pyamg_solve, pyamg_rs, pyamg_sa).

    Returns
    -------
    solution vectors x.
    """
    t0 = time.time()
    options = _parse_options(options, solver_options(), default_solver)
    # b = np.ones((A.shape[0], 1))
    b = np.random.rand(A.shape[0])
    x0 = np.random.rand(A.shape[0])
    res = []

    if options['type'] == 'direct':
        # Direct solver from scipy.sparse.linalg
        A = A.tocsr()
        x, ml= splinalg.spsolve(A, b), 0
    elif options['type'] == 'pyamg_solve':
        # Uses PyAMG blackbox solver, which implies that the program detect automatically the best solver parameters for the case.
        # Very useful for debugging and "difficult" to converge cases.
        # Does not allow to set max. iterations or this kind of stuff.
        x, ml = pyamg.solve(A,
                        b,
                        verb=options['verb'],
                        tol=options['tol'],
                        maxiter=options['maxiter'],
                        return_solver=True,
                        residuals=res)
    elif options['type'] == 'pyamg_rs':
        # Uses PyAMG Ruge-Stuben solver, which allows to set max. iterations and verbosity.
        # Create a multilevel solver using Classical AMG (Ruge-Stuben AMG).
        ml = pyamg.ruge_stuben_solver(A,
                                      strength=options['strength'],
                                      CF=options['CF'],
                                      presmoother=options['presmoother'],
                                      postsmoother=options['postsmoother'],
                                      max_levels=options['max_levels'],
                                      max_coarse=options['max_coarse'],
                                      coarse_solver=options['coarse_solver'])
        x = ml.solve(b,
                     tol=options['tol'],
                     maxiter=options['maxiter'],
                     cycle=options['cycle'],
                     accel=options['accel'],
                     residuals=res)
    elif options['type'] == 'pyamg_sa':
        # Uses PyAMG Smoothed-Aggregation solver, which allows to set max. iterations and verbosity.
        # Create a multilevel solver using Smoothed Aggregation (SA)
        ml = pyamg.smoothed_aggregation_solver(A,
                                               symmetry=options['symmetry'],
                                               strength=options['strength'],
                                               aggregate=options['aggregate'],
                                               smooth=options['smooth'],
                                               presmoother=options['presmoother'],
                                               postsmoother=options['postsmoother'],
                                               improve_candidates=options['improve_candidates'],
                                               max_levels=options['max_levels'],
                                               max_coarse=options['max_coarse'],
                                               diagonal_dominance=options['diagonal_dominance'],
                                               coarse_solver=options['coarse_solver'])
        x = ml.solve(b,
                     tol=options['tol'],
                     maxiter=options['maxiter'],
                     cycle=options['cycle'],
                     accel=options['accel'],
                     residuals=res)
    elif options['type'] == 'pyamg_asa':
        # Uses PyAMG Smoothed-Aggregation solver, which allows to set max. iterations and verbosity.
        # Create a multilevel solver using Smoothed Aggregation (SA)
        [ml, work] = pyamg.aggregation.adaptive_sa_solver(A,
                                                    num_candidates=options['num_candidates'],
                                                    symmetry=options['symmetry'],
                                                    strength=options['strength'],
                                                    aggregate=options['aggregate'],
                                                    smooth=options['smooth'],
                                                    max_levels=options['max_levels'],
                                                    max_coarse=options['max_coarse'],
                                                    keep=options['keep'],
                                                    epsilon=options['epsilon'],
                                                    coarse_solver=options['coarse_solver'])
        x = ml.solve(b,
                     x0=x0,
                     tol=options['tol'],
                     maxiter=options['maxiter'],
                     cycle=options['cycle'],
                     accel=options['accel'],
                     residuals=res)
    else:
        raise ValueError('Unknown solver type')

    t1 = time.time()
    timer = t1 - t0
    residual_norm = np.linalg.norm(b - A * x)
    return ml, res, residual_norm, timer

def test(A, solvers, options=None):
    fig, ax = plt.subplots(figsize=(8, 6), dpi=100)
    for solver in solvers:
        ml, res, residual_norm, timer = solveEqSystem(
            A, options=options, default_solver=solver)

        print("=======================")
        print('Solver: %s' % solver)

        # compute norm of residual vector
        print('Residual norm: %.2e' % residual_norm)
        print('Time: %.3f' % timer)
        print(ml)
        # print("The Multigrid Hierarchy")
        # print("-----------------------")
        # for l in range(len(ml.levels)):
        #     An = ml.levels[l].A.shape[0]
        #     Am = ml.levels[l].A.shape[1]
        #     if l == (len(ml.levels) - 1):
        #         print(f"A_{l}: {An:>10}x{Am:<10}")
        #     else:
        #         Pn = ml.levels[l].P.shape[0]
        #         Pm = ml.levels[l].P.shape[1]
        #         print(f"A_{l}: {An:>10}x{Am:<10}   P_{l}: {Pn:>10}x{Pm:<10}")

        ax.semilogy(res, '-o', label=solver)
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Relative Residual')
        ax.grid(True)
        plt.legend()
    plt.show()

if __name__ == '__main__':
    # Generate the matrix
    stencil = [[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]]
    A = pyamg.gallery.stencil_grid(
        stencil, (150, 150), dtype=float, format='bsr')

    # A = generateA()

    # Set the solvers
    solvers = ["direct", "pyamg_solve", "pyamg_rs", "pyamg_sa", "pyamg_asa"]
    # solvers = ["pyamg_asa"]
    test(A, solvers)
