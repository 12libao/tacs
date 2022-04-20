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

def generateA():
    # CRM_box_2nd.bdf
    # Load structural mesh from BDF file
    bdfFile = os.path.join(os.path.dirname(__file__), './mesh_file/CRM_box_2nd.bdf')
    tacs_comm = MPI.COMM_WORLD
    struct_mesh = TACS.MeshLoader(tacs_comm)
    struct_mesh.scanBDFFile(bdfFile)

    # Set constitutive properties
    rho = 2500.0  # density, kg/m^3
    E = 70e9  # elastic modulus, Pa
    nu = 0.3  # poisson's ratio
    kcorr = 5.0 / 6.0  # shear correction factor
    ys = 350e6  # yield stress, Pa
    min_thickness = 0.02
    max_thickness = 0.20
    thickness = 0.01

    # Loop over components, creating stiffness and element object for each
    num_components = struct_mesh.getNumComponents()
    for i in range(num_components):
        descriptor = struct_mesh.getElementDescript(i)
        # Setup (isotropic) property and constitutive objects
        prop = constitutive.MaterialProperties(rho=rho, E=E, nu=nu, ys=ys)
        # Set one thickness dv for every component
        stiff = constitutive.IsoShellConstitutive(
            prop, t=thickness, tMin=min_thickness, tMax=max_thickness, tNum=i)

        element = None
        transform = None
        if descriptor in ["CQUAD", "CQUADR", "CQUAD4"]:
            element = elements.Quad4Shell(transform, stiff)
        struct_mesh.setElement(i, element)

    # Create tacs assembler object from mesh loader
    assembler = struct_mesh.createTACS(6)

    # Assemble the Jacobian matrix on the refined mesh
    res = assembler.createVec()
    mat = assembler.createMat()
    assembler.assembleJacobian(1.0, 0.0, 0.0, res, mat)
    mat_list = mat.getMat()
    A = mat_list[0]

    # stencil = [[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]]
    # A = pyamg.gallery.stencil_grid(
    #     stencil, (1000, 1000), dtype=float, format='bsr')

    fig = plt.figure(figsize=(10, 10), dpi=100)
    plt.spy(A, precision=0.1, markersize=0.01, color="blue", marker=".")
    # plt.show()
    plt.savefig('./output/spy_A.png')
    print(f'Number of nonzeros: {A.nnz} \n'
        f'Shape of the matrix: {A.shape[0]} x {A.shape[1]} \n'
        f'Format of the matrix: {A.format}\n')
    print(A)

    # pick a random right hand side & construct the multigrid hierarchy
    B = np.ones((A.shape[0], 1))
    # ml = pyamg.smoothed_aggregation_solver(A, B, BH=None,                     # the representation of the left near null space
    #                                     symmetry='hermitian',              # indicate that the matrix Hermitian
    #                                     strength='evolution',              # change the strength of connection
    #                                     aggregate='standard',              # use a standard aggregation method
    #                                     # prolongation smoothing
    #                                     smooth=('jacobi', {'omega': 4.0 / 3.0, 'degree': 2}),
    #                                     presmoother=('block_gauss_seidel', {'sweep': 'symmetric'}),
    #                                     postsmoother=('block_gauss_seidel', {'sweep': 'symmetric'}),
    #                                     improve_candidates=[('block_gauss_seidel', {'sweep': 'symmetric', 'iterations': 10}), None],
    #                                     max_levels=10,                     # maximum number of levels
    #                                     max_coarse=5,                      # maximum number on a coarse level
    #                                     keep=False)

    [ml, work] = pyamg.aggregation.adaptive_sa_solver(
        A,
        num_candidates=1,
        symmetry='hermitian',  # indicate that the matrix is Hermitian
        strength='evolution',  # change the strength of connection
        aggregate='standard',  # use a standard aggregation method
        # prolongation smoothing
        smooth=('jacobi', {
            'omega': 4.0 / 3.0,
            'degree': 2
        }),
        max_levels=10,  # maximum number of levels
        max_coarse=5,  # maximum number on a coarse level
        keep=True,
        epsilon=1e-12)

    residuals = []
    b = np.random.rand(A.shape[0])
    x0 = np.random.rand(A.shape[0])
    # Krylov solvers (cg, bicgstab, gmres, fgmres, cgnr)
    x = ml.solve(b=b,
                 x0=x0,
                 tol=1e-12,
                 residuals=residuals,
                 accel='gmres',
                 maxiter=100,
                 cycle='V')

    # M = ml.aspreconditioner(cycle='W')
    # x, info = cg(A, b, atol=1e-8,  maxiter=300, M=M)

    print("\n")
    print("Details: Default AMG")
    print("--------------------")
    print(ml)  # print hierarchy information

    # compute norm of residual vector
    print("The residual norm is {}".format(np.linalg.norm(b - A * x)))

    print("\n")
    print("The Multigrid Hierarchy")
    print("-----------------------")
    for l in range(len(ml.levels)):
        An = ml.levels[l].A.shape[0]
        Am = ml.levels[l].A.shape[1]
        if l == (len(ml.levels) - 1):
            print(f"A_{l}: {An:>10}x{Am:<10}")
        else:
            Pn = ml.levels[l].P.shape[0]
            Pm = ml.levels[l].P.shape[1]
            print(f"A_{l}: {An:>10}x{Am:<10}   P_{l}: {Pn:>10}x{Pm:<10}")

    fig, ax = plt.subplots()
    # ax.semilogy(res1, label='Default AMG solver')
    ax.semilogy(residuals, label='Specialized AMG solver')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Relative Residual')
    ax.grid(True)
    plt.legend()

    figname = f'./output/amg_convergence.png'
    if '--savefig' in sys.argv:
        plt.savefig(figname, bbox_inches='tight', dpi=150)
    else:

        plt.show()

    return A

if __name__ == "__main__":
    a = generateA()