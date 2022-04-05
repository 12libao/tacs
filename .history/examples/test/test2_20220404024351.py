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

# CRM_box_2nd.bdf
# Load structural mesh from BDF file
bdfFile = os.path.join(os.path.dirname(__file__), 'CRM_box_2nd.bdf')
tacs_comm = MPI.COMM_WORLD
struct_mesh = TACS.MeshLoader(tacs_comm)
struct_mesh.scanBDFFile(bdfFile)

# Set constitutive properties
rho = 2500.0  # density, kg/m^3
E = 70e9  # elastic modulus, Pa
nu = 0.3  # poisson's ratio
kcorr = 5.0 / 6.0  # shear correction factor
ys = 350e6  # yield stress, Pa
min_thickness = 0.002
max_thickness = 0.20
thickness = 0.02

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
print(f'Number of nonzeros: {A.nnz} \n'
      f'Shape of the matrix: {A.shape[0]} x {A.shape[1]} \n'
      f'Format of the matrix: {A.format}\n')

# pick a random right hand side & construct the multigrid hierarchy

B = np.ones((A.shape[0], 1))
ml = pyamg.smoothed_aggregation_solver(A, B, BH=None,                     # the representation of the left near null space
                                       symmetry='hermitian',              # indicate that the matrix is Hermitian
                                       strength='evolution',              # change the strength of connection
                                       aggregate='standard',              # use a standard aggregation method
                                       # prolongation smoothing
                                       smooth=(
                                           'jacobi', {'omega': 4.0 / 3.0, 'degree': 2}),
                                       presmoother=('block_gauss_seidel', {
                                                    'sweep': 'symmetric'}),
                                       postsmoother=('block_gauss_seidel', {
                                                     'sweep': 'symmetric'}),
                                       improve_candidates=[('block_gauss_seidel',
                                                           {'sweep': 'symmetric', 'iterations': 10}), None],
                                       max_levels=10,                     # maximum number of levels
                                       max_coarse=5,                      # maximum number on a coarse level
                                       keep=False)

# [ml, work] = pyamg.aggregation.adaptive_sa_solver(A, num_candidates=1, symmetry='hermitian',              # indicate that the matrix is Hermitian
#                                                 strength='evolution',              # change the strength of connection
#                                                 aggregate='standard',              # use a standard aggregation method
#                                                 # prolongation smoothing
#                                                 smooth=(
#                                                     'jacobi', {'omega': 4.0 / 3.0, 'degree': 2}),
#                                                 max_levels=10,                     # maximum number of levels
#                                                 max_coarse=5,                      # maximum number on a coarse level
#                                                   keep=True, epsilon=1e-12)

residuals = []
b = np.random.rand(A.shape[0])
x0 = np.random.rand(A.shape[0])
x = ml.solve(b=b, x0=x0, tol=1e-12, residuals=residuals,
             accel='cg', maxiter=1000, cycle='V')

M = ml.aspreconditioner(cycle='V')


print("\n")
print("Details: Default AMG")
print("--------------------")
print(ml)                                 # print hierarchy information

# compute norm of residual vector
print("The residual norm is {}".format(np.linalg.norm(b - A * x)))

print("\n")
print("The Multigrid Hierarchy")
print("-----------------------")
for l in range(len(ml.levels)):
    An = ml.levels[l].A.shape[0]
    Am = ml.levels[l].A.shape[1]
    if l == (len(ml.levels)-1):
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

# sio.savemat('test.mat', {'mydata': A0})

# data = sio.loadmat('test.mat')

# A = data['A'].tocsr()                              # matrix
# V = data['vertices'][:A.shape[0]]                  # vertices of each variable
# E = np.vstack((A.tocoo().row, A.tocoo().col)).T  # edges of the matrix graph

# # Create a multigrid solver
# ml = pyamg.smoothed_aggregation_solver(
#     A, max_levels=2, max_coarse=1, keep=True)

# # AggOp[i,j] is 1 iff node i belongs to aggregate j
# AggOp = ml.levels[0].AggOp

# # determine which edges lie entirely inside an aggregate
# # AggOp.indices[n] is the aggregate to which vertex n belongs
# inner_edges = AggOp.indices[E[:, 0]] == AggOp.indices[E[:, 1]]
# outer_edges = ~inner_edges

# # set up a figure
# fig, ax = plt.subplots()

# # non aggregate edges
# nonaggs = V[E[outer_edges].ravel(), :].reshape((-1, 2, 2))
# col = mplt.collections.LineCollection(nonaggs,
#                                       color=[232.0/255, 74.0/255, 39.0/255],
#                                       linewidth=1.0)
# ax.add_collection(col, autolim=True)

# # aggregate edges
# aggs = V[E[inner_edges].ravel(), :].reshape((-1, 2, 2))
# col = mplt.collections.LineCollection(aggs,
#                                       color=[19.0/255, 41.0/255, 75.0/255],
#                                       linewidth=4.0)
# ax.add_collection(col, autolim=True)

# ax.autoscale_view()
# ax.axis('equal')

# figname = './output/aggregates.png'
# if len(sys.argv) > 1:
#     if sys.argv[1] == '--savefig':
#         plt.savefig(figname, bbox_inches='tight', dpi=150)
# else:
#     plt.show()
