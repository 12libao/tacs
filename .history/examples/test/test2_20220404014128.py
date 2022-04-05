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

n = A.shape[0]
X, Y = np.meshgrid(np.linspace(0, 1, n), np.linspace(0, 1, n)) 
# pick a random right hand side
b = np.random.rand(A.shape[0])

# ------------------------------------------------------------------
# Step 3: setup of the multigrid hierarchy
# ------------------------------------------------------------------
ml = pyamg.smoothed_aggregation_solver(A)   # construct the multigrid hierarchy

# ------------------------------------------------------------------
# Step 4: solve the system
# ------------------------------------------------------------------
res1 = []
# solve Ax=b to a tolerance of 1e-12
x = ml.solve(b, tol=1e-12, residuals=res1)

# ------------------------------------------------------------------
# Step 5: print details
# ------------------------------------------------------------------
print("\n")
print("Details: Default AMG")
print("--------------------")
print(ml)                                 # print hierarchy information

# compute norm of residual vector
print("The residual norm is {}".format(np.linalg.norm(b - A * x)))

# notice that there are 5 (or maybe 6) levels in the hierarchy
#
# we can look at the data in each of the levels
# e.g. the multigrid components on the finest (0) level
#      A: operator on level 0
#      P: prolongation operator mapping from level 1 to level 0
#      R: restriction operator mapping from level 0 to level 1
#      B: near null-space modes for level 0
#      presmoother: presmoothing function taking arguments (A,x,b)
#      postsmoother: postsmoothing function taking arguments (A,x,b)
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

# ------------------------------------------------------------------
# Step 6: change the hierarchy
# ------------------------------------------------------------------
# we can also change the details of the hierarchy
ml = pyamg.smoothed_aggregation_solver(A,  # the matrix
                                       # the representation of the near null space (this is a poor choice)
                                       B=X.reshape(n * n, 1),
                                       BH=None,                           # the representation of the left near null space
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
                                                           {'sweep': 'symmetric', 'iterations': 4}), None],
                                       max_levels=10,                     # maximum number of levels
                                       max_coarse=5,                      # maximum number on a coarse level
                                       keep=False)                        # keep extra operators around in the hierarchy (memory)

# ------------------------------------------------------------------
# Step 7: print details
# ------------------------------------------------------------------
# keep the residual history in the solve
res2 = []
# solve Ax=b to a tolerance of 1e-12
x = ml.solve(b, tol=1e-12, residuals=res2)
print("\n")
print("Details: Specialized AMG")
print("------------------------")
# print hierarchy information
print(ml)
# compute norm of residual vector
print("The residual norm is {}".format(np.linalg.norm(b - A * x)))
print("\n")

# ------------------------------------------------------------------
# Step 8: plot convergence history
# ------------------------------------------------------------------
fig, ax = plt.subplots()
ax.semilogy(res1, label='Default AMG solver')
ax.semilogy(res2, label='Specialized AMG solver')
ax.set_xlabel('Iteration')
ax.set_ylabel('Relative Residual')
ax.grid(True)
plt.legend()

figname = f'./output/amg_convergence.png'
if '--savefig' in sys.argv:
    plt.savefig(figname, bbox_inches='tight', dpi=150)
else:
    plt.show()


# # pick a random right hand side & construct the multigrid hierarchy
# ml = pyamg.smoothed_aggregation_solver(A)

# # using amg smoothed aggregation to solve the system
# res1 = []
# x = ml.solve(b, tol=1e-12, residuals=res1)

# print("\n")
# print("Details: Default AMG")
# print("--------------------")
# print(ml)                                 # print hierarchy information

# # compute norm of residual vector
# print("The residual norm is {}".format(np.linalg.norm(b - A * x)))

# # cgs = pyamg.coarse_grid_solver('lu')
# # print(cgs)
# # x = asa.solve(b=np.ones((A.shape[0],)), x0=np.ones(
# #     (A.shape[0],)), residuals=residuals, accel='cg')

# print((res1[-1]/res1[0])**(1.0/(len(res1)-1)))
# plt.semilogy(res1/res1[0], 'o-')
# plt.xlabel('iterations')
# plt.ylabel('normalized residual')
# plt.show()

