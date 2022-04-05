from __future__ import print_function   

# Import necessary libraries
import numpy as np
import pyamg
import os
from mpi4py import MPI
from tacs import TACS, elements, constitutive, functions
from tmr import TMR, TopOptUtils
import matplotlib.pyplot as plt
import scipy.sparse as sp

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
ml = pyamg.smoothed_aggregation_solver(A, B, smooth='energy')
print(ml)

residuals = []
b = np.random.rand(A.shape[0])
x0 = np.random.rand(A.shape[0])

x = ml.solve(b=b, x0=x0, tol=1e-10, residuals=residuals)
print((residuals[-1]/residuals[0])**(1.0/(len(residuals)-1)))

x = ml.solve(b=b, x0=x0, tol=1e-10, residuals=residuals, accel='cg')
print((residuals[-1]/residuals[0])**(1.0/(len(residuals)-1)))

plt.semilogy(residuals/residuals[0], 'o-')
plt.xlabel('iterations')
plt.ylabel('normalized residual')
plt.show()

# retrieve the problem
A0 = A.tocsr()
# vertices of each variable for A
V = A[:A.shape[0]]
# values of each variable for A
E2V = nP.vstack((A.tocoo().row, A.tocoo().col)).T

# perform smoothed aggregation
AggOp, rootnodes = pyamg.aggregation.standard_aggregation(A0)

# create the vtk file of aggregates
pyamg.vis.vis_coarse.vis_aggregate_groups(V=V, E2V=E2V, AggOp=AggOp,
                                          mesh_type='tri', fname='output_aggs.vtu')

# create the vtk file for a mesh
pyamg.vis.vtk_writer.write_basic_mesh(V=V, E2V=E2V,
                                      mesh_type='tri', fname='output_mesh.vtu')

try:
    import vedo
    gmesh = vedo.load('output_mesh.vtu')
    gaggs = vedo.load('output_aggs.vtu')

    gmesh = gmesh.tomesh().color('w').alpha(0.1)
    gmesh.color('gray')
    gmesh.lw(3.0)

    agg3 = []
    agg2 = []
    for cell in gaggs.cells():
        if len(cell) == 2:
            agg2.append(cell)
        else:
            agg3.append(cell)

    mesh2 = vedo.Mesh([gaggs.points(), agg2])
    mesh3 = vedo.Mesh([gaggs.points(), agg3])
    mesh2.lineColor('b').lineWidth(8)
    mesh3.color('b').lineWidth(0)

    figname = './output/vis_aggs2.png'
    import sys
    if len(sys.argv) > 1:
        if sys.argv[1] == '--savefig':
            plt = vedo.Plotter(offscreen=True)
            plt += gmesh
            plt += mesh2
            plt += mesh3
            plt.show().screenshot(figname)
    else:
        plt = vedo.Plotter()
        plt += gmesh
        plt += mesh2
        plt += mesh3
        plt.show()
except:
    pass
