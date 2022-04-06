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

    return A

if __name__ == "__main__":
    a = generateA()