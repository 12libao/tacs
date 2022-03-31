from __future__ import print_function
import os
from pprint import pprint

import numpy as np
from mpi4py import MPI

from tacs import functions, constitutive, elements, pyTACS, TACS

comm = MPI.COMM_WORLD

# Optional arguments: these to output the f5 file to visualize the solution and which element type to use
structOptions = {
    'writeSolution': True,
    'outputElement': TACS.PLANE_STRESS_ELEMENT,
    'printtiming': True
}

# Name of the bdf file to get the mesh
bdfFile = os.path.join(os.path.dirname(__file__), 'mesh_tacs.bdf')

# Instantiate the pyTACS object
FEAAssembler = pyTACS(bdfFile, comm, options=structOptions)
