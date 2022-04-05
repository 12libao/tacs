from mpi4py import MPI
from tmr import TMR, TopOptUtils
from paropt import ParOpt
from tacs import TACS, elements, constitutive, functions
import numpy as np
import argparse
import os


class OctCreator(TMR.OctConformTopoCreator):
    """
    An instance of an OctCreator class.

    This creates discretization for a Largange type filter, where the density is
    interpolated from the nodes of a coarser finite-element mesh. In this type of
    creator, the filter element mesh and the octree element mesh need be the same.
    (In a conformal filter, they must have the same element mesh but may have
    different degree of approximation.)
    """

    def __init__(self, bcs, filt, props=None):
        TMR.OctConformTopoCreator.__init__(bcs, filt)
        self.props = props

        # Create the constitutive object - one for the entire mesh
        self.con = TMR.OctConstitutive(props=props, forest=filt)

        # Create the model (the type of physics we're using)
        self.model = elements.LinearElasticity3D(self.con)

        # Set the basis functions and create the element
        self.basis = elements.LinearHexaBasis()
        self.element = elements.Element3D(self.model, self.basis)

        return

    def createElement(self, order, octant, index, weights):
        """
        Create the element for the given octant.

        This callback provides the global indices for the filter mesh and the weights
        applied to each nodal density value to obtain the element density. The
        local octant is also provided (but not used here).

        Args:
            order (int): Order of the underlying mesh
            octant (Octant): The TMR.Octant class
            index (list): List of the global node numbers referenced by the element
            weights (list): List of weights to compute the element density

        Returns:
            TACS.Element: Element for the given octant
        """
        return self.element


class CreatorCallback:
    def __init__(self, bcs, props):
        self.bcs = bcs
        self.props = props

    def creator_callback(self, forest):
        """
        Create the creator class and filter for the provided OctForest object.

        This is called for every mesh level when the topology optimization
        problem is created.

        Args:
            forest (OctForest): The OctForest for this mesh level

        Returns:
            OctTopoCreator, OctForest: The creator and filter for this forest
        """
        creator = OctCreator(self.bcs, forest, props=self.props)
        return creator, forest


def create_forest(comm, depth, htarget=5.0, filename='airfoil_39.stp'):
    """
    Create an initial forest for analysis and optimization

    This code loads in the model, sets names, meshes the geometry and creates
    a QuadForest from the mesh. The forest is populated with quadtrees with
    the specified depth.

    Args:
        comm (MPI_Comm): MPI communicator
        depth (int): Depth of the initial trees
        htarget (float): Target global element mesh size

    Returns:
        OctForest: Initial forest for topology optimization
    """
    # Load the geometry model
    geo = TMR.LoadModel(filename)

    # Mark the boundary condition faces
    vols = geo.getVolumes()
    faces = geo.getFaces()
    edges = geo.getEdges()
    verts = geo.getVertices()

    # Set all of the matching faces
    TMR.setMatchingFaces(geo)
    # Create the geometry
    geo = TMR.Model(verts, edges, faces, vols)

    # Create the mesh
    mesh = TMR.Mesh(comm, geo)

    # Set the meshing options
    # Mesh the part
    opts = TMR.MeshOptions()
    opts.num_smoothing_steps = 10
    opts.write_mesh_quality_histogram = 1
    opts.triangularize_print_iter = 500

    # Mesh the geometry with the given target size
    htarget = 5
    mesh.mesh(htarget, opts=opts)

    # Create the surface mesh
    mesh.mesh(htarget, opts)

    # Create a model from the mesh
    model = mesh.createModelFromMesh()

    # Create the corresponding mesh topology from the mesh-model
    topo = TMR.Topology(comm, model)

    # Create the quad forest and set the topology of the forest
    forest = TMR.OctForest(comm)
    forest.setTopology(topo)

    # Create the trees, rebalance the elements and repartition
    forest.createTrees(depth)

    return forest


class OutputCallback:
    def __init__(self, assembler, iter_offset=0):
        self.fig = None
        self.assembler = assembler
        self.xt = self.assembler.createDesignVec()

        # Set the output file name
        flag = (TACS.OUTPUT_CONNECTIVITY |
                TACS.OUTPUT_NODES |
                TACS.OUTPUT_EXTRAS)
        self.f5 = TACS.ToFH5(self.assembler, TACS.SOLID_ELEMENT, flag)
        self.iter_offset = iter_offset

        return

    def write_output(self, prefix, itr, oct_forest, quad_forest, x):

        self.f5.writeToFile(os.path.join(
            prefix, 'output%d.f5' % (itr + self.iter_offset)))

        self.assembler.getDesignVars(self.xt)
        TMR.writeSTLToBin(os.path.join(prefix, 'level_set_output%d.bstl' % (itr + self.iter_offset)),
                          oct_forest, self.xt)

        return


class MFilterCreator:
    def __init__(self, r0_frac, N, a=0.1):
        self.a = a
        self.r0_frac = r0_frac
        self.N = N

    def filter_callback(self, assemblers, filters):
        """
        Create and initialize a filter with the specified parameters
        """
        # Find the characteristic length of the domain and set the filter length scale
        r0 = self.r0_frac*self.a
        mfilter = TopOptUtils.Mfilter(self.N, assemblers, filters, dim=3, r=r0)
        mfilter.initialize()
        return mfilter


def create_assembler(forest, bcs, props, nlevels, vol_frac=0.25,
                   density=2600.0, iter_offset=0):
    """
    Create the TMRTopoProblem object and set up the topology optimization problem.

    This code is given the forest, boundary conditions, material properties and
    the number of multigrid levels. Based on this info, it creates the TMRTopoProblem
    and sets up the mass-constrained compliance minimization problem. Before
    the problem class is returned it is initialized so that it can be used for
    optimization.

    Args:
        forest (OctForest): Forest object
        bcs (BoundaryConditions): Boundary condition object
        props (StiffnessProperties): Material properties object
        nlevels (int): number of multigrid levels
        vol_frac (float): Volume fraction for the mass constraint
        density (float): Density to use for the mass computation
        iter_offset (int): iteration counter offset

    Returns:
        TopoProblem: Topology optimization problem instance
    """

    # Characteristic length of the domain
    len0 = 10.0
    r0_frac = 0.05
    N = 20

    # Create the problem and filter object
    mfilter = MFilterCreator(r0_frac, N, a=len0)
    filter_type = mfilter.filter_callback
    obj = CreatorCallback(bcs, props)
    problem = TopOptUtils.createTopoProblem(forest, obj.creator_callback,
                                            filter_type, use_galerkin=True,
                                            nlevels=nlevels)

    # Get the assembler object we just created
    assembler = problem.getAssembler()

    return assembler

asfdwew
# Set the communicator
comm = MPI.COMM_WORLD

nlevels = 4  # Number of multigrid levels
forest = create_forest(comm, nlevels-1)

# Set the boundary conditions for the problem
bcs = TMR.BoundaryConditions()
bcs.addBoundaryCondition('fixed')

# Create the material properties
material_properties = constitutive.MaterialProperties(rho=2600.0, E=70e9, nu=0.3, ys=350e6)
props = TMR.StiffnessProperties(material_properties, q=8.0)

# Create tacs assembler object
assembler = create_assembler(forest, bcs, props, nlevels)

# Assemble the Jacobian matrix
res = assembler.createVec()
mat = assembler.createFEMat()
assembler.assembleJacobian(1.0, 0.0, 0.0, res, mat)
mat_list = mat.getMat()
A = mat_list[0]
print(A)

