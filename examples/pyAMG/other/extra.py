from mpi4py import MPI
from egads4py import egads
from tmr import TMR

# Set the communicator
comm = MPI.COMM_WORLD

# Load the model from the STEP file
# geo = TMR.LoadModel('airfoil_39_3d.stp')
geo = TMR.LoadModel('airfoil_39_3d.stp')
vols = geo.getVolumes()
faces = geo.getFaces()
edges = geo.getEdges()
verts = geo.getVertices()

# Set all of the matching faces
TMR.setMatchingFaces(geo)

# Create the geometry
geo = TMR.Model(verts, edges, faces, vols)

# Create the geometry
mesh = TMR.Mesh(comm, geo)

# Mesh the part
opts = TMR.MeshOptions()
opts.num_smoothing_steps = 10
opts.write_mesh_quality_histogram = 1
opts.triangularize_print_iter = 500

# Mesh the geometry with the given target size
htarget = 4
mesh.mesh(htarget, opts=opts)

# Write the surface mesh to a file
mesh.writeToVTK('mesh_tacs.vtk', 'hex')
mesh.writeToBDF('mesh_tacs.bdf', 'hex', bcs=None)

# # Create the model from the unstructured volume mesh
# model = mesh.createModelFromMesh()

# # Create the corresponding mesh topology from the mesh-model
# topo = TMR.Topology(comm, model)

# # Create the quad forest and set the topology of the forest
# forest = TMR.OctForest(comm)
# forest.setTopology(topo)

# # Create random trees and balance the mesh. Print the output file
# forest.createRandomTrees(nrand=1, max_lev=3)
# forest.balance(1)
# filename = 'mesh_forest_1%d.vtk'%(comm.rank)
# forest.writeForestToVTK(filename)
