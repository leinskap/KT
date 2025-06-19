import pygmsh
from math import pi, cos
import pygmsh

with pygmsh.occ.Geometry() as geom:
    geom.characteristic_length_max = 0.1
    r = 0.5
    disks = [
        geom.add_disk([-0.5 * cos(7 / 6 * pi), -0.25], 1.0),
        geom.add_disk([+0.5 * cos(7 / 6 * pi), -0.25], 1.0),
        geom.add_disk([0.0, 0.5], 1.0),
    ]
    geom.boolean_intersection(disks)

    mesh = geom.generate_mesh()
# # Create a geometry object
# geom = pygmsh.occ.Geometry(
#     characteristic_length_min=0.1,
#     characteristic_length_max=0.1
# )

# Define a rectangle as the base geometry
rectangle = geom.add_rectangle([-1.0, -1.0, 0.0], 2.0, 2.0)

# Add a line to represent the crack
crack_start = geom.add_point([-0.5, 0.0, 0.0])
crack_end = geom.add_point([0.5, 0.0, 0.0])
crack_line = geom.add_line(crack_start, crack_end)

# Boolean operations can be tricky for cracks; focus on meshing around the crack line
domain = geom.add_rectangle([-1.0, -1.0, 0.0], 2.0, 2.0)
crack_domain = geom.add_rectangle([-0.5, -0.01, 0.0], 1.0, 0.02)
domain = geom.cut([(2, domain)], [(2, crack_domain)])

# Synchronize and generate mesh
geom.synchronize()
mesh = pygmsh.generate_mesh(geom, dim=2)

# Add a distance field around the crack
gmsh.model.mesh.field.add("Distance", 1)
gmsh.model.mesh.field.setNumber(1, "NodesList", [crack_start.id, crack_end.id])

# Add a threshold field to refine the mesh
gmsh.model.mesh.field.add("Threshold", 2)
gmsh.model.mesh.field.setNumber(2, "IField", 1)
gmsh.model.mesh.field.setNumber(2, "LcMin", 0.01)
gmsh.model.mesh.field.setNumber(2, "LcMax", 0.1)

# Set the threshold field as the background field
gmsh.model.mesh.field.add("Min", 3)
gmsh.model.mesh.field.setNumbers(3, "FieldsList", [2])
gmsh.model.mesh.field.setAsBackgroundMesh(3)

import dolfin

# Load the mesh
mesh = dolfin.Mesh("cracked_mesh.msh")

# Define function spaces
V = dolfin.FunctionSpace(mesh, "CG", 1)

# Define boundary conditions and material properties
u = dolfin.TrialFunction(V)
v = dolfin.TestFunction(V)

# Phase field variable
phi = dolfin.Function(V)

# Energy functional
E = (1 - phi)**2 * E0  # Degraded elasticity

# Weak form of the phase field equation
F_phi = (E * phi.dx(0) * v.dx(0) + E * phi.dx(1) * v.dx(1)) * dolfin.dx

# Solve the phase field equation
dolfin.solve(F_phi == 0, phi)


res_file = File('crack/solution.pvd')

# Time-stepping
u = dolfin.Function(V)
t = 0
num_steps  = 10
for n in range(num_steps):

    # Update current time
    t += dt
    u_D.t = t

    # Compute solution
    solve(F_phi == 0, u, bc)

    # Save solution to VTK
    res_file << u

    # Update previous solution
    u_n.assign(u)
