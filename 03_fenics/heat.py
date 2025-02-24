from fenics import *
import dolfin

T = 4.0            # final time
num_steps = 100     # number of time steps
dt = T / num_steps # time step size
alpha = 3          # parameter alpha
beta = 1.2         # parameter beta

# Create mesh and define function space
circle_x = 0.5
circle_y = 0.5
circle_r = 0.25


nx = ny = 8
from dolfin import *
from mshr import *
import numpy as np
import matplotlib.pyplot as plt


L = 8
R = 2
N = 50

domain = Rectangle(Point(0., 0.), Point(L, L)) - Circle(Point(4, 4), R, 100)
mesh = generate_mesh(domain, N)

V = FunctionSpace(mesh, 'P', 1)

# Define boundary condition
u_D = Expression('1 + x[0]*x[0] + alpha*x[1]*x[1] + beta*t', degree=2, alpha=alpha, beta=beta, t=0)

def boundary(x, on_boundary):
    return on_boundary

bc = DirichletBC(V, u_D, boundary)

# Define initial value
u_n = interpolate(u_D, V)
#u_n = project(u_D, V)

# Define variational problem
u = TrialFunction(V)
v = TestFunction(V)
f = Constant(beta - 2 - 1000*alpha)

F = u * v * dx + dt * dot ( grad(u), grad(v) ) * dx - (u_n + dt*f)*dx
# F = v*u*dx + dt*dot(grad(u), grad(v))*dx - (u_n + dt*f)*dx
# F = (u + dt*f) * v * dx + u * v * dx + dt * dot ( grad(u), grad(v) ) * dx
a, L = lhs(F), rhs(F)




res_file = File('heat/solution.pvd')

# Time-stepping
u = Function(V)
t = 0
for n in range(num_steps):

    # Update current time
    t += dt
    u_D.t = t

    # Compute solution
    solve(a == L, u, bc)

    # Save solution to VTK
    res_file << u

    # Update previous solution
    u_n.assign(u)
