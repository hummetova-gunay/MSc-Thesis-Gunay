from mpi4py import MPI
import numpy as np
import ufl
from basix.ufl import element, mixed_element
from dolfinx import default_real_type, log, plot
from dolfinx.fem import Function, functionspace
from dolfinx.fem.petsc import NonlinearProblem
from dolfinx.io import XDMFFile
from dolfinx.mesh import CellType, create_unit_square
from dolfinx.nls.petsc import NewtonSolver
import pyvista as pv
from petsc4py import PETSc
import os 

try:
    import pyvista as pv
    from dolfinx import plot
    have_pyvista = True
except ImportError:
    have_pyvista = False

# Create mesh
msh = create_unit_square(MPI.COMM_WORLD, 96, 96, CellType.triangle)
P1 = element("Lagrange", msh.basix_cell(), 1, dtype=default_real_type)
ME = functionspace(msh, mixed_element([P1, P1]))

# Parameters
lmbda = 1.0e-02  # surface parameter
dt = 2.0e-06     # reduced time step for better stability
theta = 0.5      # theta method parameter, time stepping family 

# Define test and trial functions
q, v = ufl.TestFunctions(ME) 
u = Function(ME)   # current solution
u0 = Function(ME)  # solution from previous step

# Split mixed functions
c, mu = ufl.split(u)
c0, mu0 = ufl.split(u0)

# FIXED: Interpolate initial condition for concentration ONLY
# Use smaller perturbation to stay within bounds
rng = np.random.default_rng(42)
u.sub(0).interpolate(lambda x: 0.5 + 0.01 * (0.5 - rng.random(x.shape[1])))

# FIXED: Initialize chemical potential to zero (sub(1) is mu)
u.sub(1).interpolate(lambda x: np.zeros(x.shape[1]))

# Synchronize across processes
u.x.scatter_forward()

# Copy initial condition to u0
u0.x.array[:] = u.x.array[:]

# FIXED: Define chemical potential correctly
c_var = ufl.variable(c)
f = 100 * c_var**2 * (1 - c_var)**2
dfdc = ufl.diff(f, c_var)

# Theta-weighted chemical potential
mu_mid = (1.0 - theta) * mu0 + theta * mu

# FIXED: Weak formulation with proper time discretization
# Concentration equation: dc/dt + div(grad(mu)) = 0
F0 = (
    ufl.inner((c - c0) / dt, q) * ufl.dx + 
    ufl.inner(ufl.grad(mu_mid), ufl.grad(q)) * ufl.dx
)

# Chemical potential equation: mu = df/dc - lambda * laplacian(c)
F1 = (
    ufl.inner(mu, v) * ufl.dx - 
    ufl.inner(dfdc, v) * ufl.dx - 
    lmbda * ufl.inner(ufl.grad(c), ufl.grad(v)) * ufl.dx
)

F = F0 + F1

# Create nonlinear problem and Newton solver
problem = NonlinearProblem(F, u)
solver = NewtonSolver(MPI.COMM_WORLD, problem)
solver.convergence_criterion = "incremental"
solver.rtol = np.sqrt(np.finfo(default_real_type).eps) * 1e-2

# Configure linear solver
ksp = solver.krylov_solver
opts = PETSc.Options()
option_prefix = ksp.getOptionsPrefix()
opts[f"{option_prefix}ksp_type"] = "preonly"
opts[f"{option_prefix}pc_type"] = "lu"

# Prefer MUMPS for factorization
sys = PETSc.Sys()
use_superlu = PETSc.IntType == np.int64
if sys.hasExternalPackage("mumps") and not use_superlu:
    opts[f"{option_prefix}pc_factor_mat_solver_type"] = "mumps"
elif sys.hasExternalPackage("superlu_dist"):
    opts[f"{option_prefix}pc_factor_mat_solver_type"] = "superlu_dist"
ksp.setFromOptions()

# Output file
file = XDMFFile(MPI.COMM_WORLD, "binary.xdmf", "w")
file.write_mesh(msh)

# Time stepping
t = 0.0
if "CI" in os.environ.keys() or "GITHUB_ACTIONS" in os.environ.keys():
    T = 3 * dt
else:
    T = 50 * dt

# Get sub-space for concentration
V0, dofs = ME.sub(0).collapse()

# FIXED: Prepare visualization with proper initial data
if have_pyvista:
    topology, cell_types, x = plot.vtk_mesh(V0)
    grid = pv.UnstructuredGrid(topology, cell_types, x)
    
    # Set initial concentration data
    c_values = u.x.array[dofs].real
    grid.point_data["c"] = c_values
    grid.set_active_scalars("c")
    
    print(f"Initial concentration range: [{c_values.min():.4f}, {c_values.max():.4f}]")

# Get concentration function for output
c_func = u.sub(0)

# Write initial condition
file.write_function(c_func, t)

# Time stepping loop
step = 0
while t < T:
    t += dt
    step += 1
    
    # Solve nonlinear system
    r = solver.solve(u)
    print(f"Step {step}: time = {t:.2e}, iterations = {r[0]}")
    
    # Update previous solution
    u0.x.array[:] = u.x.array[:]
    
    # Write solution
    file.write_function(c_func, t)
    
    # Print concentration statistics for debugging
    c_values = u.x.array[dofs].real
    print(f"  Concentration range: [{c_values.min():.4f}, {c_values.max():.4f}]")

file.close()

# # Final visualization
# if have_pyvista:
#     u.x.scatter_forward()
#     c_values = u.x.array[dofs].real
#     grid.point_data["c"] = c_values
    
#     print(f"\nFinal concentration range: [{c_values.min():.4f}, {c_values.max():.4f}]")
    
#     screenshot = None
#     if pv.OFF_SCREEN:
#         screenshot = "cahn_hilliard_final.png"
    
#     # Create plotter for better control
#     plotter = pv.Plotter(title="Cahn-Hilliard: Final Concentration")
#     plotter.add_mesh(grid, show_edges=True, clim=[c_values.min(), c_values.max()], 
#                      scalar_bar_args={'title': 'Concentration'})
    
#     if screenshot:
#         plotter.screenshot(screenshot)
#     else:
#         plotter.show()