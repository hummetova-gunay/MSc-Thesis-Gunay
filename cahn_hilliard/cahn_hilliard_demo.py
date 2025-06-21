from mpi4py import MPI
import numpy as np
import ufl
from basix.ufl import element, mixed_element
"""
Cahn-Hilliard equation is 4th order PDE, we split it into two 2nd order PDEs. Now we have two variables c (concentration) and mu (chemical potential)
V=V_c*V_μ
mixed_element creates this product space so we can solve for both variables simultaneously.
"""
from dolfinx import default_real_type, log, plot
from dolfinx.fem import Function, functionspace
from dolfinx.fem.petsc import NonlinearProblem
from dolfinx.io import XDMFFile
from dolfinx.mesh import CellType, create_unit_square
from dolfinx.nls.petsc import NewtonSolver
"""The Cahn-Hilliard equation is nonlinear, because of the term f'(c)=c^3-c.SO linear solvers won't work here"""
import pyvista as pv
from petsc4py import PETSc
import os 
try:
    import pyvista as pv
    # import pyvistaqt as pvqt  
    from dolfinx import plot
    have_pyvista = True
except ImportError:
    have_pyvista = False


msh = create_unit_square(MPI.COMM_WORLD, 96, 96, CellType.triangle) #If you omit CellType.triangle, FEniCSx will raise an error or you’ll have to provide a default yourself.
P1 = element("Lagrange", msh.basix_cell(), 1, dtype=default_real_type)
"""Basix is the finite element backend library used by FEniCSx to represent finite elements.
 Msh.basix_cell() tells the element function exactly what cell type your mesh uses, so it can create the right reference element."""
ME = functionspace(msh, mixed_element([P1, P1]))

lmbda = 1.0e-02  # surface parameter (related to interface thickness)
"""lambda - interfacial energy, higher lambda wider interfaces, smoother transitions between phases. Lower λ → sharper interfaces (but harder numerically — requires finer mesh and smaller time steps)."""
dt = 5.0e-06  # time step
"""Time step size, Appears in the time discretization of ∂c/∂t. Controls how far you "step forward" in time at each iteration."""
theta = 0.5 
"""The theta-method approximates the time derivative. And evaluates the right-hand side (RHS) of your PDE at a weighted average in time.
You want to predict tomorrow's value based on today's data. You can use only today's info (θ=0), only tomorrow's estimate (θ=1), or a bit of both (θ=0.5). 
The more you lean toward the future (θ → 1), the more stable your prediction becomes — but you have to solve harder equations (implicit).
In stiff systems like Cahn-Hilliard, this is worth it.
"""

# does it make any difference if we change time stepping family from Crank to euler?
"""θ=1 is Backward Euler (fully implicit, first order accurate).
θ=0.5 is Crank-Nicolson (semi-implicit trapezoidal, second order accurate).
θ=0 is Forward Euler (explicit, usually unstable for stiff problems).
"""

q, v = ufl.TestFunctions(ME) 

u = Function(ME)  # current solution
u0 = Function(ME)  # solution from previous converged step

# Split mixed functions
c, mu = ufl.split(u)
c0, mu0 = ufl.split(u0)

# Zero u
u.x.array[:] = 0.0

# Interpolate initial condition
rng = np.random.default_rng(42) #random number generator
u.sub(0).interpolate(lambda x: 0.63 + 0.02 * (0.5 - rng.random(x.shape[1])))
"""
u.sub(0)- Access first component of mixed function u (e.g., concentration)
"""
u.x.scatter_forward() #Synchronize distributed DoFs across MPI processes

# Zero u
u.x.array[:] = 0.0 #Reset all function DoFs to zero (clears previous data)

# Compute the chemical potential df/dc
c = ufl.variable(c)
f = 100 * c**2 * (1 - c) ** 2
dfdc = ufl.diff(f, c)

# mu_(n+theta)
mu_mid = (1.0 - theta) * mu0 + theta * mu # the theta-weighted average of chemical potential between steps

# Weak statement of the equations
F0 = (
    ufl.inner(c, q) * ufl.dx - ufl.inner(c0, q) * ufl.dx + dt * ufl.inner(ufl.grad(mu_mid), ufl.grad(q)) * ufl.dx
) #inner dot product 
F1 = (
    ufl.inner(mu, v) * ufl.dx - ufl.inner(dfdc, v) * ufl.dx - lmbda * ufl.inner(ufl.grad(c), ufl.grad(v)) * ufl.dx
)
F = F0 + F1

# Create nonlinear problem and Newton solver
problem = NonlinearProblem(F, u)
solver = NewtonSolver(MPI.COMM_WORLD, problem)
solver.convergence_criterion = "incremental" #at each iteration, it computes an update step δu, and stops iterating when this update is "small enough."
solver.rtol = np.sqrt(np.finfo(default_real_type).eps) * 1e-2
"""
rtol- relative tolerance.
np.finfo(default_real_type).eps gives machine epsilon — the smallest number. The solver stops when δu/u < rtol
"""

# We can customize the linear solver used inside the NewtonSolver by
# modifying the PETSc options
ksp = solver.krylov_solver
opts = PETSc.Options()  # type: ignore
option_prefix = ksp.getOptionsPrefix()
opts[f"{option_prefix}ksp_type"] = "preonly"
opts[f"{option_prefix}pc_type"] = "lu"
sys = PETSc.Sys()  # type: ignore
# For factorisation prefer MUMPS, then superlu_dist, then default
use_superlu = PETSc.IntType == np.int64  # or PETSc.ScalarType == np.complex64
if sys.hasExternalPackage("mumps") and not use_superlu:
    opts[f"{option_prefix}pc_factor_mat_solver_type"] = "mumps"
elif sys.hasExternalPackage("superlu_dist"):
    opts[f"{option_prefix}pc_factor_mat_solver_type"] = "superlu_dist"
ksp.setFromOptions()

# Output file
file = XDMFFile(MPI.COMM_WORLD, "demo_ch.xdmf", "w")
file.write_mesh(msh)

# Step in time
t = 0.0

#  Reduce run time if on test (CI) server
if "CI" in os.environ.keys() or "GITHUB_ACTIONS" in os.environ.keys():
    T = 3 * dt
else:
    T = 50 * dt

# Get the sub-space for c and the corresponding dofs in the mixed space
# vector
V0, dofs = ME.sub(0).collapse()

# Prepare viewer for plotting the solution during the computation
if have_pyvista:
    # Create a VTK 'mesh' with 'nodes' at the function dofs
    topology, cell_types, x = plot.vtk_mesh(V0)
    grid = pv.UnstructuredGrid(topology, cell_types, x)

    # Set output data
    grid.point_data["c"] = u.x.array[dofs].real
    grid.set_active_scalars("c")

    # p = pvqt.BackgroundPlotter(title="concentration", auto_update=True)
    # p.add_mesh(grid, clim=[0, 1])
    # p.view_xy(True)
    # p.add_text(f"time: {t}", font_size=12, name="timelabel")  # <--- all commented as you asked

c = u.sub(0)
u0.x.array[:] = u.x.array
while t < T:
    t += dt
    r = solver.solve(u)
    print(f"Step {int(t / dt)}: num iterations: {r[0]}")
    u0.x.array[:] = u.x.array
    file.write_function(c, t)

    # Update the plot window
    # if have_pyvista:
        # p.add_text(f"time: {t:.2e}", font_size=12, name="timelabel")
        # grid.point_data["c"] = u.x.array[dofs].real
        # p.app.processEvents()

file.close()

# Update ghost entries and plot
if have_pyvista:
    u.x.scatter_forward()
    grid.point_data["c"] = u.x.array[dofs].real
    screenshot = None
    if pv.OFF_SCREEN:
        screenshot = "c.png"
    pv.plot(grid, show_edges=True, screenshot=screenshot)
