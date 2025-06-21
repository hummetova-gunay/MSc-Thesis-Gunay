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

msh = create_unit_square(MPI.COMM_WORLD, 96, 96, CellType.triangle)
P1 = element("Lagrange", msh.basix_cell(), 1, dtype=default_real_type)
# Mixed element: c1, c2, mu1, mu2 (c3 = 1 - c1 - c2)
ME = functionspace(msh, mixed_element([P1, P1, P1, P1]))

lmbda = 1.0e-02 
dt = 2.0e-06   
theta = 0.5      
chi12 = 2.0      # interaction parameter between components 1 and 2
chi13 = 2.0      # interaction parameter between components 1 and 3
chi23 = 2.0      # interaction parameter between components 2 and 3
M11 = 1.0        
M22 = 1.0        
M12 = 0.0        # cross-mobility (coupling between fluxes)

q1, q2, v1, v2 = ufl.TestFunctions(ME) 
u = Function(ME)  
u0 = Function(ME)  


c1, c2, mu1, mu2 = ufl.split(u)
c1_0, c2_0, mu1_0, mu2_0 = ufl.split(u0)


c3 = 1.0 - c1 - c2
c3_0 = 1.0 - c1_0 - c2_0

# random perturbations around c1=c2=c3=1/3
rng = np.random.default_rng(42)
u.sub(0).interpolate(lambda x: 1.0/3.0 + 0.02 * (0.5 - rng.random(x.shape[1])))  # c1
u.sub(1).interpolate(lambda x: 1.0/3.0 + 0.02 * (0.5 - rng.random(x.shape[1])))  # c2
u.sub(2).interpolate(lambda x: np.zeros(x.shape[1]))  # mu1
u.sub(3).interpolate(lambda x: np.zeros(x.shape[1]))  # mu2

def fix_concentrations(u_func):
    """Ensure c1, c2 > 0 and c1 + c2 < 1"""
    # Get DOF indices for c1 and c2
    V0, dofs0 = ME.sub(0).collapse()  
    V1, dofs1 = ME.sub(1).collapse()  
    
    c1_vals = u_func.x.array[dofs0].real
    c2_vals = u_func.x.array[dofs1].real
    
    c1_vals = np.maximum(c1_vals, 0.01)
    c2_vals = np.maximum(c2_vals, 0.01)
    
    # c1 + c2 < 0.95 (leave room for c3)
    total = c1_vals + c2_vals
    mask = total > 0.95
    scale_factor = 0.95 / total[mask]
    c1_vals[mask] *= scale_factor
    c2_vals[mask] *= scale_factor
    
    # Update the function
    u_func.x.array[dofs0] = c1_vals
    u_func.x.array[dofs1] = c2_vals

fix_concentrations(u)

u.x.scatter_forward()

u0.x.array[:] = u.x.array[:]

# free energy derivatives for ternary Flory-Huggins model
# f = c1*ln(c1) + c2*ln(c2) + c3*ln(c3) + chi12*c1*c2 + chi13*c1*c3 + chi23*c2*c3
eps = 1e-8  # regularization to avoid log(0)

dfdc1 = (ufl.ln(c1 + eps) - ufl.ln(c3 + eps) + 
         chi12 * c2 + chi13 * c3 - chi12 * c1 - chi13 * c1)

dfdc2 = (ufl.ln(c2 + eps) - ufl.ln(c3 + eps) + 
         chi12 * c1 + chi23 * c3 - chi12 * c2 - chi23 * c2)


# Theta-weighted chemical potentials
mu1_mid = (1.0 - theta) * mu1_0 + theta * mu1
mu2_mid = (1.0 - theta) * mu2_0 + theta * mu2

# Weak formulation
# Cahn-Hilliard equations for c1 and c2
F0 = (
    ufl.inner((c1 - c1_0) / dt, q1) * ufl.dx + 
    ufl.inner(M11 * ufl.grad(mu1_mid) + M12 * ufl.grad(mu2_mid), ufl.grad(q1)) * ufl.dx
)

F1 = (
    ufl.inner((c2 - c2_0) / dt, q2) * ufl.dx + 
    ufl.inner(M12 * ufl.grad(mu1_mid) + M22 * ufl.grad(mu2_mid), ufl.grad(q2)) * ufl.dx
)

# Chemical potential equations
F2 = (
    ufl.inner(mu1, v1) * ufl.dx - 
    ufl.inner(dfdc1, v1) * ufl.dx - 
    lmbda * ufl.inner(ufl.grad(c1), ufl.grad(v1)) * ufl.dx
)

F3 = (
    ufl.inner(mu2, v2) * ufl.dx - 
    ufl.inner(dfdc2, v2) * ufl.dx - 
    lmbda * ufl.inner(ufl.grad(c2), ufl.grad(v2)) * ufl.dx
)

F = F0 + F1 + F2 + F3

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
file = XDMFFile(MPI.COMM_WORLD, "ternary.xdmf", "w")
file.write_mesh(msh)

# Time stepping
t = 0.0
if "CI" in os.environ.keys() or "GITHUB_ACTIONS" in os.environ.keys():
    T = 3 * dt
else:
    T = 50 * dt  # Run longer to see phase separation

# Get sub-spaces for concentrations
V0, dofs0 = ME.sub(0).collapse()  # c1
V1, dofs1 = ME.sub(1).collapse()  # c2

if have_pyvista:
    topology, cell_types, x = plot.vtk_mesh(V0)
    grid = pv.UnstructuredGrid(topology, cell_types, x)
    
    # Set initial concentration data
    c1_values = u.x.array[dofs0].real
    c2_values = u.x.array[dofs1].real
    c3_values = 1.0 - c1_values - c2_values
    
    grid.point_data["c1"] = c1_values
    grid.point_data["c2"] = c2_values
    grid.point_data["c3"] = c3_values
    grid.set_active_scalars("c1")
    
    print(f"Initial c1 range: [{c1_values.min():.4f}, {c1_values.max():.4f}]")
    print(f"Initial c2 range: [{c2_values.min():.4f}, {c2_values.max():.4f}]")
    print(f"Initial c3 range: [{c3_values.min():.4f}, {c3_values.max():.4f}]")

# Get concentration functions for output
c1_func = u.sub(0)
c2_func = u.sub(1)

# Create c3 function for output
V_single = functionspace(msh, P1)
c3_func = Function(V_single)

# Write initial condition
c1_func.name = "c1"
c2_func.name = "c2"
c3_func.name = "c3"

# Calculate c3 for initial output
c3_func.x.array[:] = 1.0 - u.x.array[dofs0] - u.x.array[dofs1]

file.write_function(c1_func, t)
file.write_function(c2_func, t)
file.write_function(c3_func, t)

# Time stepping loop
step = 0
while t < T:
    t += dt
    step += 1
    
    # Solve nonlinear system
    try:
        r = solver.solve(u)
        print(f"Step {step}: time = {t:.2e}, iterations = {r[0]}")
        
        # Fix concentrations to ensure physical bounds
        fix_concentrations(u)
        
        # Update previous solution
        u0.x.array[:] = u.x.array[:]
        
        # Calculate c3 for output
        c3_func.x.array[:] = 1.0 - u.x.array[dofs0] - u.x.array[dofs1]
        
        # Write solution every 10 steps
        if step % 10 == 0:
            file.write_function(c1_func, t)
            file.write_function(c2_func, t)
            file.write_function(c3_func, t)
        
        # Print concentration statistics for debugging
        c1_values = u.x.array[dofs0].real
        c2_values = u.x.array[dofs1].real
        c3_values = c3_func.x.array.real
        
        print(f"  c1 range: [{c1_values.min():.4f}, {c1_values.max():.4f}]")
        print(f"  c2 range: [{c2_values.min():.4f}, {c2_values.max():.4f}]")
        print(f"  c3 range: [{c3_values.min():.4f}, {c3_values.max():.4f}]")
        
        # Check mass conservation
        total_mass = c1_values.mean() + c2_values.mean() + c3_values.mean()
        print(f"  Total mass: {total_mass:.6f} (should be ~1.0)")
        
    except Exception as e:
        print(f"Solver failed at step {step}: {e}")
        break

file.close()

print("\nSimulation completed!")


# Final visualization
if have_pyvista:
    u.x.scatter_forward()
    c1_values = u.x.array[dofs0].real
    c2_values = u.x.array[dofs1].real
    c3_values = 1.0 - c1_values - c2_values
    
    grid.point_data["c1"] = c1_values
    grid.point_data["c2"] = c2_values  
    grid.point_data["c3"] = c3_values
    
    print(f"\nFinal c1 range: [{c1_values.min():.4f}, {c1_values.max():.4f}]")
    print(f"Final c2 range: [{c2_values.min():.4f}, {c2_values.max():.4f}]")
    print(f"Final c3 range: [{c3_values.min():.4f}, {c3_values.max():.4f}]")