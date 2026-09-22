import numpy as np
import traceback
import pyvista
from dolfinx.io import VTXWriter
from dolfinx.io import gmsh as gmshio
import gmsh 
from petsc4py import PETSc
from mpi4py import MPI
from ufl import *
from dolfinx.fem import *
from dolfinx.fem.petsc import *
from dolfinx.mesh import *
from dolfinx import default_scalar_type, plot, default_real_type
from dolfinx.fem.petsc import (
    create_matrix, assemble_matrix, apply_lifting, 
    assemble_vector, set_bc, LinearProblem, create_vector
)
from basix.ufl import element, mixed_element

# gmsh model.geo -2 -o model.msh

file_name = "hertzian_mon1" 
linesearch = 'bt'
mesh_order = 1
target_load = -600
target_node = (1.1, 2.0)

# Nonlinear solver parameters
atol_newton = 1e-18; rtol_newton = 1e-17; stol_newton = 1e-20; max_it_newton = 10

# Adaptive load stepping parameters
Nsteps = 4000; Nsteps_min = 25; Nsteps_max = 40000
initial_step_size = abs(target_load)/Nsteps
min_step_size = abs(target_load)/Nsteps_max
max_step_size = abs(target_load)/Nsteps_min 
increase_factor = 2.0; decrease_factor = 0.5
max_attempts = 10

# Parameters for the model
mu_c = default_scalar_type(35000/13); K_c = default_scalar_type(17500/3) # Cylinder
mu_e = default_scalar_type(700000/29); K_e = default_scalar_type(700000/3) # Elastic half space
gamma = default_scalar_type(1.3e-3)
beta_1 = default_scalar_type(0.01); beta_2 = default_scalar_type(1.0e-5)

# Mesh
geo_name         = "hertzian0_4"
mesh_file_name   = geo_name + ".msh"
mesh_data = gmshio.read_from_msh(mesh_file_name, MPI.COMM_WORLD, 0, gdim=2)
mesh0 = mesh_data.mesh
cell_tags = mesh_data.cell_tags
facet_tags = mesh_data.facet_tags

metadata = {"quadrature_degree": 4}
dx = Measure("dx", domain=mesh0, subdomain_data=cell_tags, metadata=metadata)
ds = Measure("ds", domain=mesh0, subdomain_data=facet_tags, metadata=metadata)
bottom_facet = facet_tags.find(10)

# Parameter d
coordinates = mesh0.geometry.x
x_coords = coordinates[:, 0]; y_coords = coordinates[:, 1]
x_range = np.max(x_coords) - np.min(x_coords)
y_range = np.max(y_coords) - np.min(y_coords)
d = max(x_range, y_range)

# equal-order elements with Taylor-Hood (P2-P1)
degree_u = mesh_order
degree_p = max(1, mesh_order - 1)

# Mixed function space for unknowns
P_u = element("Lagrange", mesh0.basix_cell(), degree_u, shape=(mesh0.geometry.dim,), dtype=default_real_type)
P_p = element("Lagrange", mesh0.basix_cell(), degree_p, dtype=default_real_type)
P_mx = mixed_element([P_u, P_p])
V = functionspace(mesh0, P_mx)
(v, q) = TestFunctions(V)

w_V = Function(V); w_V_prev = Function(V)
u, p = split(w_V)

Vu0 = V.sub(0); Vp0 = V.sub(1)
Vu, dofu = Vu0.collapse(); Vp, dofp = Vp0.collapse()

# Apply BCs
ub = Constant(mesh0, default_scalar_type(0))
bcb = dirichletbc(ub, locate_dofs_topological(Vu0.sub(1), mesh0.topology.dim - 1, bottom_facet), Vu0.sub(1))

lower_left_corner = lambda x: np.isclose(x[0], 0.0) & np.isclose(x[1], 0.0)
lower_left_facet = locate_entities_boundary(mesh0, 0, lower_left_corner)
ul = Constant(mesh0, default_scalar_type(0))
bcl = dirichletbc(ul, locate_dofs_topological(Vu0.sub(0), 0, lower_left_facet), Vu0.sub(0))
bcu = [bcb, bcl]

# Create a force that can be updated
pload = Constant(mesh0, (0.0, 0.0))

# Kinematics
dim = len(u)
I = variable(Identity(dim))             # Identity tensor
F_0 = variable(I + grad(u))             # Deformation gradient in 2D
F = variable(as_tensor([[F_0[0, 0], F_0[0, 1], 0],
                        [F_0[1, 0], F_0[1, 1], 0],
                        [0, 0, 1]]))                  # Deformation gradient in 3D
C = variable(F.T*F)                     # Right Cauchy-Green tensor
Ic = variable(tr(C))
J  = variable(det(F))

# Strain energy density in 3D

J_safe = conditional(gt(J, 1e-12), J, 1e-12)

def psi(mu, K): 
    return (K/2)*(ln(J_safe))**2 + (mu/2)*(J_safe**(-2/3)*Ic - 3)

def psi_2(mu):
    return (mu/2)*(J_safe**(-2/3)*Ic - 3)
# Potential energy
# dx(1): elastic half space; dx(2): cylinder; dx(3): third medium
# ds(11): top facet for pressure load; ds(10): bottom facet for displacement BC
Pi_body = psi(mu_e, K_e) * dx(1) + psi(mu_c, K_c) * dx(2) - inner(pload, u) * ds(11)
Pi_m = gamma * psi_2(mu_c) * dx(3)

psi_tan_R = ( beta_1/2 * ((F[0,1] - F[1,0]) / (F[0,0] + F[1,1]) - 1/d * p)**2 
    + beta_2/2 * inner(grad(p), grad(p)) )
Pi_tan_R = psi_tan_R * dx(3)

Pi_medium = Pi_m + Pi_tan_R
Pi_total = Pi_body + Pi_medium

L_u = derivative(Pi_total, u, v)

# To solve p
epsilon_penalty = 1e-16
Pi_penalty_R = epsilon_penalty * inner(p, p) * dx(1) + epsilon_penalty * inner(p, p) * dx(2)
Pi_total_R = Pi_tan_R + Pi_penalty_R

L_p = derivative(Pi_total_R, p, q)

# Combine the two parts of the weak form
L = L_u + L_p

# 1st Piola-Kirchhoff stress and cauchy stress
P_el = diff(psi(mu_e, K_e), F) 
sigma_el = (1/J) * P_el * F.T

P_cyliner = diff(psi(mu_c, K_c), F)
sigma_cyliner = (1/J) * P_cyliner * F.T

psi_medium = gamma * psi_2(mu_c) + psi_tan_R
P_medium = diff(psi_medium, F)
sigma_medium = (1/J) * P_medium * F.T

class NonlinearPDE_SNESProblem:
    def __init__(self, F, u, bcs):
        V = u.function_space
        du = TrialFunction(V)
        self.L = form(F)
        self.a = form(derivative(F, u, du))
        self.bcs = bcs
        self._F, self._J = None, None
        self.u = u

    def F(self, snes, x, F):
        x.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
        x.copy(self.u.x.petsc_vec)
        self.u.x.petsc_vec.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
            
        with F.localForm() as f_local:
            f_local.set(0.0)
        assemble_vector(F, self.L)
        apply_lifting(F, [self.a], bcs=[self.bcs], x0=[x], alpha=-1.0)
        F.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        set_bc(F, self.bcs, x, -1.0)

    def J(self, snes, x, J, P):
        J.zeroEntries()
        assemble_matrix(J, self.a, bcs=self.bcs)
        J.assemble()


def get_interpolation_points(elem):
    """dolfinx API changed across versions: interpolation_points is a
    *method* in some releases and a precomputed *array attribute* in
    others. This works with either."""
    ip = elem.interpolation_points
    return ip() if callable(ip) else ip


def find_node_by_coordinates(mesh, target_coords, tolerance=1e-10):
    coordinates = mesh.geometry.x
    distances = np.sqrt((coordinates[:, 0] - target_coords[0])**2 + 
                       (coordinates[:, 1] - target_coords[1])**2)
    
    closest_node_id = np.argmin(distances)
    min_distance = distances[closest_node_id]
    
    if min_distance <= tolerance:
        return closest_node_id
    else:
        return closest_node_id

V_point = functionspace(mesh0, ("CG", 1))
sigma22_total_func = Function(V_point, name="sigma22_total_nodal")

def evaluate_stress_at_node(node_id, mesh):
    sigma22_total_expr = conditional(eq(material_function, 1), sigma_el[1, 1],
                                    conditional(eq(material_function, 2), sigma_cyliner[1, 1],
                                                sigma_medium[1, 1]))
    sigma22_total_func.interpolate(Expression(sigma22_total_expr, get_interpolation_points(V_point.element)))
    sigma22_total_value = sigma22_total_func.x.array[node_id]

    return sigma22_total_value, sigma22_total_func

def compute_cauchy_stress_direct():
    sigma22_el = sigma_el[1,1]
    sigma22_cyliner = sigma_cyliner[1,1]
    sigma22_medium = sigma_medium[1,1]
    sigma22_total = stress_total(sigma22_el, sigma22_cyliner, sigma22_medium)
    return sigma22_total, sigma22_el, sigma22_cyliner, sigma22_medium

def stress_total(sigma_el, sigma_cyliner, sigma_medium):
    sigma_total = conditional(eq(material_function, 1), sigma_el,
        conditional(eq(material_function, 2), sigma_cyliner, sigma_medium))
    return sigma_total

def stress_interpolate(expr_func_pairs, space):
    for expr, func in expr_func_pairs:
        func.interpolate(Expression(expr, get_interpolation_points(space.element)))

def write_stress(xdmf_file, func_sigma, n):
    for sigma in func_sigma:
        xdmf_file.write_function(sigma, n)

# Material markers function
V0 = functionspace(mesh0, ("DG", 0))
material_function = Function(V0, name="material")
material_function.x.array[:] = cell_tags.values
material_function.x.scatter_forward()

# Stress functions DG0
Vs_scalar = functionspace(mesh0, ("DG", 0))
func_sigma22_el = Function(Vs_scalar, name="sigma22_el")
func_sigma22_cyliner = Function(Vs_scalar, name="sigma22_cyliner")
func_sigma22_medium = Function(Vs_scalar, name="sigma22_medium")
func_sigma22_total = Function(Vs_scalar, name="sigma22_total")

func_sigma22 = [func_sigma22_el, func_sigma22_cyliner, func_sigma22_medium, func_sigma22_total]

target_node_ID = find_node_by_coordinates(mesh0, target_node)

u_out = Function(Vu, name="u_out")
p_out = Function(Vp, name="p_out")

output_functions = [u_out, p_out, sigma22_total_func] + func_sigma22

# context manager so the xdmf file is always properly closed

from contextlib import ExitStack
 
with ExitStack() as stack:
    vtx_u = stack.enter_context(VTXWriter(mesh0.comm, f"{file_name}/results_{file_name}_u.bp", [u_out], engine="BP4"))
    vtx_p = stack.enter_context(VTXWriter(mesh0.comm, f"{file_name}/results_{file_name}_p.bp", [p_out], engine="BP4"))
    vtx_sigma_node = stack.enter_context(VTXWriter(mesh0.comm, f"{file_name}/results_{file_name}_sigma_node.bp", [sigma22_total_func], engine="BP4"))
    vtx_sigma_dg0 = stack.enter_context(VTXWriter(mesh0.comm, f"{file_name}/results_{file_name}_sigma_dg0.bp", func_sigma22, engine="BP4"))
 
    def write_all(t):
        vtx_u.write(t)
        vtx_p.write(t)
        vtx_sigma_node.write(t)
        vtx_sigma_dg0.write(t)
 
    write_all(0.0)
 
    # Adaptive load stepping loop
    current_load = 0.0; step_size = initial_step_size; n = 0
    while current_load > target_load:
        k = 0; j = 0
        success_ad = False
 
        while not success_ad and k < max_attempts:
            k += 1
            next_load = max(current_load - step_size, target_load)  # New load step
            pload.value = [0.0, next_load]
 
            # Save current state before attempting the step
            w_V_prev.x.array[:] = w_V.x.array[:]
 
            # Create SNES solver
            problem = NonlinearPDE_SNESProblem(L, w_V, bcs=bcu)
            b = create_vector(V)
            J_L = create_matrix(problem.a)
            snes = PETSc.SNES().create()
            snes.setFunction(problem.F, b)
            snes.setJacobian(problem.J, J_L)
            snes.setTolerances(max_it=max_it_newton)
            snes.getKSP().setType("preonly")
            snes.getKSP().getPC().setType("lu")
 
            opts = PETSc.Options()
            opts["pc_factor_mat_solver_type"] = "mumps"
            opts['snes_linesearch_type'] = linesearch
            opts['snes_max_linear_solve_fail'] = 20
            opts['snes_linesearch_max_it'] = 20
            opts['snes_monitor'] = None
            opts['snes_linesearch_monitor'] = None
            opts['snes_converged_reason'] = None
            snes.setFromOptions()
 
            x = None
            try:
                print("n=: ", n)
                x = w_V.x.petsc_vec.copy()
                x.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
                snes.solve(None, x)  # Compute u and p
                if snes.getConvergedReason() > 0:
                    x.copy(w_V.x.petsc_vec)
                    w_V.x.petsc_vec.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
                else:
                    raise RuntimeError(f"Newton failed due to error:{snes.getConvergedReason()}")
                print(f"Iterations: {snes.getIterationNumber()}, Residual: {snes.getFunctionNorm()}")
 
                # Update u_out/p_out from the current solution
                u_out.x.array[:] = w_V.sub(0).collapse().x.array[:]
                p_out.x.array[:] = w_V.sub(1).collapse().x.array[:]
 
                # Stress fields (updates func_sigma22_* in place)
                sigma22_total, sigma22_el, sigma22_cyliner, sigma22_medium = compute_cauchy_stress_direct()
                stress_interpolate([(sigma22_total, func_sigma22_total),
                                    (sigma22_el, func_sigma22_el),
                                    (sigma22_cyliner, func_sigma22_cyliner),
                                    (sigma22_medium, func_sigma22_medium)], Vs_scalar)
 
                # Nodal stress at target node (updates sigma22_total_func in place)
                sigma22_total_at_node = evaluate_stress_at_node(target_node_ID, mesh0)
 
                # Single call: writes u_out, p_out, sigma22_total_func,
                # and all four func_sigma22 fields together (via their
                # four separate writers), in their just-updated state.
                write_all(float(n + 1))
 
                success_ad = True
                n += 1
                current_load = next_load
 
                step_size = min(step_size * increase_factor, max_step_size)
 
            except Exception as e:
                print(f"Step failed with error: {e}")
                traceback.print_exc()
 
                # Restore to previous state and reduce step size
                w_V.x.array[:] = w_V_prev.x.array[:]
                step_size = max(step_size * decrease_factor, min_step_size)
                print(f"Reducing step size to {step_size:.6f}")
 
                if step_size == min_step_size:
                    j += 1
 
                if j == 2:
                    print(f"Already reached the minimum step size {min_step_size} and still failing.")
 
            finally:
                snes.destroy()
                J_L.destroy()
                b.destroy()
                if x is not None:
                    x.destroy()
 
            if j == 2:
                break
 
        if j == 2 or (not success_ad and k >= max_attempts):
            print(f"Step {n + 1} failed after {k} attempts.")
            print(f"Final displacement achieved: {current_load:.6f}")
            break
 
    print(f"Finished. Total successful steps written: {n}, final load: {current_load:.6f}")
    
# vtx_writer is guaranteed closed/flushed here.

# ============================================================
# POST-PROCESSING
# ============================================================
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from dolfinx import plot

topology_u, _, geometry_u = plot.vtk_mesh(Vu)
topology_p, _, geometry_p = plot.vtk_mesh(Vp)

V_p1 = functionspace(mesh0, ("Lagrange", 1, (mesh0.geometry.dim,)))
u_p1 = Function(V_p1)
u_p1.interpolate(u_out)

topology_u, _, geometry_u = plot.vtk_mesh(V_p1)
tri_u = topology_u.reshape(-1, 4)[:, 1:]

u_vals = u_p1.x.array.reshape(-1, mesh0.geometry.dim)
u_mag = np.linalg.norm(u_vals, axis=1)

x_def = geometry_u[:, 0] + u_vals[:, 0]
y_def = geometry_u[:, 1] + u_vals[:, 1]

target_width = 10.0
x_span = x_def.max() - x_def.min()
y_span = y_def.max() - y_def.min()
aspect_ratio = y_span / x_span if x_span > 0 else 1.0
fig_height = max(target_width * aspect_ratio, 2.0)  

triang_def = mtri.Triangulation(x_def, y_def, tri_u)

fig1, ax1 = plt.subplots(figsize=(target_width, fig_height))
tpc1 = ax1.tricontourf(triang_def, u_mag, levels=20, cmap="viridis")
ax1.triplot(triang_def, color="k", linewidth=0.1, alpha=0.3)
ax1.set_aspect("equal")
ax1.set_title("Deformation – |u|")
fig1.colorbar(tpc1, ax=ax1, label="|u|")
fig1.tight_layout()
fig1.savefig("displacement_deformed_hertz.png", dpi=200)

print("Saved: displacement_deformed_hertz.png")
plt.show()