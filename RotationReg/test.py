import numpy as np
import traceback
from contextlib import ExitStack

from dolfinx.io import VTXWriter
from petsc4py import PETSc
from mpi4py import MPI
from ufl import *
from dolfinx.fem import *
from dolfinx.fem.petsc import *
from dolfinx.mesh import *
from dolfinx import default_scalar_type, default_real_type
from dolfinx.fem.petsc import (
    create_matrix, assemble_matrix, apply_lifting,
    assemble_vector, set_bc, create_vector
)
from basix.ufl import element, mixed_element

from hertzian_mesh import generate_mesh

# ============================================================
# Fixed physical/solver parameters (unchanged from the original script)
# ============================================================
linesearch = 'bt'
mesh_order = 1
target_load = -600

atol_newton = 1e-18; rtol_newton = 1e-17; stol_newton = 1e-20; max_it_newton = 10

Nsteps = 4000; Nsteps_min = 25; Nsteps_max = 40000
increase_factor = 2.0; decrease_factor = 0.5
max_attempts = 10

mu_c = default_scalar_type(35000/13); K_c = default_scalar_type(17500/3)   # Cylinder
mu_e = default_scalar_type(700000/29); K_e = default_scalar_type(700000/3)  # Elastic half space
gamma = default_scalar_type(1.3e-3)
beta_1 = default_scalar_type(0.01); beta_2 = default_scalar_type(1.0e-5)

target_node = (1.1, 2.0)


class NonlinearPDE_SNESProblem:
    def __init__(self, F, u, bcs):
        V = u.function_space
        du = TrialFunction(V)
        self.L = form(F)
        self.a = form(derivative(F, u, du))
        self.bcs = bcs
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
    ip = elem.interpolation_points
    return ip() if callable(ip) else ip


def find_node_by_coordinates(mesh, target_coords, tolerance=1e-10):
    coordinates = mesh.geometry.x
    distances = np.sqrt((coordinates[:, 0] - target_coords[0])**2 +
                         (coordinates[:, 1] - target_coords[1])**2)
    closest_node_id = np.argmin(distances)
    return closest_node_id


def run_case(lc, target_load=target_load, Nsteps=Nsteps, write_outputs=True,
             plot_deformation=True, comm=MPI.COMM_WORLD):
    """Run the full nonlinear Hertzian-contact simulation on a mesh built
    at resolution `lc`, and return a dict of convergence diagnostics."""

    initial_step_size = abs(target_load) / Nsteps
    min_step_size = abs(target_load) / Nsteps_max
    max_step_size = abs(target_load) / Nsteps_min

    file_name = f"hertzian_lc_{lc:g}"

    # ---- Mesh (built in memory at this resolution, no .msh file needed) ----
    mesh_data = generate_mesh(lc=lc, comm=comm, gdim=2)
    mesh0 = mesh_data.mesh
    cell_tags = mesh_data.cell_tags
    facet_tags = mesh_data.facet_tags

    metadata = {"quadrature_degree": 4}
    dx = Measure("dx", domain=mesh0, subdomain_data=cell_tags, metadata=metadata)
    ds = Measure("ds", domain=mesh0, subdomain_data=facet_tags, metadata=metadata)
    bottom_facet = facet_tags.find(10)

    coordinates = mesh0.geometry.x
    x_range = np.max(coordinates[:, 0]) - np.min(coordinates[:, 0])
    y_range = np.max(coordinates[:, 1]) - np.min(coordinates[:, 1])
    d = max(x_range, y_range)

    degree_u = mesh_order
    degree_p = max(1, mesh_order - 1)

    P_u = element("Lagrange", mesh0.basix_cell(), degree_u, shape=(mesh0.geometry.dim,), dtype=default_real_type)
    P_p = element("Lagrange", mesh0.basix_cell(), degree_p, dtype=default_real_type)
    P_mx = mixed_element([P_u, P_p])
    V = functionspace(mesh0, P_mx)
    (v, q) = TestFunctions(V)

    w_V = Function(V); w_V_prev = Function(V)
    u, p = split(w_V)

    Vu0 = V.sub(0); Vp0 = V.sub(1)
    Vu, dofu = Vu0.collapse(); Vp, dofp = Vp0.collapse()

    ub = Constant(mesh0, default_scalar_type(0))
    bcb = dirichletbc(ub, locate_dofs_topological(Vu0.sub(1), mesh0.topology.dim - 1, bottom_facet), Vu0.sub(1))

    lower_left_corner = lambda x: np.isclose(x[0], 0.0) & np.isclose(x[1], 0.0)
    lower_left_facet = locate_entities_boundary(mesh0, 0, lower_left_corner)
    ul = Constant(mesh0, default_scalar_type(0))
    bcl = dirichletbc(ul, locate_dofs_topological(Vu0.sub(0), 0, lower_left_facet), Vu0.sub(0))
    bcu = [bcb, bcl]

    pload = Constant(mesh0, (0.0, 0.0))

    dim = len(u)
    I = variable(Identity(dim))
    F_0 = variable(I + grad(u))
    F = variable(as_tensor([[F_0[0, 0], F_0[0, 1], 0],
                             [F_0[1, 0], F_0[1, 1], 0],
                             [0, 0, 1]]))
    C = variable(F.T * F)
    Ic = variable(tr(C))
    J = variable(det(F))

    J_safe = conditional(gt(J, 1e-12), J, 1e-12)

    def psi(mu, K):
        return (K / 2) * (ln(J_safe))**2 + (mu / 2) * (J_safe**(-2/3) * Ic - 3)

    def psi_2(mu):
        return (mu / 2) * (J_safe**(-2/3) * Ic - 3)

    Pi_body = psi(mu_e, K_e) * dx(1) + psi(mu_c, K_c) * dx(2) - inner(pload, u) * ds(11)
    Pi_m = gamma * psi_2(mu_c) * dx(3)

    psi_tan_R = (beta_1 / 2 * ((F[0, 1] - F[1, 0]) / (F[0, 0] + F[1, 1]) - 1 / d * p)**2
                 + beta_2 / 2 * inner(grad(p), grad(p)))
    Pi_tan_R = psi_tan_R * dx(3)

    Pi_medium = Pi_m + Pi_tan_R
    Pi_total = Pi_body + Pi_medium

    L_u = derivative(Pi_total, u, v)

    epsilon_penalty = 1e-16
    Pi_penalty_R = epsilon_penalty * inner(p, p) * dx(1) + epsilon_penalty * inner(p, p) * dx(2)
    Pi_total_R = Pi_tan_R + Pi_penalty_R

    L_p = derivative(Pi_total_R, p, q)

    L = L_u + L_p

    P_el = diff(psi(mu_e, K_e), F)
    sigma_el = (1 / J) * P_el * F.T

    P_cyliner = diff(psi(mu_c, K_c), F)
    sigma_cyliner = (1 / J) * P_cyliner * F.T

    psi_medium = gamma * psi_2(mu_c) + psi_tan_R
    P_medium = diff(psi_medium, F)
    sigma_medium = (1 / J) * P_medium * F.T

    V0 = functionspace(mesh0, ("DG", 0))
    material_function = Function(V0, name="material")
    material_function.x.array[:] = cell_tags.values
    material_function.x.scatter_forward()

    def stress_total(sigma_el_, sigma_cyliner_, sigma_medium_):
        return conditional(eq(material_function, 1), sigma_el_,
                            conditional(eq(material_function, 2), sigma_cyliner_, sigma_medium_))

    def compute_cauchy_stress_direct():
        sigma22_el = sigma_el[1, 1]
        sigma22_cyliner = sigma_cyliner[1, 1]
        sigma22_medium = sigma_medium[1, 1]
        sigma22_total = stress_total(sigma22_el, sigma22_cyliner, sigma22_medium)
        return sigma22_total, sigma22_el, sigma22_cyliner, sigma22_medium

    def stress_interpolate(expr_func_pairs, space):
        for expr, func in expr_func_pairs:
            func.interpolate(Expression(expr, get_interpolation_points(space.element)))

    V_point = functionspace(mesh0, ("CG", 1))
    sigma22_total_func = Function(V_point, name="sigma22_total_nodal")

    def evaluate_stress_at_node(node_id):
        sigma22_total_expr = conditional(eq(material_function, 1), sigma_el[1, 1],
                                          conditional(eq(material_function, 2), sigma_cyliner[1, 1],
                                                      sigma_medium[1, 1]))
        sigma22_total_func.interpolate(Expression(sigma22_total_expr, get_interpolation_points(V_point.element)))
        return sigma22_total_func.x.array[node_id]

    Vs_scalar = functionspace(mesh0, ("DG", 0))
    func_sigma22_el = Function(Vs_scalar, name="sigma22_el")
    func_sigma22_cyliner = Function(Vs_scalar, name="sigma22_cyliner")
    func_sigma22_medium = Function(Vs_scalar, name="sigma22_medium")
    func_sigma22_total = Function(Vs_scalar, name="sigma22_total")
    func_sigma22 = [func_sigma22_el, func_sigma22_cyliner, func_sigma22_medium, func_sigma22_total]

    target_node_ID = find_node_by_coordinates(mesh0, target_node)

    u_out = Function(Vu, name="u_out")
    p_out = Function(Vp, name="p_out")

    num_dofs = V.dofmap.index_map.size_global * V.dofmap.index_map_bs
    num_cells = mesh0.topology.index_map(mesh0.topology.dim).size_global

    sigma22_at_node = None
    current_load = 0.0
    n = 0

    stack = ExitStack()
    try:
        if write_outputs:
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
        else:
            def write_all(t):
                pass

        step_size = initial_step_size
        while current_load > target_load:
            k = 0; j = 0
            success_ad = False

            while not success_ad and k < max_attempts:
                k += 1
                next_load = max(current_load - step_size, target_load)
                pload.value = [0.0, next_load]

                w_V_prev.x.array[:] = w_V.x.array[:]

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
                    print(f"[lc={lc:g}] n=: ", n)
                    x = w_V.x.petsc_vec.copy()
                    x.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
                    snes.solve(None, x)
                    if snes.getConvergedReason() > 0:
                        x.copy(w_V.x.petsc_vec)
                        w_V.x.petsc_vec.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
                    else:
                        raise RuntimeError(f"Newton failed due to error:{snes.getConvergedReason()}")
                    print(f"Iterations: {snes.getIterationNumber()}, Residual: {snes.getFunctionNorm()}")

                    u_out.x.array[:] = w_V.sub(0).collapse().x.array[:]
                    p_out.x.array[:] = w_V.sub(1).collapse().x.array[:]

                    sigma22_total, sigma22_el, sigma22_cyliner, sigma22_medium = compute_cauchy_stress_direct()
                    stress_interpolate([(sigma22_total, func_sigma22_total),
                                         (sigma22_el, func_sigma22_el),
                                         (sigma22_cyliner, func_sigma22_cyliner),
                                         (sigma22_medium, func_sigma22_medium)], Vs_scalar)

                    sigma22_at_node = evaluate_stress_at_node(target_node_ID)

                    write_all(float(n + 1))

                    success_ad = True
                    n += 1
                    current_load = next_load

                    step_size = min(step_size * increase_factor, max_step_size)

                except Exception as e:
                    print(f"Step failed with error: {e}")
                    traceback.print_exc()
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

        print(f"[lc={lc:g}] Finished. Total successful steps written: {n}, final load: {current_load:.6f}")
    finally:
        stack.close()

    if plot_deformation and comm.rank == 0:
        try:
            import matplotlib.pyplot as plt
            import matplotlib.tri as mtri
            from dolfinx import plot

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
            ax1.set_title(f"Deformation – |u| (lc={lc:g})")
            fig1.colorbar(tpc1, ax=ax1, label="|u|")
            fig1.tight_layout()
            fig1.savefig(f"displacement_deformed_hertz_lc_{lc:g}.png", dpi=200)
            plt.close(fig1)
        except Exception as e:
            print(f"Post-processing plot failed for lc={lc:g}: {e}")

    return {
        "lc": lc,
        "num_dofs": int(num_dofs),
        "num_cells": int(num_cells),
        "sigma22_at_target_node": float(sigma22_at_node) if sigma22_at_node is not None else None,
        "final_load_reached": float(current_load),
        "n_steps": int(n),
    }


if __name__ == "__main__":
    comm = MPI.COMM_WORLD

    # Mesh sizes to sweep for the convergence study (coarse -> fine).
    # Adjust as needed; each finer step roughly 4x the elements of the last.
    lc_values = [0.08, 0.06, 0.04, 0.03, 0.02]

    results = []
    for lc in lc_values:
        res = run_case(lc, write_outputs=True, plot_deformation=True, comm=comm)
        results.append(res)
        if comm.rank == 0:
            print(res)

    if comm.rank == 0:
        import json
        with open("convergence_results.json", "w") as f:
            json.dump(results, f, indent=2)

        try:
            import matplotlib.pyplot as plt
            h = [r["lc"] for r in results]
            sigma = [r["sigma22_at_target_node"] for r in results]
            dofs = [r["num_dofs"] for r in results]

            fig, ax1 = plt.subplots()
            ax1.plot(h, sigma, "o-")
            ax1.set_xlabel("Mesh size lc")
            ax1.set_ylabel(r"$\sigma_{22}$ at target node")
            ax1.invert_xaxis()
            ax1.set_title("Mesh convergence: stress at contact node")
            fig.tight_layout()
            fig.savefig("convergence_sigma22_vs_lc.png", dpi=200)

            fig2, ax2 = plt.subplots()
            ax2.plot(dofs, sigma, "o-")
            ax2.set_xscale("log")
            ax2.set_xlabel("Number of DOFs")
            ax2.set_ylabel(r"$\sigma_{22}$ at target node")
            ax2.set_title("Mesh convergence vs. DOF count")
            fig2.tight_layout()
            fig2.savefig("convergence_sigma22_vs_dofs.png", dpi=200)
        except Exception as e:
            print(f"Convergence plot failed: {e}")

        print("Saved: convergence_results.json, convergence_sigma22_vs_lc.png, convergence_sigma22_vs_dofs.png")