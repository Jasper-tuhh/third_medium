import numpy as np
import traceback
from contextlib import ExitStack
import os

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
from basix.ufl import element
from hertzian_mesh import generate_mesh

# ============================================================
# Fixed physical/solver parameters
# ============================================================
file_name = "hertz_huhu"
os.makedirs(file_name, exist_ok=True)

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

# HuHu-LuLu regularization coefficients
k_HuHu = default_scalar_type(1.0e-6)

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


def run_case(lc, fe_degree, target_load=target_load, Nsteps=Nsteps, write_outputs=True,
             plot_deformation=True, comm=MPI.COMM_WORLD):
    """Run the full nonlinear Hertzian-contact simulation on a mesh built
    at resolution `lc`, and return a dict of convergence diagnostics."""

    initial_step_size = abs(target_load) / Nsteps
    min_step_size = abs(target_load) / Nsteps_max
    max_step_size = abs(target_load) / Nsteps_min

    # ---- Mesh ----
    mesh_data = generate_mesh(lc=lc, comm=comm, gdim=2)
    mesh0 = mesh_data.mesh
    cell_tags = mesh_data.cell_tags
    facet_tags = mesh_data.facet_tags

    metadata = {"quadrature_degree": 2 * fe_degree}
    dx = Measure("dx", domain=mesh0, subdomain_data=cell_tags, metadata=metadata)
    ds = Measure("ds", domain=mesh0, subdomain_data=facet_tags, metadata=metadata)
    bottom_facet = facet_tags.find(10)

    coordinates = mesh0.geometry.x
    x_range = np.max(coordinates[:, 0]) - np.min(coordinates[:, 0])
    y_range = np.max(coordinates[:, 1]) - np.min(coordinates[:, 1])
    d = max(x_range, y_range)

    degree_u = fe_degree

    P_u = element("Lagrange", mesh0.basix_cell(), degree_u, shape=(mesh0.geometry.dim,), dtype=default_real_type)
    V = functionspace(mesh0, P_u)
    v = TestFunction(V)

    u = Function(V); u_prev = Function(V)

    ub = Constant(mesh0, default_scalar_type(0))
    bcb = dirichletbc(ub, locate_dofs_topological(V.sub(1), mesh0.topology.dim - 1, bottom_facet), V.sub(1))

    lower_left_corner = lambda x: np.isclose(x[0], 0.0) & np.isclose(x[1], 0.0)
    lower_left_facet = locate_entities_boundary(mesh0, 0, lower_left_corner)
    ul = Constant(mesh0, default_scalar_type(0))
    bcl = dirichletbc(ul, locate_dofs_topological(V.sub(0), 0, lower_left_facet), V.sub(0))
    bcu = [bcb, bcl]

    pload = Constant(mesh0, (0.0, 0.0))

    dim = mesh0.geometry.dim
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

    # HuHu regularization on the third medium (dx(3)):
    Hu = grad(grad(u))         

    Pi_HuHu = (
        k_HuHu * inner(Hu, Hu)) * dx(3)

    Pi_total = Pi_body + Pi_m + Pi_HuHu

    L = derivative(Pi_total, u, v)

    P_el = diff(psi(mu_e, K_e), F)
    sigma_el = (1 / J) * P_el * F.T

    P_cyliner = diff(psi(mu_c, K_c), F)
    sigma_cyliner = (1 / J) * P_cyliner * F.T

    P_medium = diff(gamma * psi_2(mu_c), F)
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

    u_out = u

    num_dofs = V.dofmap.index_map.size_global * V.dofmap.index_map_bs
    num_cells = mesh0.topology.index_map(mesh0.topology.dim).size_global

    sigma22_at_node = None
    current_load = 0.0
    n = 0

    stack = ExitStack()
    try:
        if write_outputs:
            vtx_u = stack.enter_context(VTXWriter(mesh0.comm, f"{file_name}/results_{file_name}_u.bp", [u_out], engine="BP4"))
            vtx_sigma_node = stack.enter_context(VTXWriter(mesh0.comm, f"{file_name}/results_{file_name}_sigma_node.bp", [sigma22_total_func], engine="BP4"))
            vtx_sigma_dg0 = stack.enter_context(VTXWriter(mesh0.comm, f"{file_name}/results_{file_name}_sigma_dg0.bp", func_sigma22, engine="BP4"))

            def write_all(t):
                vtx_u.write(t)
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

                u_prev.x.array[:] = u.x.array[:]

                problem = NonlinearPDE_SNESProblem(L, u, bcs=bcu)
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
                    x = u.x.petsc_vec.copy()
                    x.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
                    snes.solve(None, x)
                    if snes.getConvergedReason() > 0:
                        x.copy(u.x.petsc_vec)
                        u.x.petsc_vec.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
                    else:
                        raise RuntimeError(f"Newton failed due to error:{snes.getConvergedReason()}")
                    print(f"Iterations: {snes.getIterationNumber()}, Residual: {snes.getFunctionNorm()}")

                    # u_out is an alias of u -- no copy needed.

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
                    u.x.array[:] = u_prev.x.array[:]
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

    energy_bodies = assemble_scalar(form(Pi_body))
    energy_bodies = comm.allreduce(energy_bodies, op=MPI.SUM)

    return {
    "lc": lc,
    "fe_degree": fe_degree,
    "num_dofs": int(num_dofs),
    "num_cells": int(num_cells),
    "sigma22_at_target_node": float(sigma22_at_node) if sigma22_at_node is not None else None,
    "final_load_reached": float(current_load),
    "n_steps": int(n),
    "energy_bodies": float(energy_bodies),
    }


if __name__ == "__main__":
    comm = MPI.COMM_WORLD

    lc_values = [0.08, 0.06, 0.04, 0.03, 0.02]
    fe_degree = [1 , 2, 3]

    results = []
    for lc in lc_values:
        for degree in fe_degree:
            res = run_case(lc, degree, write_outputs=True, plot_deformation=True, comm=comm)
            results.append(res)
        if comm.rank == 0:
            print(res)

    if comm.rank == 0:
        import json
        with open("convergence_results.json", "w") as f:
            json.dump(results, f, indent=2)

        try:
            import matplotlib.pyplot as plt

            fe_degrees = sorted(set(r["fe_degree"] for r in results))
            markers = ['o', 's', '^', 'd', 'v', '*']  # provide enough markers

            plt.figure()
            for i, degree in enumerate(fe_degrees):
                filtered = [r for r in results if r["fe_degree"] == degree]
                dofs = [r["num_dofs"] for r in filtered]
                energy = [r["energy_bodies"] for r in filtered]

                # Sort by increasing DOFs for better plot lines
                sorted_pairs = sorted(zip(dofs, energy))
                dofs_sorted, energy_sorted = zip(*sorted_pairs)

                plt.plot(dofs_sorted, energy_sorted, marker=markers[i % len(markers)], label=f"p={degree}", linestyle='-')

            plt.xlabel('Number of DOFs')
            plt.ylabel('Bodies Potential Energy')
            plt.xscale('log')
            plt.title('Convergence of Bodies Potential Energy vs. DOFs')
            plt.legend()
            plt.tight_layout()
            plt.savefig(f"{file_name}/bodies_potential_energy_vs_dofs.png", dpi=200)
            plt.show()

            plt.figure()
            for i, degree in enumerate(fe_degrees):
                filtered = [r for r in results if r["fe_degree"] == degree]
                h = [r["lc"] for r in results]
                energy = [r["energy_bodies"] for r in filtered]

                # Sort by increasing DOFs for better plot lines
                sorted_pairs = sorted(zip(dofs, energy))
                dofs_sorted, energy_sorted = zip(*sorted_pairs)

                plt.plot(h, energy_sorted, marker=markers[i % len(markers)], label=f"p={degree}", linestyle='-')

            plt.xlabel('Mesh size lc')
            plt.ylabel('Bodies Potential Energy')
            plt.xscale('log')
            plt.title('Convergence of Bodies Potential Energy vs. Mesh size')
            plt.legend()
            plt.tight_layout()
            plt.savefig(f"{file_name}/bodies_potential_energy_vs_mesh.png", dpi=200)
            plt.show()

        except Exception as e:
            print(f"Convergence plot failed: {e}")

        print("Saved: convergence_results.json, bodies_potential_energy_vs_dofs.png, bodies_potential_energy_vs_mesh.png")
