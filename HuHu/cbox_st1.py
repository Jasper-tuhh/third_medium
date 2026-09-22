import time
import traceback
import numpy as np

from mpi4py import MPI
from petsc4py import PETSc

from dolfinx import default_scalar_type, default_real_type
from dolfinx.mesh import *
from dolfinx.fem import *
from dolfinx.fem.petsc import (
    create_matrix, assemble_matrix, apply_lifting,
    assemble_vector, set_bc, create_vector
)
from ufl import *
import os

file_name = "cbox"
folder_name = "cbox_huhu"
relative_path = os.path.join(".", folder_name)

if not os.path.exists(relative_path):
    os.makedirs(relative_path)
    print(f"Created folder: {relative_path}")

dir_name = relative_path 
linesearch = "bt"

def mpi_time(comm, t0):
    local_time = time.perf_counter() - t0
    return comm.allreduce(local_time, op=MPI.MAX)

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


def run_case(Nx, Ny, fe_degree, target_displacement=-0.50,
             verbose=False):
    """
    Runs one full staggered hyperelastic solve at the given mesh
    resolution (Nx, Ny) and polynomial degree (fe_degree), on the
    "cbox" domain: the classic C-shape TMC benchmark (fixed left wall,
    prescribed vertical displacement at the top vertex, open medium
    "mouth" to the right of the solid bracket).

    Returns a dict with h, dof, potential, final_displacement, n_steps,
    plus the solved u/p Functions and the mesh (for plotting).
    """

    comm = MPI.COMM_WORLD
    t_total_start = time.perf_counter()
    t_setup_start = time.perf_counter()

    timings = {
        "setup": 0.0,
        "snes": 0.0,
        "p_update": 0.0,
        "staggered": 0.0,
        "total": 0.0,
    }

    stats = {
        "newton_iterations": 0,
        "staggered_iterations": 0,
        "load_steps": 0,
        "failed_attempts": 0,
        "successful_attempts": 0,
    }

    degree_u = max(1, fe_degree)
    degree_p = max(1, fe_degree)
    p_space = "Lagrange"

    # ---- Nonlinear / staggered / load-stepping parameters ----
    atol_newton = 1e-12
    rtol_newton = 1e-10
    stol_newton = 1e-12

    max_it_newton = 18
    max_it_staggered = 100
    err_tol_staggered = 1e-3
    Nsteps, Nsteps_min, Nsteps_max = 100, 40, 800

    crit_displacement = -0.8
    initial_step_size = abs(target_displacement) / Nsteps
    min_step_size = abs(target_displacement) / Nsteps_max
    max_step_size = abs(target_displacement) / Nsteps_min

    increase_factor = 1.5
    decrease_factor = 0.5
    max_attempts = 10

    # ---- Model parameters ----
    mu = default_scalar_type(5 / 14)
    K = default_scalar_type(5 / 3)
    gamma = default_scalar_type(1.0e-6)
    beta_1 = default_scalar_type(0.01)
    beta_2 = default_scalar_type(1e-5)

    # ---- Mesh ----
    p0 = [0.0, 0.0]
    p1 = [1.0, 0.5]
    mesh0 = create_rectangle(MPI.COMM_WORLD, [p0, p1], [Nx, Ny], cell_type=CellType.triangle)

    tdim = mesh0.topology.dim
    num_local_cells = mesh0.topology.index_map(tdim).size_local
    h = mesh0.comm.allreduce(
        float(np.min(mesh0.h(tdim, np.arange(num_local_cells)))), op=MPI.MIN
    )

    coords = mesh0.geometry.x
    x_range = np.max(coords[:, 0]) - np.min(coords[:, 0])
    y_range = np.max(coords[:, 1]) - np.min(coords[:, 1])
    d = max(x_range, y_range)

    # ---- Function spaces ----
    Vu = functionspace(mesh0, ("Lagrange", degree_u, (mesh0.geometry.dim,)))
    v = TestFunction(Vu)
    u = Function(Vu, name="displacement")
    u_old = Function(Vu)
    u_prev = Function(Vu)
    u_old_prev = Function(Vu)

    Vp = functionspace(mesh0, (p_space, degree_p))
    dp = TrialFunction(Vp)
    q = TestFunction(Vp)
    p = Function(Vp, name="p")
    p_old = Function(Vp)
    p_prev = Function(Vp)
    p_old_prev = Function(Vp)

    # ---- Material tagging (cbox: classic open C-shape bracket) ----
    tol = 1e-8
    marker = np.zeros(num_local_cells, dtype=np.int32)
    cells_medium1 = locate_entities(
        mesh0, tdim,
        lambda x: (x[0] >= 0.1 - tol) & (x[1] >= 0.1 - tol) & (x[1] <= 0.4 + tol),
    )
    marker[cells_medium1] = 2

    potential_body1 = locate_entities(mesh0, tdim, lambda x: (x[0] <= 0.1 + tol))
    potential_body2 = locate_entities(
        mesh0, tdim, lambda x: (x[1] <= 0.1 + tol) & (x[0] <= 1.0 + tol)
    )
    potential_body3 = locate_entities(
        mesh0, tdim, lambda x: (x[1] >= 0.4 - tol) & (x[0] <= 1.0 + tol)
    )
    all_potential_body = np.concatenate([potential_body1, potential_body2, potential_body3])
    for cell_id in all_potential_body:
        if marker[cell_id] == 0:
            marker[cell_id] = 1

    marker[marker == 0] = 2

    material_tags = meshtags(mesh0, tdim, np.arange(num_local_cells), marker)

    quadrature_degree = 2 * fe_degree + 2
    metadata = {"quadrature_degree": quadrature_degree}
    dx = Measure("dx", domain=mesh0, subdomain_data=material_tags, metadata=metadata)

    # ---- BCs (cbox: fixed left wall + prescribed top-vertex displacement) ----
    left = lambda x: np.isclose(x[0], 0.0)
    top_right_corner = lambda x: np.isclose(x[0], 1.0) & np.isclose(x[1], 0.5)
    left_facet = locate_entities_boundary(mesh0, tdim - 1, left)
    top_vertex = locate_entities_boundary(mesh0, 0, top_right_corner)

    ul = np.array((0,) * mesh0.geometry.dim, dtype=default_scalar_type)
    bcl = dirichletbc(ul, locate_dofs_topological(Vu, tdim - 1, left_facet), Vu)

    # ---- Kinematics / energy ----
    I = variable(Identity(len(u)))
    F_0 = variable(I + grad(u))
    F = as_tensor([[F_0[0, 0], F_0[0, 1], 0],
                   [F_0[1, 0], F_0[1, 1], 0],
                   [0, 0, 1]])
    C = variable(F.T * F)
    Ic = variable(tr(C))
    Jdet = variable(det(F))

    psi = (K / 2) * (ln(Jdet)) ** 2 + (mu / 2) * (Jdet ** (-2 / 3) * Ic - 3)
    psi_2 = (mu / 2) * (Jdet ** (-2 / 3) * Ic - 3)

    Pi_body = psi * dx(1) 
    Pi_m = gamma * psi_2 * dx(2)

    # Regularization Hessian Term
    Hu = grad(grad(u))

    k_HuHu = default_scalar_type(1.0e-6)

    Pi_HuHu = k_HuHu * inner(Hu, Hu) * dx(2)

    Pi_total = Pi_body + Pi_m + Pi_HuHu

    L = derivative(Pi_total, u, v)
    bilin = beta_1 / d ** 2 * inner(dp, q) * dx(2) + beta_2 * inner(grad(dp), grad(q)) * dx(2)
    lin = beta_1 / d * inner(((F[0, 1] - F[1, 0]) / (F[0, 0] + F[1, 1])), q) * dx(2)

    dof_count = (
        Vu.dofmap.index_map.size_global * Vu.dofmap.index_map_bs
        + Vp.dofmap.index_map.size_global
    )
    potential = MPI.COMM_WORLD.allreduce(
        assemble_scalar(form(Pi_total)),
        op=MPI.SUM
    )

    timings["total"] = mpi_time(mesh0.comm, t_total_start)

    def update_p():
        t0 = time.perf_counter()

        A = create_matrix(form(bilin))
        b = create_vector(Vp)

        assemble_matrix(A, form(bilin))
        A.assemble()
        A.shift(3e-14)

        with b.localForm() as b_local:
            b_local.set(0.0)

        assemble_vector(b, form(lin))
        b.ghostUpdate(
            addv=PETSc.InsertMode.ADD,
            mode=PETSc.ScatterMode.REVERSE
        )

        solver = PETSc.KSP().create(mesh0.comm)
        solver.setOperators(A)
        solver.setType(PETSc.KSP.Type.PREONLY)
        solver.getPC().setType(PETSc.PC.Type.LU)
        solver.setFromOptions()

        x = create_vector(Vp)
        solver.solve(b, x)

        x.copy(p.x.petsc_vec)
        p.x.petsc_vec.ghostUpdate(
            addv=PETSc.InsertMode.INSERT,
            mode=PETSc.ScatterMode.FORWARD
        )

        solver.destroy()
        A.destroy()
        b.destroy()
        x.destroy()

        timings["p_update"] += mpi_time(mesh0.comm, t0)

    def compute_error(a, a_old):
        a_diff = a.x.array - a_old.x.array
        a_norm = np.linalg.norm(a.x.array)
        err_a = np.linalg.norm(a_diff)
        return err_a, err_a / (a_norm + 1e-14), a_norm, a_diff

    def staggered_residuals():
        err_u, u_rel_diff, u_norm, u_diff = compute_error(u, u_old)
        err_p, p_rel_diff, p_norm, p_diff = compute_error(p, p_old)
        concat_norm = np.linalg.norm(np.concatenate((u.x.array, p.x.array)))
        err = np.linalg.norm(np.concatenate((u_diff, p_diff)))
        return err, err_u, err_p, u_norm, p_norm, err / (concat_norm + 1e-14), u_rel_diff, p_rel_diff

    timings["setup"] = mpi_time(mesh0.comm, t_setup_start)

    # ---- adaptive load stepping ----
    current_displacement = 0.0
    step_size = initial_step_size
    n = 0
    while current_displacement > target_displacement:
        k = 0
        j = 0
        success_ad = False

        while not success_ad and k < max_attempts:
            k += 1
            next_displacement = max(current_displacement - step_size, target_displacement)

            ut = default_scalar_type(next_displacement)
            bct = dirichletbc(ut, locate_dofs_topological(Vu.sub(1), 0, top_vertex), Vu.sub(1))
            bcu = [bcl, bct]

            u_prev.x.array[:] = u.x.array[:]; u_old_prev.x.array[:] = u_old.x.array[:]
            p_prev.x.array[:] = p.x.array[:]; p_old_prev.x.array[:] = p_old.x.array[:]

            problem = NonlinearPDE_SNESProblem(L, u, bcu)
            b = create_vector(Vu)
            J_L = create_matrix(problem.a)
            snes = PETSc.SNES().create()
            snes.setFunction(problem.F, b)
            snes.setJacobian(problem.J, J_L)
            snes.setTolerances(atol=atol_newton, rtol=rtol_newton,
                                stol=stol_newton, max_it=max_it_newton)
            snes.getKSP().setType("preonly")
            snes.getKSP().getPC().setType("lu")

            opts = PETSc.Options()
            opts["pc_factor_mat_solver_type"] = "mumps"
            opts["snes_linesearch_type"] = linesearch
            opts["snes_max_linear_solve_fail"] = 20
            opts["snes_linesearch_max_it"] = 20
            if verbose:
                opts["snes_monitor"] = None
                opts["snes_linesearch_monitor"] = None
                opts["snes_converged_reason"] = None
            snes.setFromOptions()

            x = None
            try:
                itr = 0
                c0 = False
                staggered_local = 0

                while not c0:

                    t_staggered = time.perf_counter()

                    staggered_local += 1
                    stats["staggered_iterations"] += 1
                    itr += 1
                    x = u.x.petsc_vec.copy()
                    x.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)

                    t_snes = time.perf_counter()
                    snes.solve(None, x)
                    timings["snes"] += mpi_time(mesh0.comm, t_snes)

                    newton_it = snes.getIterationNumber()
                    stats["newton_iterations"] += newton_it

                    if snes.getConvergedReason() > 0:
                        x.copy(u.x.petsc_vec)
                        u.x.petsc_vec.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
                    else:
                        raise RuntimeError(f"Newton failed: reason={snes.getConvergedReason()}")
                    if verbose:
                        print(f"  Newton iters: {snes.getIterationNumber()}, "
                              f"residual: {snes.getFunctionNorm():.3e}")
                    x.destroy(); x = None

                    update_p()

                    err, err_u, err_p, u_norm, p_norm, err_rel, u_rel, p_rel = staggered_residuals()
                    c0 = err < err_tol_staggered
                    if itr > max_it_staggered and not c0:
                        raise RuntimeError("Staggered iterations failed, must reduce load step size")

                    u_old.x.array[:] = u.x.array[:]; p_old.x.array[:] = p.x.array[:]

                    timings["staggered"] += mpi_time(
                        mesh0.comm,
                        t_staggered
                    )
                success_ad = True
                n += 1
                stats["load_steps"] += 1
                stats["successful_attempts"] += 1

                current_displacement = next_displacement
                if current_displacement <= crit_displacement:
                    increase_factor = 1.1
                step_size = min(step_size * increase_factor, max_step_size)

            except Exception as e:

                if verbose:
                    print(f"Step failed: {e}")
                    traceback.print_exc()
                    stats["failed_attempts"] += 1
                u.x.array[:] = u_prev.x.array[:]; u_old.x.array[:] = u_old_prev.x.array[:]
                p.x.array[:] = p_prev.x.array[:]; p_old.x.array[:] = p_old_prev.x.array[:]
                step_size = max(step_size * decrease_factor, min_step_size)
                if step_size == min_step_size:
                    j += 1

            finally:
                snes.destroy(); J_L.destroy(); b.destroy()
                if x is not None:
                    x.destroy()

            if j == 2:
                break

        if j == 2 or (not success_ad and k == max_attempts):
            if verbose:
                print(f"Step {n + 1} failed after {k} attempts; "
                      f"final displacement {current_displacement:.6f}")
            break

    potential = MPI.COMM_WORLD.allreduce(assemble_scalar(form(Pi_total)), op=MPI.SUM)

    if verbose:
        print(f"[p={fe_degree}, {Nx}x{Ny}] steps={n}, "
              f"final displacement={current_displacement:.6f}, dof={dof_count}, "
              f"h={h:.3e}, potential={potential:.3e}")

    return {
        "fe_degree": fe_degree,
        "Nx": Nx,
        "Ny": Ny,
        "h": h,
        "dof": dof_count,
        "final_displacement": current_displacement,
        "n_steps": n,
        "potential": potential,

        "time_total": timings["total"],
        "time_setup": timings["setup"],
        "time_snes": timings["snes"],
        "time_p_update": timings["p_update"],
        "time_staggered": timings["staggered"],

        "newton_iterations": stats["newton_iterations"],
        "staggered_iterations": stats["staggered_iterations"],
        "failed_attempts": stats["failed_attempts"],
        "successful_attempts": stats["successful_attempts"],

        "u": u,
        "p": p,
        "Vu": Vu,
        "mesh": mesh0,
    }


if __name__ == "__main__":

    import platform
    import psutil
    import os
    import matplotlib.pyplot as plt

    # ------------------------------------------------------------
    # Hardware
    # ------------------------------------------------------------

    def print_hardware_info():
        print("\n--- Hardware information ---")
        print(f"OS              : {platform.system()} {platform.release()}")
        print(f"Architecture    : {platform.machine()}")
        print(f"Processor       : {platform.processor()}")
        print(f"CPU cores       : {os.cpu_count()}")
        print(f"Physical cores  : {psutil.cpu_count(logical=False)}")
        print(f"Logical cores   : {psutil.cpu_count(logical=True)}")

        mem = psutil.virtual_memory()
        print(f"Total memory    : {mem.total / (1024**3):.2f} GB")

        print(f"Python          : {platform.python_version()}")
        print(f"Hostname        : {platform.node()}")
        print("-----------------------------\n")

    # ------------------------------------------------------------
    # Reference solution
    # ------------------------------------------------------------

    print("\n" + "=" * 70)
    print("REFERENCE SOLUTION")
    print("=" * 70)

    reference = run_case(
        80,
        40,
        1,
        verbose=False,
    )

    potential_ref = reference["potential"]

    print("\nReference solution:")
    print("  FE degree : p=1")
    print("  Mesh      : 80x40")
    print(f"  potential_ref   : {potential_ref:.3e}")
    print(f"  Runtime   : {reference['time_total']:.3f} s")

    # ------------------------------------------------------------
    # h/p convergence sweep
    # ------------------------------------------------------------

    sweep_results = []

    degrees = [1,2,3]
    meshes = [
        (40, 20),
        (80, 40),
        (120, 60),
        (160, 80),
        (200, 100),
    ]

    print("\n" + "=" * 70)
    print("H/P CONVERGENCE SWEEP")
    print("=" * 70)

    for fe_degree in degrees:

        print("\n" + "-" * 70)
        print(f"Polynomial degree p = {fe_degree}")
        print("-" * 70)

        potential_coarse = None

        for mesh_id, (Nx, Ny) in enumerate(meshes):

            res = run_case(
                Nx,
                Ny,
                fe_degree,
                verbose=False,
            )

            if potential_coarse is None:
                potential_coarse = res["potential"]

            potential_error = abs(res["potential"] - potential_ref)
            res["potential_error"] = potential_error
            sweep_results.append(res)

            print(f"\n[p={fe_degree}] Mesh={Nx}x{Ny}")
            print(f"  potential       = {res['potential']:.3e}")
            print(f"  Error    = {potential_error:.3e}")
            print(f"  DOF       = {res['dof']}")
            print(f"  Run Time  = {res['time_total']:.3f} s")
            print(f"  Newton    = {res['newton_iterations']}")
            print(f"  Staggered = {res['staggered_iterations']}")

    # ============================================================
    # OBSERVED CONVERGENCE RATES
    # ============================================================

    print("\n" + "=" * 70)
    print("OBSERVED CONVERGENCE RATES")
    print("=" * 70)

    rates = []

    for fe_degree in degrees:

        data = sorted(
            [r for r in sweep_results if r["fe_degree"] == fe_degree],
            key=lambda r: r["h"],
            reverse=True
        )

        for i in range(len(data) - 1):

            coarse = data[i]
            fine = data[i + 1]

            E_coarse = coarse["potential_error"]
            E_fine = fine["potential_error"]

            h_coarse = coarse["h"]
            h_fine = fine["h"]

            if E_coarse > 0.0 and E_fine > 0.0:

                rate = (
                    np.log(E_coarse / E_fine)
                    / np.log(h_coarse / h_fine)
                )

                rates.append({
                    "p": fe_degree,
                    "h_coarse": h_coarse,
                    "h_fine": h_fine,
                    "rate": rate
                })

                print(
                    f"p={fe_degree} | "
                    f"h: {h_coarse:.3e} -> {h_fine:.3e} | "
                    f"rate = {rate:.3f}"
                )

    # ============================================================
    # PLOT SETTINGS
    # ============================================================

    cBlue = np.array([0, 193, 212]) / 255
    cRed = np.array([255, 79, 79]) / 255
    cYellow = np.array([255, 220, 54]) / 255
    cPink = np.array([255, 174, 162]) / 255

    plot_colors = [cBlue, cRed, cYellow, cPink]

    a4_w_mm = 210
    a4_h_mm = 297

    fig_w_in = (a4_w_mm / 2) / 25.4
    fig_h_in = (a4_h_mm / 2) / 25.4

    plt.rcParams.update({
        "font.size": 12,
        "axes.labelsize": 12,
        "axes.titlesize": 12,
        "legend.fontsize": 11,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "lines.linewidth": 1.5,
        "lines.markersize": 5,
    })

    # ============================================================
    # ACCURACY VS COMPUTATIONAL COST
    # ============================================================

    print_hardware_info()

    fig, ax = plt.subplots(figsize=(fig_w_in, fig_h_in))

    for i, fe_degree in enumerate(degrees):

        data = sorted(
            [r for r in sweep_results if r["fe_degree"] == fe_degree],
            key=lambda r: r["time_total"]
        )

        runtime = np.array([r["time_total"] for r in data])
        error = np.array([r["potential_error"] for r in data])

        mask = (runtime > 0.0) & (error > 0.0)

        ax.loglog(
            runtime[mask],
            error[mask],
            marker="o",
            color=plot_colors[i % len(plot_colors)],
            label=f"$p={fe_degree}$"
        )

    ax.set_xlabel("Computational time [s]")
    ax.set_ylabel(r"$|E_h-E_{\mathrm{ref}}|$")
    ax.set_title("Error vs. computational time")

    ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.6)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(dir_name, "Error_vs_Time.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved to directory: {os.path.join(dir_name, 'Error_vs_Time.png')}")

    # ============================================================
    # ERROR VS DOF
    # ============================================================

    fig, ax = plt.subplots(figsize=(fig_w_in, fig_h_in))

    for i, fe_degree in enumerate(degrees):

        data = sorted(
            [r for r in sweep_results if r["fe_degree"] == fe_degree],
            key=lambda r: r["dof"]
        )

        dofs = np.array([r["dof"] for r in data])
        errors = np.array([r["potential_error"] for r in data])

        mask = (dofs > 0) & (errors > 0)

        ax.loglog(
            dofs[mask],
            errors[mask],
            marker="o",
            color=plot_colors[i % len(plot_colors)],
            label=f"$p={fe_degree}$"
        )

    ax.set_xlabel("DOF")
    ax.set_ylabel(r"$|E_h-E_{\mathrm{ref}}|$")
    ax.set_title("Error vs. degrees of freedom")

    ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.6)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(dir_name, "Error_vs_DOF.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved to directory: {os.path.join(dir_name, 'Error_vs_DOF.png')}")