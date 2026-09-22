"""
Finger sliding transversally over a fixed half-cylinder obstacle -- a
Third-Medium-Contact (TMC) snap-through problem, geometrically and
materially aligned with:

  Frederiksen, Dalklint, Sigmund, Poulios (2025),
  "Improved third medium formulation for 3D topology optimization
  with contact", CMAME 436, 117595.

Material parameters (mu, K, alpha, gamma0) and the two length scales
L, t are taken verbatim from that paper's Table 3 (reused there for
the "snap example" too). The finger/obstacle geometry itself is NOT
given numerically in the paper (only sketched in Fig. 6) -- the sizes
below are reasonable defaults built around L and t; rescale freely.

The half-cylinder obstacle is treated as perfectly rigid and is never
discretized: it is boolean-cut out of the mesh, leaving only its
curved boundary (a fixed Dirichlet BC) behind. Only the finger (an
elastic solid) and the third medium filling the gap around the
obstacle are meshed.

Snap-through handling follows the paper's Algorithm 1: plain
displacement-controlled Newton first; on failure, a temporary
(1/dt) * inner(u - u0, v) * dx damping term is added and dt is grown
from a small value back towards infinity, recovering the true static
equilibrium across the unstable branch.
"""

import time
import os
import traceback
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri

import gmsh
from mpi4py import MPI
from petsc4py import PETSc

from ufl import *

from dolfinx.io import gmsh as gmshio
from dolfinx import default_scalar_type, plot
from dolfinx.mesh import *
from dolfinx.fem import *
from dolfinx.fem.petsc import (
    create_matrix, assemble_matrix, apply_lifting,
    assemble_vector, set_bc, create_vector
)

file_name = "snap_finger"
linesearch = "bt"

def mpi_time(comm, t0):
    return comm.allreduce(time.perf_counter() - t0, op=MPI.MAX)

# ============================================================
# GEOMETRY (mm) -- L and t are from the paper's Table 3;
# everything else is a reasonable default, not paper-sourced.
# ============================================================
L = 4.0                 # domain height                      [paper: Table 3, "Width L"]
t_gap = 0.40             # initial finger-obstacle clearance  [paper: Table 3, "Gap height t"]
Lx = 1.5 * L             # domain width -- generous on both sides, not just a thin buffer [assumed]
obstacle_r = 0.5          # half-cylinder radius                [assumed]
finger_w = 0.5           # finger width                        [assumed]

obstacle_xc = Lx / 2.0
finger_bottom = obstacle_r + t_gap
finger_x0 = obstacle_xc - obstacle_r - finger_w / 2.0 - t_gap
sweep_target = (obstacle_xc + obstacle_r + finger_w / 2.0 + t_gap) - (finger_x0 + finger_w / 2.0)

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


def build_mesh(comm, lc):
    """
    Fragmented gmsh mesh containing only the finger (elastic solid,
    physical group 1) and the third medium (physical group 2). The
    half-cylinder obstacle is treated as perfectly rigid and is never
    discretized: it is boolean-cut out of the domain, leaving only its
    curved boundary behind, where a fixed Dirichlet BC is applied.
    """
    gmsh.initialize()
    if comm.rank == 0:
        gmsh.model.add("snap_finger")
        occ = gmsh.model.occ

        domain = occ.addRectangle(0, 0, 0, Lx, L)
        obstacle = occ.addDisk(obstacle_xc, 0, 0, obstacle_r, obstacle_r)
        finger = occ.addRectangle(finger_x0, finger_bottom, 0, finger_w, L - finger_bottom)

        occ.synchronize()
        # Remove the rigid obstacle's interior entirely -- only its
        # curved boundary remains, as a hole in the domain.
        domain_cut, _ = occ.cut([(2, domain)], [(2, obstacle)], removeTool=True)
        occ.synchronize()

        # Fragment the cut domain with the finger to get two conformal,
        # non-overlapping regions: finger (solid) and everything else (medium).
        occ.fragment(domain_cut, [(2, finger)])
        occ.synchronize()

        solid_tags, medium_tags = [], []
        for dim, tag in gmsh.model.getEntities(2):
            xc, yc, _ = gmsh.model.occ.getCenterOfMass(dim, tag)
            in_finger = (finger_x0 <= xc <= finger_x0 + finger_w) and (yc >= finger_bottom)
            if in_finger:
                solid_tags.append(tag)
            else:
                medium_tags.append(tag)

        gmsh.model.addPhysicalGroup(2, solid_tags, 1, name="solid")
        gmsh.model.addPhysicalGroup(2, medium_tags, 2, name="medium")

        gmsh.model.mesh.setSize(gmsh.model.getEntities(0), lc)
        gmsh.model.mesh.generate(2)

    mesh0, cell_tags, *_ = gmshio.model_to_mesh(gmsh.model, comm, 0, gdim=2)
    gmsh.finalize()
    return mesh0, cell_tags


def compute_reaction_force(L_form, Vu, dofs_owned):
    """
    Reaction force at a set of prescribed dofs, extracted from the raw
    (unconstrained) internal-force residual -- no Lagrange multiplier
    needed. After assembling and folding ghost contributions back to
    their owning rank, the residual entries at the constrained dofs
    equal the reaction force required to enforce that constraint.
    """
    b = create_vector(Vu)
    with b.localForm() as b_local:
        b_local.set(0.0)
    assemble_vector(b, form(L_form))
    b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    b_array = b.getArray(readonly=True)
    local_sum = float(np.sum(b_array[dofs_owned])) if len(dofs_owned) else 0.0
    total = Vu.mesh.comm.allreduce(local_sum, op=MPI.SUM)
    b.destroy()
    return total


def save_deformation_figure(u, mesh0, step, u_Dx, output_dir):
    """Save one PNG of the current deformed configuration, colored by |u|."""
    Vu_vis = functionspace(mesh0, ("Lagrange", 1, (mesh0.geometry.dim,)))
    u_vis = Function(Vu_vis)
    u_vis.interpolate(u)

    topology, _, geometry = plot.vtk_mesh(Vu_vis)
    tri = topology.reshape(-1, 4)[:, 1:]

    bs = Vu_vis.dofmap.index_map_bs
    u_vals = u_vis.x.array.reshape(-1, bs)
    u_mag = np.linalg.norm(u_vals, axis=1)

    x_def = geometry[:, 0] + u_vals[:, 0]
    y_def = geometry[:, 1] + u_vals[:, 1]

    triang = mtri.Triangulation(x_def, y_def, tri)

    fig, ax = plt.subplots(figsize=(6, 6))
    tpc = ax.tricontourf(triang, u_mag, levels=20, cmap="viridis")
    ax.triplot(triang, color="k", linewidth=0.1, alpha=0.3)

    # The half-cylinder is rigid and unmeshed -- draw its fixed outline
    # so it doesn't just look like empty space in the plot.
    theta = np.linspace(np.pi, 0, 100)
    arc_x = obstacle_xc + obstacle_r * np.cos(theta)
    arc_y = obstacle_r * np.sin(theta)
    ax.fill_between(arc_x, 0.0, arc_y, color="0.35", zorder=5)

    ax.set_aspect("equal")
    ax.set_xlim(-0.2, Lx + 0.2)
    ax.set_ylim(-0.2, L + 0.2)
    ax.set_title(f"step {step:04d}, u_Dx = {u_Dx:.3f} mm")
    fig.colorbar(tpc, ax=ax, label="|u| [mm]")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, f"{file_name}_step_{step:04d}.png"), dpi=150)
    plt.close(fig)


def solve_pseudo_time_step(u, u_anchor, L_form, Vu, bcu,
                            atol_newton, rtol_newton, stol_newton, max_it_newton,
                            dt_init=1e-3, dt_max=1e9, verbose=False):
    """
    Algorithm 1 (Frederiksen et al. 2025): damped pseudo time-stepping
    fallback for a single Dirichlet-BC set (bcu) where plain Newton
    fails -- i.e. a snap-through/snap-back point. Solves in place on
    `u`; returns True/False. On failure, `u` is left at u_anchor.
    """
    v = TestFunction(Vu)
    dt = Constant(u.function_space.mesh, default_scalar_type(dt_init))
    u0 = Function(Vu)
    u0.x.array[:] = u_anchor.x.array[:]

    dx_full = Measure("dx", domain=u.function_space.mesh)
    L_damped = L_form + (1.0 / dt) * inner(u - u0, v) * dx_full
    problem = NonlinearPDE_SNESProblem(L_damped, u, bcu)

    dt_val = dt_init
    while True:
        dt.value = dt_val

        b = create_vector(Vu)
        J_L = create_matrix(problem.a)
        snes = PETSc.SNES().create()
        snes.setFunction(problem.F, b)
        snes.setJacobian(problem.J, J_L)
        snes.setTolerances(atol=atol_newton, rtol=rtol_newton, stol=stol_newton, max_it=max_it_newton)
        snes.getKSP().setType("preonly")
        snes.getKSP().getPC().setType("lu")
        opts = PETSc.Options()
        opts["pc_factor_mat_solver_type"] = "mumps"
        opts["snes_linesearch_type"] = linesearch
        snes.setFromOptions()

        x = u.x.petsc_vec.copy()
        x.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
        snes.solve(None, x)
        converged = snes.getConvergedReason() > 0
        n_it = snes.getIterationNumber()

        if converged:
            x.copy(u.x.petsc_vec)
            u.x.petsc_vec.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)

        snes.destroy(); J_L.destroy(); b.destroy(); x.destroy()

        if converged:
            u0.x.array[:] = u.x.array[:]
            if verbose:
                print(f"    [damped] dt={dt_val:.3e} converged in {n_it} its")
            if dt_val >= dt_max:
                return True                      # damping vanished -> static solution recovered
            dt_val = min(dt_val * 10.0 / 9.0, dt_max) if n_it < 7 else dt_val
            if dt_val >= dt_max:
                dt_val = dt_max
        else:
            u.x.array[:] = u0.x.array[:]
            dt_val /= 10.0
            if verbose:
                print(f"    [damped] failed, reducing dt to {dt_val:.3e}")
            if dt_val < 1e-14:
                u.x.array[:] = u_anchor.x.array[:]
                return False


def run_snap(fe_degree=2, lc=0.05, target_displacement=None,
             Nsteps=100, Nsteps_min=20, Nsteps_max=800,
             output_dir=None, verbose=True):
    comm = MPI.COMM_WORLD
    if target_displacement is None:
        target_displacement = sweep_target
    if output_dir is None:
        output_dir = f"{file_name}_frames"
    os.makedirs(output_dir, exist_ok=True)

    degree_u = max(1, fe_degree)
    degree_p = max(1, fe_degree)

    atol_newton, rtol_newton, stol_newton, max_it_newton = 1e-12, 1e-10, 1e-12, 18
    max_it_staggered, err_tol_staggered = 100, 1e-3

    initial_step_size = target_displacement / Nsteps
    min_step_size = target_displacement / Nsteps_max
    max_step_size = target_displacement / Nsteps_min
    increase_factor, decrease_factor, max_attempts = 1.5, 0.5, 10

    # ---- Model parameters (paper Table 3: E=1 MPa, nu=0.40) ----
    mu = default_scalar_type(5 / 14)
    K = default_scalar_type(5 / 3)
    gamma0 = default_scalar_type(1e-7)       # paper Table 3 void scaling for the mechanics-only examples
    beta_1 = default_scalar_type(0.01)
    beta_2 = default_scalar_type(1e-5)

    mesh0, cell_tags = build_mesh(comm, lc)
    tdim = mesh0.topology.dim
    d = L  # characteristic length for the tangential regularization term

    quadrature_degree = 2 * fe_degree + 2
    dx = Measure("dx", domain=mesh0, subdomain_data=cell_tags,
                 metadata={"quadrature_degree": quadrature_degree})

    Vu = functionspace(mesh0, ("Lagrange", degree_u, (mesh0.geometry.dim,)))
    v = TestFunction(Vu)
    u = Function(Vu, name="displacement")
    u_prev = Function(Vu)

    Vp = functionspace(mesh0, ("Lagrange", degree_p))
    dp = TrialFunction(Vp)
    q = TestFunction(Vp)
    p = Function(Vp, name="p")
    p_old = Function(Vp)
    p_prev = Function(Vp)

    # ---- BCs ----
    def on_top(x): return np.isclose(x[1], L)
    def on_bottom(x): return np.isclose(x[1], 0.0)
    def under_finger(x): return (x[0] >= finger_x0 - 1e-9) & (x[0] <= finger_x0 + finger_w + 1e-9)
    def under_obstacle(x): return np.abs(x[0] - obstacle_xc) <= obstacle_r + 1e-9
    def on_obstacle_arc(x):
        r_from_center = np.sqrt((x[0] - obstacle_xc) ** 2 + x[1] ** 2)
        return np.isclose(r_from_center, obstacle_r, atol=1e-6)

    finger_top = locate_entities_boundary(
        mesh0, tdim - 1, lambda x: on_top(x) & under_finger(x))
    roller_top = locate_entities_boundary(
        mesh0, tdim - 1, lambda x: on_top(x) & ~under_finger(x))
    obstacle_arc = locate_entities_boundary(mesh0, tdim - 1, on_obstacle_arc)
    roller_bottom = locate_entities_boundary(
        mesh0, tdim - 1, lambda x: on_bottom(x) & ~under_obstacle(x))

    zero_vec = np.array((0.0, 0.0), dtype=default_scalar_type)
    bc_obstacle = dirichletbc(zero_vec, locate_dofs_topological(Vu, tdim - 1, obstacle_arc), Vu)
    bc_roller_top = dirichletbc(default_scalar_type(0.0),
                                 locate_dofs_topological(Vu.sub(1), tdim - 1, roller_top), Vu.sub(1))
    bc_roller_bottom = dirichletbc(default_scalar_type(0.0),
                                    locate_dofs_topological(Vu.sub(1), tdim - 1, roller_bottom), Vu.sub(1))
    bc_finger_y = dirichletbc(default_scalar_type(0.0),
                               locate_dofs_topological(Vu.sub(1), tdim - 1, finger_top), Vu.sub(1))

    fixed_bcs = [bc_obstacle, bc_roller_top, bc_roller_bottom, bc_finger_y]

    # dofs for the finger's driven x-displacement -- computed once and reused
    # both for the BC each step and for reaction-force extraction.
    finger_x_dofs = locate_dofs_topological(Vu.sub(0), tdim - 1, finger_top)
    owned_limit = Vu.dofmap.index_map.size_local * Vu.dofmap.index_map_bs
    finger_x_dofs_owned = finger_x_dofs[finger_x_dofs < owned_limit]

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

    Pi_solid = psi * dx(1)
    Pi_medium = gamma0 * psi_2 * dx(2)
    Pi_tan_R = (
        beta_1 / 2 * ((F[0, 1] - F[1, 0]) / (F[0, 0] + F[1, 1]) - 1 / d * p) ** 2
        + beta_2 / 2 * inner(grad(p), grad(p))
    ) * dx(2)
    Pi_total = Pi_solid + Pi_medium + Pi_tan_R

    L_form = derivative(Pi_total, u, v)
    bilin = beta_1 / d ** 2 * inner(dp, q) * dx(2) + beta_2 * inner(grad(dp), grad(q)) * dx(2)
    lin = beta_1 / d * inner(((F[0, 1] - F[1, 0]) / (F[0, 0] + F[1, 1])), q) * dx(2)

    def update_p():
        A = create_matrix(form(bilin))
        b = create_vector(Vp)
        assemble_matrix(A, form(bilin)); A.assemble(); A.shift(3e-14)
        with b.localForm() as b_local:
            b_local.set(0.0)
        assemble_vector(b, form(lin))
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        solver = PETSc.KSP().create(mesh0.comm)
        solver.setOperators(A)
        solver.setType(PETSc.KSP.Type.PREONLY); solver.getPC().setType(PETSc.PC.Type.LU)
        solver.setFromOptions()
        x = create_vector(Vp)
        solver.solve(b, x)
        x.copy(p.x.petsc_vec)
        p.x.petsc_vec.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
        solver.destroy(); A.destroy(); b.destroy(); x.destroy()

    def staggered_err():
        diff = np.concatenate((u.x.array - u_old_arr[0], p.x.array - p_old.x.array))
        norm = np.linalg.norm(np.concatenate((u.x.array, p.x.array)))
        return np.linalg.norm(diff) / (norm + 1e-14)

    current_displacement = 0.0
    step_size = initial_step_size
    n = 0
    stats = {"snap_events": 0}

    history = {"u_Dx": [0.0], "Fx": [0.0]}
    save_deformation_figure(u, mesh0, 0, 0.0, output_dir)

    while current_displacement < target_displacement:
        k = 0
        j = 0
        success_ad = False

        while not success_ad and k < max_attempts:
            k += 1
            next_displacement = min(current_displacement + step_size, target_displacement)

            ux = default_scalar_type(next_displacement)
            bc_finger_x = dirichletbc(ux, finger_x_dofs, Vu.sub(0))
            bcu = fixed_bcs + [bc_finger_x]

            u_prev.x.array[:] = u.x.array[:]
            p_prev.x.array[:] = p.x.array[:]

            try:
                itr, c0 = 0, False
                p_old.x.array[:] = p.x.array[:]
                u_old_arr = [u.x.array.copy()]

                while not c0:
                    itr += 1
                    problem = NonlinearPDE_SNESProblem(L_form, u, bcu)
                    b = create_vector(Vu)
                    J_L = create_matrix(problem.a)
                    snes = PETSc.SNES().create()
                    snes.setFunction(problem.F, b)
                    snes.setJacobian(problem.J, J_L)
                    snes.setTolerances(atol=atol_newton, rtol=rtol_newton,
                                        stol=stol_newton, max_it=max_it_newton)
                    snes.getKSP().setType("preonly"); snes.getKSP().getPC().setType("lu")
                    opts = PETSc.Options()
                    opts["pc_factor_mat_solver_type"] = "mumps"
                    opts["snes_linesearch_type"] = linesearch
                    snes.setFromOptions()

                    x = u.x.petsc_vec.copy()
                    x.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
                    snes.solve(None, x)

                    if snes.getConvergedReason() > 0:
                        x.copy(u.x.petsc_vec)
                        u.x.petsc_vec.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
                    else:
                        snes.destroy(); J_L.destroy(); b.destroy(); x.destroy()
                        # --- true snap-through / snap-back: fall back to Algorithm 1 ---
                        if verbose:
                            print(f"  Newton failed at u_Dx={next_displacement:.4f} mm "
                                  f"-> damped pseudo-time stepping")
                        stats["snap_events"] += 1
                        ok = solve_pseudo_time_step(
                            u, u_prev, L_form, Vu, bcu,
                            atol_newton, rtol_newton, stol_newton, max_it_newton,
                            verbose=verbose)
                        if not ok:
                            raise RuntimeError("Damped pseudo-time stepping also failed")
                        break

                    x.destroy()
                    snes.destroy(); J_L.destroy(); b.destroy()

                    update_p()
                    c0 = staggered_err() < err_tol_staggered
                    if itr > max_it_staggered and not c0:
                        raise RuntimeError("Staggered iterations failed")
                    u_old_arr[0] = u.x.array.copy()

                success_ad = True
                n += 1
                current_displacement = next_displacement
                step_size = min(step_size * increase_factor, max_step_size)

                reaction = compute_reaction_force(L_form, Vu, finger_x_dofs_owned)
                history["u_Dx"].append(current_displacement)
                history["Fx"].append(reaction)
                save_deformation_figure(u, mesh0, n, current_displacement, output_dir)

            except Exception as e:
                if verbose:
                    print(f"Step failed: {e}")
                u.x.array[:] = u_prev.x.array[:]
                p.x.array[:] = p_prev.x.array[:]
                step_size = max(step_size * decrease_factor, min_step_size)
                if step_size == min_step_size:
                    j += 1

            if j == 2:
                break

        if j == 2 or (not success_ad and k == max_attempts):
            if verbose:
                print(f"Step {n + 1} failed after {k} attempts; "
                      f"final u_Dx={current_displacement:.4f} mm")
            break

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(history["u_Dx"], history["Fx"], marker="o", markersize=3)
    ax.set_xlabel("u_Dx [mm]")
    ax.set_ylabel("Reaction force F_x (residual units)")
    ax.set_title("Force-displacement curve")
    ax.grid(True, linestyle="--", alpha=0.5)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, f"{file_name}_force_displacement.png"), dpi=200)
    plt.close(fig)

    if verbose:
        print(f"Finished: {n} accepted steps, final u_Dx={current_displacement:.4f} mm, "
              f"snap events handled: {stats['snap_events']}")
        print(f"Frames and force-displacement plot saved to: {output_dir}/")

    return {"u": u, "p": p, "Vu": Vu, "mesh": mesh0,
            "final_displacement": current_displacement, "n_steps": n,
            "snap_events": stats["snap_events"],
            "history": history, "output_dir": output_dir}


if __name__ == "__main__":
    print("Geometry (mm):")
    print(f"  L={L}, Lx={Lx}, obstacle_r={obstacle_r}, finger_w={finger_w}, "
          f"gap t={t_gap}, sweep target={sweep_target:.3f}")
    result = run_snap(fe_degree=2, lc=0.05, verbose=True)