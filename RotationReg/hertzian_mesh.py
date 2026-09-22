"""
hertzian_mesh.py

In-memory, resolution-parametrized version of hertzian0_4.geo.

Instead of maintaining several fixed .msh files (hertzian0_1.msh,
hertzian0_2.msh, ...) for a mesh-convergence study, this module rebuilds the
same geometry with the gmsh Python API for any base mesh size `lc` you ask
for, and hands the result straight to dolfinx -- no file is written unless
you explicitly ask for one.

`lc=0.03` reproduces hertzian0_4.msh almost exactly (106457 vs 106459 nodes),
so this is a faithful translation, not an approximation.

Usage
-----
    from hertzian_mesh import generate_mesh
    from mpi4py import MPI

    mesh_data = generate_mesh(lc=0.03, comm=MPI.COMM_WORLD, gdim=2)
    mesh0, cell_tags, facet_tags = mesh_data.mesh, mesh_data.cell_tags, mesh_data.facet_tags

This is a drop-in replacement for:

    mesh_data = gmshio.read_from_msh(mesh_file_name, MPI.COMM_WORLD, 0, gdim=2)
"""
import math
import gmsh
# dolfinx is only needed inside generate_mesh(), imported lazily there so
# this module's geometry-only helpers can still be used/tested without a
# dolfinx installation available.


def _build_geometry(lc):
    """Re-creates hertzian0_4.geo in the currently-initialized gmsh model.
    Everything (including the local Ball-field refinement) scales off the
    single parameter `lc`, exactly as in the original .geo file."""
    p0x, p0y = 0.0, 0.0
    p1x, p1y = 2.2, 2.0
    p2y = 4.0
    R = 1.0

    gmsh.model.add("hertzian")
    geo = gmsh.model.geo

    def pt(x, y, size):
        return geo.addPoint(x, y, 0, size)

    p1 = pt(p0x, p0y, lc)
    p2 = pt(p0x, p1y, lc)
    p3 = pt(p1x, p1y, lc)
    p4 = pt(p1x, p0y, lc)
    p5 = pt(p0x, p2y, lc)
    p6 = pt(p1x, p2y, lc)
    p7 = pt(p1x / 2, p1y + R + 0.05, lc)
    p8 = pt(1.1 - math.sqrt(0.0975), 4, lc)
    p9 = pt(1.1 + math.sqrt(0.0975), 4, lc)
    p10 = pt(p1x / 2, p1y + 0.05, lc)
    p11 = pt(p1x / 2, p1y, lc)          # contact node, target_node = (1.1, 2.0)
    p12 = pt(0.85, 4, lc)
    p13 = pt(1.35, 4, lc)
    p14 = pt(p1x / 2 - 0.12, p1y, 1 * lc)
    p15 = pt(p1x / 2 + 0.12, p1y, 1 * lc)

    l1 = geo.addLine(p1, p2)
    l2 = geo.addLine(p2, p14)
    l15 = geo.addLine(p14, p11)
    l3 = geo.addLine(p3, p4)
    l4 = geo.addLine(p4, p1)
    l5 = geo.addLine(p5, p2)
    l6 = geo.addLine(p11, p15)
    l16 = geo.addLine(p15, p3)
    l7 = geo.addLine(p3, p6)
    l8 = geo.addCircleArc(p8, p7, p10)
    l9 = geo.addCircleArc(p10, p7, p9)
    l10 = geo.addLine(p9, p13)
    l13 = geo.addLine(p13, p12)
    l14 = geo.addLine(p12, p8)
    l11 = geo.addLine(p8, p5)
    l12 = geo.addLine(p6, p9)

    cl1 = geo.addCurveLoop([l1, l2, l15, l6, l16, l3, l4])   # elastic half space
    s1 = geo.addPlaneSurface([cl1])

    cl2 = geo.addCurveLoop([l10, l13, l14, l8, l9])          # cylinder (hole)
    s2 = geo.addPlaneSurface([cl2])

    cl3 = geo.addCurveLoop([l5, l2, l15, l6, l16, l7, l12, l10, l13, l14, l11])
    s3 = geo.addPlaneSurface([cl3, cl2])                     # third medium

    geo.synchronize()

    gmsh.model.addPhysicalGroup(2, [s1], 1, "elastic_half_space")
    gmsh.model.addPhysicalGroup(2, [s2], 2, "contact_body")
    gmsh.model.addPhysicalGroup(2, [s3], 3, "third_medium")
    gmsh.model.addPhysicalGroup(1, [l4], 10, "bottom_boundary")
    gmsh.model.addPhysicalGroup(1, [l13], 11, "top_boundary")

    field = gmsh.model.mesh.field.add("Ball")
    gmsh.model.mesh.field.setNumber(field, "VIn", 0.02 * lc)
    gmsh.model.mesh.field.setNumber(field, "VOut", lc)
    gmsh.model.mesh.field.setNumber(field, "XCenter", 1.1)
    gmsh.model.mesh.field.setNumber(field, "YCenter", 2.025)
    gmsh.model.mesh.field.setNumber(field, "Radius", 0.1)
    gmsh.model.mesh.field.setNumber(field, "Thickness", 0.2)
    gmsh.model.mesh.field.setAsBackgroundMesh(field)

    gmsh.option.setNumber("Mesh.ElementOrder", 1)
    gmsh.option.setNumber("Mesh.Algorithm", 8)


def generate_mesh(lc, comm, gdim=2, model_rank=0, write_msh_path=None, verbose=False):
    """Build the Hertzian-contact geometry at mesh size `lc` and return a
    dolfinx MeshData object (mesh, cell_tags, facet_tags), exactly like
    gmshio.read_from_msh would.

    Only rank `model_rank` builds/meshes with gmsh; gmshio.model_to_mesh
    handles distributing the resulting mesh across the MPI communicator.

    write_msh_path : optional str -- if given, also writes the mesh to disk
                      at that path (rank `model_rank` only).
    """
    from dolfinx.io import gmsh as gmshio

    gmsh.initialize()
    if not verbose:
        gmsh.option.setNumber("General.Terminal", 0)

    if comm.rank == model_rank:
        _build_geometry(lc)
        gmsh.model.mesh.generate(2)
        if write_msh_path is not None:
            gmsh.write(write_msh_path)

    mesh_data = gmshio.model_to_mesh(gmsh.model, comm, model_rank, gdim=gdim)
    gmsh.finalize()
    return mesh_data


if __name__ == "__main__":
    # Quick standalone sanity check (no MPI/dolfinx needed for this part).
    for lc in [0.06, 0.03, 0.015]:
        gmsh.initialize()
        gmsh.option.setNumber("General.Terminal", 0)
        _build_geometry(lc)
        gmsh.model.mesh.generate(2)
        n_nodes = len(gmsh.model.mesh.getNodes()[0])
        print(f"lc={lc:<6} nodes={n_nodes}")
        gmsh.finalize()