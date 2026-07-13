import gmsh
from dolfinx.io import XDMFFile
from dolfinx.io import gmsh as gmshio
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.rank

# Tamanho aproximado dos elementos.
# Diminua para refinar a malha.
h = 0.05

gmsh.initialize()
gmsh.option.setNumber("General.Terminal", 1)

if rank == 0:
    gmsh.model.add("rounded_cross")

    geo = gmsh.model.geo

    def point(x, y):
        return geo.addPoint(x, y, 0.0, h)

    # ============================================================
    # Pontos do retângulo central
    # ============================================================

    A = point(-1.0, -2.0)
    B = point(1.0, -2.0)

    R0 = point(1.0, 0.0)
    C = point(1.0, 2.0)

    D = point(-1.0, 2.0)
    L0 = point(-1.0, 0.0)

    # ============================================================
    # Pontos da parte superior
    # ============================================================

    top_center = point(0.0, 2.4)

    top_left = point(-1.0, 2.4)
    top_right = point(1.0, 2.4)
    top_apex = point(0.0, 3.2)

    # ============================================================
    # Pontos do lóbulo esquerdo
    # ============================================================

    left_center = point(-1.0, 1.0)
    left_outer = point(-2.2, 1.0)

    # ============================================================
    # Pontos do lóbulo direito
    # ============================================================

    right_center = point(1.0, 1.0)
    right_outer = point(2.2, 1.0)

    # ============================================================
    # Pontos da parte inferior
    # ============================================================

    bottom_center = point(0.0, -2.0)
    bottom_apex = point(0.0, -3.2)

    # ============================================================
    # Linhas do retângulo central
    # ============================================================

    line_bottom = geo.addLine(A, B)

    line_right_lower = geo.addLine(B, R0)
    line_right_upper = geo.addLine(R0, C)

    line_top = geo.addLine(C, D)

    line_left_upper = geo.addLine(D, L0)
    line_left_lower = geo.addLine(L0, A)

    central_loop = geo.addCurveLoop(
        [
            line_bottom,
            line_right_lower,
            line_right_upper,
            line_top,
            line_left_upper,
            line_left_lower,
        ]
    )

    central_surface = geo.addPlaneSurface([central_loop])

    # ============================================================
    # Parte superior
    # ============================================================

    top_right_side = geo.addLine(C, top_right)

    # Cada semielipse é dividida em dois arcos, pois cada arco
    # elíptico deve ter abertura menor que pi.
    top_arc_right = geo.addEllipseArc(
        top_right,
        top_center,
        top_right,
        top_apex,
    )

    top_arc_left = geo.addEllipseArc(
        top_apex,
        top_center,
        top_right,
        top_left,
    )

    top_left_side = geo.addLine(top_left, D)

    top_loop = geo.addCurveLoop(
        [
            -line_top,
            top_right_side,
            top_arc_right,
            top_arc_left,
            top_left_side,
        ]
    )

    top_surface = geo.addPlaneSurface([top_loop])

    # ============================================================
    # Lóbulo esquerdo
    # ============================================================

    left_arc_upper = geo.addEllipseArc(
        D,
        left_center,
        left_outer,
        left_outer,
    )

    left_arc_lower = geo.addEllipseArc(
        left_outer,
        left_center,
        left_outer,
        L0,
    )

    left_loop = geo.addCurveLoop(
        [
            -line_left_upper,
            left_arc_upper,
            left_arc_lower,
        ]
    )

    left_surface = geo.addPlaneSurface([left_loop])

    # ============================================================
    # Lóbulo direito
    # ============================================================

    right_arc_lower = geo.addEllipseArc(
        R0,
        right_center,
        right_outer,
        right_outer,
    )

    right_arc_upper = geo.addEllipseArc(
        right_outer,
        right_center,
        right_outer,
        C,
    )

    right_loop = geo.addCurveLoop(
        [
            -line_right_upper,
            right_arc_lower,
            right_arc_upper,
        ]
    )

    right_surface = geo.addPlaneSurface([right_loop])

    # ============================================================
    # Parte inferior
    # ============================================================

    bottom_arc_left = geo.addEllipseArc(
        A,
        bottom_center,
        bottom_apex,
        bottom_apex,
    )

    bottom_arc_right = geo.addEllipseArc(
        bottom_apex,
        bottom_center,
        bottom_apex,
        B,
    )

    bottom_loop = geo.addCurveLoop(
        [
            -line_bottom,
            bottom_arc_left,
            bottom_arc_right,
        ]
    )

    bottom_surface = geo.addPlaneSurface([bottom_loop])

    # Sincroniza a geometria com o modelo do Gmsh
    geo.synchronize()

    # ============================================================
    # Tags das regiões
    # ============================================================

    physical_surfaces = [
        (1, central_surface, "central_region"),
        (2, top_surface, "top_region"),
        (3, left_surface, "left_region"),
        (4, right_surface, "right_region"),
        (5, bottom_surface, "bottom_region"),
    ]

    for physical_tag, surface, name in physical_surfaces:
        gmsh.model.addPhysicalGroup(
            dim=2,
            tags=[surface],
            tag=physical_tag,
        )
        gmsh.model.setPhysicalName(2, physical_tag, name)

    # ============================================================
    # Tag do contorno externo
    # ============================================================

    outer_boundary = [
        top_right_side,
        top_arc_right,
        top_arc_left,
        top_left_side,
        left_arc_upper,
        left_arc_lower,
        right_arc_lower,
        right_arc_upper,
        line_left_lower,
        line_right_lower,
        bottom_arc_left,
        bottom_arc_right,
    ]

    gmsh.model.addPhysicalGroup(
        dim=1,
        tags=outer_boundary,
        tag=10,
    )
    gmsh.model.setPhysicalName(1, 10, "outer_boundary")

    # ============================================================
    # Tags das interfaces internas
    # ============================================================

    internal_interfaces = [
        line_top,
        line_left_upper,
        line_right_upper,
        line_bottom,
    ]

    gmsh.model.addPhysicalGroup(
        dim=1,
        tags=internal_interfaces,
        tag=20,
    )
    gmsh.model.setPhysicalName(1, 20, "internal_interfaces")

    # ============================================================
    # Geração da malha triangular
    # ============================================================

    # Frontal-Delaunay para malhas 2D
    gmsh.option.setNumber("Mesh.Algorithm", 6)

    # Desativa qualquer recombinação para quadriláteros
    gmsh.option.setNumber("Mesh.RecombineAll", 0)

    gmsh.option.setNumber("Mesh.MeshSizeMin", h)
    gmsh.option.setNumber("Mesh.MeshSizeMax", h)

    gmsh.model.mesh.generate(2)
    gmsh.model.mesh.setOrder(1)

    gmsh.write("rounded_cross.msh")


# ================================================================
# Conversão para uma malha distribuída do DOLFINx
# ================================================================

mesh_data = gmshio.model_to_mesh(
    gmsh.model,
    comm,
    rank=0,
    gdim=2,
)

domain = mesh_data.mesh
cell_tags = mesh_data.cell_tags
facet_tags = mesh_data.facet_tags

gmsh.finalize()

# ================================================================
# Salva em XDMF
# ================================================================

domain.name = "rounded_cross"

if cell_tags is not None:
    cell_tags.name = "cell_tags"

if facet_tags is not None:
    facet_tags.name = "facet_tags"

# Necessário para escrever corretamente as tags das arestas
tdim = domain.topology.dim
domain.topology.create_connectivity(tdim - 1, tdim)

geometry_xpath = "/Xdmf/Domain/Grid[@Name='rounded_cross']/Geometry"

with XDMFFile(comm, "rounded_cross.xdmf", "w") as xdmf:
    xdmf.write_mesh(domain)

    if cell_tags is not None:
        xdmf.write_meshtags(
            cell_tags,
            domain.geometry,
            geometry_xpath=geometry_xpath,
        )

    if facet_tags is not None:
        xdmf.write_meshtags(
            facet_tags,
            domain.geometry,
            geometry_xpath=geometry_xpath,
        )

if rank == 0:
    print("Malha criada:")
    print("  rounded_cross.msh")
    print("  rounded_cross.xdmf")
