# Essa é minha primeira tentativa de resolver um sistema de EDPs usando o FEniCSx
# Para tal, escolhi iniciar com um sistema simples, estacionário, porém acoplado
from pathlib import Path

import basix.ufl
import dolfinx.io
import numpy as np
import pyvista
import ufl
from dolfinx import default_real_type, default_scalar_type, fem, mesh, plot
from dolfinx.fem import problems
from dolfinx.fem.petsc import LinearProblem
from mpi4py import MPI


# Lembrando que para o FEniCSx X[0] = x e X[1] = y
# Vou definir aqui as funções para marcar as regiões da fronteira
#
def top(X):
    return np.isclose(X[1], 1.0)


def bottom(X):
    return np.isclose(X[1], 0.0)


def right(X):
    return np.isclose(X[0], 1.0)


def left(X):
    return np.isclose(X[0], 0.0)


domain = mesh.create_unit_square(MPI.COMM_WORLD, 12, 12, mesh.CellType.quadrilateral)


# Agora preciso definir um espaço misto de elementos de FEM
# Assim:
P1 = basix.ufl.element(
    "Lagrange", domain.basix_cell(), degree=1, dtype=default_real_type
)
mixed_elements = basix.ufl.mixed_element([P1, P1])
W = fem.functionspace(domain, mixed_elements)

# Definindo as funções de teste e experimento
u, v = ufl.TrialFunctions(W)
q, p = ufl.TestFunctions(W)

# Vou definir aqui os termos fontes
f_u = fem.Constant(domain, default_scalar_type(6))
f_v = fem.Constant(domain, default_scalar_type(7))

tdim = domain.topology.dim
fdim = tdim - 1

# Faço agora o mapeamento das fronteiras
facets_top = mesh.locate_entities_boundary(domain, fdim, top)
facets_bottom = mesh.locate_entities_boundary(domain, fdim, bottom)
facets_right = mesh.locate_entities_boundary(domain, fdim, right)
facets_left = mesh.locate_entities_boundary(domain, fdim, left)

# Vou agora dar um nome a elas
TOP = 1
BOTTOM = 2
RIGHT = 3
LEFT = 4

# Montando as mesh-tags
facet_indices = np.hstack(
    [
        facets_left,
        facets_right,
        facets_bottom,
        facets_top,
    ]
).astype(np.int32)


facet_values = np.hstack(
    [
        np.full(facets_left.size, LEFT, dtype=np.int32),
        np.full(facets_right.size, RIGHT, dtype=np.int32),
        np.full(facets_bottom.size, BOTTOM, dtype=np.int32),
        np.full(facets_top.size, TOP, dtype=np.int32),
    ]
)


order = np.argsort(facet_indices)

facet_tags = mesh.meshtags(
    domain,
    fdim,
    facet_indices[order],
    facet_values[order],
)

ds = ufl.Measure(
    "ds",
    domain=domain,
    subdomain_data=facet_tags,
)

# print(ds(TOP))
# print(ds(BOTTOM))
# print(ds(RIGHT))
# print(ds(LEFT))
# Seleciona a componente u dentro do espaço misto
#
Wu = W.sub(0)

# Cria um espaço escalar independente equivalente ao espaço de u
Vu, map_u = Wu.collapse()

# Função que armazena o valor prescrito u = 1
u_D = fem.Function(Vu)

u_D.interpolate(  # type: ignore[attr-defined]
    lambda x: np.full(
        x.shape[1],
        1.0,
        dtype=default_scalar_type,
    )
)

# Relaciona os dofs de u no sistema misto
# com os dofs da função u_D
dofs_u_left = fem.locate_dofs_topological(
    (Wu, Vu),
    fdim,
    facets_left,
)

# Impõe u_D nos dofs de u localizados na esquerda
bc_u = fem.dirichletbc(
    u_D,  # type: ignore[attr-defined]
    dofs_u_left,
    Wu,
)

bcs = [bc_u]


a = (
    -ufl.dot(ufl.grad(q), ufl.grad(u))
    + 1 / 6 * v * q
    - 1 / 2 * u * q
    - ufl.dot(ufl.grad(p), ufl.grad(v))
    + 1 / 8 * v * p
    - u * p
) * ufl.dx - 2 * p * v * ds
L = (-f_u * q - f_v * p) * ufl.dx - (q * ds(RIGHT) + p * ds)

problem = LinearProblem(
    a,
    L,
    bcs=bcs,
    petsc_options={
        "ksp_type": "preonly",
        "pc_type": "lu",
        "ksp_error_if_not_converged": True,
    },
    petsc_options_prefix="coupled_system_",
)

# problem = LinearProblem(
#     a,
#     L,
#     bcs=bcs,
#     petsc_options={"ksp_type": "cgs", "pc_type": "lu"},
#     petsc_options_prefix="coupled_system_",
# )

wh = problem.solve()

# Separando as soluções de u e v
uh = wh.sub(0).collapse()
vh = wh.sub(1).collapse()

uh.name = "u"
vh.name = "v"

# Salvando para visualizar no Paraview
results_folder = Path("Mini-Project/results_stationary_system")
results_folder.mkdir(exist_ok=True, parents=True)
# with dolfinx.io.XDMFFile(MPI.COMM_WORLD, results_folder / "ssystem.xdmf", "w") as xdmf:
#     xdmf.write_mesh(domain)
#     xdmf.write_function(uh)  # type: ignore[attr-defined]
#     xdmf.write_function(vh)  # type: ignore[attr-defined]

with dolfinx.io.XDMFFile(
    domain.comm,
    results_folder / "u_solution.xdmf",
    "w",
) as xdmf:
    xdmf.write_mesh(domain)
    xdmf.write_function(uh)

with dolfinx.io.XDMFFile(
    domain.comm,
    results_folder / "v_solution.xdmf",
    "w",
) as xdmf:
    xdmf.write_mesh(domain)
    xdmf.write_function(vh)
