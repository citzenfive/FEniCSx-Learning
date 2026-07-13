from pathlib import Path

import dolfinx.io
import numpy as np
import pyvista
import ufl
from dolfinx import default_real_type, default_scalar_type, fem, mesh, plot
from dolfinx.fem import problems
from dolfinx.fem.petsc import LinearProblem
from dolfinx.io import XDMFFile
from mpi4py import MPI

with XDMFFile(MPI.COMM_WORLD, "rounded_cross.xdmf", "r") as xdmf:
    # Malha
    domain = xdmf.read_mesh(name="rounded_cross")

    tdim = domain.topology.dim
    fdim = tdim - 1

    # Cria as conectividades necessárias para as facetas
    domain.topology.create_connectivity(tdim, fdim)
    domain.topology.create_connectivity(fdim, tdim)

    # Tags das regiões 2D
    cell_tags = xdmf.read_meshtags(
        domain,
        name="cell_tags",
    )

    # Tags das arestas
    facet_tags = xdmf.read_meshtags(
        domain,
        name="facet_tags",
    )

# (domain)  # malha
# cell_tags  # tags das regiões
# facet_tags  # tags dos contornos e interfaces

V = fem.functionspace(
    domain, ("Lagrange", 1)
)  # cria o espaço de funções de aproximação de elementos finitos

# Para o FEniCSx, quando resolvendo problema 2D X é o vetor de variáveis, enquanto X[0] é x e X[1] é y

# Aqui vou definir a condição de contorno de Dirichlet

uD = fem.Function(V)  # define uD como uma função de fem no espaço V
uD.interpolate(lambda X: 1 + X[0] ** 2 + 2 * X[1] ** 2)  # type: ignore[attr-defined]

# Preciso agora atribuir o valor uD em todas as "linhas" da fronteira
# Para isso, preciso identificar todas os segmentos da malha que estão nessa fronteira

tdim = domain.topology.dim
fdim = tdim - 1
domain.topology.create_connectivity(fdim, tdim)
boundary_facets = mesh.exterior_facet_indices(domain.topology)
boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
bc = fem.dirichletbc(uD, boundary_dofs)  # type: ignore[attr-defined]


# Agora, posso definir as funções de teste e experimento

u = ufl.TrialFunction(V)  # variável do problema
v = ufl.TestFunction(V)  # função de testes

# Definimos agora um termo fonte

f = fem.Constant(domain, default_scalar_type(-6))


# Definindo agora a formulação variacional: a(u, v) = L(v)
#   -> a(u, v) = termo que queremos saber
#   -> L(v) = termo que já sabemos

a = ufl.dot(ufl.grad(v), ufl.grad(u)) * ufl.dx
L = f * v * ufl.dx

# Agora juntamos o problema e resolvemos o sistema linear

problem = LinearProblem(
    a,
    L,
    bcs=[bc],
    petsc_options={"ksp_type": "preonly", "pc_type": "lu"},
    petsc_options_prefix="Poisson",
)

uh = problem.solve()


# Salvando para visualizar no Paraview
results_folder = Path("Fundamentals/results")
results_folder.mkdir(exist_ok=True, parents=True)

uh.name = "Poisson"  # type: ignore[attr-defined]
with dolfinx.io.XDMFFile(
    MPI.COMM_WORLD, results_folder / "poisson_custom_msh.xdmf", "w"
) as xdmf:
    xdmf.write_mesh(domain)
    xdmf.write_function(uh)  # type: ignore[attr-defined]
