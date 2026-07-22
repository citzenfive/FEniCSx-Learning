import math as mt
import time
from pathlib import Path

import dolfinx.io
import numpy as np
import pyvista
import ufl
from dolfinx import default_real_type, default_scalar_type, fem, mesh, plot
from dolfinx.fem import problems
from dolfinx.fem.petsc import LinearProblem
from mpi4py import MPI
from numpy._core import ceil
from scipy import constants

# Salvando para visualizar no Paraview
results_folder = Path("Mechanical_Model/results_linear_elasticity")
results_folder.mkdir(exist_ok=True, parents=True)


def epsilon(u):
    return ufl.sym(ufl.grad(u))


def sigma(u, lambda_, mu_):
    return lambda_ * ufl.nabla_div(u) * ufl.Identity(
        u.ufl_shape[0]
    ) + 2 * mu_ * epsilon(u)


def lame_parameters(E, nu):
    mu = E / (2 * (1 + nu))
    lamb = E * nu / ((1 + nu) * (1 - 2 * nu))

    return mu, lamb


def clamped_boundary(X):
    return np.isclose(X[0], 0)


# Fazendo a simulação para uma barra de Poliestireno
# Dados retirados de https://www.sonelastic.com/en/fundamentals/tables-of-materials-properties/polymers.html
# E = 2.78  # GPa
# nu = 3.3e-1

# rho = 1.0
# g = constants.g
#
# Sistema SI:
# comprimento: m
# força: N
# tensão: Pa
# densidade: kg/m³

E = 2.78e9  # Pa = 2.78 GPa
nu = 0.33  # adimensional
rho = 1050.0  # kg/m³
g = constants.g  # m/s²
P = rho * g

f_z = -1.5e3 * P


mu_, lambda_ = lame_parameters(E=E, nu=nu)
print(
    f"Using parameters: lambda = {lambda_:.5f}, mu = {mu_:.5f}, g = {g:.5f}, P = {P:.5f}, f_z = {f_z:.5f}"
)

# Tamanhos da barra
L = 1.0
W = 0.2
L = 1.0
W = 0.2

domain = mesh.create_box(
    MPI.COMM_WORLD,
    [np.array([0, 0, 0]), np.array([L, W, W])],
    [100, 50, 50],
    cell_type=mesh.CellType.hexahedron,
)


# Criando aqui o espaço do problema
V = fem.functionspace(domain, ("Lagrange", 1, (domain.geometry.dim,)))

fdim = domain.topology.dim - 1
boundary_facets = mesh.locate_entities_boundary(domain, fdim, clamped_boundary)

u_D = np.array([0, 0, 0], dtype=default_scalar_type)
bc = fem.dirichletbc(u_D, fem.locate_dofs_topological(V, fdim, boundary_facets), V)  # type: ignore[attr-defined]

T = fem.Constant(domain, default_scalar_type((0, 0, 0)))  # type: ignore[attr-defined]

ds = ufl.Measure("ds", domain=domain)

# Agora, posso definir as funções de teste e experimento
u = ufl.TrialFunction(V)  # variável do problema
v = ufl.TestFunction(V)  # função de testes

# Agora definimos os parâmetros para o problema, da maneira que o FEniCSx espera
f = fem.Constant(domain, default_scalar_type((0, 0, f_z)))  # type: ignore[attr-defined]

# Definindo nossa forma bilinear
a = ufl.inner(sigma(u, lambda_, mu_), epsilon(v)) * ufl.dx
L = ufl.dot(f, v) * ufl.dx + ufl.dot(T, v) * ds  # type: ignore[attr-defined]

petsc_options = {
    "ksp_type": "gmres",
    "ksp_gmres_restart": 30,
    "ksp_rtol": 1e-8,
    "ksp_max_it": 1000,
    "ksp_monitor": None,
}

# Resolvendo o problema
problem = LinearProblem(
    a,
    L,
    bcs=[bc],
    # petsc_options={"ksp_type": "preonly", "pc_type": "lu"},
    petsc_options=petsc_options,
    petsc_options_prefix="linear_elasticity",
)

start = time.time()
uh = problem.solve()

with dolfinx.io.XDMFFile(
    MPI.COMM_WORLD, results_folder / "LNElasticity.xdmf", "w"
) as xdmf:
    xdmf.write_mesh(domain)
    uh.name = "Deformation"  # type: ignore[attr-defined]
    xdmf.write_function(uh)  # type: ignore[attr-defined]

# Calculando as tensões
s = sigma(uh, lambda_, mu_) - 1.0 / 3 * ufl.tr(sigma(uh, lambda_, mu_)) * ufl.Identity(
    len(uh)
)
von_Mises = ufl.sqrt(3.0 / 2 * ufl.inner(s, s))  # type: ignore[attr-defined]

V_von_mises = fem.functionspace(domain, ("DG", 0))
stress_expr = fem.Expression(von_Mises, V_von_mises.element.interpolation_points)
stresses = fem.Function(V_von_mises)
stresses.interpolate(stress_expr)  # type: ignore[attr-defined]

stresses.name = "Von Mises"  # type: ignore[attr-defined]

with dolfinx.io.XDMFFile(
    MPI.COMM_WORLD,
    results_folder / "VMLNStress.xdmf",
    "w",
) as xdmf:
    xdmf.write_mesh(domain)
    xdmf.write_function(stresses)  # type: ignore[attr-defined]
end = time.time()

print(f"Time to solve = {end - start} seconds")
