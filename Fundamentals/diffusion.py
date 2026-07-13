import math as mt
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

# Vou resolver aqui uma Reação-Difusão 2D simples, com condição de Neumann em todas as fronteiras
# Essa será a minha primeira tentativa em fazer uma solução "sozinho", apenas seguindo o exemplo do código que fiz pra Poisson e usando, pela primeira vez na
# minha vida o método de Crank-Nicolson no tempo


def initial_condition(x):
    """
    Distribuição gaussiana centrada em (0.5, 0.5).
    """
    xc = 0.5
    yc = 0.5
    sigma = 0.1

    r2 = (x[0] - xc) ** 2 + (x[1] - yc) ** 2

    return np.exp(-r2 / (2.0 * sigma**2))


# Criando o domínio e o espaço de funções de aproximações de MEF
domain = mesh.create_unit_square(MPI.COMM_WORLD, 100, 100, mesh.CellType.triangle)
V = fem.functionspace(domain, ("Lagrange", 1))

# Definindo função de teste e experimento
u = ufl.TrialFunction(V)  # variável do problema
v = ufl.TestFunction(V)  # função de testes

# Escolhi uma fonte constante e igual a 0.05, apenas para testes
f = fem.Constant(domain, default_scalar_type(0.05))
D = fem.Constant(domain, default_scalar_type(0.001))

g = fem.Constant(domain, default_scalar_type(0.0))
dt = fem.Constant(domain, default_scalar_type(1e-3))
dt_real = 1e-3


# Para condição inicial
u_n = fem.Function(V)
u_n.name = "u"  # type: ignore[attr-defined]
u_n.interpolate(initial_condition)  # type: ignore[attr-defined]

# Montagem das formas bilineares (como f é constante, ela entra apenas no L -> termos conhecidos)
a = ((u) / dt * v) * ufl.dx + ((D / 2.0) * ufl.dot(ufl.grad(u), ufl.grad(v))) * ufl.dx
L = (
    ((u_n) / dt * v) * ufl.dx
    - ((D / 2.0) * ufl.dot(ufl.grad(u_n), ufl.grad(v))) * ufl.dx
    + (f * v) * ufl.dx
)

# Montagem do problema (eu aqui não sei mexer muito nessas opções... depois tenho que aprender!)
# TODO aprender a mexer isso aqui
problem = LinearProblem(
    a,
    L,
    bcs=[],
    petsc_options_prefix="diffusion_reaction_",
    petsc_options={
        "ksp_type": "preonly",
        "pc_type": "lu",
        "ksp_error_if_not_converged": True,
    },
)

# Agora entro no laço temporal

t0 = 0.0
tf = 1.0
t = 0.0

num_steps = mt.ceil((tf - t0) / dt_real)

print("Number of steps: {}".format(num_steps))

u_n.name = "u_old"  # type: ignore[attr-defined]

# Salvando para visualizar no Paraview
results_folder = Path("Fundamentals/results")
results_folder.mkdir(exist_ok=True, parents=True)

with dolfinx.io.XDMFFile(
    MPI.COMM_WORLD, results_folder / "diffusion.xdmf", "w"
) as xdmf:
    xdmf.write_mesh(domain)
    xdmf.write_function(u_n)  # type: ignore[attr-defined]

    for n in range(num_steps):
        t += dt_real  # Apenas atualizando
        u_h = problem.solve()  # Resolvendo o problema para u_{n+1}
        u_h.name = "u_new"  # type: ignore[attr-defined]

        xdmf.write_function(u_h, t)  # salvando

        u_n.x.array[:] = u_h.x.array  # type: ignore[attr-defined]
        if n % 100:
            print("Step = {}".format(n))
