import os
from math import gamma
from pathlib import Path

import basix.ufl
import dolfinx.io
import numpy as np
import pyvista
import ufl
from dolfinx import default_real_type, default_scalar_type, fem, mesh, plot
from dolfinx.fem import problems
from dolfinx.fem.petsc import NonlinearProblem
from mpi4py import MPI
from read_mesh import read2DMesh

# Aqui, estou fazendo algumas configurações iniciais do problema
results_folder = Path("Mini_Project/results_pathogen_diffusion_short_axis")
results_folder.mkdir(exist_ok=True, parents=True)

pathogen_file = results_folder / "pathogen.xdmf"

X_MIN = -5.0
X_MAX = 4.99366688
Y_MIN = -4.02153262
Y_MAX = 4.02786574


def initial_condition_pathogen(x):
    amplitude = 1.0e-1

    # Centro escolhido em coordenadas normalizadas
    center_x_normalized = 0.78
    center_y_normalized = 0.28
    sigma_normalized = 0.035

    # Normaliza as coordenadas da malha
    x_normalized = (x[0] - X_MIN) / (X_MAX - X_MIN)

    y_normalized = (x[1] - Y_MIN) / (Y_MAX - Y_MIN)

    distance_squared = (x_normalized - center_x_normalized) ** 2 + (
        y_normalized - center_y_normalized
    ) ** 2

    return amplitude * np.exp(-distance_squared / (2.0 * sigma_normalized**2))


t0 = 0.0
tf = 10.0
t = 0.0

Dp_real = 1.0
gammap_real = 2.67
lambdap_real = 1.50


dt_real = 1e-3

domain, cell_tags, facet_tags = read2DMesh(
    path_to_mesh="meshes/heart_meshes/h_0_05/short_axis/heart_cross_section.xdmf",
    name="heart",
)

Dp = fem.Constant(domain, default_scalar_type(Dp_real))
gammap = fem.Constant(domain, default_scalar_type(gammap_real))
lambdap = fem.Constant(domain, default_scalar_type(lambdap_real))
dt = fem.Constant(domain, default_scalar_type(dt_real))

# Criando o espaço de funções de aproximação
V = fem.functionspace(domain, ("Lagrange", 1))

# Definindo função de teste
v = ufl.TestFunction(V)  # função de testes

# Para condição inicial e variável conhecida
p_n = fem.Function(V)
p_n.name = "pathogen_old"  # type: ignore[attr-defined]
p_n.interpolate(initial_condition_pathogen)  # type: ignore[attr-defined]

# Aqui fica a variável desconhecida
p_n1 = fem.Function(V)
p_n1.name = "Pathogen"  # type: ignore[attr-defined]

# Iniciando o vetor da resposta com a condição inicial
p_n1.x.array[:] = p_n.x.array  # type: ignore[attr-defined]


reaction_n = gammap * p_n - lambdap * p_n  # type: ignore[attr-defined]
reaction_n1 = gammap * p_n1 - lambdap * p_n1  # type: ignore[attr-defined]
reaction_mid = 0.5 * (reaction_n + reaction_n1)  # type: ignore[attr-defined]

# Definimos aqui o nosso residual
F = (
    ((p_n1 - p_n) / dt) * v  # type: ignore[attr-defined]
    + Dp * ufl.dot(ufl.grad(0.5 * (p_n + p_n1)), ufl.grad(v))  # type: ignore[attr-defined]
    - reaction_mid * v
) * ufl.dx

# Vou configurar aqui os petsc_options... como não sei muito sobre essa parte de resolver sistema não-linear no PETSc, vou deixar o padrão... preciso aprender mais sobre isso dps
petsc_options = {
    "snes_type": "newtonls",
    "snes_linesearch_type": "none",
    "snes_atol": 1e-6,
    "snes_rtol": 1e-6,
    # "snes_monitor": None,
    # "ksp_monitor": None,
    "ksp_error_if_not_converged": True,
    "ksp_type": "gmres",
    "ksp_rtol": 1e-8,
    "pc_type": "hypre",
    "pc_hypre_type": "boomeramg",
    "pc_hypre_boomeramg_max_iter": 1,
    "pc_hypre_boomeramg_cycle_type": "v",
}

problem = NonlinearProblem(
    F,
    p_n1,  # type: ignore[attr-defined]
    petsc_options=petsc_options,
    petsc_options_prefix="pathogen_diffusion_short_axis_",
)

step = 0
save_every = 100

with dolfinx.io.XDMFFile(MPI.COMM_WORLD, pathogen_file, "w") as xdmf:
    xdmf.write_mesh(domain)
    xdmf.write_function(p_n1, t)  # type: ignore[attr-defined]

    while t < tf:
        t += dt_real
        step += 1

        # Chute inicial para o Newton: usei o mesmo que eu vi o pessoal usando... tenho que ver com o Bernardo dps se tem algum melhor
        p_n1.x.array[:] = p_n.x.array  # type: ignore[attr-defined]

        problem.solve()
        converged = problem.solver.getConvergedReason()
        num_iter = problem.solver.getIterationNumber()
        # assert converged > 0, f"Solver did not converge for reasons:\n{converged}"  # type: ignore[attr-defined]
        # print(
        #     f"Solver converged after {num_iter} iteractions with converged reason {converged}"
        # )

        p_n.x.array[:] = p_n1.x.array  # type: ignore[attr-defined]

        if step % save_every == 0:
            print(f"t = {t:.2f}...")
            xdmf.write_function(p_n1, t)  # type: ignore[attr-defined]
