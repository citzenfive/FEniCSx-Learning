# Resolvendo aqui uma reação-difusão com reação não linear
# Peguei o exemplo da KPP-Fisher da wikipédia para ter uma ideia de acertei ou não

from pathlib import Path

import dolfinx.io
import numpy as np
import pyvista
import ufl
from dolfinx import default_real_type, default_scalar_type, fem, mesh, plot
from dolfinx.fem import problems
from dolfinx.fem.petsc import NonlinearProblem
from mpi4py import MPI


def initial_condition(x):
    amplitude = 0.1
    beta = 100.0

    return amplitude * np.exp(-beta * ((x[0] - 0.5) ** 2 + (x[1] - 0.5) ** 2))


# Aqui, estou fazendo algumas configurações iniciais do problema
results_folder = Path("Non_Linear/results")
results_folder.mkdir(exist_ok=True, parents=True)

results_file = results_folder / "kpp_fisher.xdmf"

D_real = 5e-3
r_real = 1.0
dt_real = 1e-3

t0 = 0.0
tf = 3.0
t = 0.0

save_every = 100

num_steps = int(np.ceil((tf - t0) / dt_real))

print(f"We will execute {num_steps} time steps...")

# resolvi fazer um pouco diferente aqui... para me acostumar com a ordem de passagem de parâmetros da função
nx = 100
ny = 100
domain = mesh.create_unit_square(MPI.COMM_WORLD, nx, ny, mesh.CellType.triangle)

D = fem.Constant(domain, default_scalar_type(D_real))
r = fem.Constant(domain, default_scalar_type(r_real))
dt = fem.Constant(domain, default_scalar_type(dt_real))

g = fem.Constant(domain, default_scalar_type(0.0))  # para a condição de Neumann

# Criando o espaço de funções de aproximação
V = fem.functionspace(domain, ("Lagrange", 1))

# Definindo função de teste
v = ufl.TestFunction(V)  # função de testes

# Para condição inicial e variável conhecida
u_n = fem.Function(V)
u_n.name = "u_old"  # type: ignore[attr-defined]
u_n.interpolate(initial_condition)  # type: ignore[attr-defined]

# Aqui fica a variável desconhecida
u = fem.Function(V)
u.name = "u_new"  # type: ignore[attr-defined]
u.x.array[:] = u_n.x.array  # type: ignore[attr-defined]


# Precisamos agr usar uma média do termo reativo não linear
# Para isso fazemos o seguinte
reaction_n = r * u_n * (1.0 - u_n)  # type: ignore[attr-defined]
reaction_n1 = r * u * (1.0 - u)  # type: ignore[attr-defined]
mid_reaction = 0.5 * (reaction_n1 + reaction_n)

# Definimos aqui o nosso residual
F = (
    ((u - u_n) / dt) * v  # type: ignore[attr-defined]
    + D * ufl.dot(ufl.grad(0.5 * (u + u_n)), ufl.grad(v))  # type: ignore[attr-defined]
    - mid_reaction * v
) * ufl.dx


# Vou configurar aqui os petsc_options... como não sei muito sobre essa parte de resolver sistema não-linear no PETSc, vou deixar o padrão... preciso aprender mais sobre isso dps
petsc_options = {
    "snes_type": "newtonls",
    "snes_linesearch_type": "none",
    "snes_atol": 1e-6,
    "snes_rtol": 1e-6,
    "snes_monitor": None,
    "ksp_monitor": None,
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
    u,  # type: ignore[attr-defined]
    petsc_options=petsc_options,
    petsc_options_prefix="kpp_fisher",
)

step = 0

with dolfinx.io.XDMFFile(MPI.COMM_WORLD, results_file, "w") as xdmf:
    xdmf.write_mesh(domain)
    xdmf.write_function(u, t)  # type: ignore[attr-defined]

    while t < tf:
        t += dt_real
        step += 1

        # Chute inicial para o Newton: usei o mesmo que eu vi o pessoal usando... tenho que ver com o Bernardo dps se tem algum melhor
        u.x.array[:] = u_n.x.array  # type: ignore[attr-defined]

        problem.solve()
        converged = problem.solver.getConvergedReason()
        num_iter = problem.solver.getIterationNumber()
        assert converged > 0, f"Solver did not converge for reasons:\n{converged}"  # type: ignore[attr-defined]
        print(
            f"Solver converged after {num_iter} iteractions with converged reason {converged}"
        )

        u_n.x.array[:] = u.x.array  # type: ignore[attr-defined]

        if step % save_every:
            print(f"t = {t}...")
            xdmf.write_function(u, t)  # type: ignore[attr-defined]
