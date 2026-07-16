import numpy as np
import ufl
from dolfinx import fem, mesh
from dolfinx.fem.petsc import NonlinearProblem
from mpi4py import MPI

q = lambda u: 1 + u**2  # Primeiro, defini uma função lambda para dar os valores de q(u)

# Assim como em todos os casos, vou definir meu domínio
domain = mesh.create_unit_square(MPI.COMM_WORLD, 12, 12, mesh.CellType.quadrilateral)
# Criando o espaço de funções de aproximação
V = fem.functionspace(domain, ("Lagrange", 1))

X = ufl.SpatialCoordinate(domain)
u_ufl = 1 + X[0] + 2 * X[1]  # type: ignore[attr-defined]

f = -ufl.div(q(u_ufl) * ufl.grad(u_ufl))


def u_exac(x):
    return eval(str(u_ufl))


# Aplicando as condições de contorno
uD = fem.Function(V)
uD.interpolate(u_exac)  # type: ignore[attr-defined]
fdim = domain.topology.dim - 1
boundary_facets = mesh.locate_entities_boundary(
    domain, fdim, lambda X: np.full(X.shape[1], True, dtype=bool)
)
bc = fem.dirichletbc(uD, fem.locate_dofs_topological(V, fdim, boundary_facets))  # type: ignore[attr-defined]

# Como agora temos um problema não-linear, ao invés de trial function iremos definir uma função em V, que irá servir como nossa variável não-conhecida do problema
uh = fem.Function(V)
v = ufl.TestFunction(V)
F = (
    q(uh) * ufl.dot(ufl.grad(uh), ufl.grad(v)) * ufl.dx - f * v * ufl.dx
)  # Definimos aqui o nosso residual

# Precisamos usar uma definição de problema não linear
# Como é o usual, usei o método de Newton

# Vou configurar aqui os petsc_options... como não sei muito sobre essa parte de resolver sistema não-linear no PETSc, vou deixar o padrão... preciso aprender mais sobre isso dps
petsc_options = {
    "snes_type": "newtonls",
    "snes_linesearch_type": "none",
    "snes_atol": 1e-6,
    "snes_rtol": 1e-6,
    "snes_monitor": None,
    "ksp_error_if_not_converged": True,
    "ksp_type": "gmres",
    "ksp_rtol": 1e-8,
    "ksp_monitor": None,
    "pc_type": "hypre",
    "pc_hypre_type": "boomeramg",
    "pc_hypre_boomeramg_max_iter": 1,
    "pc_hypre_boomeramg_cycle_type": "v",
}

# Agora definimos o problem
problem = NonlinearProblem(
    F,
    uh,  # type: ignore[attr-defined]
    bcs=[bc],
    petsc_options=petsc_options,
    petsc_options_prefix="nonlinear_poisson",
)

# Resolvemos o problema
problem.solve()
converged = problem.solver.getConvergedReason()
num_iter = problem.solver.getIterationNumber()
assert converged > 0, f"Solver did not converge for reasons:\n{converged}"  # type: ignore[attr-defined]
print(
    f"Solver converged after {num_iter} iteractions with converged reason {converged}"
)
