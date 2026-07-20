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

# Aqui, estou fazendo algumas configurações iniciais do problema
results_folder = Path("Mini_Project/results_pathogen_leukocyte")
results_folder.mkdir(exist_ok=True, parents=True)

results_file = results_folder / "pathogen_leukocyte.xdmf"


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


def initial_condition_pathogen(x):
    amplitude = 0.001
    sigma = 0.03

    centers = [
        (0.0, 0.0),
        (1.0, 0.0),
        (0.0, 1.0),
        (1.0, 1.0),
        (0.5, 0.5),
    ]

    values = np.zeros(x.shape[1], dtype=default_scalar_type)

    for cx, cy in centers:
        distance_squared = (x[0] - cx) ** 2 + (x[1] - cy) ** 2

        values += amplitude * np.exp(-distance_squared / (2.0 * sigma**2))

    return values


def initial_condition_leukocytes(x):
    return np.zeros(x.shape[1])


nx = 100
ny = 100

t0 = 0.0
tf = 30.0
dt_real = 1e-3


num_steps = int(np.ceil((tf - t0) / dt_real))

domain = mesh.create_unit_square(MPI.COMM_WORLD, nx, ny, mesh.CellType.triangle)

# Agora preciso definir um espaço misto de elementos de FEM
# Assim:
P1 = basix.ufl.element(
    "Lagrange", domain.basix_cell(), degree=1, dtype=default_real_type
)
mixed_elements = basix.ufl.mixed_element([P1, P1])
W = fem.functionspace(domain, mixed_elements)

# Definindo as funções de teste
v1, v2 = ufl.TestFunctions(W)  # type: ignore[attr-defined]

# Solução conhecida no instante anterior:
# solution_n = (pathogen_n, leukocytes_n)
solution_n = fem.Function(W)
solution_n.name = "solution_old"  # type: ignore[attr-defined]

# Solução desconhecida no instante atual:
# solution_n1 = (pathogen_n1, leukocytes_n1)
solution_n1 = fem.Function(W)
solution_n1.name = "solution"  # type: ignore[attr-defined]

p_n_function = solution_n.sub(0)  # type: ignore[attr-defined]
l_n_function = solution_n.sub(1)  # type: ignore[attr-defined]

p_n1_function = solution_n1.sub(0)  # type: ignore[attr-defined]
l_n1_function = solution_n1.sub(1)  # type: ignore[attr-defined]

p_n_function.name = "pathogen_old"
p_n1_function.name = "Pathogen"
l_n_function.name = "leukocytes_old"
l_n1_function.name = "Leukocytes"

solution_n.x.array[:] = 0.0  # type: ignore[attr-defined]
p_n_function.interpolate(initial_condition_pathogen)
l_n_function.interpolate(initial_condition_leukocytes)

solution_n1.x.array[:] = solution_n.x.array  # type: ignore[attr-defined]

p_n, l_n = ufl.split(solution_n)  # type: ignore[attr-defined]
p_n1, l_n1 = ufl.split(solution_n1)  # type: ignore[attr-defined]

F_p = () * ufl.dx
F_l = () * ufl.dx

F_final = F_p + F_l

# Vou configurar aqui os petsc_options... como não sei muito sobre essa parte de resolver sistema não-linear no PETSc, vou deixar o padrão... preciso aprender mais sobre isso dps
petsc_options = {
    "snes_type": "newtonls",
    "snes_linesearch_type": "bt",
    "snes_atol": 1e-6,
    "snes_rtol": 1e-6,
    "snes_monitor": None,
    "ksp_monitor": None,
    "ksp_error_if_not_converged": True,
    "ksp_type": "bcgs",
    "ksp_rtol": 1e-8,
    "pc_type": "hypre",
    "pc_hypre_type": "boomeramg",
    "pc_hypre_boomeramg_max_iter": 1,
    "pc_hypre_boomeramg_cycle_type": "v",
}

problem = NonlinearProblem(
    F_final,
    solution_n1,  # type: ignore[attr-defined]
    petsc_options=petsc_options,
    petsc_options_prefix="pathogen_leukocytes_",
)


with dolfinx.io.XDMFFile(MPI.COMM_WORLD, results_file, "w") as xdmf:
    xdmf.write_mesh(domain)
    pathogen_output = solution_n1.sub(0).collapse()  # type: ignore[attr-defined]
    leukocytes_output = solution_n1.sub(1).collapse()  # type: ignore[attr-defined]
    pathogen_output.name = "Pathogen"
    leukocytes_output.name = "Leukocytes"
    xdmf.write_function(pathogen_output, t0)
    xdmf.write_function(leukocytes_output, t0)

    t = 0
    step = 0
    save_every = 1000

    while t < tf:
        t += dt_real
        step += 1

        # Chute inicial para o Newton: usei o mesmo que eu vi o pessoal usando... tenho que ver com o Bernardo dps se tem algum melhor
        solution_n1.x.array[:] = solution_n.x.array  # type: ignore[attr-defined]

        problem.solve()
        converged = problem.solver.getConvergedReason()
        num_iter = problem.solver.getIterationNumber()
        assert converged > 0, f"Solver did not converge for reasons:\n{converged}"  # type: ignore[attr-defined]
        print(
            f"Solver converged after {num_iter} iteractions with converged reason {converged}"
        )

        solution_n.x.array = solution_n1.x.array[:]  # type: ignore[attr-defined]

        if step % save_every:
            pathogen_output = solution_n1.sub(0).collapse()  # type: ignore[attr-defined]
            leukocytes_output = solution_n1.sub(1).collapse()  # type: ignore[attr-defined]
            pathogen_output.name = "Pathogen"
            leukocytes_output.name = "Leukocytes"
            xdmf.write_function(pathogen_output, t0)
            xdmf.write_function(leukocytes_output, t0)
