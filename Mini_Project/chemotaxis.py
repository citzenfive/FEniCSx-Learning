import argparse
import os
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
from ufl.classes import ufl_classes

parser = argparse.ArgumentParser()
parser.add_argument("--chemotaxis_mode", help="chemotaxis treatment mode")
args = parser.parse_args()

type_chemotaxis = ""
SUPG_on = False

if args.chemotaxis_mode == "SUPG":
    type_chemotaxis = "SUPG"
    SUPG_on = True
else:
    type_chemotaxis = "USUALGALERKIN"
    SUPG_on = False


# Aqui, estou fazendo algumas configurações iniciais do problema
results_folder = Path("Mini_Project/results_chemotaxis/1D")
results_folder.mkdir(exist_ok=True, parents=True)

pathogen_file_name = "pathogen_" + type_chemotaxis + "_.xdmf"
leukocytes_file_name = "leukocytes_" + type_chemotaxis + "_.xdmf"

# results_file = results_folder / "pathogen_leukocyte.xdmf"
pathogen_file = results_folder / pathogen_file_name
leukocytes_file = results_folder / leukocytes_file_name


def initial_condition_pathogen1D(x):
    values = np.zeros(
        x.shape[1],
        dtype=default_scalar_type,
    )

    infected_region = (x[0] >= 0.92) & (x[0] <= 1.0)

    values[infected_region] = 1.0e-3

    return values


def initial_condition_leukocytes(x):
    return np.zeros(
        x.shape[1],
        dtype=default_scalar_type,
    )


def fb(lambda_lp, gamma_p, l, p):
    return -lambda_lp * l * p + p * gamma_p


def fl(lambda_pl, gamma_l, mu_l, Clmax, l, p):
    return -(lambda_pl * l * p + mu_l * l) + gamma_l * p * (Clmax - l)


nx = 100
ny = 100

t0 = 0.0
tf = 30.0

# Passo temporal
dt_real = 1.0e-3

# Porosidade
phi_f_real = 2.0e-1

# Difusão
D_p_real = 1.0e-4
D_l_real = 1.0e-4

# Dinâmica da bactéria
gamma_p_real = 1.54e-1  # c_b: reprodução bacteriana
lambda_lp_real = 1.8  # lambda_nb: fagocitose

# Dinâmica do neutrófilo
gamma_l_real = 1.0e-1  # gamma_n: recrutamento/permeabilidade capilar
lambda_pl_real = 1.0e-1  # lambda_bn: apoptose induzida
mu_l_real = 2.0e-1  # mu_n: apoptose natural

# Concentração de neutrófilos no sangue
Clmax_real = 5.5e-1

# Sensibilidade quimiotática
chi_real = 1e-4

num_steps = int(np.ceil((tf - t0) / dt_real))
L = 1.0


# domain = mesh.create_unit_square(MPI.COMM_WORLD, nx, ny, mesh.CellType.triangle)

domain = mesh.create_interval(
    MPI.COMM_WORLD,
    nx,
    np.array([0.0, L], dtype=default_real_type),
)

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
p_n_function.interpolate(initial_condition_pathogen1D)
l_n_function.interpolate(initial_condition_leukocytes)

solution_n1.x.array[:] = solution_n.x.array  # type: ignore[attr-defined]

p_n, l_n = ufl.split(solution_n)  # type: ignore[attr-defined]
p_n1, l_n1 = ufl.split(solution_n1)  # type: ignore[attr-defined]

D_p = fem.Constant(
    domain,
    default_scalar_type(D_p_real),
)

D_l = fem.Constant(
    domain,
    default_scalar_type(D_l_real),
)

gamma_p = fem.Constant(
    domain,
    default_scalar_type(gamma_p_real),
)

lambda_lp = fem.Constant(
    domain,
    default_scalar_type(lambda_lp_real),
)

lambda_pl = fem.Constant(
    domain,
    default_scalar_type(lambda_pl_real),
)

mu_l = fem.Constant(
    domain,
    default_scalar_type(mu_l_real),
)

gamma_l = fem.Constant(
    domain,
    default_scalar_type(gamma_l_real),
)

Clmax = fem.Constant(
    domain,
    default_scalar_type(Clmax_real),
)

phi_f = fem.Constant(
    domain,
    default_scalar_type(phi_f_real),
)

chi = fem.Constant(
    domain,
    default_scalar_type(chi_real),
)

dt = fem.Constant(
    domain,
    default_scalar_type(dt_real),
)

Fp = (
    (p_n1 - p_n) / dt * v1  # type: ignore[attr-defined]
    + (D_p / phi_f) * ufl.inner(ufl.grad(p_n1), ufl.grad(v1))  # type: ignore[attr-defined]
    - (fb(lambda_lp, gamma_p, l_n1, p_n1) / phi_f) * v1
) * ufl.dx

Fl_cg = (
    (l_n1 - l_n) / dt * v2
    + (D_l / phi_f) * ufl.inner(ufl.grad(l_n1), ufl.grad(v2))  # type: ignore[attr-defined]
    - l_n1 * ufl.inner((chi * ufl.grad(p_n1)) / phi_f, ufl.grad(v2))
    - (fl(lambda_pl, gamma_l, mu_l, Clmax, l_n1, p_n1) / phi_f) * v2
) * ufl.dx


strong_residual = (l_n1 - l_n) / dt - (1 / phi_f) * (
    (ufl.div(D_l * ufl.grad(l_n1) - chi * l_n1 * ufl.grad(p_n1)))
    + fl(lambda_pl, gamma_l, mu_l, Clmax, l_n1, p_n1)
)
stream_line = ufl.inner((chi * ufl.grad(p_n1)) / phi_f, ufl.grad(v2))

# Definindo o parâmetro tau
h = ufl.CellDiameter(domain)
epsilon = fem.Constant(domain, default_scalar_type(1.0e-16))
vel_norm = ufl.sqrt(
    epsilon + ufl.inner((chi * ufl.grad(p_n1)) / phi_f, (chi * ufl.grad(p_n1)) / phi_f)
)
tau = ufl.sqrt(
    (2.0 / dt) ** 2 + (2 * vel_norm / h) ** 2 + (4 * (D_l / phi_f) / (h**2)) ** 2  # type: ignore[attr-defined]
) ** (-1)


Fl_supg = tau * strong_residual * stream_line * ufl.dx

Fl = 0

if SUPG_on == True:
    Fl = Fl_cg + Fl_supg
else:
    Fl = Fl_cg

F_model = Fp + Fl

# Vou configurar aqui os petsc_options... como não sei muito sobre essa parte de resolver sistema não-linear no PETSc, vou deixar o padrão... preciso aprender mais sobre isso dps
# petsc_options = {
#     "snes_type": "newtonls",
#     "snes_linesearch_type": "bt",
#     "snes_atol": 1e-6,
#     "snes_rtol": 1e-6,
#     # "snes_monitor": None,
#     # "ksp_monitor": None,
#     "snes_error_if_not_converged": True,
#     "ksp_error_if_not_converged": True,
#     "ksp_type": "gmres",
#     "ksp_rtol": 1e-8,
#     "pc_type": "hypre",
#     "pc_hypre_type": "boomeramg",
#     "pc_hypre_boomeramg_max_iter": 1,
#     "pc_hypre_boomeramg_cycle_type": "v",
# }
#
petsc_options = {
    "snes_type": "newtonls",
    "snes_linesearch_type": "bt",
    "snes_atol": 1.0e-10,
    "snes_rtol": 1.0e-8,
    "snes_max_it": 30,
    "snes_error_if_not_converged": True,
    "ksp_error_if_not_converged": True,
    "ksp_type": "preonly",
    "pc_type": "lu",
}

problem = NonlinearProblem(
    F_model,
    solution_n1,  # type: ignore[attr-defined]
    petsc_options=petsc_options,
    petsc_options_prefix="pathogen_leukocytes_",
)


with (
    dolfinx.io.XDMFFile(
        domain.comm,
        pathogen_file,
        "w",
    ) as xdmf_p,
    dolfinx.io.XDMFFile(
        domain.comm,
        leukocytes_file,
        "w",
    ) as xdmf_l,
):
    xdmf_p.write_mesh(domain)
    xdmf_l.write_mesh(domain)
    pathogen_output = solution_n1.sub(0).collapse()  # type: ignore[attr-defined]
    leukocytes_output = solution_n1.sub(1).collapse()  # type: ignore[attr-defined]
    pathogen_output.name = "Pathogen"
    leukocytes_output.name = "Leukocytes"
    xdmf_p.write_function(pathogen_output, t0)
    xdmf_l.write_function(leukocytes_output, t0)

    t = 0
    step = 0
    save_interval = 6.0
    save_every = int(round(save_interval / dt_real))

    while t < tf:
        t += dt_real
        step += 1

        # Chute inicial para o Newton: usei o mesmo que eu vi o pessoal usando... tenho que ver com o Bernardo dps se tem algum melhor
        solution_n1.x.array[:] = solution_n.x.array  # type: ignore[attr-defined]

        problem.solve()
        converged = problem.solver.getConvergedReason()
        num_iter = problem.solver.getIterationNumber()
        # assert converged > 0, f"Solver did not converge for reasons:\n{converged}"  # type: ignore[attr-defined]
        # print(
        #     f"Solver converged after {num_iter} iteractions with converged reason {converged}"
        # )

        solution_n.x.array[:] = solution_n1.x.array  # type: ignore[attr-defined]

        if step % save_every == 0 or step == num_steps:
            print(f"Saving step {step} of {num_steps}...")
            pathogen_output = solution_n1.sub(0).collapse()  # type: ignore[attr-defined]
            leukocytes_output = solution_n1.sub(1).collapse()  # type: ignore[attr-defined]
            pathogen_output.name = "Pathogen"
            leukocytes_output.name = "Leukocytes"
            xdmf_p.write_function(pathogen_output, t)
            xdmf_l.write_function(leukocytes_output, t)
