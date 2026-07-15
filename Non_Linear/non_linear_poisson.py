import numpy as np
import ufl
from dolfinx import fem, mesh
from dolfinx.fem.petsc import NonlinearProblem
from mpi4py import MPI

q = lambda u: 1 + u**2  # Primeiro, defini uma função lambda para dar os valores de q(u)
