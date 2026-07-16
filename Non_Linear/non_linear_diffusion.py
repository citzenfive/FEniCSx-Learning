import dolfinx.io
import numpy as np
import pyvista
import ufl
from dolfinx import default_real_type, default_scalar_type, fem, mesh, plot
from dolfinx.fem import problems
from dolfinx.fem.petsc import NonlinearProblem
from mpi4py import MPI
from numpy._core import ceil
