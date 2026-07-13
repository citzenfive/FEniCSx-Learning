from mpi4py import MPI
from petsc4py import PETSc

opts = PETSc.Options()
opts["help"] = None
ksp = PETSc.KSP().create()
ksp.setFromOptions()
