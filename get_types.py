from mpi4py import MPI
from petsc4py import PETSc

opts = PETSc.Options()
opts["help"] = None  # type: ignore[attr-defined]
ksp = PETSc.KSP().create()
ksp.setFromOptions()
