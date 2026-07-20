from pathlib import Path

from dolfinx.io import XDMFFile
from mpi4py import MPI


def read2DMesh(
    path_to_mesh: str | Path = "rounded_cross.xdmf",
    name: str = "rounded_cross",
):
    with XDMFFile(
        MPI.COMM_WORLD,
        path_to_mesh,
        "r",
    ) as xdmf:
        # Lê a malha
        domain = xdmf.read_mesh(name=name)

        tdim = domain.topology.dim
        fdim = tdim - 1

        # Conectividades necessárias
        domain.topology.create_connectivity(tdim, fdim)
        domain.topology.create_connectivity(fdim, tdim)

        # Tags das regiões 2D
        cell_tags = xdmf.read_meshtags(
            domain,
            name="cell_tags",
        )

        # Tags das arestas/fronteiras
        facet_tags = xdmf.read_meshtags(
            domain,
            name="facet_tags",
        )

    return domain, cell_tags, facet_tags
