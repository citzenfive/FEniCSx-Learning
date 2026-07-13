from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import gmsh
import numpy as np
from dolfinx.io import XDMFFile
from dolfinx.io import gmsh as gmshio
from mpi4py import MPI


def keep_largest_connected_component(mask: np.ndarray) -> np.ndarray:
    """Mantém somente o maior objeto branco da máscara."""
    nlabels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)

    if nlabels <= 1:
        raise RuntimeError("Nenhuma região não preta foi encontrada na imagem.")

    largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    return np.where(labels == largest_label, 255, 0).astype(np.uint8)


def extract_outer_contour_and_holes(
    mask: np.ndarray,
    min_hole_area_fraction: float,
) -> tuple[np.ndarray, list[np.ndarray]]:
    """
    Extrai o contorno externo do coração e os contornos pretos internos.

    A máscara deve conter:
      - 255 na região calculada;
      - 0 no fundo e nas cavidades.
    """
    contours, hierarchy = cv2.findContours(
        mask,
        cv2.RETR_TREE,
        cv2.CHAIN_APPROX_NONE,
    )

    if hierarchy is None or len(contours) == 0:
        raise RuntimeError("Não foi possível detectar contornos na imagem.")

    hierarchy = hierarchy[0]

    external_indices = [i for i, data in enumerate(hierarchy) if data[3] == -1]

    if not external_indices:
        raise RuntimeError("Não foi encontrado um contorno externo.")

    outer_index = max(
        external_indices,
        key=lambda i: cv2.contourArea(contours[i]),
    )

    outer = contours[outer_index]
    outer_area = cv2.contourArea(outer)
    minimum_hole_area = min_hole_area_fraction * outer_area

    hole_indices = [
        i
        for i, data in enumerate(hierarchy)
        if data[3] == outer_index and cv2.contourArea(contours[i]) >= minimum_hole_area
    ]

    holes = [contours[i] for i in hole_indices]

    # Ordena da esquerda para a direita na imagem.
    holes.sort(key=contour_centroid_x)

    return outer, holes


def contour_centroid_x(contour: np.ndarray) -> float:
    """Coordenada x do centroide de um contorno."""
    moments = cv2.moments(contour)

    if abs(moments["m00"]) < 1.0e-14:
        return float(np.mean(contour[:, 0, 0]))

    return float(moments["m10"] / moments["m00"])


def simplify_contour(
    contour: np.ndarray,
    epsilon_fraction: float,
) -> np.ndarray:
    """Reduz o número de pontos sem alterar muito o formato."""
    perimeter = cv2.arcLength(contour, closed=True)
    epsilon = epsilon_fraction * perimeter

    simplified = cv2.approxPolyDP(
        contour,
        epsilon,
        closed=True,
    )

    points = simplified[:, 0, :].astype(np.float64)

    if len(points) < 3:
        raise RuntimeError(
            "Um contorno ficou com menos de três pontos após a simplificação."
        )

    return points


def pixels_to_physical_coordinates(
    pixel_points: np.ndarray,
    bounding_box: tuple[int, int, int, int],
    physical_width: float,
) -> np.ndarray:
    """
    Converte pixels para coordenadas cartesianas.

    O eixo y da imagem é invertido para ficar orientado para cima.
    """
    x0, y0, width_pixels, height_pixels = bounding_box

    scale = physical_width / float(width_pixels)

    center_x = x0 + 0.5 * width_pixels
    center_y = y0 + 0.5 * height_pixels

    x = (pixel_points[:, 0] - center_x) * scale
    y = (center_y - pixel_points[:, 1]) * scale

    return np.column_stack((x, y))


def add_polygon_loop(
    geometry,
    points_xy: np.ndarray,
    mesh_size: float,
) -> tuple[int, list[int]]:
    """Adiciona um contorno poligonal fechado ao modelo geométrico do Gmsh."""
    point_tags = [
        geometry.addPoint(
            float(x),
            float(y),
            0.0,
            mesh_size,
        )
        for x, y in points_xy
    ]

    line_tags = []

    for i in range(len(point_tags)):
        start = point_tags[i]
        end = point_tags[(i + 1) % len(point_tags)]
        line_tags.append(geometry.addLine(start, end))

    loop_tag = geometry.addCurveLoop(
        line_tags,
        reorient=True,
    )

    return loop_tag, line_tags


def save_segmentation_preview(
    image: np.ndarray,
    mask: np.ndarray,
    outer: np.ndarray,
    holes: list[np.ndarray],
    output_directory: Path,
) -> None:
    """Salva imagens para conferir a segmentação antes de usar a malha."""
    cv2.imwrite(
        str(output_directory / "heart_binary_mask.png"),
        mask,
    )

    preview = image.copy()

    cv2.drawContours(
        preview,
        [outer],
        contourIdx=-1,
        color=(0, 255, 0),
        thickness=4,
    )

    cv2.drawContours(
        preview,
        holes,
        contourIdx=-1,
        color=(255, 0, 255),
        thickness=4,
    )

    cv2.imwrite(
        str(output_directory / "heart_detected_contours.png"),
        preview,
    )


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Cria uma malha triangular 2D a partir da região não preta "
            "de uma imagem de corte transversal do coração."
        )
    )

    parser.add_argument(
        "image",
        nargs="?",
        default="Heart_short_axis_transgastric_view.jpg",
        help="Arquivo de imagem de entrada.",
    )

    parser.add_argument(
        "--output-dir",
        default="heart_mesh",
        help="Pasta onde os arquivos serão salvos.",
    )

    parser.add_argument(
        "--black-threshold",
        type=int,
        default=15,
        help=(
            "Pixels cujo maior canal RGB seja menor ou igual a esse valor "
            "serão tratados como preto."
        ),
    )

    parser.add_argument(
        "--physical-width",
        type=float,
        default=10.0,
        help="Largura física atribuída ao coração, em unidades arbitrárias.",
    )

    parser.add_argument(
        "--mesh-size",
        type=float,
        default=0.12,
        help="Tamanho aproximado dos elementos triangulares.",
    )

    parser.add_argument(
        "--simplify",
        type=float,
        default=8.0e-4,
        help=(
            "Fração usada para simplificar os contornos. "
            "Valores menores preservam mais detalhes."
        ),
    )

    parser.add_argument(
        "--min-hole-area-fraction",
        type=float,
        default=2.0e-3,
        help=("Área mínima de uma cavidade como fração da área do contorno externo."),
    )

    parser.add_argument(
        "--morph-kernel",
        type=int,
        default=3,
        help=(
            "Tamanho do kernel morfológico usado para fechar pequenos defeitos. "
            "Use 0 ou 1 para desativar."
        ),
    )

    return parser.parse_args()


def main() -> None:
    args = parse_arguments()

    comm = MPI.COMM_WORLD
    rank = comm.rank

    image_path = Path(args.image)
    output_directory = Path(args.output_dir)

    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 1)

    error_message: str | None = None

    if rank == 0:
        try:
            output_directory.mkdir(parents=True, exist_ok=True)

            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)

            if image is None:
                raise FileNotFoundError(
                    f"Não foi possível abrir a imagem: {image_path}"
                )

            # Como o preto possui todos os canais próximos de zero,
            # a ordem BGR/RGB não altera esse teste.
            maximum_channel = np.max(image, axis=2)

            mask = np.where(
                maximum_channel > args.black_threshold,
                255,
                0,
            ).astype(np.uint8)

            if args.morph_kernel > 1:
                kernel_size = int(args.morph_kernel)

                if kernel_size % 2 == 0:
                    kernel_size += 1

                kernel = np.ones(
                    (kernel_size, kernel_size),
                    dtype=np.uint8,
                )

                mask = cv2.morphologyEx(
                    mask,
                    cv2.MORPH_CLOSE,
                    kernel,
                )

            # Remove pequenos componentes isolados e mantém somente o coração.
            mask = keep_largest_connected_component(mask)

            outer_contour, hole_contours = extract_outer_contour_and_holes(
                mask,
                min_hole_area_fraction=args.min_hole_area_fraction,
            )

            if len(hole_contours) != 2:
                print(
                    "Aviso: foram detectadas "
                    f"{len(hole_contours)} cavidades internas, não exatamente 2."
                )

            save_segmentation_preview(
                image,
                mask,
                outer_contour,
                hole_contours,
                output_directory,
            )

            outer_pixels = simplify_contour(
                outer_contour,
                epsilon_fraction=args.simplify,
            )

            hole_pixels = [
                simplify_contour(
                    contour,
                    epsilon_fraction=args.simplify,
                )
                for contour in hole_contours
            ]

            bounding_box = cv2.boundingRect(outer_contour)

            outer_xy = pixels_to_physical_coordinates(
                outer_pixels,
                bounding_box,
                physical_width=args.physical_width,
            )

            holes_xy = [
                pixels_to_physical_coordinates(
                    points,
                    bounding_box,
                    physical_width=args.physical_width,
                )
                for points in hole_pixels
            ]

            gmsh.model.add("heart_cross_section")
            geometry = gmsh.model.geo

            outer_loop, outer_lines = add_polygon_loop(
                geometry,
                outer_xy,
                mesh_size=args.mesh_size,
            )

            hole_loops = []
            hole_line_groups = []

            for points_xy in holes_xy:
                loop_tag, line_tags = add_polygon_loop(
                    geometry,
                    points_xy,
                    mesh_size=args.mesh_size,
                )

                hole_loops.append(loop_tag)
                hole_line_groups.append(line_tags)

            # O primeiro loop é externo; os demais são buracos.
            surface_tag = geometry.addPlaneSurface([outer_loop, *hole_loops])

            geometry.synchronize()

            # Região onde as EDPs serão resolvidas.
            gmsh.model.addPhysicalGroup(
                dim=2,
                tags=[surface_tag],
                tag=1,
            )
            gmsh.model.setPhysicalName(
                2,
                1,
                "heart_tissue",
            )

            # Contorno externo.
            gmsh.model.addPhysicalGroup(
                dim=1,
                tags=outer_lines,
                tag=10,
            )
            gmsh.model.setPhysicalName(
                1,
                10,
                "outer_boundary",
            )

            # Cavidades internas ordenadas da esquerda para a direita da imagem.
            for index, line_group in enumerate(hole_line_groups):
                physical_tag = 20 + index

                gmsh.model.addPhysicalGroup(
                    dim=1,
                    tags=line_group,
                    tag=physical_tag,
                )

                gmsh.model.setPhysicalName(
                    1,
                    physical_tag,
                    f"cavity_{index + 1}",
                )

            gmsh.option.setNumber("Mesh.Algorithm", 6)
            gmsh.option.setNumber("Mesh.RecombineAll", 0)
            gmsh.option.setNumber(
                "Mesh.MeshSizeMin",
                args.mesh_size,
            )
            gmsh.option.setNumber(
                "Mesh.MeshSizeMax",
                args.mesh_size,
            )

            gmsh.model.mesh.generate(2)
            gmsh.model.mesh.setOrder(1)

            gmsh.write(str(output_directory / "heart_cross_section.msh"))

            print("\nContornos detectados:")
            print(f"  externo: {len(outer_xy)} pontos")
            print(f"  cavidades: {len(holes_xy)}")

            for i, points in enumerate(holes_xy, start=1):
                print(f"    cavidade {i}: {len(points)} pontos")

        except Exception as error:
            error_message = f"{type(error).__name__}: {error}"

    error_message = comm.bcast(error_message, root=0)

    if error_message is not None:
        gmsh.finalize()
        raise RuntimeError(error_message)

    # Converte a malha do modelo Gmsh para uma malha distribuída do DOLFINx.
    mesh_data = gmshio.model_to_mesh(
        gmsh.model,
        comm,
        rank=0,
        gdim=2,
    )

    gmsh.finalize()

    domain = mesh_data.mesh
    cell_tags = mesh_data.cell_tags
    facet_tags = mesh_data.facet_tags

    domain.name = "heart"

    if cell_tags is not None:
        cell_tags.name = "cell_tags"

    if facet_tags is not None:
        facet_tags.name = "facet_tags"

    tdim = domain.topology.dim
    fdim = tdim - 1

    domain.topology.create_connectivity(fdim, tdim)
    domain.topology.create_connectivity(tdim, fdim)

    xdmf_path = output_directory / "heart_cross_section.xdmf"
    geometry_xpath = "/Xdmf/Domain/Grid[@Name='heart']/Geometry"

    with XDMFFile(
        comm,
        xdmf_path,
        "w",
    ) as xdmf:
        xdmf.write_mesh(domain)

        if cell_tags is not None:
            xdmf.write_meshtags(
                cell_tags,
                domain.geometry,
                geometry_xpath=geometry_xpath,
            )

        if facet_tags is not None:
            xdmf.write_meshtags(
                facet_tags,
                domain.geometry,
                geometry_xpath=geometry_xpath,
            )

    if rank == 0:
        print("\nArquivos criados:")
        print(f"  {output_directory / 'heart_cross_section.msh'}")
        print(f"  {xdmf_path}")
        print(f"  {output_directory / 'heart_cross_section.h5'}")
        print(f"  {output_directory / 'heart_binary_mask.png'}")
        print(f"  {output_directory / 'heart_detected_contours.png'}")
        print("\nTags:")
        print("  células 1: tecido cardíaco")
        print("  facetas 10: contorno externo")

        for index in range(len(hole_contours)):
            print(f"  facetas {20 + index}: cavidade {index + 1}")


if __name__ == "__main__":
    main()
