"""
Plot one-dimensional time-dependent FEniCSx solutions stored in XDMF files.

Example
-------
python Mini_Project/plot1D.py \
    Mini_Project/results_pathogen_leukocyte/pathogen.xdmf \
    Mini_Project/results_pathogen_leukocyte/leukocytes.xdmf \
    --times 0 6 12 18 24 30 \
    --labels "Bacteria" "Neutrophils" \
    --xlim 0 1 \
    --formats png pdf svg \
    --output-dir Mini_Project/results_pathogen_leukocyte/plots
"""

from __future__ import annotations

import argparse
from itertools import cycle
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import meshio
import numpy as np
import seaborn as sns
from matplotlib.ticker import ScalarFormatter
from meshio.xdmf.common import xdmf_to_meshio_type

# DOLFINx writes "PolyLine", while meshio normally expects "Polyline".
xdmf_to_meshio_type["PolyLine"] = "line"


Profile = tuple[float, np.ndarray, np.ndarray]


def configure_publication_style(
    font_size: float = 10.0,
) -> None:
    """
    Configure Matplotlib and Seaborn for publication-quality figures.

    The selected configuration:

    - uses a colorblind-friendly palette;
    - uses serif fonts;
    - preserves editable text in PDF and SVG files;
    - creates thin axes and grid lines;
    - provides consistent font sizes.
    """

    sns.set_theme(
        context="paper",
        style="ticks",
        palette="colorblind",
        font="serif",
    )

    mpl.rcParams.update(
        {
            # Font configuration
            "font.family": "serif",
            "font.serif": [
                "STIX Two Text",
                "DejaVu Serif",
                "Times New Roman",
            ],
            "mathtext.fontset": "stix",
            "font.size": font_size,
            "axes.labelsize": font_size,
            "axes.titlesize": font_size + 1,
            "legend.fontsize": font_size - 1,
            "xtick.labelsize": font_size - 1,
            "ytick.labelsize": font_size - 1,
            # Axes and lines
            "axes.linewidth": 0.8,
            "lines.linewidth": 1.8,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "xtick.minor.width": 0.6,
            "ytick.minor.width": 0.6,
            # Figure output
            "figure.dpi": 120,
            "savefig.dpi": 600,
            "savefig.bbox": "tight",
            # Keep text editable in vector formats
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def read_xdmf_time_series(
    xdmf_file: Path,
    field_name: str | None = None,
) -> tuple[str, list[Profile]]:
    """
    Read a scalar time series stored in a one-dimensional XDMF file.

    Parameters
    ----------
    xdmf_file
        Path to the XDMF file.

    field_name
        Name of the point-data field to read. If omitted, the first
        available field is selected.

    Returns
    -------
    selected_field
        Name of the selected field.

    profiles
        List containing tuples with:

        ``(time, x_coordinates, field_values)``.
    """

    if not xdmf_file.exists():
        raise FileNotFoundError(f"File not found: {xdmf_file}")

    profiles: list[Profile] = []

    with meshio.xdmf.TimeSeriesReader(str(xdmf_file)) as reader:
        points, _ = reader.read_points_cells()

        if points.ndim != 2 or points.shape[1] < 1:
            raise RuntimeError(f"Invalid coordinates in file: {xdmf_file}")

        x_coordinates = points[:, 0].copy()
        selected_field = field_name

        for step in range(reader.num_steps):
            time, point_data, _ = reader.read_data(step)

            if not point_data:
                raise RuntimeError(
                    f"No point-data field was found at step {step} in {xdmf_file}."
                )

            if selected_field is None:
                selected_field = next(iter(point_data))

            if selected_field not in point_data:
                available_fields = ", ".join(point_data.keys())

                raise KeyError(
                    f"Field '{selected_field}' was not found "
                    f"in {xdmf_file}.\n"
                    f"Available fields: {available_fields}"
                )

            values = np.asarray(point_data[selected_field]).squeeze()

            if values.ndim != 1:
                raise ValueError(
                    f"Field '{selected_field}' is not scalar. "
                    f"Detected shape: {values.shape}"
                )

            if len(values) != len(x_coordinates):
                raise ValueError(
                    "The number of field values does not match "
                    "the number of mesh points."
                )

            order = np.argsort(x_coordinates)

            profiles.append(
                (
                    float(time),
                    x_coordinates[order].copy(),
                    values[order].copy(),
                )
            )

    if selected_field is None:
        raise RuntimeError(f"No field was found in {xdmf_file}.")

    return selected_field, profiles


def select_profiles(
    profiles: list[Profile],
    requested_times: list[float] | None,
) -> list[Profile]:
    """
    Select profiles corresponding to the requested times.

    When an exact time is unavailable, the closest available time
    is selected and a warning is printed.
    """

    if requested_times is None:
        return profiles

    available_times = np.array(
        [profile[0] for profile in profiles],
        dtype=float,
    )

    selected_profiles: list[Profile] = []
    selected_indices: set[int] = set()

    for requested_time in requested_times:
        closest_index = int(np.argmin(np.abs(available_times - requested_time)))

        closest_time = available_times[closest_index]

        tolerance = max(
            1.0e-10,
            1.0e-8
            * max(
                1.0,
                abs(requested_time),
            ),
        )

        if abs(closest_time - requested_time) > tolerance:
            print(
                f"Warning: time {requested_time:g} was not "
                f"found. Using the closest available time: "
                f"{closest_time:g}."
            )

        if closest_index not in selected_indices:
            selected_profiles.append(profiles[closest_index])

            selected_indices.add(closest_index)

    selected_profiles.sort(key=lambda profile: profile[0])

    return selected_profiles


def infer_species_label(
    file_path: Path,
    field_name: str,
) -> str:
    """
    Infer a publication-friendly species name from the file or field name.
    """

    name = (f"{file_path.stem} {field_name}").lower()

    if "pathogen" in name or "bacteria" in name or "bacterium" in name:
        return "Bacteria"

    if "leukocyte" in name or "neutro" in name or "neutrophil" in name:
        return "Neutrophils"

    return field_name


def plot_profiles(
    profiles: list[Profile],
    species_label: str,
    output_stem: Path,
    formats: list[str],
    x_limits: tuple[float, float] | None = None,
    y_limits: tuple[float, float] | None = None,
    figure_size: tuple[float, float] = (6.5, 4.0),
    dpi: int = 600,
    title: str | None = None,
    legend_location: str = "best",
    scientific_y: bool = False,
    transparent: bool = False,
    show: bool = False,
) -> None:
    """
    Plot one-dimensional profiles at multiple time points.

    The plot uses both colors and line styles, allowing the curves
    to remain distinguishable in grayscale printing.
    """

    if not profiles:
        raise ValueError("No profiles were provided for plotting.")

    figure, axis = plt.subplots(
        figsize=figure_size,
        constrained_layout=True,
    )

    number_of_profiles = len(profiles)

    colors = sns.color_palette(
        "colorblind",
        n_colors=max(number_of_profiles, 3),
    )

    line_styles = cycle(
        [
            "-",
            "--",
            "-.",
            ":",
            (0, (5, 1)),
            (0, (3, 1, 1, 1)),
            (0, (1, 1)),
        ]
    )

    for index, (
        time,
        x_coordinates,
        values,
    ) in enumerate(profiles):
        axis.plot(
            x_coordinates,
            values,
            color=colors[index],
            linestyle=next(line_styles),
            linewidth=1.9,
            label=rf"$t = {time:g}\,\mathrm{{s}}$",
            solid_capstyle="round",
        )

    axis.set_xlabel(r"Position, $x$ (cm)")

    axis.set_ylabel("Concentration")

    if title is not None:
        axis.set_title(
            title,
            pad=8.0,
        )

    if x_limits is None:
        x_min = min(np.min(profile[1]) for profile in profiles)

        x_max = max(np.max(profile[1]) for profile in profiles)

        axis.set_xlim(x_min, x_max)
    else:
        axis.set_xlim(x_limits)

    if y_limits is not None:
        axis.set_ylim(y_limits)
    else:
        minimum_value = min(np.min(profile[2]) for profile in profiles)

        if minimum_value >= 0.0:
            axis.set_ylim(bottom=0.0)

    axis.margins(x=0.0)

    axis.minorticks_on()

    axis.tick_params(
        axis="both",
        which="major",
        direction="in",
        top=True,
        right=True,
        length=4.0,
        width=0.8,
    )

    axis.tick_params(
        axis="both",
        which="minor",
        direction="in",
        top=True,
        right=True,
        length=2.2,
        width=0.6,
    )

    axis.grid(
        visible=True,
        which="major",
        linewidth=0.45,
        alpha=0.25,
    )

    if scientific_y:
        formatter = ScalarFormatter(useMathText=True)

        formatter.set_scientific(True)
        formatter.set_powerlimits((-3, 3))

        axis.yaxis.set_major_formatter(formatter)

    number_of_legend_columns = 2 if number_of_profiles > 4 else 1

    axis.legend(
        loc=legend_location,
        frameon=False,
        ncol=number_of_legend_columns,
        handlelength=2.8,
        columnspacing=1.2,
        borderaxespad=0.5,
    )

    for spine in axis.spines.values():
        spine.set_linewidth(0.8)

    output_stem.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    for file_format in formats:
        output_file = output_stem.with_suffix(f".{file_format}")

        figure.savefig(
            output_file,
            dpi=dpi,
            bbox_inches="tight",
            transparent=transparent,
            facecolor=("none" if transparent else "white"),
        )

        print(f"Figure saved to: {output_file}")

    if show:
        plt.show()

    plt.close(figure)


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description=(
            "Create publication-quality one-dimensional "
            "plots from FEniCSx XDMF time series."
        )
    )

    parser.add_argument(
        "xdmf_files",
        nargs="+",
        type=Path,
        help=("XDMF files to process. Example: pathogen.xdmf leukocytes.xdmf"),
    )

    parser.add_argument(
        "--times",
        nargs="+",
        type=float,
        default=None,
        help=(
            "Times to include in the figure. Example: "
            "--times 0 6 12 18 24 30. "
            "All available times are used by default."
        ),
    )

    parser.add_argument(
        "--fields",
        nargs="+",
        default=None,
        help=("Field name for each XDMF file, in the same order as the input files."),
    )

    parser.add_argument(
        "--labels",
        nargs="+",
        default=None,
        help=(
            "Publication labels for each input file. Example: "
            '--labels "Bacteria" "Neutrophils"'
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("plots_1d"),
        help=("Directory where figures will be saved. Default: plots_1d"),
    )

    parser.add_argument(
        "--formats",
        nargs="+",
        choices=["png", "pdf", "svg"],
        default=["png", "pdf"],
        help=("Output formats. Default: png pdf"),
    )

    parser.add_argument(
        "--xlim",
        nargs=2,
        type=float,
        metavar=("XMIN", "XMAX"),
        default=None,
        help=("Horizontal axis limits. Example: --xlim 0 1"),
    )

    parser.add_argument(
        "--ylim",
        nargs=2,
        type=float,
        metavar=("YMIN", "YMAX"),
        default=None,
        help=("Vertical axis limits. Example: --ylim 0 0.5"),
    )

    parser.add_argument(
        "--figsize",
        nargs=2,
        type=float,
        metavar=("WIDTH", "HEIGHT"),
        default=(6.5, 4.0),
        help=("Figure size in inches. Default: 6.5 4.0"),
    )

    parser.add_argument(
        "--font-size",
        type=float,
        default=10.0,
        help=("Base font size. Default: 10"),
    )

    parser.add_argument(
        "--dpi",
        type=int,
        default=600,
        help=("Raster output resolution. Default: 600 DPI"),
    )

    parser.add_argument(
        "--legend-location",
        choices=[
            "best",
            "upper right",
            "upper left",
            "lower right",
            "lower left",
            "center right",
            "center left",
        ],
        default="best",
        help=("Legend location. Default: best"),
    )

    parser.add_argument(
        "--no-title",
        action="store_true",
        help=(
            "Do not add a title to the figure. "
            "This is often preferable for journal articles."
        ),
    )

    parser.add_argument(
        "--scientific-y",
        action="store_true",
        help=("Use scientific notation on the vertical axis."),
    )

    parser.add_argument(
        "--transparent",
        action="store_true",
        help=("Save figures with a transparent background."),
    )

    parser.add_argument(
        "--show",
        action="store_true",
        help=("Display the figures after saving them."),
    )

    return parser.parse_args()


def validate_optional_list(
    values: list[str] | None,
    number_of_files: int,
    argument_name: str,
) -> None:
    """Check that an optional list contains one value per input file."""

    if values is not None and len(values) != number_of_files:
        raise ValueError(
            f"{argument_name} must contain exactly "
            f"{number_of_files} value(s), one for each XDMF file."
        )


def main() -> None:
    """Read the files and create all requested figures."""

    arguments = parse_arguments()

    configure_publication_style(font_size=arguments.font_size)

    number_of_files = len(arguments.xdmf_files)

    validate_optional_list(
        arguments.fields,
        number_of_files,
        "--fields",
    )

    validate_optional_list(
        arguments.labels,
        number_of_files,
        "--labels",
    )

    for index, xdmf_file in enumerate(arguments.xdmf_files):
        field_name = arguments.fields[index] if arguments.fields is not None else None

        detected_field, profiles = read_xdmf_time_series(
            xdmf_file=xdmf_file,
            field_name=field_name,
        )

        profiles = select_profiles(
            profiles=profiles,
            requested_times=arguments.times,
        )

        if arguments.labels is not None:
            species_label = arguments.labels[index]
        else:
            species_label = infer_species_label(
                file_path=xdmf_file,
                field_name=detected_field,
            )

        if arguments.no_title:
            title = None
        else:
            title = f"Spatial dynamics of {species_label.lower()}"

        output_stem = arguments.output_dir / f"{xdmf_file.stem}_profiles"

        x_limits = tuple(arguments.xlim) if arguments.xlim is not None else None

        y_limits = tuple(arguments.ylim) if arguments.ylim is not None else None

        plot_profiles(
            profiles=profiles,
            species_label=species_label,
            output_stem=output_stem,
            formats=arguments.formats,
            x_limits=x_limits,
            y_limits=y_limits,
            figure_size=tuple(arguments.figsize),
            dpi=arguments.dpi,
            title=title,
            legend_location=arguments.legend_location,
            scientific_y=arguments.scientific_y,
            transparent=arguments.transparent,
            show=arguments.show,
        )


if __name__ == "__main__":
    main()
