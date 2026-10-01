import numpy as np
from dfttoolkit.geometry import XYZGeometry


def read_xyz_animation(filename: str) -> list[str]:
    """
    Read an XYZ animation file.

    i.e. an XYZ file containing serveral geometries.

    Parameters
    ----------
    filename : str
        path to xyz animation file

    Returns
    -------
    list[str]
        list of XYZGeometries
    """
    text_list = []
    geometry_text = ""

    with open(filename) as f:
        text = f.readlines()

        for ind, line in enumerate(text):
            line_split = line.split()

            n_entries = len(line_split)

            if ind != 0 and n_entries == 1:
                text_list.append(geometry_text)
                geometry_text = ""

            geometry_text += line

    geometry_list = []
    for text in text_list:
        geometry = XYZGeometry()
        geometry.parse(text)
        geometry_list.append(geometry)

    return geometry_list


def get_cubefile_grid(geom, spacing, origin=None, verbose=True):
    """
    Creates cube file grid with a target spacing in each direction.
    If only one spacing value is given, it will be used in all directions.
    If origin is given the vectors will be aligned around this value.

    Returns a formatted string containing the FHI-aims cube file settings.
    """

    # Handle the case where a single integer or float is provided for spacing
    if isinstance(spacing, (int, float)):
        spacing = [float(spacing)] * 3
    else:
        spacing = [float(s) for s in spacing]

    if origin is None:
        origin = (
            geom.lattice_vectors[0, :] / 2
            + geom.lattice_vectors[1, :] / 2
            + geom.lattice_vectors[2, :] / 2
        )

    # Calculate the number of divisions based on the target spacing
    # We use ceil to ensure the final spacing is <= the requested target spacing
    divisions = np.zeros(3, dtype=int)
    for i in range(3):
        vec_length = np.linalg.norm(geom.lattice_vectors[i, :])
        divisions[i] = max(1, int(np.ceil(vec_length / spacing[i])))

    # calculate exact vectors (dx, dy, dz for each dimension)
    vecs = np.zeros([3, 3])
    vecs[0, :] = geom.lattice_vectors[0, :] / divisions[0]
    vecs[1, :] = geom.lattice_vectors[1, :] / divisions[1]
    vecs[2, :] = geom.lattice_vectors[2, :] / divisions[2]

    # Format the output string to match FHI-aims requirements
    cube_settings = (
        f"cube origin    {origin[0]:.10f}    {origin[1]:.10f}    {origin[2]:.10f}\n"
        f"cube edge {divisions[0]:<4}   {vecs[0, 0]: >14.10f}   {vecs[0, 1]: >14.10f}   {vecs[0, 2]: >14.10f}\n"
        f"cube edge {divisions[1]:<4}   {vecs[1, 0]: >14.10f}   {vecs[1, 1]: >14.10f}   {vecs[1, 2]: >14.10f}\n"
        f"cube edge {divisions[2]:<4}   {vecs[2, 0]: >14.10f}   {vecs[2, 1]: >14.10f}   {vecs[2, 2]: >14.10f}"
    )

    if verbose:
        print("Generated FHI-aims cube settings:")
        print(cube_settings)

    return cube_settings


def get_k_grid(geom_bulk, geom_slab, k_density, verbose=True):
    """
    Calculates the appropriate k-point grid for a slab geometry based on the
    reciprocal lattice vector of a reference bulk geometry and a target k-density.

    Parameters
    ----------
    geom_bulk : object
        The bulk geometry object (must have a `get_reciprocal_lattice_vectors()` method).
    geom_slab : object
        The slab geometry object (must have a `get_reciprocal_lattice_vectors()` method).
    k_density : int or float, default=64
        The target k-point density (referenced to the bulk cell).

    Returns
    -------
    tuple[int, int, int]
        A tuple of 3 integers representing the k-point grid (k_x, k_y, 1) for the slab.
    """
    r_lattice_bulk = geom_bulk.get_reciprocal_lattice_vectors()
    r_lattice_slab = geom_slab.get_reciprocal_lattice_vectors()

    # Calculate norms of the reciprocal lattice vectors
    l_0 = np.linalg.norm(r_lattice_bulk[0])
    l_1 = np.linalg.norm(r_lattice_slab[0])
    l_2 = np.linalg.norm(r_lattice_slab[1])
    l_3 = np.linalg.norm(r_lattice_slab[2])

    # Calculate the k-grid for the slab, keeping 1 in the z-direction
    k_grid = (
        int(np.ceil(k_density * l_1 / l_0)),
        int(np.ceil(k_density * l_2 / l_0)),
        int(np.ceil(k_density * l_3 / l_0)),
    )

    if verbose:
        print("Generated FHI-aims k-grid settings:")
        print(f"k_grid {k_grid[0]} {k_grid[1]} {k_grid[2]}")

    return k_grid
