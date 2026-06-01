import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from matplotlib import axes, figure
from matplotlib.ticker import MaxNLocator
from weas_widget import WeasWidget

from .cube import Cube
from .output import AimsOutput


class VisualiseAims(AimsOutput):
    """
    FHI-aims visualisation tools.

    ...

    Attributes
    ----------
    scf_convergence : dict[str, NDArray[float64]]
    convergence_params : dict[str, float]

    Examples
    --------
    >>> from dfttoolkit.visualise import VisualiseAims
    >>> vis = VisualiseAims(aims_out="path/to/aims.out")
    """

    def __init__(self, aims_out: str = "aims.out"):
        super().__init__(aims_out=aims_out)

    @staticmethod
    def _plot_charge_convergence(
        ax: axes.Axes,
        charge_density: float,
        tot_scf_iters: npt.NDArray[np.int64],
        delta_charge: npt.NDArray[np.float64],
        delta_charge_sd: npt.NDArray[np.float64] | None = None,
        title: str | None = None,
    ) -> None:
        """
        Create a subplot for the charge convergence of an FHI-aims calculation.

        Parameters
        ----------
        ax : Axes
            matplotlib subplot index
        charge_density : float
            convergence criterion for the charge density
        tot_scf_iters : NDArray[int64]
            cumulative SCF iterations
        delta_charge : NDArray[float64]
            change of spin-up or total spin (if the calculation was spin none)
            eigenvalues
        delta_charge_sd : NDArray[float64] | None, default=None
            change of spin-down eigenvalues
        title : str | None, default=None
            system name to include in title
        """
        ax.plot(tot_scf_iters, delta_charge, label=r"$\Delta$ charge")

        # Only plot delta_charge_sd if the calculation is spin polarised
        if delta_charge_sd is not None and np.all(delta_charge_sd) != 0.0:
            ax.plot(
                tot_scf_iters,
                delta_charge_sd,
                label=r"$\Delta$ charge/spin density",
            )

        # Add the convergence parameters
        if charge_density != 0.0:
            ax.axhline(
                charge_density,
                ls="--",
                c="gray",
                label="charge convergence criterion",
            )

        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.set_yscale("log")
        ax.set_xlabel("cumulative SCF iteration")
        ax.set_ylabel(r"Charge / $e a_0^{-3}$")
        ax.legend()

        if title is None:
            title = ""
        else:
            title += " "

        ax.set_title(rf"{title}Charge Convergence")

    @staticmethod
    def _plot_energy_convergence(
        ax: axes.Axes,
        sum_eigenvalues_conv_param: float,
        total_energy_conv_param: float,
        tot_scf_iters: npt.NDArray[np.int64],
        delta_sum_eigenvalues: npt.NDArray[np.float64],
        delta_total_energies: npt.NDArray[np.float64],
        absolute: bool = True,
        title: str | None = None,
    ) -> None:
        """
        Create a subplot for the energy convergence of an FHI-aims calculation.

        Parameters
        ----------
        ax : Axes
            matplotlib subplot index
        sum_eigenvalues_conv_param : float
            convergence criterion for the sum of eigenvalues
        total_energy_conv_param : float
            convergence criterion for the total energy
        tot_scf_iters : NDArray[int64]
            cumulative SCF iterations
        delta_sum_eigenvalues : NDArray[float64]
            change of sum of eigenvalues
        delta_total_energies : NDArray[float64]
            change of total energies
        absolute : bool, default=True
            whether to plot the absolute value of the energies
        title : str | None, default=None
            system name to include in title
        """
        if absolute:
            delta_sum_eigenvalues = np.abs(delta_sum_eigenvalues)
            delta_total_energies = np.abs(delta_total_energies)

        ax.plot(
            tot_scf_iters,
            delta_sum_eigenvalues,
            label=r"$\Delta \; \Sigma$ eigenvalues",
            c="C1",
        )
        ax.plot(
            tot_scf_iters,
            delta_total_energies,
            label=r"$\Delta$ total energies",
            c="C0",
        )

        # Add the convergence parameters
        if sum_eigenvalues_conv_param != 0:
            ax.axhline(
                sum_eigenvalues_conv_param,
                ls="-.",
                c="darkgray",
                label=r"$\Delta \; \Sigma$ eigenvalues convergence criterion",
            )
        if total_energy_conv_param != 0:
            ax.axhline(
                total_energy_conv_param,
                ls="--",
                c="gray",
                label=r"$\Delta$ total energies convergence criterion",
            )

        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.set_yscale("log")
        ax.set_xlabel("cumulative SCF iteration")
        ax.set_ylabel(r"absolute energy / $| \mathrm{eV} |$")
        ax.legend()

        if title is None:
            title = ""
        else:
            title += " "

        ax.set_title(rf"{title}Energies and Eigenvalues Convergence")

    @staticmethod
    def _plot_forces_convergence(
        ax: axes.Axes,
        change_max_force_conv_param: float,
        max_force_per_atom_conv_param: float,
        delta_forces: npt.NDArray[np.float64],
        forces_on_atoms: npt.NDArray[np.float64],
        title: str | None = None,
    ) -> None:
        """
        Create a subplot for the forces convergence of an FHI-aims calculation.

        Parameters
        ----------
        ax : Axes
            matplotlib subplot index
        change_max_force_conv_param : float
            convergence criterion specified by `sc_accuracy_forces`
        max_force_per_atom_conv_param : float
            convergence criterion specified by `relax_geometry`
        delta_forces: NDArray[float64] | list[float],
            change in forces on each atom
        forces_on_atoms : NDArray[float64]
            all forces acting on each atom
        title : str | None, default=None
            system name to include in title
        """
        ax.plot(forces_on_atoms, label=r"max force atom$^{-1}$")
        ax.plot(delta_forces, label=r"$\Delta$ max force")

        # Add the convergence parameters
        if max_force_per_atom_conv_param != 0:
            ax.axhline(
                max_force_per_atom_conv_param,
                ls="--",
                c="gray",
                label=r"max force atom$^{-1}$ convergence criterion",
            )

        if change_max_force_conv_param != 0:
            ax.axhline(
                change_max_force_conv_param,
                ls="-.",
                c="darkgray",
                label=r"$\Delta$ max force convergence criterion",
            )

        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.set_yscale("log")
        ax.set_xlabel("geometry relaxation step")
        ax.set_ylabel(r"force / $\mathrm{eV} \mathrm{\AA}^{-1}$")
        ax.legend()

        if title is None:
            title = ""
        else:
            title += " "

        ax.set_title(rf"{title}Force Convergence")

    @staticmethod
    def _plot_ks_states_convergence(
        ax: axes.Axes,
        ks_eigenvals: (
            dict[str, npt.NDArray[np.int64 | np.float64]]
            | tuple[
                dict[str, npt.NDArray[np.int64 | np.float64]],
                dict[str, npt.NDArray[np.int64 | np.float64]],
            ]
        ),
        title: str | None = None,
    ) -> None:
        """
        Create a subplot for the energy changes of the KS eigenstates.

        Parameters
        ----------
        ax : Axes
            matplotlib subplot index
        ks_eigenvals: (
            dict[str, NDArray[int64 | float64]]
            | tuple[NDArray, NDArray]
        )
            state, occupation, and eigenvalue of each KS state at each SCF iteration
        title : str | None, default=None
            system name to include in title
        """
        if isinstance(ks_eigenvals, dict):
            # Don't include last eigenvalue as it only prints after final SCF iteration
            # Add 1 to total SCF iterations to match the length of the eigenvalues and
            # we want to include the first pre SCF iteration
            for ev in ks_eigenvals["eigenvalue_eV"].T:
                ax.plot(np.arange(len(ks_eigenvals["eigenvalue_eV"])), ev)

        elif isinstance(ks_eigenvals, tuple):
            su_ks_eigenvals = ks_eigenvals[0]
            sd_ks_eigenvals = ks_eigenvals[1]

            for i, ev in enumerate(su_ks_eigenvals["eigenvalue_eV"].T):
                ax.plot(
                    np.arange(len(su_ks_eigenvals["eigenvalue_eV"])),
                    ev,
                    c="C0",
                    label="Spin-up Eigenstates" if i == 0 else None,
                )

            for i, ev in enumerate(sd_ks_eigenvals["eigenvalue_eV"].T):
                ax.plot(
                    np.arange(len(su_ks_eigenvals["eigenvalue_eV"])),
                    ev,
                    c="C1",
                    label="Spin-down Eigenstates" if i == 0 else None,
                )

            ax.legend()

        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.set_yscale("symlog")
        ax.set_xlabel("cumulative SCF iteration")
        ax.set_ylabel("energy / eV")

        if title is None:
            title = ""
        else:
            title += " "

        ax.set_title(rf"{title}KS State Convergence")

    def convergence(
        self,
        title: str | None = None,
        forces: bool = False,
        ks_eigenvalues: bool = False,
        fig_size: tuple[int, int] = (20, 5),
    ) -> figure.Figure:
        """
        Plot the SCF convergence.

        Parameters
        ----------
        title : str | None, default=None
            system name to use in title of the plot
        forces : bool, default=False
            whether to plot the change of forces and forces on atoms
        ks_eigenvalues : bool, default=False
            whether to plot the kohn-sham eigenvalues
        fig_size : tuple[int, int], default=(24, 6)
            the total size of the figure

        Returns
        -------
        Figure
            matplotlib figure object

        Examples
        --------
        >>> from dfttoolkit.visualise import VisualiseAims
        >>> vis = VisualiseAims("path/to/aims.out")
        >>> fig = vis.convergence(
        ...     title="My System", forces=True, ks_eigenvalues=True, fig_size=(12, 8)
        ... )
        """
        # Get the SCF convergence values and parameters
        self.get_scf_convergence()
        self.get_convergence_parameters()

        # Get the total number of SCF iterations
        tot_scf_iters = np.arange(
            1, len(self.scf_convergence["SCF iterations"]) + 1
        )

        # Change the number of subplots if forces and ks_eigenvalues are to be plotted
        subplots = [True, True, forces, ks_eigenvalues]
        i_subplot = 1

        # Setup the figure subplots
        fig, ax = plt.subplots(1, subplots.count(True), figsize=fig_size)

        # Plot the change of charge
        self._plot_charge_convergence(
            ax[0],
            self.convergence_params["charge density"],
            tot_scf_iters,
            self.scf_convergence["change of charge"],
            self.scf_convergence["change of charge spin density"],
            title,
        )

        # Plot the change of total energies and sum of eigenvalues
        self._plot_energy_convergence(
            ax[1],
            self.convergence_params["sum eigenvalues"],
            self.convergence_params["total energy"],
            tot_scf_iters,
            self.scf_convergence["change of sum eigenvalues"],
            self.scf_convergence["change of total energy"],
            title=title,
        )

        # Plot the forces
        if forces:
            delta_forces = self.scf_convergence["change of max force"]
            delta_forces = np.delete(
                delta_forces, np.argwhere(delta_forces == 0.0)
            )
            forces_on_atoms = self.scf_convergence["forces on atoms"]
            forces_on_atoms = np.delete(
                forces_on_atoms, np.argwhere(forces_on_atoms == 0.0)
            )
            i_subplot += 1
            self._plot_forces_convergence(
                ax[i_subplot],
                self.convergence_params["change of max force"],
                self.convergence_params["max force per atom"],
                delta_forces,
                forces_on_atoms,
                title,
            )

        # Plot the KS state energies
        if ks_eigenvalues:
            i_subplot += 1
            ks_eigenvals = self.get_all_ks_eigenvalues()

            self._plot_ks_states_convergence(
                ax[i_subplot], ks_eigenvals, title
            )

        return fig


class VisualiseCube(Cube):
    """
    Cube visualisation tools.

    ...

    Attributes
    ----------
    path : str
        path to the .cube file
    lines : list[str]
        contents of the .cube file
    atoms : Atom | Atoms
        ASE atom or atoms object
    volume : NDArray[float64]
        volumetric data of the cube file

    Parameters
    ----------
    cube : str
        path to the .cube file
    """

    def __init__(self, filename: str):
        super().__init__(filename=filename)

    def weas_core_hole(
        self, viewer: WeasWidget | None = None, **kwargs: str
    ) -> WeasWidget | None:
        """
        Visualise the core hole as an isosurface.

        Parameters
        ----------
        viewer : WeasWidget | None, default=None
            viewer instance
        **kwargs
            viewer settings keyword arguments

        Returns
        -------
        WeasWidget | None
            viewer instance or None if not running in Jupyter or Marimo
        """
        # Check if running in Jupyter
        # TODO
        # try:
        #     from IPython import (  # type: ignore[unresolved-import]
        #         get_ipython,
        #     )

        #     if (
        #         not mo.running_in_notebook()
        #         or get_ipython().__class__.__name__ != "ZMQInteractiveShell"
        #     ):
        #         raise NotImplementedError(
        #             "The core_hole method is only available in a Jupyter session."
        #         )

        # except (ImportError, AttributeError):
        #     return None

        # Create the viewer
        ch_viewer = WeasWidget(**kwargs) if viewer is None else viewer

        ch_viewer.from_ase(self.atoms)
        ch_viewer.avr.model_style = 1  # Ball and stick
        ch_viewer.avr.iso.volumetric_data = {"values": self.volume}
        ch_viewer.avr.iso.settings = {
            "positive": {"isovalue": -0.03, "color": "red"},
            "negative": {"isovalue": 0.03, "color": "blue"},
        }

        return ch_viewer

    def visualise_plane(
        self,
        plane_centre: npt.NDArray,
        plane_normal: npt.NDArray,
        extent: list[float] | npt.NDArray,
        plane_points: int = 100,
        cmap: str = "RdBu_r",
        vmin: float | None = None,
        vmax: float | None = None,
    ) -> None:
        """
        Extract and plot a 2D heatmap slice of the cube file grid values.

        Parameters
        ----------
        plane_centre : npt.NDArray
            Cartesian coordinates [x, y, z] of the plane's center point.
        plane_normal : npt.NDArray
            Vector perpendicular to the viewing plane.
        plane_extent : float
            Half-width of the square viewport window in Angstroms.
        plane_points : int, default=100
            Resolution grid size for the extraction slice.
        cmap : str, default="RdBu_r"
            The matplotlib colormap palette string.
        show_atoms : bool, default=True
            If True, projects and overlays nearby atoms within the tolerance gap.
        atom_tolerance : float, default=1.0
            Maximum perpendicular distance (Angstroms) for an atom to be rendered.
        save_path : str | Path | None, default=None
            If provided, saves the output plot to the given filepath destination.
        """
        xmin, xmax, ymin, ymax = extent
        plane_extent = np.max(np.abs(extent))

        # 1. Extract 2D grid slice from the volumetric array data
        values = self.get_values_on_plane(
            plane_centre=plane_centre,
            plane_normal=plane_normal,
            plane_extent=plane_extent,
            plane_points=plane_points,
        )

        # 3. Create a coordinate mapping of the calculated square rows/columns
        # Note: get_values_on_plane uses: np.linspace(-plane_extent, plane_extent, plane_points)
        square_coords = np.linspace(-plane_extent, plane_extent, plane_points)

        # 4. Find the array index slices corresponding to your custom bounding box
        # We find where the coordinates fall inside the [xmin, xmax] and [ymin, ymax] windows
        x_indices = np.where(
            (square_coords >= xmin) & (square_coords <= xmax)
        )[0]
        y_indices = np.where(
            (square_coords >= ymin) & (square_coords <= ymax)
        )[0]

        if len(x_indices) == 0 or len(y_indices) == 0:
            raise ValueError(
                "The requested extent lies completely outside the sampled plane_extent."
            )

        idx_x_min, idx_x_max = x_indices[0], x_indices[-1] + 1
        idx_y_min, idx_y_max = y_indices[0], y_indices[-1] + 1

        # 5. Slice the matrix down to the requested rectangular size
        # Because values is generated as values[ind_1, ind_2] where ind_1 is x and ind_2 is y:
        cropped_values = values[idx_x_min:idx_x_max, idx_y_min:idx_y_max]

        # 6. Render the cropped matrix
        ax = plt.gca()
        img = ax.imshow(
            cropped_values.T,  # Transpose to map rows to Y and columns to X
            extent=extent,
            origin="lower",
            cmap=cmap,
            aspect="equal",
            vmin=vmin,
            vmax=vmax,
        )
