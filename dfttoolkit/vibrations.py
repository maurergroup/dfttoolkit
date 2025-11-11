import copy
import functools
import multiprocessing as mp
import sys
from pathlib import Path

import numpy as np
import numpy.typing as npt

from .geometry import AimsGeometry, VaspGeometry
from .utils import units
from .utils import vibrations_utils as vu
from .utils.periodic_table import PeriodicTable


class Vibrations:
    """TODO."""

    def __init__(self):
        # TODO This is currently a placeholder and should be developed further
        self.wave_vector = np.array([0.0, 0.0, 0.0])
        self._vibration_coords = []
        self._vibration_forces = []

    def get_instance_of_other_type(
        self, vibrations_type: str
    ) -> "AimsVibrations | VaspVibrations":
        if vibrations_type == "aims":
            new_vibration = AimsVibrations()
        elif vibrations_type == "vasp":
            new_vibration = VaspVibrations()
        else:
            msg = f"Unsupported vibration type: {vibrations_type}"
            raise ValueError(msg)

        new_vibration.__dict__ = self.__dict__

        return new_vibration

    @property
    def vibration_coords(self) -> list[npt.NDArray[np.float64]]:
        return self._vibration_coords

    @vibration_coords.setter
    def vibration_coords(self, vibration_coords: list[npt.NDArray[np.float64]]) -> None:
        self._vibration_coords = vibration_coords

    @property
    def vibration_forces(self) -> list[npt.NDArray[np.float64]]:
        return self._vibration_forces

    @vibration_forces.setter
    def vibration_forces(self, vibration_forces: list[npt.NDArray[np.float64]]) -> None:
        self._vibration_forces = vibration_forces

    @property
    def hessian(self) -> npt.NDArray[np.float64]:
        return self._hessian

    @hessian.setter
    def hessian(self, hessian: npt.NDArray[np.float64]) -> None:
        self._hessian = hessian

    @property
    def eigenvalues(self) -> npt.NDArray[np.float64]:
        return self._eigenvalues

    @eigenvalues.setter
    def eigenvalues(self, eigenvalues: npt.NDArray[np.float64]) -> None:
        self._eigenvalues = eigenvalues

    @property
    def eigenvectors(self) -> npt.NDArray[np.float64]:
        return self._eigenvectors

    @eigenvectors.setter
    def eigenvectors(self, eigenvectors: npt.NDArray[np.float64]) -> None:
        self._eigenvectors = eigenvectors

    def get_displacements(
        self, displacement: float = 0.0025, directions: list | None = None
    ) -> list:  # pyright:ignore
        """
        Apply a given displacement for each degree of freedom of self and
        generates a new vibration with it.

        Parameters
        ----------
        displacement : float, default=0.0025
            Displacement for finite difference calculation of vibrations in
            Angstrom.

        Returns
        -------
        list
            List of geometries where atoms have been displaced.

        """  # noqa: D205
        geometries_displaced = [self]

        if directions is None:
            directions = [1]

        for i in range(self.n_atoms):  # pyright:ignore
            for dim in range(3):
                if self.constrain_relax[i, dim]:  # pyright:ignore
                    continue

                for direction in directions:
                    geometry_displaced = copy.deepcopy(self)
                    geometry_displaced.coords[i, dim] += (  # pyright:ignore
                        displacement * direction
                    )

                    geometries_displaced.append(geometry_displaced)

        return geometries_displaced

    def get_mass_tensor(self) -> npt.NDArray[np.float64]:
        """
        Determine a NxN tensor containing sqrt(m_i*m_j).

        Returns
        -------
        mass_tensor : np.array
            Mass tensor in atomic units.
        """
        mass_vector = [
            PeriodicTable.get_atomic_mass(s)  # pyright:ignore
            for s in self.species  # pyright:ignore
        ]
        mass_vector = np.repeat(mass_vector, 3)

        mass_tensor = np.tile(mass_vector, (len(mass_vector), 1))

        return np.sqrt(mass_tensor * mass_tensor.T)

    def get_hessian(
        self, set_constrained_atoms_zero: bool = False
    ) -> npt.NDArray[np.float64]:
        """
        Calculate the Hessian from the forces.

        This includes the atomic masses since F = m*a.

        Parameters
        ----------
        set_constrained_atoms_zero : bool, default=False
            Set elements in Hessian that code for constrained atoms to zero.

        Returns
        -------
        H : np.array
            Hessian.
        """
        N = len(self) * 3  # pyright:ignore
        H = np.zeros([N, N])

        if not np.allclose(self.coords, self.vibration_coords[0]):  # pyright:ignore
            raise ValueError(
                "The first entry in vibration_coords must be identical to the "
                "undispaced geometry."
            )

        coords_0 = self.vibration_coords[0].flatten()
        F_0 = self.vibration_forces[0].flatten()

        n_forces = np.zeros(N, np.int64)

        for c, F in zip(self.vibration_coords, self.vibration_forces, strict=False):
            dF = F.flatten() - F_0
            dx = c.flatten() - coords_0
            ind = np.argmax(np.abs(dx))
            n_forces[ind] += 1
            displacement = dx[ind]

            if np.abs(displacement) < 1e-5:
                continue

            H[ind, :] -= dF / displacement

        for row in range(H.shape[0]):
            if n_forces[row] > 0:
                H[row, :] /= n_forces[row]  # prevent div by zero for unknown forces

        if set_constrained_atoms_zero:
            constrained = self.constrain_relax.flatten()  # pyright:ignore
            H[constrained, :] = 0
            H[:, constrained] = 0

        self.hessian = H

        return H

    def get_symmetrized_hessian(
        self, hessian: npt.NDArray[np.float64] | None = None
    ) -> npt.NDArray[np.float64]:
        """
        Symmetrieses the Hessian by using the lower triangular matrix.

        Parameters
        ----------
        hessian : npt.NDArray[np.float64] | None, default=None
            Hessian matrix to be symmetrized. If None, the Hessian of the object

        Returns
        -------
        npt.NDArray[np.float64]
            Symmetrized Hessian matrix.
        """
        if hessian is None:
            hessian = copy.deepcopy(self.hessian)

        hessian_new = hessian + hessian.T

        all_inds = list(range(len(self) * 3))  # pyright:ignore

        constrain = self.constrain_relax.flatten()  # pyright:ignore
        constrained_inds = [i for i, c in enumerate(constrain) if c]
        constrained_inds = np.array(constrained_inds)

        unconstrained_inds = np.array(list(set(all_inds) - set(constrained_inds)))

        for i in unconstrained_inds:
            for j in unconstrained_inds:
                hessian_new[i, j] *= 0.5

        return hessian_new

    def get_eigenvalues_and_eigenvectors(
        self,
        hessian: npt.NDArray[np.float64] | None = None,  # pyright:ignore
        only_real: bool = True,
        symmetrize_hessian: bool = True,
        eigenvectors_to_cartesian: bool = False,
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """
        Get all eigenvalues and eigenvectors of the hessian.

        Note that the eigenvectors are mass weighted.

        Parameters
        ----------
        hessian : npt.NDArray[np.float64], optional
            Hessian. The default is None.
        only_real : bool, default=True
            Returns only real valued eigenfrequencies + eigenmodes
            (ATTENTION: if you want to also include instable modes, you have to
            symmetrize the hessian as provided below).
        symmetrize_hessian : bool, default=True
            Symmetrise the hessian only for this function (no global change).

        Returns
        -------
        omega2 : np.array
            Direct eigenvalues as squared angular frequencies instead of
            inverse wavelengths.
        eigenvectors : np.array
            Mass weighted eigenvectors of the Hessian given as a list of numpy
            arrays, where each array is a normalized displacement for the
            corresponding eigenfrequency.
        """
        if symmetrize_hessian:
            hessian = self.get_symmetrized_hessian(hessian=hessian)
        elif hessian is None:
            hessian = copy.deepcopy(self.hessian)

        if not hasattr(self, "hessian") or hessian is None:
            raise ValueError("Hessian must be given to calculate the Eigenvalues!")

        M = 1 / self.get_mass_tensor()

        omega2, X = np.linalg.eig(M * hessian)

        # only real valued eigen modes
        if only_real:
            real_mask = np.isreal(omega2)
            min_omega2 = 1e-3
            min_mask = omega2 >= min_omega2
            mask = np.logical_and(real_mask, min_mask)

            omega2 = np.real(omega2[mask])
            X = np.real(X[:, mask])

        eigenvectors = [column.reshape(-1, 3) for column in X.T]

        # sort modes by energies (ascending)
        ind_sort = np.argsort(omega2)
        eigenvectors = np.array(eigenvectors)[ind_sort, :, :]
        omega2 = omega2[ind_sort]

        self.eigenvalues = omega2
        self.eigenvectors = eigenvectors

        # Convert eigenvector to Cartesian coordinates
        if eigenvectors_to_cartesian:
            m = np.tile(np.sqrt(self.get_atomic_masses()), (3, 1)).T  # pyright:ignore

            for index in range(len(eigenvectors)):
                eigenvectors[index] /= m

        return omega2, eigenvectors

    def get_eigenvalues_in_Hz(  # noqa: N802
        self, omega2: npt.NDArray[np.float64] | None = None
    ) -> npt.NDArray[np.float64]:
        """
        Determine angular vibration frequencies in Hz.

        Parameters
        ----------
        omega2 : npt.NDArray[np.float64] | None, default=None
            Eigenvalues of the mass weighted hessian.

        Returns
        -------
        omega_SI : np.array
            Array of the eigenfrequencies in Hz.
        """
        if omega2 is None:
            omega2 = self.get_eigenvalues_and_eigenvectors()[0]

        omega = np.sign(omega2) * np.sqrt(np.abs(omega2))  # pyright:ignore

        conversion = np.sqrt(
            (units.EV_IN_JOULE) / (units.ATOMIC_MASS_IN_KG * units.ANGSTROM_IN_METER**2)
        )
        return omega * conversion

    def get_eigenvalues_in_inverse_cm(
        self, omega2: npt.NDArray[np.float64] | None = None
    ) -> npt.NDArray[np.float64]:
        """
        Determine vibration frequencies in cm^-1.

        Parameters
        ----------
        omega2 : Union[None, np.array]
            Eigenvalues of the mass weighted hessian.

        Returns
        -------
        f_inv_cm : np.array
            Array of the eigenfrequencies in cm^(-1).
        """
        omega_SI = self.get_eigenvalues_in_Hz(omega2=omega2)
        return omega_SI * units.INVERSE_CM_IN_HZ / (2 * np.pi)

    def get_eigenvalues_in_eV(  # noqa: N802
        self, omega2: npt.NDArray[np.float64] | None = None
    ) -> npt.NDArray[np.float64]:
        omega_SI = self.get_eigenvalues_in_Hz(omega2=omega2)
        return omega_SI * units.PLANCK_CONSTANT / (2 * np.pi) / units.JOULE_IN_EV

    def get_thermally_displaced_geometry(
        self, temperature: np.float64, classical: bool = True
    ) -> list:  # pyright:ignore
        """
        Generate thermally displaced structures from vibrational modes.

        Parameters
        ----------
        modes : np.ndarray
            Normalized mode eigenvectors, shape (n_modes, 3N).
        frequencies : np.ndarray
            Frequencies in Hz or rad/s, shape (n_modes,).
        masses : np.ndarray
            Atomic masses in kg, shape (N_atoms,).
        temperature : float
            Temperature in Kelvin.
        n_samples : int
            Number of thermally displaced geometries to generate.
        classical : bool
            If True, uses classical approximation (kT), else quantum formula.

        Returns
        -------
        displacements : list of np.ndarray
            List of atomic displacements (shape (3N,)) for each snapshot.
        """
        kB = 1.380649e-23  # J/K

        eigenvalues = self.get_eigenvalues_in_Hz()
        n_modes = len(eigenvalues)

        new_geometry = copy.deepcopy(self)

        displacement = np.zeros((len(self), 3))
        for i in range(n_modes):
            freq = eigenvalues[i]
            if freq < 1e-12:
                continue  # skip zero or imaginary modes

            omega = 2 * np.pi * freq
            if classical:
                var_qi = kB * temperature / (omega**2)
            else:
                hbar = 1.054571817e-34
                # var_qi = (
                #     (hbar / (2 * omega))
                #     * np.cosh(hbar * omega / (2 * kB * temperature))
                #     / np.sinh(hbar * omega / (2 * kB * temperature))
                # )
                var_qi = (hbar / (2 * omega)) / np.tanh(
                    hbar * omega / (2 * kB * temperature)
                )

            # Convert to atomic units
            # var_qi (kg m²) -> var_qi (u A²)
            var_qi *= 1e20 / units.ATOMIC_MASS_IN_KG

            amp = np.random.Generator(0.0, np.sqrt(var_qi))  # pyright:ignore
            print(amp)
            displacement += amp * self.eigenvectors[i]

        # Convert from mass weighted to Cartesian coordinates
        m = np.tile(np.sqrt(self.get_atomic_masses()), (3, 1)).T  # pyright:ignore

        displacement /= m

        new_geometry.coords += displacement

        return new_geometry

    def get_atom_type_index(self) -> npt.NDArray[np.int64]:
        n_atoms = len(self)  # pyright:ignore

        # Tolerance for accepting equivalent atoms in super cell
        masses = self.get_mass_of_all_atoms()  # pyright:ignore
        tolerance = 0.001

        primitive_cell_inverse = np.linalg.inv(self.lattice_vectors)  # pyright:ignore

        atom_type_index = np.array([None] * n_atoms)
        counter = 0
        for i in range(n_atoms):
            if atom_type_index[i] is None:
                atom_type_index[i] = counter
                counter += 1
            for j in range(i + 1, n_atoms):
                coordinates_atom_i = self.coords[i]  # pyright:ignore
                coordinates_atom_j = self.coords[j]  # pyright:ignore

                difference_in_cell_coordinates = np.around(
                    np.dot(
                        primitive_cell_inverse.T,
                        (coordinates_atom_j - coordinates_atom_i),
                    )
                )

                projected_coordinates_atom_j = coordinates_atom_j - np.dot(
                    self.lattice_vectors.T,  # pyright:ignore
                    difference_in_cell_coordinates,
                )
                separation = pow(
                    np.linalg.norm(projected_coordinates_atom_j - coordinates_atom_i),
                    2,
                )

                if separation < tolerance and masses[i] == masses[j]:
                    atom_type_index[j] = atom_type_index[i]

        return np.array(atom_type_index, dtype=int)

    def get_velocity_mass_average(
        self, velocities: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        """
        Weighs velocities by atomic masses.

        Parameters
        ----------
        velocities : npt.NDArray[np.float64]

        Returns
        -------
        velocities_mass_average : np.array
            Velocities weighted by atomic masses.
        """
        velocities_mass_average = np.zeros_like(velocities)

        for i in range(velocities.shape[1]):
            velocities_mass_average[:, i, :] = velocities[:, i, :] * np.sqrt(
                self.get_atomic_masses()[i]  # pyright:ignore
            )

        return velocities_mass_average

    def project_onto_wave_vector(
        self,
        velocities: npt.NDArray[np.float64],
        wave_vector: npt.NDArray[np.float64],
        project_on_atom: int = -1,
    ) -> npt.NDArray[np.float64]:
        number_of_primitive_atoms = len(self)  # pyright:ignore
        number_of_atoms = velocities.shape[1]
        number_of_dimensions = velocities.shape[2]

        coordinates = self.coords  # pyright:ignore
        atom_type = self.get_atom_type_index()

        velocities_projected = np.zeros(
            (
                velocities.shape[0],
                number_of_primitive_atoms,
                number_of_dimensions,
            ),
            dtype=complex,
        )

        if wave_vector.shape[0] != coordinates.shape[1]:
            print("Warning!! Q-vector and coordinates dimension do not match")
            sys.exit()

        # Projection onto the wave vector
        for i in range(number_of_atoms):
            # Projection on atom
            if project_on_atom > -1 and atom_type[i] != project_on_atom:
                continue

            for k in range(number_of_dimensions):
                velocities_projected[:, atom_type[i], k] += velocities[
                    :, i, k
                ] * np.exp(-1j * np.dot(wave_vector, coordinates[i, :]))

        # Normalize the velocities
        number_of_primitive_cells = number_of_atoms / number_of_primitive_atoms
        velocities_projected /= np.sqrt(number_of_primitive_cells)

        return velocities_projected

    def get_normal_mode_decomposition(
        self,
        velocities: npt.NDArray,
        use_numba: bool = True,
    ) -> npt.NDArray:
        """
        Calculate the normal-mode-decomposition of the velocities.

        This is done by projecting the atomic velocities onto the vibrational
        eigenvectors. See equation 10 in: https://doi.org/10.1016/j.cpc.2017.08.017

        Parameters
        ----------
        velocities : npt.NDArray[np.float64]
            Array containing the velocities from an MD trajectory structured in
            the following way:
            [number of time steps, number of atoms, number of dimensions].

        Returns
        -------
        velocities_projected : npt.NDArray[np.float64]
            Velocities projected onto the eigenvectors structured as follows:
            [number of time steps, number of frequencies]
        """
        velocities = np.array(velocities, dtype=np.complex128)

        velocities_mass_averaged = self.get_velocity_mass_average(velocities)

        return vu.get_normal_mode_decomposition(
            velocities_mass_averaged,
            self.eigenvectors,
            use_numba=use_numba,
        )

    def get_cross_spectrum(
        self,
        velocities: npt.NDArray[np.float64],
        index_pair: tuple,
        time_step: float,
        bootstrapping_blocks: int = 1,
        bootstrapping_overlap: int = 0,
        cutoff_at_last_maximum: bool = True,
        window_function: str = "hann",
    ) -> tuple[npt.NDArray, npt.NDArray]:
        """
        PLACEHOLDE.

        Parameters
        ----------
        velocities : npt.NDArray[np.float64]
            Velocity time series.
        index_pair : tuple
            Indices of the two vibration between which the corss spectrum
            should be calculated. For instance (1, 4).
        time_step : float
            Time step in the velocity time series.
        bootstrapping_blocks : int, optional
            DESCRIPTION. The default is 1.
        bootstrapping_overlap : int, optional
            DESCRIPTION. The default is 0.

        Returns
        -------
        frequencies : np.array
            DESCRIPTION.
        cross_spectrum : np.array
            DESCRIPTION.
        """
        index_0, index_1 = index_pair

        velocities_proj = self.get_normal_mode_decomposition(velocities)

        frequencies, cross_spectrum = vu.get_cross_spectrum(
            velocities_proj[:, index_0],
            velocities_proj[:, index_1],
            time_step,
            bootstrapping_blocks=bootstrapping_blocks,
            bootstrapping_overlap=bootstrapping_overlap,
            cutoff_at_last_maximum=cutoff_at_last_maximum,
            window_function=window_function,
        )

        return frequencies, cross_spectrum

    def output_cross_spectrum(
        self,
        velocities: npt.NDArray[np.float64],
        time_step: float,
        use_mem: bool = False,
        bootstrapping_blocks: int = 1,
        bootstrapping_overlap: int = 0,
        model_order: int = 15,
        processes: int = 1,
        frequency_cutoff: float | None = None,
        dirname: Path = Path("cross_spectrum"),
    ) -> None:
        """TODO."""
        velocities_proj = self.get_normal_mode_decomposition(velocities)

        n_points = len(self.eigenvectors)

        if use_mem:
            frequencies = vu.get_cross_spectrum_mem(
                velocities_proj[:, 0],
                velocities_proj[:, 0],
                time_step,  # pyright: ignore
                model_order,
                n_freqs=len(velocities_proj),
            )[0]
        else:
            frequencies = vu.get_cross_spectrum(
                velocities_proj[:, 0],
                velocities_proj[:, 0],
                time_step,
                bootstrapping_blocks=bootstrapping_blocks,
                bootstrapping_overlap=bootstrapping_overlap,
            )[0]

        if not dirname.is_dir():
            dirname.mkdir()

        cutoff = -1
        if frequency_cutoff is not None:
            f_inv_cm = frequencies * units.INVERSE_CM_IN_HZ
            L = f_inv_cm < frequency_cutoff  # pyright: ignore
            cutoff = np.sum(L)

        np.savetxt(dirname / "frequencies.csv", frequencies[:cutoff])

        index = []
        for index_0 in range(n_points):
            for index_1 in range(n_points):
                if index_0 < index_1:
                    continue

                index.append((index_0, index_1))

        func = functools.partial(
            _output_cross_spectrum,
            velocities_proj=velocities_proj,
            time_step=time_step,
            use_mem=use_mem,
            bootstrapping_blocks=bootstrapping_blocks,
            bootstrapping_overlap=bootstrapping_overlap,
            model_order=model_order,
            cutoff=cutoff,  # pyright: ignore
            dirname=dirname,
        )

        with mp.Pool(processes) as pool:
            pool.map(func, index)


def _output_cross_spectrum(
    index: npt.NDArray,
    velocities_proj: npt.NDArray,
    time_step: float,
    use_mem: bool,
    bootstrapping_blocks: int,
    bootstrapping_overlap: int,
    model_order: int,
    cutoff: int,
    dirname: Path,
) -> None:
    index_0 = index[0]
    index_1 = index[1]

    if use_mem:
        cross_spectrum = vu.get_cross_spectrum_mem(
            velocities_proj[:, index_0],
            velocities_proj[:, index_1],
            time_step,  # pyright: ignore
            model_order,
            n_freqs=len(velocities_proj),
        )[1]
    else:
        cross_spectrum = vu.get_cross_spectrum(
            velocities_proj[:, index_0],
            velocities_proj[:, index_1],
            time_step,
            bootstrapping_blocks=bootstrapping_blocks,
            bootstrapping_overlap=bootstrapping_overlap,
        )[1]

    np.savetxt(
        dirname / f"cross_spectrum_{index_0}_{index_1}.csv",
        cross_spectrum[:cutoff],
    )


class AimsVibrations(Vibrations, AimsGeometry):  # pyright: ignore
    """TODO."""

    def __init__(self, filename: str | None = None):
        Vibrations.__init__(self)
        AimsGeometry.__init__(self, filename=filename)


class VaspVibrations(Vibrations, VaspGeometry):  # pyright: ignore
    """TODO."""

    def __init__(self, filename: str | None = None):
        Vibrations.__init__(self)
        VaspGeometry.__init__(self, filename=filename)
