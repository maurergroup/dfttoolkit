import copy
import warnings

import numpy as np
import numpy.typing as npt
import scipy.sparse as sp

from .base import Parser
from .geometry import AimsGeometry
from .utils.exceptions import ItemNotFoundError


class Output(Parser):
    """
    Parse output files from electronic structure calculations.

    If contributing a new parser, please subclass this class, add the new supported file
    type to _supported_files, call the super().__init__ method, include the new file
    type as a kwarg in the super().__init__ call. Optionally include the self.lines line
    in examples.

    ...

    Attributes
    ----------
    _supported_files : dict
        List of supported file types.
    """

    def __init__(self, **kwargs: str):
        # Parse file information and perform checks
        super().__init__(self._supported_files, **kwargs)

        # Check that the files are in the correct format
        match self._format:
            case "aims_out":
                self._check_binary(False)
            case "elsi_csc":
                self._check_binary(True)

    @property
    def _supported_files(self) -> dict[str, str]:
        # FHI-aims, ELSI, ...
        return {"aims_out": ".out", "elsi_csc": ".csc"}

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self._format}={self._path})"

    def __init_subclass__(cls, **kwargs: str):
        # Override the parent's __init_subclass__ without calling it
        pass


class AimsOutput(Output):
    """
    FHI-aims output file parser.

    ...

    Attributes
    ----------
    path: str
        path to the aims.out file
    lines: list[str]
        contents of the aims.out file

    Parameters
    ----------
    aims_out : str, default="aims.out"
        path to the FHI-aims output file

    Examples
    --------
    >>> ao = AimsOutput(aims_out="./aims.out")
    """

    def __init__(self, aims_out: str = "aims.out"):
        super().__init__(aims_out=aims_out)

    def get_number_of_atoms(self) -> int:
        """
        Return number of atoms in unit cell.

        Returns
        -------
        int
            Number of atoms in the unit cell
        """
        n_atoms = None

        for line in self.lines:
            if "| Number of atoms" in line:
                n_atoms = int(line.strip().split()[5])

        if n_atoms is None:
            raise ValueError("Number of atoms not found in aims.out file")

        return n_atoms

    def get_geometry(self) -> AimsGeometry:
        """
        Extract the geometry file from the aims output.

        Returns
        -------
        AimsGeometry
            Geometry object
        """
        geometry_lines = []
        read_trigger = False
        for line in self.lines:
            if (
                "Parsing geometry.in "
                "(first pass over file, find array dimensions only)." in line
            ):
                read_trigger = True

            if read_trigger:
                geometry_lines.append(line)

            if "Completed first pass over input file geometry.in ." in line:
                break

        geometry_text = "\n".join(geometry_lines[6:-3])

        geometry = AimsGeometry()
        geometry.parse(geometry_text)

        return geometry

    def get_geometry_steps_of_optimisation(
        self, n_occurrence: int | None = None
    ) -> AimsGeometry | list[AimsGeometry]:
        """
        Get a list of all geometry steps performed.

        Parameters
        ----------
        n_occurrence : int or None
            If there are multiple energies in a file (e.g. during a geometry
            optimization) this parameters allows to select which energy is
            returned. If set to -1 the last one is returned (e.g. result of a
            geometry optimization), if set to None, all values will be returned
            as a numpy array.

        Returns
        -------
        geometry_files : list
            List of geometry objects.
        """
        geometry_files = [self.get_geometry()]  # append initial geometry
        geometry_lines = []

        state = 0
        # 0... before geometry file,
        # 1... between start of geometry file and lattice section
        # 2... in lattice section of geometry file
        # 3... in atoms section of geometry file

        for line in self.lines:
            if (
                "Updated atomic structure:" in line
                or "Atomic structure that was used in the preceding time step of the "
                "wrapper"
                in line
            ):
                state = 1
                geometry_lines = []

            if state > 0 and "atom " in line:
                state = 3
            if state == 1 and "lattice_vector  " in line:
                state = 2

            if state > 0:
                geometry_lines.append(line)

            if state == 3 and "atom " not in line:
                state = 0
                geometry_text = "".join(geometry_lines[2:-1])
                g = AimsGeometry()
                g.parse(geometry_text)
                geometry_files.append(g)

        if n_occurrence is not None:
            geometry_files = geometry_files[n_occurrence]

        return geometry_files

    def get_control_file(self) -> list[str]:
        """
        Extract the control file from the aims output.

        Returns
        -------
        list[str]
            Lines from the control file found in the aims output
        """
        control_lines = []
        control_file_reached = False
        for line in self.lines:
            if (
                "Parsing control.in (first pass over file, find array dimensions only)."
                in line
            ):
                control_file_reached = True

            if control_file_reached:
                control_lines.append(line.strip())

            if "Completed first pass over input file control.in ." in line:
                break

        return control_lines[6:-3]

    def get_parameters(self) -> dict:
        """
        Parse the parameters of the FHI-aims control file from the aims output.

        Returns
        -------
        dict
            The parameters of the FHI-aims control file found in the aims output
        """
        # Find where the parameters start
        start_params_line = 0
        for i, line in enumerate(self.lines):
            if (
                "Parsing control.in (first pass over file, find array dimensions only)."
                in line
            ):
                start_params_line = i
                break

        parameters = {}

        # If the file was written with ASE, there is an extra header/keyword delimiter
        extra_lines = (
            11 if self.lines[start_params_line + 8].split()[-1] == "(ASE)" else 6
        )

        for line in self.lines[start_params_line + extra_lines :]:
            # End of parameters and start of basis sets
            if line.strip() == "#" * 80 or line.strip() == "#" + ("=" * 79):
                break

            spl = line.split()
            if len(spl) > 1:
                parameters[spl[0]] = " ".join(spl[1:])

        return parameters

    def get_basis_sets(self) -> dict[str, str]: ...

    def check_exit_normal(self) -> bool:
        """
        Check if the FHI-aims calculation exited normally.

        Returns
        -------
        bool
            whether the calculation exited normally or not
        """
        exit_normal = False

        if len(self.lines) > 8:
            for i in range(1, 10):  # only read last few lines
                if self.lines[-i].strip() == "Have a nice day.":
                    exit_normal = True
                    break

        return exit_normal

    def get_time_per_scf(self) -> npt.NDArray[np.float64]:
        """
        Calculate the average time taken per SCF iteration.

        Returns
        -------
        NDArray[float64]
            The average time taken per SCF iteration.
        """
        # Get the number of SCF iterations
        n_scf_iters = self.get_n_scf_iters()
        scf_iter_times = np.zeros(n_scf_iters)

        # Get the time taken for each SCF iteration
        iter_num = 0
        for line in self.lines:
            if "Time for this iteration" in line:
                scf_iter_times[iter_num] = float(line.split()[-4])
                iter_num += 1

        return scf_iter_times

    def check_geometry_optimisation_has_completed(self) -> bool:
        """Check whether present geometry is converged."""
        geometry_optimisation_has_completed = False
        for line in reversed(self.lines):
            if "Present geometry is converged." in line:
                geometry_optimisation_has_completed = True

        return geometry_optimisation_has_completed

    ###############################################################################
    #                                   Energies                                  #
    ###############################################################################
    def _get_energy(  # noqa: PLR0912
        self,
        n_occurrence: int | None,
        search_string: str,
        token_nr: int | None = None,
        energy_invalid_indicator: list | int | str | None = None,
        energy_valid_indicator: list | int | str | None = None,
    ) -> float | npt.NDArray[np.float64]:
        """
        Generalized energy parser.

        Parameters
        ----------
        n_occurrence : int | None
            If there are multiple energies in a file (e.g. during a geometry
            optimization) this parameters allows to select which energy is returned.
            If set to -1 the last one is returned (e.g. result of a geometry
            optimization), if set to None, all values will be returned as a numpy array.
        search_string : str
            string to be searched in the output file
        token_nr : int | None
            take n-th element of found line
        energy_invalid_indicator : list | int | str | None = None
            In some cases an energy value can be found in the output file although it is
            invalid -> ignore this value. For example, a line having
            'restarting mixer to attempt better convergence' indicates that this
            SCF-cycle leads to invalid energies.
        param energy_valid_indicator : list | int | str | None = None
            In some cases the value is only valid after a certain phrase is used ->
            ignore all values before. For example, the post-SCF vdW energy correction is
            0.00 until the SCF is converged.

        Returns
        -------
        energies : float | NDArray[float64]
            Energies that have been grepped
        """
        skip_next_energy = False  # only relevant if energy_invalid_indicator != None
        use_next_energy = False  # only relevant if energy_valid_indicator != None

        if skip_next_energy and use_next_energy:
            raise ValueError(
                "AIMSOutput._get_energy: usage of skip_next_energy and "
                "use_next_energy at the same function call is undefined!"
            )

        # energy (in)valid indicator allows now for multiple values, if a list is
        # provided. Otherwise, everything works out as before.
        if energy_valid_indicator is not None and not isinstance(
            energy_valid_indicator, list
        ):
            energy_valid_indicator = [energy_valid_indicator]

        if energy_invalid_indicator is not None and not isinstance(
            energy_invalid_indicator, list
        ):
            energy_invalid_indicator = [energy_invalid_indicator]

        energies = []

        for line_text in self.lines:
            # check for energy_invalid_indicator:
            if energy_invalid_indicator is not None:
                for ind in energy_invalid_indicator:
                    if ind in line_text:
                        skip_next_energy = True

            if energy_valid_indicator is not None:
                for ind in energy_valid_indicator:
                    if ind in line_text:
                        use_next_energy = True
            else:
                use_next_energy = True

            if search_string in line_text:
                if skip_next_energy is True:
                    skip_next_energy = False  # reset this 'counter'
                elif use_next_energy:
                    if token_nr is None:
                        token_nr = len(search_string.split()) + 3
                    energies.append(float(line_text.strip().split()[token_nr]))
                    use_next_energy = False
                else:
                    pass

        if len(energies) == 0:
            msg = f"Energy not found in aims.out file for {search_string}"
            raise ValueError(msg)

        energies = np.array(energies)

        if n_occurrence is None:
            return energies
        return energies[n_occurrence]

    def get_change_of_total_energy(
        self,
        n_occurrence: int | None = -1,
        energy_invalid_indicator: str | None = None,
    ) -> float | npt.NDArray[np.float64]:
        return self._get_energy(
            n_occurrence,
            "Change of total energy",
            token_nr=6,
            energy_invalid_indicator=energy_invalid_indicator,
        )

    def get_change_of_forces(
        self,
        n_occurrence: int | None = -1,
        energy_invalid_indicator: str | None = None,
    ) -> float | npt.NDArray[np.float64]:
        return self._get_energy(
            n_occurrence,
            "Change of forces",
            token_nr=5,
            energy_invalid_indicator=energy_invalid_indicator,
        )

    def get_change_of_sum_of_eigenvalues(
        self,
        n_occurrence: int | None = -1,
        energy_invalid_indicator: str | None = None,
    ) -> float | npt.NDArray[np.float64]:
        return self._get_energy(
            n_occurrence,
            "Change of sum of eigenvalues",
            token_nr=7,
            energy_invalid_indicator=energy_invalid_indicator,
        )

    def get_change_of_charge_density(
        self,
        n_occurrence: int | None = -1,
        energy_invalid_indicator: str | None = None,
    ) -> float | npt.NDArray[np.float64]:
        return self._get_energy(
            n_occurrence,
            "Change of charge density",
            token_nr=6,
            energy_invalid_indicator=energy_invalid_indicator,
        )

    def get_maximum_force(
        self,
        n_occurrence: int | None = -1,
        energy_invalid_indicator: str | None = None,
    ) -> float | npt.NDArray[np.float64]:
        return self._get_energy(
            n_occurrence,
            "Maximum force component",
            token_nr=4,
            energy_invalid_indicator=energy_invalid_indicator,
        )

    def get_energy_corrected(
        self,
        n_occurrence: int | None = -1,
        skip_E_after_mixer: bool = True,
        all_scfs: bool = False,
        energy_invalid_indicator: list[str] | None = None,
    ) -> float | npt.NDArray[np.float64]:
        """
        Return the total corrected energy.

        Parameters
        ----------
        n_occurrence : int | None
            If there are multiple energies in a file (e.g. during a geometry opt)
            this parameters allows to select which energy is returned.
            If set to -1 the last one is returned (e.g. result of a geometry opt)
            if set to None, all values will be returned as a numpy array.

        skip_E_after_mixer : bool, default=True
            If the scf cycles of one geometry optimisation step didn't converge,
            aims will restart the mixer and this optimisation step.
            However, it still prints out the total energy, which can be totally
            nonsense. if skip_E_after_mixer==True ignore first total energy after
            'restarting mixer to attempt better convergence'

        Examples
        --------
        >>> AimsOutput.get_energy_corrected()
        -2080.83225450528
        """
        if energy_invalid_indicator is None:
            energy_invalid_indicator = []
        if skip_E_after_mixer:
            energy_invalid_indicator += [
                "restarting mixer to attempt better convergence"
            ]

        if all_scfs:
            return self.get_total_energy_T0(n_occurrence, skip_E_after_mixer)
        return self._get_energy(
            n_occurrence,
            search_string="| Total energy corrected",
            energy_invalid_indicator=energy_invalid_indicator,
            token_nr=5,
        )

    def get_total_energy_T0(  # noqa: N802
        self,
        n_occurrence: None | int = -1,
        skip_E_after_mixer: bool = True,
        energy_invalid_indicator: list[str] | None = None,
    ) -> float | npt.NDArray[np.float64]:
        if energy_invalid_indicator is None:
            energy_invalid_indicator = []
        if skip_E_after_mixer:
            energy_invalid_indicator += [
                "restarting mixer to attempt better convergence"
            ]

        return self._get_energy(
            n_occurrence,
            search_string="| Total energy, T -> 0",
            energy_invalid_indicator=energy_invalid_indicator,
            token_nr=9,
        )

    def get_energy_uncorrected(
        self,
        n_occurrence: None | int = -1,
        skip_E_after_mixer: bool = True,
        energy_invalid_indicator: list[str] | None = None,
    ) -> float | npt.NDArray[np.float64]:
        """
        Return uncorrected (without smearing correction) energy.

        Parameters
        ----------
        n_occurrence : Union[int, None]
            see getEnergyCorrected()

        skip_E_after_mixer : bool
            If the scf cycles of one geometry optimisation step didn't converge,
            aims will restart the mixer and this optimisation step.
            However, it still prints out the total energy, which can be totally
            nonsense. if skip_E_after_mixer==True: ignore first total energy after
            'restarting mixer to attempt better convergence'.

        Returns
        -------
        float | npt.NDArray[np.float64]
            Uncorrected energy
        """
        if energy_invalid_indicator is None:
            energy_invalid_indicator = []
        if skip_E_after_mixer:
            energy_invalid_indicator += [
                "restarting mixer to attempt better convergence"
            ]

        return self._get_energy(
            n_occurrence,
            search_string="| Total energy uncorrected",
            energy_invalid_indicator=energy_invalid_indicator,
            token_nr=5,
        )

    def get_energy_without_vdw(
        self,
        n_occurrence: None | int = -1,
        energy_invalid_indicator: list[str] | None = None,
    ) -> float | npt.NDArray[np.float64]:
        if energy_invalid_indicator is None:
            energy_invalid_indicator = []
        energy = self.get_energy_corrected(
            n_occurrence=n_occurrence,
            energy_invalid_indicator=energy_invalid_indicator,
        )

        energy_vdw = self.get_vdw_energy(
            n_occurrence=n_occurrence,
            energy_invalid_indicator=energy_invalid_indicator,
        )

        return energy - energy_vdw

    def get_HOMO_energy(  # noqa: N802
        self,
        n_occurrence: None | int = -1,
        energy_invalid_indicator: list[str] | None = None,
    ) -> float | npt.NDArray[np.float64]:
        return self._get_energy(
            n_occurrence,
            "Highest occupied state",
            token_nr=5,
            energy_invalid_indicator=energy_invalid_indicator,
        )

    def get_LUMO_energy(  # noqa: N802
        self,
        n_occurrence: None | int = -1,
        energy_invalid_indicator: list[str] | None = None,
    ) -> float | npt.NDArray[np.float64]:
        if energy_invalid_indicator is None:
            energy_invalid_indicator = []
        return self._get_energy(
            n_occurrence,
            "Lowest unoccupied state",
            token_nr=5,
            energy_invalid_indicator=energy_invalid_indicator,
        )

    def get_vdw_energy(
        self,
        n_occurrence: None | int = -1,
        energy_invalid_indicator: list[str] | None = None,
    ) -> float | npt.NDArray[np.float64]:
        if energy_invalid_indicator is None:
            energy_invalid_indicator = []
        search_keyword = "| vdW energy correction"
        token_nr = None

        return self._get_energy(
            n_occurrence,
            search_keyword,
            token_nr=token_nr,
            energy_invalid_indicator=energy_invalid_indicator,
            energy_valid_indicator="Self-consistency cycle converged",
        )

    def get_exchange_correlation_energy(
        self,
        n_occurrence: None | int = -1,
        energy_invalid_indicator: list[str] | None = None,
    ) -> float | npt.NDArray[np.float64]:
        if energy_invalid_indicator is None:
            energy_invalid_indicator = []
        return self._get_energy(
            n_occurrence,
            "| XC energy correction",
            energy_invalid_indicator=energy_invalid_indicator,
        )

    def get_electrostatic_energy(
        self,
        n_occurrence: None | int = -1,
        energy_invalid_indicator: list[str] | None = None,
    ) -> float | npt.NDArray[np.float64]:
        if energy_invalid_indicator is None:
            energy_invalid_indicator = []
        return self._get_energy(
            n_occurrence,
            "| Electrostatic energy ",
            energy_invalid_indicator=energy_invalid_indicator,
        )

    def get_kinetic_energy(
        self,
        n_occurrence: None | int = -1,
        energy_invalid_indicator: list[str] | None = None,
    ) -> float | npt.NDArray[np.float64]:
        if energy_invalid_indicator is None:
            energy_invalid_indicator = []
        return self._get_energy(
            n_occurrence,
            "| Kinetic energy ",
            energy_invalid_indicator=energy_invalid_indicator,
        )

    def get_sum_of_eigenvalues(
        self,
        n_occurrence: None | int = -1,
        energy_invalid_indicator: list[str] | None = None,
    ) -> float | npt.NDArray[np.float64]:
        if energy_invalid_indicator is None:
            energy_invalid_indicator = []
        return self._get_energy(
            n_occurrence,
            "| Sum of eigenvalues  ",
            energy_invalid_indicator=energy_invalid_indicator,
        )

    def get_cx_potential_correction(
        self,
        n_occurrence: None | int = -1,
        energy_invalid_indicator: list[str] | None = None,
    ) -> float | npt.NDArray[np.float64]:
        if energy_invalid_indicator is None:
            energy_invalid_indicator = []
        return self._get_energy(
            n_occurrence,
            "| XC potential correction",
            energy_invalid_indicator=energy_invalid_indicator,
        )

    def get_free_atom_electrostatic_energy(
        self,
        n_occurrence: None | int = -1,
        energy_invalid_indicator: list[str] | None = None,
    ) -> float | npt.NDArray[np.float64]:
        if energy_invalid_indicator is None:
            energy_invalid_indicator = []
        return self._get_energy(
            n_occurrence,
            "| Free-atom electrostatic energy:",
            token_nr=6,
            energy_invalid_indicator=energy_invalid_indicator,
        )

    def get_entropy_correction(
        self,
        n_occurrence: None | int = -1,
        energy_invalid_indicator: list[str] | None = None,
    ) -> float | npt.NDArray[np.float64]:
        if energy_invalid_indicator is None:
            energy_invalid_indicator = []
        return self._get_energy(
            n_occurrence,
            "| Entropy correction ",
            energy_invalid_indicator=energy_invalid_indicator,
        )

    def get_hartree_energy_correction(
        self,
        n_occurrence: None | int = -1,
        energy_invalid_indicator: list[str] | None = None,
    ) -> float | npt.NDArray[np.float64]:
        if energy_invalid_indicator is None:
            energy_invalid_indicator = []
        return self._get_energy(
            n_occurrence,
            "| Hartree energy correction",
            energy_invalid_indicator=energy_invalid_indicator,
        )

    def get_ionic_embedding_energy(
        self,
        n_occurrence: None | int = -1,
        energy_invalid_indicator: list[str] | None = None,
    ) -> float | npt.NDArray[np.float64]:
        """Get the energy of the nuclei in the potential of the electric field."""
        if energy_invalid_indicator is None:
            energy_invalid_indicator = []
        return self._get_energy(
            n_occurrence,
            "| Ionic    embedding energy",
            energy_invalid_indicator=energy_invalid_indicator,
        )

    def get_density_embedding_energy(
        self,
        n_occurrence: None | int = -1,
        energy_invalid_indicator: list[str] | None = None,
    ) -> float | npt.NDArray[np.float64]:
        """
        Get the energy of the electrons in the potential of the electric field.

        Electrons given as electron density
        """
        if energy_invalid_indicator is None:
            energy_invalid_indicator = []
        return self._get_energy(
            n_occurrence,
            "| Density  embedding energy",
            energy_invalid_indicator=energy_invalid_indicator,
        )

    def get_nonlocal_embedding_energy(
        self,
        n_occurrence: None | int = -1,
        energy_invalid_indicator: list[str] | None = None,
    ) -> float | npt.NDArray[np.float64]:
        """Non-local electron interaction energy in the electric field potential."""
        if energy_invalid_indicator is None:
            energy_invalid_indicator = []
        return self._get_energy(
            n_occurrence,
            "| Nonlocal embedding energy",
            energy_invalid_indicator=energy_invalid_indicator,
        )

    def get_external_embedding_energy(
        self,
        n_occurrence: None | int = -1,
        energy_invalid_indicator: list[str] | None = None,
    ) -> float | npt.NDArray[np.float64]:
        """
        Calculate the sum of all the embedding energies.

        Ionic + (electronic) density + nonlocal.
        """
        if energy_invalid_indicator is None:
            energy_invalid_indicator = []
        return self._get_energy(
            n_occurrence,
            "| External embedding energy",
            energy_invalid_indicator=energy_invalid_indicator,
        )

    def get_forces(self, n_occurrence: None | int = -1) -> npt.NDArray[np.float64]:
        """Return forces on all atoms."""
        natoms = self.get_number_of_atoms()
        all_force_values = []

        for j, line in enumerate(self.lines):
            if "Total atomic forces" in line:
                force_values = np.ones([natoms, 3]) * np.nan
                for i in range(natoms):
                    force_values[i, :] = [
                        float(x) for x in self.lines[j + i + 1].split()[2:5]
                    ]
                all_force_values.append(np.array(force_values))

        if len(all_force_values) == 0:
            msg = f"Forces not found in {self.path} file"
            raise ValueError(msg)

        if n_occurrence is None:
            return np.array(all_force_values)
        return all_force_values[n_occurrence]

    def get_vdw_forces(self, nr_of_occurrence: int = -1) -> dict:  # noqa: ARG002
        components_of_gradients = self.get_force_components()

        return components_of_gradients["van_der_waals"]
        # TODO: implement occurences
        # if nr_of_occurrence is None:
        #     return components_of_gradients["van_der_waals"]
        # else:
        #     return components_of_gradients["van_der_waals"][nr_of_occurrence]

    def get_forces_without_vdw(self, nr_of_occurrence: int = -1) -> npt.NDArray:  # noqa: ARG002
        """
        Return the uncleaned forces with the vdW component.

        Note that for the final output, which you get when envoding `self.get_forces()`
        you get the cleaned forces. Look up "final_forces_cleaned" in the AIMS manual
        for more info.

        Parameters
        ----------
        nr_of_occurrence : int, default=-1
            Currently not used. The default is -1.

        Returns
        -------
        gradients_without_vdW : npt.NDArray
        """
        components_of_gradients = self.get_force_components()

        return (
            components_of_gradients["total"] - components_of_gradients["van_der_waals"]
        )

        # TODO: implement occurences
        # if nr_of_occurrence is None:
        #     return gradients_without_vdW
        # else:
        #     return gradients_without_vdW[nr_of_occurrence]

    def get_force_components(self, nr_of_occurrence: int = -1) -> dict:  # noqa: ARG002
        """
        Return the force component specified in "component" for all atoms.

        These are the Hellmann-Feynman, Ionic, Multipole, Pulay + GGA, Van der Waals,
        and total forces

        Parameters
        ----------
        nr_of_occurrence : int, default=-1
            Currently not used. The default is -1.

        Returns
        -------
        forces : dict
            Dictionary with the force components as keys and the forces as values.
            The forces are in the form of a numpy array with shape (n_atoms, 3).
        """
        number_of_atoms = self.get_number_of_atoms()

        force_key_list = [
            "Hellmann-Feynman              :",
            "Ionic forces                  :",
            "Multipole                     :",
            "Pulay + GGA                   :",
            "Van der Waals                 :",
            "Total forces",
        ]
        force_key_list_2 = [
            "hellmann_feynman",
            "ionic",
            "multipole",
            "pulay_gga",
            "van_der_waals",
            "total",
        ]
        force_values = {}

        for ind_0, force_key in enumerate(force_key_list):
            force_key_2 = force_key_list_2[ind_0]
            force_values[force_key_2] = np.ones([number_of_atoms, 3]) * np.nan

            for ind_1 in range(3):
                force = self._get_energy(
                    None,
                    force_key,
                    token_nr=ind_1 - 3,
                    energy_invalid_indicator=None,
                )
                force_values[force_key_2][:, ind_1] = force[-number_of_atoms:]  # pyright: ignore[reportIndexIssue]

            centre_fo_mass_force = np.mean(force_values[force_key_2], axis=0)
            centre_fo_mass_force = np.tile(centre_fo_mass_force, (number_of_atoms, 1))

            force_values[force_key_2] -= centre_fo_mass_force

        return force_values

    def check_spin_polarised(self) -> bool:
        """
        Check if the FHI-aims calculation was spin polarised.

        Returns
        -------
        bool
            Whether the calculation was spin polarised or not
        """
        spin_polarised = False

        for line in self.lines:
            spl = line.split()
            if len(spl) == 2:
                # Don't break the loop if spin polarised calculation is found as if the
                # keyword is specified again, it is the last one that is used
                if spl[0] == "spin" and spl[1] == "collinear":
                    spin_polarised = True

                if spl[0] == "spin" and spl[1] == "none":
                    spin_polarised = False

        return spin_polarised

    def get_convergence_parameters(self) -> dict[str, float]:
        """
        Get the convergence parameters from the aims.out file.

        Returns
        -------
        dict[str, float]
            The convergence parameters from the aims.out file
        """
        # Setup dictionary to store convergence parameters
        self.convergence_params = {
            "charge density": 0.0,
            "sum eigenvalues": 0.0,
            "total energy": 0.0,
            "change of max force": 0.0,
            "max force per atom": 0.0,
        }

        for line in self.lines:
            spl = line.split()
            if len(spl) > 1:
                if "Convergence accuracy of self-consistent charge density:" in line:
                    self.convergence_params["charge density"] = float(spl[-1])
                if (
                    "* Based on n_atoms and forces and force-correction status, "
                    "FHI-aims chose sc_accuracy_rho =" in line
                ):
                    self.convergence_params["charge density"] = float(spl[-2])
                if "Convergence accuracy of sum of eigenvalues:" in line:
                    self.convergence_params["sum eigenvalues"] = float(spl[-1])
                if "Convergence accuracy of total energy:" in line:
                    self.convergence_params["total energy"] = float(spl[-1])
                if "Convergence accuracy of forces:" in line:
                    self.convergence_params["change of max force"] = float(spl[-1])
                if (
                    "Convergence accuracy for geometry relaxation: Maximum force <"
                    in line
                ):
                    self.convergence_params["max force per atom"] = float(spl[-2])

                # No more values to get after SCF starts
                if "Begin self-consistency loop" in line:
                    break

        return self.convergence_params

    def get_final_energy(self) -> float | None:
        """
        Get the final energy from a FHI-aims calculation.

        Returns
        -------
        float | None
            The final energy of the calculation
        """
        for line in self.lines:
            if "s.c.f. calculation      :" in line:
                return float(line.split()[-2])

        return None

    def get_final_spin_moment(self) -> tuple | None:
        """
        Get the final spin moment from a FHI-aims calculation.

        Returns
        -------
        tuple | None
            The final spin moment of the calculation, if it exists
        """
        n, s, j = None, None, None

        # Iterate through the lines in reverse order to find the final spin moment
        for i, line in enumerate(reversed(self.lines)):
            if "Current spin moment of the entire structure :" in line:
                if len(self.lines[-1 - i + 3].split()) > 0:
                    n = float(self.lines[-1 - i + 1].split()[-1])
                    s = float(self.lines[-1 - i + 2].split()[-1])
                    j = float(self.lines[-1 - i + 3].split()[-1])

                    return n, s, j

                # Non-gamma point periodic calculation
                n = float(self.lines[-1 - i + 1].split()[-1])
                s = float(self.lines[-1 - i + 2].split()[-1])

                return n, s

        if n is None or s is None:
            return None

        return None

    def get_n_relaxation_steps(self) -> int:
        """
        Get the number of relaxation steps from the aims.out file.

        Returns
        -------
        int
            the number of relaxation steps
        """
        n_relax_steps = 0
        for line in reversed(self.lines):
            if "Number of relaxation steps" in line:
                return int(line.split()[-1])

            # If the calculation did not finish normally, the number of relaxation steps
            # will not be printed. In this case, count each relaxation step as they were
            # calculated by checking when the SCF cycle converged.
            if line.strip() == "Self-consistency cycle converged.":
                n_relax_steps += 1

        return n_relax_steps

    def get_n_scf_iters(self) -> int:
        """
        Get the number of SCF iterations from the aims.out file.

        Returns
        -------
        int
            The number of scf iterations
        """
        n_scf_iters = 0
        for line in reversed(self.lines):
            if "Number of self-consistency cycles" in line:
                return int(line.split()[-1])

            # If the calculation did not finish normally, the number of SCF iterations
            # will not be printed. In this case, count each SCF iteration as they were
            # calculated
            if "Begin self-consistency iteration #" in line:
                n_scf_iters += 1

        return n_scf_iters

    def get_scf_convergence(self) -> dict[str, npt.NDArray[np.float64]]:
        """
        Get the convergence of various parameters from the SCF cycle.

        1. SCF Iterations
        2. Change of charge
        3. Change of charge/spin density
        4. Change of sum of eigenvalues
        5. Change of total energy
        6. Change of forces (currently not fully implemented)
        7. Forces on atoms

        Returns
        -------
        dict[str, NDArray[float64]]
            convergence data
        """
        # Read the total number of SCF iterations
        n_scf_iters = self.get_n_scf_iters()
        n_relax_steps = self.get_n_relaxation_steps() + 1

        self.scf_convergence = {
            "SCF iterations": np.zeros(n_scf_iters),
            "change of charge": np.zeros(n_scf_iters),
            "change of charge spin density": np.zeros(n_scf_iters),
            "change of sum eigenvalues": np.zeros(n_scf_iters),
            "change of total energy": np.zeros(n_scf_iters),
            "change of max force": np.zeros(n_relax_steps),
            "forces on atoms": np.zeros(n_relax_steps),
        }

        current_scf_iter = 0
        current_relax_step = 0
        new_scf_iter = True

        for line in self.lines:
            spl = line.split()
            if len(spl) > 1:
                if "Begin self-consistency iteration #" in line:
                    # save the scf iteration number
                    self.scf_convergence["SCF iterations"][current_scf_iter] = int(
                        spl[-1]
                    )
                    # use a counter rather than reading the SCF iteration number as it
                    # resets upon re-initialisation and for each geometry opt step
                    current_scf_iter += 1

                # Use spin density if spin polarised calculation
                if "Change of charge/spin density" in line:
                    self.scf_convergence["change of charge"][current_scf_iter - 1] = (
                        float(spl[-2])
                    )
                    self.scf_convergence["change of charge spin density"][
                        current_scf_iter - 1
                    ] = float(spl[-1])

                # Otherwise just use change of charge
                elif "Change of charge" in line:
                    self.scf_convergence["change of charge"][current_scf_iter - 1] = (
                        float(spl[-1])
                    )

                if "Change of sum of eigenvalues" in line:
                    self.scf_convergence["change of sum eigenvalues"][
                        current_scf_iter - 1
                    ] = float(spl[-2])

                if "Change of total energy" in line:
                    self.scf_convergence["change of total energy"][
                        current_scf_iter - 1
                    ] = float(spl[-2])

                if "Change of forces" in line:
                    # Only save the smallest change of forces for each geometry
                    # relaxation step. I have no idea why it prints multiple times but
                    # I assume it's a data race of some sort
                    if new_scf_iter:
                        self.scf_convergence["change of max force"][
                            current_relax_step - 1
                        ] = float(spl[-2])

                        new_scf_iter = False

                    elif (
                        float(spl[-2]) < self.scf_convergence["change of max force"][-1]
                    ):
                        self.scf_convergence["change of max force"][
                            current_relax_step - 1
                        ] = float(spl[-2])

                if "Forces on atoms" in line:
                    self.scf_convergence["forces on atoms"][current_relax_step - 1] = (
                        float(spl[-2])
                    )

                if line.strip() == "Self-consistency cycle converged.":
                    new_scf_iter = True
                    current_relax_step += 1

        return self.scf_convergence

    def get_n_initial_ks_states(self, include_spin_polarised: bool = True) -> int:
        """
        Get the number of Kohn-Sham states from the first SCF step.

        Parameters
        ----------
        include_spin_polarised : bool, default=True
            Whether to include the spin-down states in the count if the calculation is
            spin polarised.

        Returns
        -------
        int
            The number of kohn-sham states

        Raises
        ------
        ValueError
            No KS states found in aims.out file
        """
        target_line = "State    Occupation    Eigenvalue [Ha]    Eigenvalue [eV]"

        init_ev_start = 0
        n_ks_states = 0

        # Find the first time the KS states are printed
        init_ev_start = None
        for i, line in enumerate(self.lines):
            if target_line == line.strip():
                init_ev_start = i + 1
                break

        if init_ev_start is None:
            raise ValueError("No KS states found in aims.out file.")

        # Then count the number of lines until the next empty line
        for i, line in enumerate(self.lines[init_ev_start:]):
            if len(line.strip()) == 0:
                n_ks_states = i
                break

        if n_ks_states == 0:
            raise ValueError("No KS states found in aims.out file.")

        # Count the spin-down eigenvalues if the calculation is spin polarised
        if include_spin_polarised:
            init_ev_end = init_ev_start + n_ks_states
            if target_line == self.lines[init_ev_end + 3].strip():
                init_ev_end += 4
                for line in self.lines[init_ev_end:]:
                    if len(line) > 1:
                        n_ks_states += 1
                    else:
                        break

            else:  # If SD states are not found 4 lines below end of SU states
                warnings.warn(
                    "A spin polarised calculation was expected but not found.",
                    stacklevel=2,
                )

        return n_ks_states

    def _get_ks_states(
        self, ev_start: int, eigenvalues: dict, scf_iter: int, n_ks_states: int
    ) -> None:
        """
        Get any set of KS states, occupations, and eigenvalues.

        Parameters
        ----------
        ev_start : int
            The line number where the KS states start.
        eigenvalues : dict
            The dictionary to store the KS states, occupations, and eigenvalues.
        scf_iter : int
            The current SCF iteration.
        n_ks_states : int
            The number of KS states to save.
        """
        if (
            eigenvalues["state"].ndim == 1
            or eigenvalues["occupation"].ndim == 1
            or eigenvalues["eigenvalue_eV"].ndim == 1
        ):
            # This is the case for finding the final KS eigenvalues
            # Therefore only parse the KS states from the final SCF iteration
            for i, line in enumerate(self.lines[ev_start : ev_start + n_ks_states]):
                values = line.split()
                eigenvalues["state"][i] = int(values[0])
                eigenvalues["occupation"][i] = float(values[1])
                eigenvalues["eigenvalue_eV"][i] = float(values[3])

        else:
            for i, line in enumerate(self.lines[ev_start : ev_start + n_ks_states]):
                values = line.split()
                eigenvalues["state"][scf_iter][i] = int(values[0])
                eigenvalues["occupation"][scf_iter][i] = float(values[1])
                eigenvalues["eigenvalue_eV"][scf_iter][i] = float(values[3])

    def get_all_ks_eigenvalues(
        self,
    ) -> (
        dict[str, npt.NDArray[np.int64 | np.float64]]
        | tuple[
            dict[str, npt.NDArray[np.int64 | np.float64]],
            dict[str, npt.NDArray[np.int64 | np.float64]],
        ]
    ):
        """
        Get all Kohn-Sham eigenvalues from a calculation.

        Returns
        -------
        dict | tuple[dict, dict]
            dict
                the kohn-sham eigenvalues
            tuple[dict, dict]
                dict
                    the spin-up kohn-sham eigenvalues
                dict
                    the spin-down kohn-sham eigenvalues

        Raises
        ------
        ItemNotFoundError
            the 'output_level full' keyword was not found in the calculation
        ValueError
            could not determine if the calculation was spin polarised
        """
        # Check if the calculation was spin polarised
        spin_polarised = self.check_spin_polarised()

        # Check if output_level full was specified in the calculation
        required_item = ("output_level", "full")
        if required_item not in self.get_parameters().items():
            raise ItemNotFoundError(required_item)

        # Get the number of KS states and scf iterations
        # Add 2 to SCF iters as if output_level full is specified, FHI-aims prints the
        # KS states once before the SCF starts and once after it finishes
        n_scf_iters = self.get_n_scf_iters() + 2
        n_ks_states = self.get_n_initial_ks_states(include_spin_polarised=False)

        # Parse line to find the start of the KS eigenvalues
        target_line = "State    Occupation    Eigenvalue [Ha]    Eigenvalue [eV]"

        if not spin_polarised:
            eigenvalues = {
                "state": np.zeros((n_scf_iters, n_ks_states), dtype=int),
                "occupation": np.zeros((n_scf_iters, n_ks_states), dtype=float),
                "eigenvalue_eV": np.zeros((n_scf_iters, n_ks_states), dtype=float),
            }

            n = 0  # Count the current SCF iteration
            for i, line in enumerate(self.lines):
                if target_line in line:
                    # Get the KS states from this line until the next empty line
                    self._get_ks_states(i + 1, eigenvalues, n, n_ks_states)
                    n += 1

            return eigenvalues

        if spin_polarised:
            su_eigenvalues = {
                "state": np.zeros((n_scf_iters, n_ks_states), dtype=int),
                "occupation": np.zeros((n_scf_iters, n_ks_states), dtype=float),
                "eigenvalue_eV": np.zeros((n_scf_iters, n_ks_states), dtype=float),
            }
            sd_eigenvalues = {
                "state": np.zeros((n_scf_iters, n_ks_states), dtype=int),
                "occupation": np.zeros((n_scf_iters, n_ks_states), dtype=float),
                "eigenvalue_eV": np.zeros((n_scf_iters, n_ks_states), dtype=float),
            }

            # Count the number of SCF iterations for each spin channel
            up_n = 0
            down_n = 0
            for i, line in enumerate(self.lines):
                # Printing of KS states is weird in aims.out. Ensure that we don't add
                # more KS states than the array is long
                if up_n == n_scf_iters and down_n == n_scf_iters:
                    break

                if target_line in line:
                    # The spin-up line is two lines above the target line
                    if self.lines[i - 2].strip() == "Spin-up eigenvalues:":
                        # Get the KS states from this line until the next empty line
                        self._get_ks_states(i + 1, su_eigenvalues, up_n, n_ks_states)
                        up_n += 1

                    # The spin-down line is two lines above the target line
                    if self.lines[i - 2].strip() == "Spin-down eigenvalues:":
                        # Get the KS states from this line until the next empty line
                        self._get_ks_states(i + 1, sd_eigenvalues, down_n, n_ks_states)
                        down_n += 1

            return su_eigenvalues, sd_eigenvalues

        raise ValueError("Could not determine if calculation was spin polarised.")

    def get_final_ks_eigenvalues(
        self,
    ) -> (
        dict[str, npt.NDArray[np.int64 | np.float64]]
        | tuple[
            dict[str, npt.NDArray[np.int64 | np.float64]],
            dict[str, npt.NDArray[np.int64 | np.float64]],
        ]
    ):
        """
        Get the final Kohn-Sham eigenvalues from a calculation.

        Returns
        -------
        dict | tuple[dict, dict]]
            dict
                the final kohn-sham eigenvalues
            tuple[dict, dict]
                dict
                    the spin-up kohn-sham eigenvalues
                dict
                    the spin-down kohn-sham eigenvalues

        Raises
        ------
        ValueError
            the calculation was not spin polarised
        ValueError
            the final KS states were not found in aims.out file
        """
        # Check if the calculation was spin polarised
        spin_polarised = self.check_spin_polarised()

        # Get the number of KS states
        n_ks_states = self.get_n_initial_ks_states(include_spin_polarised=False)

        # Parse line to find the start of the KS eigenvalues
        target_line = "State    Occupation    Eigenvalue [Ha]    Eigenvalue [eV]"

        # Iterate backwards from end of aims.out to find the final KS eigenvalues
        final_ev_start = None
        for i, line in enumerate(reversed(self.lines)):
            if target_line == line.strip():
                final_ev_start = -i
                break

        if final_ev_start is None:
            raise ValueError("Final KS states not found in aims.out file.")

        if not spin_polarised:
            eigenvalues = {
                "state": np.zeros((n_ks_states), dtype=int),
                "occupation": np.zeros((n_ks_states), dtype=float),
                "eigenvalue_eV": np.zeros((n_ks_states), dtype=float),
            }
            # Get the KS states from this line until the next empty line
            self._get_ks_states(final_ev_start, eigenvalues, 0, n_ks_states)

            return eigenvalues

        if spin_polarised:
            su_eigenvalues = {
                "state": np.zeros((n_ks_states), dtype=int),
                "occupation": np.zeros((n_ks_states), dtype=float),
                "eigenvalue_eV": np.zeros((n_ks_states), dtype=float),
            }
            sd_eigenvalues = copy.deepcopy(su_eigenvalues)

            # The spin-down states start from here
            self._get_ks_states(final_ev_start, sd_eigenvalues, 0, n_ks_states)

            # Go back one more target line to get the spin-up states
            for i, line in enumerate(reversed(self.lines[: final_ev_start - 1])):
                if target_line == line.strip():
                    final_ev_start += -i - 1
                    break

            self._get_ks_states(final_ev_start, su_eigenvalues, 0, n_ks_states)

            return su_eigenvalues, sd_eigenvalues

        raise ValueError("Could not determine if calculation was spin polarised.")

    def get_pert_soc_ks_eigenvalues(self) -> dict:
        """
        Get the perturbative SOC Kohn-Sham eigenvalues from a calculation.

        Returns
        -------
        dict
            The perturbative SOC Kohn-Sham eigenvalues

        Raises
        ------
        ValueError
            the final KS states were not found in aims.out file
        """
        # Get the number of KS states
        n_ks_states = self.get_n_initial_ks_states()

        target_line = (
            "State    Occupation    Unperturbed Eigenvalue [eV]"
            "    Eigenvalue [eV]    Level Spacing [eV]"
        )

        # Iterate backwards from end of aims.out to find the perturbative SOC
        # eigenvalues
        final_ev_start = None
        for i, line in enumerate(reversed(self.lines)):
            if target_line == line.strip():
                final_ev_start = -i
                break

        if final_ev_start is None:
            raise ValueError("Final KS states not found in aims.out file.")

        eigenvalues = {
            "state": np.zeros(n_ks_states, dtype=int),
            "occupation": np.zeros(n_ks_states, dtype=float),
            "unperturbed_eigenvalue_eV": np.zeros(n_ks_states, dtype=float),
            "eigenvalue_eV": np.zeros(n_ks_states, dtype=float),
            "level_spacing_eV": np.zeros(n_ks_states, dtype=float),
        }

        for i, line in enumerate(
            self.lines[final_ev_start : final_ev_start + n_ks_states]
        ):
            spl = line.split()

            eigenvalues["state"][i] = int(spl[0])
            eigenvalues["occupation"][i] = float(spl[1])
            eigenvalues["unperturbed_eigenvalue_eV"][i] = float(spl[2])
            eigenvalues["eigenvalue_eV"][i] = float(spl[3])
            eigenvalues["level_spacing_eV"][i] = float(spl[4])

        return eigenvalues


class ELSIOutput(Output):
    """
    Parse matrix output written in a binary csc format from ELSI.

    ...

    Attributes
    ----------
    data : bytes
        The binary data from the ELSI csc file
    path : str
        Path to ELSI csc file.
    n_basis : int
        Number of basis functions
    n_non_zero : int
        Number of non-zero elements in the matrix
    """

    def __init__(self, elsi_csc: str = "elsi.csc"):
        super().__init__(elsi_csc=elsi_csc)

    def get_elsi_csc_header(self) -> npt.NDArray[np.int64]:
        """
        Get the contents of the ELSI file header.

        Returns
        -------
        tuple
            The contents of the ELSI csc file header
        """
        return np.frombuffer(self.data[0:128], dtype=np.int64)

    @property
    def n_basis(self) -> int:
        return self.get_elsi_csc_header()[3]

    @property
    def n_non_zero(self) -> int:
        return self.get_elsi_csc_header()[5]

    def read_elsi_as_csc(
        self, csc_format: bool = False
    ) -> sp.csc_array | npt.NDArray[np.float64]:
        """
        Get a CSC matrix from an ELSI output file.

        Parameters
        ----------
        csc_format : bool, default=True
            Whether to return the matrix in CSC format or a standard numpy array

        Returns
        -------
        csc_array | NDArray
            The CSC matrix or numpy array
        """
        header = self.get_elsi_csc_header()

        # Get the column pointer
        end = 128 + self.n_basis * 8
        col_i = np.frombuffer(self.data[128:end], dtype=np.int64)
        col_i = np.append(col_i, self.n_non_zero + 1)
        col_i -= 1

        # Get the row index
        start = end + self.n_non_zero * 4
        row_i = np.array(np.frombuffer(self.data[end:start], dtype=np.int32))
        row_i -= 1

        if header[2] == 0:  # real
            nnz = np.frombuffer(
                self.data[start : start + self.n_non_zero * 8],
                dtype=np.float64,
            )

        else:  # complex
            nnz = np.frombuffer(
                self.data[start : start + self.n_non_zero * 16],
                dtype=np.complex128,
            )

        if csc_format:
            return sp.csc_array((nnz, row_i, col_i), shape=(self.n_basis, self.n_basis))

        return sp.csc_array(
            (nnz, row_i, col_i), shape=(self.n_basis, self.n_basis)
        ).toarray()
