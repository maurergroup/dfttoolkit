from collections.abc import Generator

import numpy as np
import pytest
import scipy.sparse as sp

from dfttoolkit.output import AimsOutput, ELSIOutput
from dfttoolkit.utils.exceptions import ItemNotFoundError


class TestAimsOutput:
    """Tests for the AimsOutput class."""

    @property
    def _aims_fixture_no(self) -> int:
        return int(self.ao.path.split("/")[-2])

    @pytest.fixture(params=range(1, 14), autouse=True)
    def aims_out(self, cwd, request, aims_calc_dir) -> None:
        self.ao = AimsOutput(
            aims_out=f"{cwd}/fixtures/{aims_calc_dir}/{request.param}/aims.out"
        )

    @pytest.fixture
    def control_in(
        self, cwd, aims_calc_dir
    ) -> Generator[list[str], None, None]:
        """
        Get lines from a control.in fixture.

        Yields
        ------
        list[str]
            lines from the control file
        """
        with open(
            f"{cwd}/fixtures/{aims_calc_dir}/{self._aims_fixture_no}/control.in",
        ) as f:
            yield [line.strip() for line in f.readlines()]

    def test_get_number_of_atoms(self) -> None:
        if self._aims_fixture_no in [4, 6, 8, 10, 11, 12]:
            assert self.ao.get_number_of_atoms() == 2
        else:
            assert self.ao.get_number_of_atoms() == 3

    def test_get_geometry(self) -> None:
        geom = self.ao.get_geometry()

        if self._aims_fixture_no in [1, 2, 3, 5, 7, 9, 13]:
            assert len(geom) == 3
            assert geom.get_is_periodic() is False
        else:
            assert len(geom) == 2
            assert geom.get_is_periodic() is True

    def test_get_geometry_steps_of_optimisation(self) -> None:
        positions = np.array(
            [
                [0.00000004, 0.00000045, 2.95776161],
                [-0.00000010, -0.00000113, 1.03284091],
                [0.00000006, 0.00000068, -0.09060251],
            ]
        )

        if self._aims_fixture_no == 13:
            geom_steps = self.ao.get_geometry_steps_of_optimisation()
            assert isinstance(geom_steps, list)
            assert np.allclose(geom_steps[-1].coords, positions)

    def test_get_control_file(self, control_in) -> None:
        assert self.ao.get_control_file() == control_in

    def test_get_parameters(self, ref_data) -> None:
        assert (
            self.ao.get_parameters()
            == ref_data["control_params"][self._aims_fixture_no - 1]
        )

    def test_check_exit_normal(self) -> None:
        if self._aims_fixture_no in [7, 8]:
            assert self.ao.check_exit_normal() is False
        else:
            assert self.ao.check_exit_normal() is True

    def test_get_time_per_scf(self, ref_data) -> None:
        # Fail if the absolute tolerance between any values in test vs. reference array
        # is greater than 2e-3
        if self._aims_fixture_no in range(1, 13):
            assert np.allclose(
                self.ao.get_time_per_scf(),
                ref_data["timings"][self._aims_fixture_no - 1],
                atol=2e-3,
            )

    def test_get_change_of_total_energy_1(self) -> None:
        """Using default args (final energy change)."""
        final_energies = np.array(
            [
                1.599e-08,
                1.611e-09,
                1.611e-09,
                -1.492e-07,
                -5.833e-09,
                3.703e-09,
                1.509e-05,
                -0.0001144,
                6.018e-06,
                7.119e-06,
                1.96e-06,
                3.743e-09,
                -0.1131e-06,
            ]
        )

        assert (
            abs(
                self.ao.get_change_of_total_energy()
                - final_energies[self._aims_fixture_no - 1]
            )
            < 1e-8
        )

    def test_get_change_of_total_energy_2(self, ref_data) -> None:
        """Get every energy change."""
        # Fail if the absolute tolerance between any values in test vs. reference array
        # is greater than 1e-10
        if self._aims_fixture_no in range(1, 13):
            assert np.allclose(
                self.ao.get_change_of_total_energy(n_occurrence=None),
                ref_data["energy_diffs"][self._aims_fixture_no - 1],
                atol=1e-8,
            )

    def test_get_change_of_total_energy_3(self) -> None:
        """Get the 1st energy change."""
        first_energies = [
            1.408,
            -0.1508,
            -0.1508,
            0.871,
            1.277,
            0.1063,
            1.19,
            0.871,
            -5.561,
            -0.07087,
            -0.1222,
            -0.387,
            0.6123e01,
        ]

        assert (
            abs(
                self.ao.get_change_of_total_energy(n_occurrence=1)
                - first_energies[self._aims_fixture_no - 1]
            )
            < 1e-8
        )

    # TODO
    # def test_get_change_of_total_energy_4(self):
    #     """
    #     Use an energy invalid indicator
    #     """

    #     assert np.allclose(
    #         self.ao.get_change_of_total_energy(n_occurrence=1),
    #         ref_data['all_energies'][self.aims_fixture_no(self.ao) - 1],
    #         atol=1e-10,
    #     )

    # Not necessary to include every possible function argument in the next tests as all
    # of the following functions wrap around _get_energy(), which have all been tested
    # in the previous 4 tests

    def test_get_forces(self) -> None:
        forces = [
            np.array(
                [
                    [
                        -0.632469472942813e-11,
                        -0.900095529694541e-04,
                        -0.324518849313061e-27,
                    ],
                    [
                        -0.137684561607316e-03,
                        0.450047740234745e-04,
                        -0.486778273969592e-27,
                    ],
                    [
                        0.137684567932011e-03,
                        0.450047789459852e-04,
                        -0.324518849313061e-27,
                    ],
                ]
            )
        ]

        if self._aims_fixture_no in [5]:
            assert (
                np.all(
                    abs(
                        self.ao.get_forces()
                        - forces[self._aims_fixture_no - 5]
                    )
                )
                < 1e-8
            )

    def test_get_forces_without_vdw_1(self) -> None:
        forces = {
            13: np.array(
                [
                    [6.49623190e-07, 7.07838434e-06, -1.93477267e-03],
                    [-1.76149723e-06, -1.91962129e-05, 8.22860733e-03],
                    [1.11187404e-06, 1.21178286e-05, -6.29383467e-03],
                ]
            ),
        }

        if self._aims_fixture_no == 13:
            assert np.allclose(
                self.ao.get_forces_without_vdw(), forces[self._aims_fixture_no]
            )

    def test_get_change_of_forces(self) -> None:
        forces = {
            5: 0.4728,
            6: 6.684e-12,
            7: 8.772e-09,
            12: 0.1248e-07,
            13: 0.1665e-06,
        }

        aims_forces_fixtures = [5, 6, 7, 12, 13]

        if self._aims_fixture_no in aims_forces_fixtures:
            assert (
                abs(
                    self.ao.get_change_of_forces()
                    - forces[self._aims_fixture_no]
                )
                < 1e-8
            )

        else:
            with pytest.raises(
                ValueError, match="Energy not found in aims.out file for"
            ):
                self.ao.get_change_of_forces()

    # TODO
    # def get_change_of_sum_of_eigenvalues(self):

    def test_get_change_of_charge_density(self) -> None:
        """Using default args (final charge density change)."""
        charge_densities = np.array(
            [
                0.7136e-06,
            ]
        )

        if self._aims_fixture_no in [0]:
            assert (
                abs(
                    self.ao.get_change_of_charge_density()
                    - charge_densities[self._aims_fixture_no - 1]
                )
                < 1e-8
            )

    def test_check_spin_polarised(self) -> None:
        if self._aims_fixture_no in [2, 3, 11, 12]:
            assert self.ao.check_spin_polarised() is True
        else:
            assert self.ao.check_spin_polarised() is False

    def test_get_convergence_parameters(self, ref_data) -> None:
        if self._aims_fixture_no in [1, 2, 3, 4, 9, 10]:
            assert self.ao.get_convergence_parameters() == ref_data["conv_params"][0]
        elif self._aims_fixture_no == 5:
            assert self.ao.get_convergence_parameters() == ref_data["conv_params"][1]
        elif self._aims_fixture_no == 6:
            assert self.ao.get_convergence_parameters() == ref_data["conv_params"][2]
        elif self._aims_fixture_no == 7:
            assert self.ao.get_convergence_parameters() == ref_data["conv_params"][3]
        elif self._aims_fixture_no == 8:
            assert self.ao.get_convergence_parameters() == ref_data["conv_params"][4]

    def test_get_final_energy(self) -> None:
        final_energies = [
            -2080.832254505,
            -2080.832254498,
            -2080.832254498,
            -15785.832821011,
            -2080.832254506,
            -15802.654211961,
            None,
            None,
            -2081.000809207,
            -15804.824029071,
            -15783.7132844,
            -15802.654211961,
            -0.483268773784931e05,
        ]

        final_energy = self.ao.get_final_energy()

        if self._aims_fixture_no in [7, 8]:
            assert final_energy is None

        else:
            assert (
                abs(final_energy - final_energies[self._aims_fixture_no - 1])
                < 1e-8
            )

    def test_get_final_spin_moment(self) -> None:
        final_spin_moments = [
            None,
            (0.0, 0.0, 1.00),
            (0.0, 0.0, 1.00),
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            (6.0, 3.0, 7.00),
            (0, 0),
        ]

        final_spin_moment = self.ao.get_final_spin_moment()

        if self._aims_fixture_no in [2, 3, 11]:
            assert final_spin_moment is not None

            for i in range(3):
                assert (
                    abs(
                        final_spin_moment[i]
                        - final_spin_moments[self._aims_fixture_no - 1][i]
                    )
                    < 1e-8
                )

        elif self._aims_fixture_no == 12:
            for i in range(2):
                assert final_spin_moment is not None
                assert (
                    abs(
                        final_spin_moment[i]
                        - final_spin_moments[self._aims_fixture_no - 1][i]
                    )
                    < 1e-8
                )

        else:
            assert final_spin_moment is None

    def get_n_relaxation_steps_test(self) -> None:
        n_relaxation_steps = [1, 1, 1, 1, 4, 2, 3, 0, 1, 1, 1, 2]
        assert (
            self.ao.get_n_relaxation_steps()
            == n_relaxation_steps[self._aims_fixture_no - 1]
        )

    def test_get_n_scf_iters(self) -> None:
        n_scf_iters = [12, 13, 13, 10, 42, 27, 56, 8, 14, 11, 10, 29, 251]
        assert (
            self.ao.get_n_scf_iters() == n_scf_iters[self._aims_fixture_no - 1]
        )

    # TODO
    # def get_i_scf_conv_acc_test(self):

    def test_get_n_initial_ks_states(self) -> None:
        n_initial_ks_states = [
            11,
            22,
            48,
            20,
            11,
            20,
            11,
            20,
            11,
            20,
            40,
            20,
            34,
        ]

        if self._aims_fixture_no in [2, 3, 11]:
            assert (
                self.ao.get_n_initial_ks_states()
                == n_initial_ks_states[self._aims_fixture_no - 1]
            )
        else:
            with pytest.warns(UserWarning):
                assert (
                    self.ao.get_n_initial_ks_states()
                    == n_initial_ks_states[self._aims_fixture_no - 1]
                )

    def test_get_all_ks_eigenvalues(self, ref_data) -> None:
        if self._aims_fixture_no == 1:
            for key in ref_data["eigenvalues"]:
                # Check the values are within tolerance and that keys match
                assert np.allclose(
                    self.ao.get_all_ks_eigenvalues()[key],
                    ref_data["eigenvalues"][key],
                    atol=1e-8,
                )

        elif self._aims_fixture_no in [2, 3]:
            spin_up, spin_down = self.ao.get_all_ks_eigenvalues()

            # Check for both spin states
            for spin_eval, spin in zip(
                ["su_eigenvalues", "sd_eigenvalues"],
                [spin_up, spin_down],
                strict=False,
            ):
                for key in ref_data[spin_eval][self._aims_fixture_no - 2]:
                    # Check the values are within tolerance and that keys match
                    assert np.allclose(
                        spin[key],
                        ref_data[spin_eval][self._aims_fixture_no - 2][key],
                        atol=1e-8,
                    )

        else:
            with pytest.raises(ItemNotFoundError):
                self.ao.get_all_ks_eigenvalues()

    def _compare_final_ks_evals(
        self, ref_data: dict, ref: int, spin_case: str
    ) -> None:
        for key in ref_data[f"{spin_case}_final_eigenvalues"][ref]:
            if spin_case == "sn":
                test = self.ao.get_final_ks_eigenvalues()[key]
            elif spin_case == "su":
                test_nk, _ = self.ao.get_final_ks_eigenvalues()
                test = test_nk[key]
            elif spin_case == "sd":
                _, test_nk = self.ao.get_final_ks_eigenvalues()
                test = test_nk[key]
            else:
                raise ValueError("Invalid test")

            assert np.allclose(
                test,
                ref_data[f"{spin_case}_final_eigenvalues"][ref][key],
                atol=1e-8,
            )

    def test_get_final_ks_eigenvalues(self, ref_data) -> None:
        sn_refs = [1, 4, 5, 6, 7, 8, 9]
        sc_refs = [2, 3, 11, 12]

        if self._aims_fixture_no in sn_refs:
            ref = sn_refs.index(self._aims_fixture_no)
            self._compare_final_ks_evals(ref_data, ref, "sn")

        if self._aims_fixture_no in sc_refs:
            ref = sc_refs.index(self._aims_fixture_no)
            self._compare_final_ks_evals(ref_data, ref, "su")
            self._compare_final_ks_evals(ref_data, ref, "sd")

    def test_get_pert_soc_ks_eigenvalues(self, ref_data) -> None:
        if self._aims_fixture_no == 3:
            for key in ref_data["pert_soc_eigenvalues"]:
                # Check the values are within tolerance and that keys match
                assert np.allclose(
                    self.ao.get_pert_soc_ks_eigenvalues()[key],
                    ref_data["pert_soc_eigenvalues"][key],
                    atol=1e-8,
                )

        elif self._aims_fixture_no in [2, 11]:
            with pytest.raises(
                ValueError, match="Final KS states not found in aims.out file."
            ):
                self.ao.get_pert_soc_ks_eigenvalues()

        else:
            # Check that it warns and then raises an error
            with (
                pytest.warns(UserWarning),
                pytest.raises(
                    ValueError,
                    match="Final KS states not found in aims.out file.",
                ),
            ):
                self.ao.get_pert_soc_ks_eigenvalues()


class TestELSIOutput:
    """Tests for the ELSIOutput class."""

    @pytest.fixture(autouse=True)
    def elsi_csc(self, cwd) -> None:
        self.eo_csc = ELSIOutput(
            elsi_csc=f"{cwd}/fixtures/elsi_files/D_spin_01_kpt_000001.csc"
        )

    @pytest.fixture(autouse=True)
    def elsi_npz(self, cwd) -> None:
        self.eo_npz = sp.load_npz(
            f"{cwd}/fixtures/elsi_files/D_spin_01_kpt_000001.npz"
        )

    def test_get_elsi_csc_header(self) -> None:
        assert (
            self.eo_csc.get_elsi_csc_header().all()
            == np.array(
                [
                    170915,
                    -910910,
                    0,
                    42,
                    22,
                    582,
                    -910910,
                    -910910,
                    -910910,
                    -910910,
                    -910910,
                    -910910,
                    -910910,
                    -910910,
                    -910910,
                    -910910,
                ]
            ).all()
        )

    def test_read_elsi_as_csc_to_array(self) -> None:
        eo_csc_format_false = self.eo_csc.read_elsi_as_csc(csc_format=False)
        assert isinstance(eo_csc_format_false, np.ndarray)
        assert np.allclose(
            eo_csc_format_false.all(),
            self.eo_npz.toarray().all(),
            atol=1e-8,
        )

        eo_csc_format_true = self.eo_csc.read_elsi_as_csc(csc_format=True)
        assert not isinstance(eo_csc_format_true, np.ndarray)
        assert np.allclose(
            eo_csc_format_true.toarray().all(),
            self.eo_npz.toarray().all(),
        )

    @pytest.mark.xfail(
        False, reason="Direct comparison of floats without tolerance"
    )
    def test_read_elsi_as_csc_bin_compare(self) -> None:
        assert (
            self.eo_csc.read_elsi_as_csc(csc_format=True) != self.eo_npz
        )._getnnz() == 0


# ruff: noqa: ANN001, S101, ERA001
