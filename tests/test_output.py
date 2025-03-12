import os

import numpy as np
import pytest
from dfttoolkit.output import AimsOutput


class TestAimsOutput:
    @property
    def _aims_fixture_no(self) -> int:
        return int(self.ao.path.split("/")[-2])

    @pytest.fixture(params=range(1, 13), autouse=True)
    def aims_out(self, cwd, request, aims_calc_dir):

        self.ao = AimsOutput(
            aims_out=f"{cwd}/fixtures/{aims_calc_dir}/{str(request.param)}/aims.out"
        )

    def test_get_number_of_atoms(self):
        if self._aims_fixture_no in [4, 6, 8, 10]:
            assert self.ao.get_number_of_atoms() == 2
        else:
            assert self.ao.get_number_of_atoms() == 3

    def test_get_geometry(self):
        geom = self.ao.get_geometry()

        if self._aims_fixture_no in [1, 2, 3, 5, 7, 9, 13]:
            assert len(geom) == 3
            assert geom.get_is_periodic() is False
        else:
            assert len(geom) == 2
            assert geom.get_is_periodic() is True

    # TODO
    # def test_get_parameters(self):

    def test_check_exit_normal(self):
        if self._aims_fixture_no in [7, 8]:
            assert self.ao.check_exit_normal() is False
        else:
            assert self.ao.check_exit_normal() is True

    def test_get_time_per_scf(self, ref_data):
        # Fail if the absolute tolerance between any values in test vs. reference array is
        # greater than 2e-3
        if self._aims_fixture_no in range(1, 13):
            assert np.allclose(
                self.ao.get_time_per_scf(),
                ref_data["timings"][self._aims_fixture_no - 1],
                atol=2e-3,
            )

    def test_get_change_of_total_energy_1(self):
        """Using default args (final energy change)"""

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

    def test_get_change_of_total_energy_2(self, ref_data):
        """Get every energy change"""

        # Fail if the absolute tolerance between any values in test vs. reference array is
        # greater than 1e-10
        if self._aims_fixture_no in range(1, 13):
            assert np.allclose(
                self.ao.get_change_of_total_energy(n_occurrence=None),
                ref_data["energy_diffs"][self._aims_fixture_no - 1],
                atol=1e-8,
            )

    def test_get_change_of_total_energy_3(self):
        """Get the 1st energy change"""

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
    # of the following functions wrap around _get_energy(), which have all been tested in
    # the previous 4 tests

    def test_get_forces(self):
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
            print(
                abs(self.ao.get_forces() - forces[self._aims_fixture_no - 5])
            )

            assert (
                np.all(
                    abs(
                        self.ao.get_forces()
                        - forces[self._aims_fixture_no - 5]
                    )
                )
                < 1e-8
            )

    def test_get_forces_without_vdw(self):
        forces = [
            np.array(
                [
                    [6.49567857e-07, 7.07725101e-06, 6.02880000e-04],
                    [-1.76155257e-06, -1.91973462e-05, 1.07662600e-02],
                    [1.11181871e-06, 1.21166952e-05, -3.75618200e-03],
                ]
            )
        ]

        if self._aims_fixture_no in [13]:
            # print(self.ao.get_forces_without_vdw())
            # print(self.ao.get_forces() - self.ao.get_vdw_forces())

            assert (
                np.all(abs(self.ao.get_forces_without_vdw() - forces[0]))
                < 1e-8
            )

    def test_get_change_of_forces(self):
        forces = {5: 0.4728, 6: 6.684e-12, 7: 8.772e-09, 13: 0.1665e-06}

        if self._aims_fixture_no in [5, 6, 7, 13]:
            print(self.ao.get_change_of_forces())
            assert (
                abs(
                    self.ao.get_change_of_forces()
                    - forces[self._aims_fixture_no]
                )
                < 1e-8
            )

        else:
            with pytest.raises(ValueError):
                self.ao.get_change_of_forces()

    # TODO
    # def get_change_of_sum_of_eigenvalues(self):

    def test_get_change_of_charge_density(self):
        """Using default args (final charge density change)"""

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

    def test_check_spin_polarised(self):
        if self._aims_fixture_no in [2, 3]:
            assert self.ao.check_spin_polarised() is True
        else:
            assert self.ao.check_spin_polarised() is False

    def test_get_convergence_parameters(self, ref_data):
        if self._aims_fixture_no in [7, 8]:
            assert (
                self.ao.get_convergence_parameters()
                == ref_data["conv_params"][1]
            )
        elif self._aims_fixture_no in [1, 2, 3, 4, 5, 6, 9, 10]:
            assert (
                self.ao.get_convergence_parameters()
                == ref_data["conv_params"][0]
            )

    def test_get_final_energy(self):
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

    def get_n_relaxation_steps_test(self):
        n_relaxation_steps = [1, 1, 1, 1, 4, 2, 3, 0, 1, 1, 9]
        assert (
            self.ao.get_n_relaxation_steps()
            == n_relaxation_steps[self._aims_fixture_no - 1]
        )

    def test_get_n_scf_iters(self):
        n_scf_iters = [12, 13, 13, 10, 42, 27, 56, 8, 14, 11, 251]
        assert (
            self.ao.get_n_scf_iters() == n_scf_iters[self._aims_fixture_no - 1]
        )

    # TODO
    # def get_i_scf_conv_acc_test(self):

    def test_get_n_initial_ks_states(self):
        n_initial_ks_states = [11, 22, 48, 20, 11, 20, 11, 20, 11, 20, 34]

        def compare_n_initial_ks_states():
            assert (
                self.ao.get_n_initial_ks_states()
                == n_initial_ks_states[self._aims_fixture_no - 1]
            )

        if self._aims_fixture_no in [2, 3]:
            compare_n_initial_ks_states()
        else:
            with pytest.warns(UserWarning):
                compare_n_initial_ks_states()

    # TODO
    # def test_get_all_ks_eigenvalues(self):
    #     if self._aims_fixture_no == 1:
    #         for key in ref_data["eigenvalues"].keys():
    #             # Check the values are within tolerance and that keys match
    #             assert np.allclose(
    #                 self.ao.get_all_ks_eigenvalues()[key],
    #                 ref_data["eigenvalues"][key],
    #                 atol=1e-8,
    #             )

    #     elif self._aims_fixture_no in [2, 3]:
    #         spin_up, spin_down = self.ao.get_all_ks_eigenvalues()

    #         for key in ref_data["su_eigenvalues"].keys():
    #             # Check the values are within tolerance and that keys match
    #             assert np.allclose(
    #                 spin_up[key], ref_data["su_eigenvalues"][key], atol=1e-8
    #             )
    #             # Repeat for spin_down
    #             assert np.allclose(
    #                 spin_down[key], ref_data["sd_eigenvalues"][key], atol=1e-8
    #             )

    #     else:
    #         with pytest.raises(ValueError):
    #             self.ao.get_all_ks_eigenvalues()

    # TODO
    # def get_final_ks_eigenvalues_test(self):

    def test_get_pert_soc_ks_eigenvalues(self, ref_data):
        if self._aims_fixture_no == 3:
            for key in ref_data["pert_soc_eigenvalues"].keys():
                # Check the values are within tolerance and that keys match
                assert np.allclose(
                    self.ao.get_pert_soc_ks_eigenvalues()[key],
                    ref_data["pert_soc_eigenvalues"][key],
                    atol=1e-8,
                )

        elif self._aims_fixture_no == 2:
            with pytest.raises(ValueError):
                self.ao.get_pert_soc_ks_eigenvalues()

        else:
            # Check that it warns and then raises an error
            with pytest.warns(UserWarning):
                with pytest.raises(ValueError):
                    self.ao.get_pert_soc_ks_eigenvalues()


# TODO
# class TestELSIOutput:
