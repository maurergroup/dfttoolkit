import os

import numpy as np
import pytest
from dfttoolkit.output import AimsOutput


class TestAimsOutput:
    @property
    def _aims_fixture_no(self) -> int:
        return int(self.ao.path.split("/")[-2])

    @pytest.fixture(params=range(1, 11), autouse=True)
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

        if self._aims_fixture_no in [1, 2, 3, 5, 7, 9]:
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

    def test_get_change_of_forces(self):
        forces = [0.4728, 8.772e-09, 6.684e-12]

        if self._aims_fixture_no in [5, 6, 7]:
            assert (
                abs(self.ao.get_change_of_forces() - forces[self._aims_fixture_no - 5])
                < 1e-8
            )

        else:
            with pytest.raises(ValueError):
                self.ao.get_change_of_forces()

    # TODO
    # def get_change_of_sum_of_eigenvalues(self):

    def test_check_spin_polarised(self):
        if self._aims_fixture_no in [2, 3]:
            assert self.ao.check_spin_polarised() is True
        else:
            assert self.ao.check_spin_polarised() is False

    def test_get_convergence_parameters(self, ref_data):
        if self._aims_fixture_no in [7, 8]:
            assert self.ao.get_convergence_parameters() == ref_data["conv_params"][1]
        else:
            assert self.ao.get_convergence_parameters() == ref_data["conv_params"][0]

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
        ]

        final_energy = self.ao.get_final_energy()

        if self._aims_fixture_no in [7, 8]:
            assert final_energy is None

        else:
            assert abs(final_energy - final_energies[self._aims_fixture_no - 1]) < 1e-8

    def get_n_relaxation_steps_test(self):
        n_relaxation_steps = [1, 1, 1, 1, 4, 2, 3, 0, 1, 1]
        assert (
            self.ao.get_n_relaxation_steps()
            == n_relaxation_steps[self._aims_fixture_no - 1]
        )

    def test_get_n_scf_iters(self):
        n_scf_iters = [12, 13, 13, 10, 42, 27, 56, 8, 14, 11]
        assert self.ao.get_n_scf_iters() == n_scf_iters[self._aims_fixture_no - 1]

    # TODO
    # def get_i_scf_conv_acc_test(self):

    def test_get_n_initial_ks_states(self):
        n_initial_ks_states = [11, 22, 48, 20, 11, 20, 11, 20, 11, 20]

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
