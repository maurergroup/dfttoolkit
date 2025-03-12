from dfttoolkit.base import Parser, File
import pytest


class TestFile:
    @pytest.fixture(params=range(1, 13), scope="module")
    def aims_out_loc(self, cwd, request, aims_calc_dir):
        return f"{cwd}/fixtures/{aims_calc_dir}/{str(request.param)}/aims.out"

    @pytest.fixture(scope="module")
    def aims_out_lines(self, aims_out_loc):
        with open(aims_out_loc, "r") as f:
            aims_out_lines = f.readlines()

        return aims_out_lines

    @pytest.fixture(scope="module")
    def aims_out_file(self, aims_out_loc):
        return File(aims_out_loc, "aims_out")

    @pytest.fixture(scope="module")
    def elsi_csc_loc(self, cwd):
        return f"{cwd}/fixtures/elsi_files/D_spin_01_kpt_000001.csc"

    @pytest.fixture(scope="module")
    def elsi_csc_file(self, elsi_csc_loc):
        return File(elsi_csc_loc, "elsi_csc")

    def test_text_file_attrs(self, aims_out_file, aims_out_loc):
        assert aims_out_file.path == aims_out_loc
        assert aims_out_file._format == "aims_out"
        assert aims_out_file._name == "aims.out"
        assert aims_out_file._extension == ".out"
        assert aims_out_file._binary is False
        assert aims_out_file.lines != []
        assert aims_out_file.data == b""

    def test_text_file_lines(self, aims_out_file, aims_out_lines):
        assert aims_out_file.lines == aims_out_lines

    def test_text_file_str(self, aims_out_file, aims_out_lines):
        assert str(aims_out_file) == "".join(aims_out_lines)

    def test_binary_file_attrs(self, elsi_csc_file, elsi_csc_loc):
        assert elsi_csc_file.path == elsi_csc_loc
        assert elsi_csc_file._format == "elsi_csc"
        assert elsi_csc_file._name == "D_spin_01_kpt_000001.csc"
        assert elsi_csc_file._extension == ".csc"
        assert elsi_csc_file._binary is True
        assert elsi_csc_file.lines == []
        assert elsi_csc_file.data != b""

    def test_binary_file_data(self, elsi_csc_file, elsi_csc_loc):
        with open(elsi_csc_loc, "rb") as f:
            assert elsi_csc_file.data == f.read()

    def test_binary_file_str(self, elsi_csc_file):
        with pytest.raises(OSError):
            str(elsi_csc_file)


# TODO
class TestParser: ...
