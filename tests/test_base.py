from contextlib import nullcontext
import pytest
from dfttoolkit.base import File, Parser
from dfttoolkit.utils.exceptions import UnsupportedFileError


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

    def test_file_not_found_error(self):
        with pytest.raises(FileNotFoundError):
            File("./aims.out", "aims_out")

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


class DummyParser(Parser):
    """
    Use dummy classes to test the Parser class as it is an abstract class and cannot be
    instantiated directly
    """

    def __init__(self, binary, **kwargs):
        super().__init__(self._supported_files, **kwargs)

        self._binary = binary
        self._name = "name"

        match self._format:
            case "arbitrary_format_1":
                self._extension = ".arb_fmt"
                self.lines = ["This is a Parser test file!"]
                self.data = b""
                self._check_binary(False)
            case "arbitrary_format_2":
                self._extension = ".csc"
                self.lines = []
                self.data = b"This is a Parser test file!"
                self._check_binary(True)

    @property
    def _supported_files(self) -> dict:
        return {
            "arbitrary_format_1": ".arb_fmt",
            "arbitrary_format_2": ".csc",
        }


class TestParser:
    @pytest.fixture
    def dummy_parser(self):
        yield DummyParser

    @pytest.mark.parametrize(
        "kwargs, binary, expectation",
        [
            ({}, False, pytest.raises(TypeError)),
            (
                {"unsupported_format": ".unsup_fmt"},
                False,
                pytest.raises(UnsupportedFileError),
            ),
            (
                {
                    "arbitrary_format_1": ".arb_fmt",
                    "arbitrary_format_2": ".csc",
                },
                True,
                pytest.raises(TypeError),
            ),
            ({"arbitrary_format_1": ".csc"}, True, pytest.raises(KeyError)),
            (
                {"arbitrary_format_1": "fixtures/base_test_files/test.arb_fmt"},
                False,
                nullcontext(
                    DummyParser(
                        False,
                        arbitrary_format_1="fixtures/base_test_files/test.arb_fmt",
                    )
                ),
            ),
            (
                {"arbitrary_format_2": "fixtures/base_test_files/test.csc"},
                True,
                nullcontext(
                    DummyParser(
                        True, arbitrary_format_2="fixtures/base_test_files/test.csc"
                    )
                ),
            ),
        ],
    )
    def test_parser_init(self, dummy_parser, kwargs, binary, expectation):
        with expectation as e:
            # print(dummy_parser(binary, **kwargs))
            assert dummy_parser(binary, **kwargs) == e

    def test_no_cls_init(self):
        with pytest.raises(TypeError):

            class DummyParser(Parser):
                @property
                def _supported_files(self) -> dict:
                    return {"arbitrary_format": ".arb_format"}

    def test_no_supported_files_property(self):
        class DummyParser(Parser):
            def __init__(self, **kwargs):
                super().__init__(**kwargs)

                self._check_binary(False)

        with pytest.raises(TypeError):
            DummyParser()  # pyright: ignore

    def test_no_check_binary(self):
        with pytest.raises(TypeError):

            class DummyParser(Parser):
                def __init__(self, **kwargs):
                    super().__init__(self._supported_files, **kwargs)

                @property
                def _supported_files(self) -> dict:
                    return {"arbitrary_format": ".arb_format"}

    @pytest.mark.parametrize(
        "kwargs, binary, expectation",
        [
            (
                {"arbitrary_format_1": "fixtures/base_test_files/test.arb_fmt"},
                False,
                nullcontext(None),
            ),
            (
                {"arbitrary_format_2": "fixtures/base_test_files/test.csc"},
                True,
                nullcontext(None),
            ),
            (
                {"arbitrary_format_1": "fixtures/base_test_files/test.arb_fmt"},
                True,
                pytest.raises(ValueError),
            ),
            (
                {"arbitrary_format_2": "fixtures/base_test_files/test.csc"},
                False,
                pytest.raises(ValueError),
            ),
        ],
    )
    def test_check_binary(self, dummy_parser, kwargs, binary, expectation):
        with expectation as e:
            dp = dummy_parser(binary, **kwargs)
            assert dp._check_binary(binary) == e
