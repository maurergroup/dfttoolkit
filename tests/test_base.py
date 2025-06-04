from contextlib import nullcontext

import pytest

from dfttoolkit.base import File, Parser
from dfttoolkit.utils.exceptions import UnsupportedFileError


class TestFile:
    """Test the File class."""

    @pytest.fixture(params=range(1, 13), scope="module")
    def aims_out_loc(self, cwd, request, aims_calc_dir) -> str:
        return f"{cwd}/fixtures/{aims_calc_dir}/{request.param!s}/aims.out"

    @pytest.fixture(scope="module")
    def aims_out_lines(self, aims_out_loc) -> list[str]:
        with open(aims_out_loc) as f:
            return f.readlines()

    @pytest.fixture(scope="module")
    def aims_out_file(self, aims_out_loc) -> File:
        return File(aims_out_loc, "aims_out")

    @pytest.fixture(scope="module")
    def elsi_csc_loc(self, cwd) -> str:
        return f"{cwd}/fixtures/elsi_files/D_spin_01_kpt_000001.csc"

    @pytest.fixture(scope="module")
    def elsi_csc_file(self, elsi_csc_loc) -> File:
        return File(elsi_csc_loc, "elsi_csc")

    def test_file_not_found_error(self) -> None:
        with pytest.raises(FileNotFoundError):
            File("./aims.out", "aims_out")

    def test_text_file_attrs(self, aims_out_file, aims_out_loc) -> None:
        assert aims_out_file.path == aims_out_loc
        assert aims_out_file._format == "aims_out"
        assert aims_out_file._name == "aims.out"
        assert aims_out_file._extension == ".out"
        assert aims_out_file._binary is False
        assert aims_out_file.lines != []
        assert aims_out_file.data == b""

    def test_text_file_lines(self, aims_out_file, aims_out_lines) -> None:
        assert aims_out_file.lines == aims_out_lines

    def test_text_file_str(self, aims_out_file, aims_out_lines) -> None:
        assert str(aims_out_file) == "".join(aims_out_lines)

    def test_binary_file_attrs(self, elsi_csc_file, elsi_csc_loc) -> None:
        assert elsi_csc_file.path == elsi_csc_loc
        assert elsi_csc_file._format == "elsi_csc"
        assert elsi_csc_file._name == "D_spin_01_kpt_000001.csc"
        assert elsi_csc_file._extension == ".csc"
        assert elsi_csc_file._binary is True
        assert elsi_csc_file.lines == []
        assert elsi_csc_file.data != b""

    def test_binary_file_data(self, elsi_csc_file, elsi_csc_loc) -> None:
        with open(elsi_csc_loc, "rb") as f:
            assert elsi_csc_file.data == f.read()

    def test_binary_file_str(self, elsi_csc_file) -> None:
        with pytest.raises(OSError, match="Is a binary file"):
            str(elsi_csc_file)


class DummyParser(Parser):
    """
    Use dummy classes to test the Parser class.

    As it is an abstract class it cannot be instantiated directly.
    """

    def __init__(self, binary, **kwargs: str):
        super().__init__(self._supported_files, **kwargs)

        self._name = "name"
        self._binary = binary

        match self._format:
            case "arbitrary_format_1":
                self.path = "tests/fixtures/base_test_files/test.arb_fmt"
                self._extension = ".arb_fmt"
                self.lines = ["This is a Parser test file!"]
                self.data = b""
                self._check_binary(False)
            case "arbitrary_format_2":
                self.path = "tests/fixtures/base_test_files/test.csc"
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
    """Test the Parser class."""

    @pytest.fixture
    def dummy_parser(self) -> type[DummyParser]:
        """Get a dummy parser class for testing."""
        return DummyParser

    @pytest.mark.parametrize(
        ("kwargs", "binary", "expectation"),
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
                {"arbitrary_format_1": "tests/fixtures/base_test_files/test.arb_fmt"},
                False,
                nullcontext(
                    DummyParser(
                        False,
                        arbitrary_format_1="tests/fixtures/base_test_files/test.arb_fmt",
                    )
                ),
            ),
            (
                {"arbitrary_format_2": "tests/fixtures/base_test_files/test.csc"},
                True,
                nullcontext(
                    DummyParser(
                        True,
                        arbitrary_format_2="tests/fixtures/base_test_files/test.csc",
                    )
                ),
            ),
        ],
    )
    def test_parser_init(self, dummy_parser, kwargs, binary, expectation) -> None:
        with expectation as e:
            assert dummy_parser(binary, **kwargs) == e

    def test_no_cls_init(self) -> None:
        with pytest.raises(TypeError):

            class DummyParser(Parser):  # pyright: ignore[reportUnusedClass]
                @property
                def _supported_files(self) -> dict:
                    return {"arbitrary_format": ".arb_format"}

    def test_no_supported_files_property(self) -> None:
        class DummyParser(Parser):
            def __init__(self, **kwargs: str):
                super().__init__(self._supported_files, **kwargs)

                self._check_binary(False)

        with pytest.raises(TypeError):
            DummyParser()  # pyright: ignore[reportAbstractUsage]

    def test_no_check_binary(self) -> None:
        with pytest.raises(TypeError):

            class DummyParser(Parser):  # pyright: ignore[reportUnusedClass]
                def __init__(self, **kwargs: str):
                    super().__init__(self._supported_files, **kwargs)

                @property
                def _supported_files(self) -> dict:
                    return {"arbitrary_format": ".arb_format"}

    @pytest.mark.parametrize(
        ("kwargs", "binary", "expectation"),
        [
            (
                {"arbitrary_format_1": "tests/fixtures/base_test_files/test.arb_fmt"},
                False,
                nullcontext(None),
            ),
            (
                {"arbitrary_format_2": "tests/fixtures/base_test_files/test.csc"},
                True,
                nullcontext(None),
            ),
            (
                {"arbitrary_format_1": "tests/fixtures/base_test_files/test.arb_fmt"},
                True,
                pytest.raises(ValueError, match="name should be text format"),
            ),
            (
                {"arbitrary_format_2": "tests/fixtures/base_test_files/test.csc"},
                False,
                pytest.raises(ValueError, match="name should be binary format"),
            ),
        ],
    )
    def test_check_binary(self, dummy_parser, kwargs, binary, expectation) -> None:
        with expectation as e:
            dp = dummy_parser(binary, **kwargs)
            assert dp._check_binary(binary) == e
