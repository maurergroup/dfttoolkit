from collections.abc import Callable
from functools import wraps
from pathlib import Path
from typing import Any
from warnings import warn


def no_repeat(
    _func: Callable[..., Any] | None = None,
    *,
    output_file: str = "aims.out",
    calc_dir: str = "./",
    force: bool = False,
    suppress_warn: bool = False,
) -> Callable[..., Any]:
    """
    Skip function execution if a specified file already exists.

    Use this decorator to avoid redundant or expensive operations when output files are
    already present.

    Arguments passed as keyword arguments to the wrapped function (`output_file`,
    `calc_dir`, `force`, `suppress_warn`) will override those defined in the decorator.

    Parameters
    ----------
    output_file : str, default="aims.out"
        Name of the file to check for existence.
    calc_dir : str, default="./"
        Directory to look for `output_file`.
    force : bool, default=False
        Always run the function regardless of file existence.
    suppress_warn : bool, default=False
        Suppress warnings when the wrapped function overrides decorator keyword
        arguments.

    Returns
    -------
    Callable[..., Any]
        A wrapped function with the same signature as the original, which returns the
        original value if executed, or `None` if the file exists and `force=False`.

    Raises
    ------
    NotADirectoryError
        If `calc_dir` is not a valid directory.

    Examples
    --------
    TODO
    """

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Callable[..., Any] | None:
            # Allow override via kwargs
            output = kwargs.pop("output_file", output_file)
            dir_ = Path(str(kwargs.pop("calc_dir", calc_dir)))
            force_run = kwargs.pop("force", force)
            suppress = kwargs.pop("suppress_warn", suppress_warn)

            # Warn on possible name collisions
            overridden = {
                "output_file": output != output_file,
                "calc_dir": str(dir_) != calc_dir,
                "force": force_run != force,
                "suppress_warn": suppress != suppress_warn,
            }

            if not suppress and any(overridden.values()):
                warn(
                    "Found keywords used in the `no_repeat` wrapper in "
                    f"`{func.__name__}`, which will override the values for the wrapper"
                    " arguments. If this is the intended behaviour, this warning can be"
                    " suppressed by specifying `suppress_warn=True` in the wrapper"
                    " arguments.",
                    stacklevel=2,
                )

            if not dir_.is_dir():
                raise NotADirectoryError("Provided `calc_dir` is not a directory.")

            if force_run or not (dir_ / Path(str(output))).is_file():
                return func(*args, **kwargs)

            print(f"Skipping `{func.__name__}`: `{output}` already exists.")
            return None

        return wrapper

    if _func is None:
        return decorator

    return decorator(_func)
