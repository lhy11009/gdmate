"""Run a prepared ASPECT case locally."""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from io import StringIO
from os import PathLike
from pathlib import Path

from gdmate.aspect_tools.io import save_parameters_from_dict


@dataclass(frozen=True)
class AspectCaseResult:
    """Command, exit code, and paths for a completed ASPECT process."""

    command: tuple[str, ...]
    case_directory: Path
    output_directory: Path
    stdout_path: Path
    stderr_path: Path
    returncode: int


def run_aspect_case(
    executable: str | PathLike[str],
    parameter_file: str | PathLike[str],
    *,
    case_directory: str | PathLike[str] | None = None,
    output_directory: str | PathLike[str] = "output",
) -> AspectCaseResult:
    """Run a prepared ASPECT parameter file and retain its command and logs.

    The executable path is relative to the caller if it is not absolute.
    By default, ASPECT runs in the parameter file's directory. If
    ``case_directory`` is given, a relative parameter path is resolved there.
    ASPECT resolves relative includes and other input paths from that working
    directory.

    The output directory is relative to the case directory and must be new.
    The original input and output directory override are supplied through
    standard input. ASPECT saves its own copy of the run parameters. Standard
    output and error are saved to separate log files.
    Nonzero process exits are returned with their logs, and an executable
    that cannot be started raises the underlying OS error.
    """
    aspect = Path(executable).absolute()
    parameters = Path(parameter_file)
    if case_directory is None:
        parameters = parameters.absolute()
        case = parameters.parent
    else:
        case = Path(case_directory).resolve()
        parameters = case / parameters
    output = (case / output_directory).resolve()

    for path in (aspect, parameters):
        if not path.is_file():
            raise FileNotFoundError(path)
    for path in (parameters, output):
        if any(character in str(path) for character in "#\r\n"):
            raise ValueError(
                f"Path cannot be written to an ASPECT parameter file: {path}"
            )

    output.mkdir(parents=True)
    run_parameters = StringIO()
    save_parameters_from_dict(
        run_parameters, {"Output directory": str(output)}, include_path=parameters
    )

    command = (str(aspect), "--")
    stdout_path = output / "stdout.log"
    stderr_path = output / "stderr.log"
    with stdout_path.open("wb") as stdout, stderr_path.open("wb") as stderr:
        completed = subprocess.run(
            command,
            cwd=case,
            input=run_parameters.getvalue().encode("utf-8"),
            stdout=stdout,
            stderr=stderr,
            check=False,
        )

    return AspectCaseResult(
        command=command,
        case_directory=case,
        output_directory=output,
        stdout_path=stdout_path,
        stderr_path=stderr_path,
        returncode=completed.returncode,
    )
