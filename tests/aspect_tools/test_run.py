"""Tests for running a prepared ASPECT case."""

import os

import pytest

from gdmate.aspect_tools import run_aspect_case

pytestmark = pytest.mark.skipif(
    os.name != "posix", reason="Uses a POSIX stub executable"
)


@pytest.fixture
def prepared_case(tmp_path):
    case = tmp_path / "case with spaces"
    case.mkdir()
    parameters = case / "case.prm"
    parameters.write_text("set Output directory = elsewhere\n", encoding="utf-8")
    executable = tmp_path / "fake aspect"
    executable.write_text(
        '#!/bin/sh\nprintf "%s\\n" "$PWD"\ntest "$1" = -- || exit 9\n'
        'cat\nprintf "diagnostic\\n" >&2\n',
        encoding="utf-8",
    )
    executable.chmod(0o755)
    return executable, parameters


def test_run_aspect_case_records_command_and_logs(prepared_case):
    executable, parameters = prepared_case
    case = parameters.parent

    result = run_aspect_case(
        executable, parameters.name, case_directory=case, output_directory="run"
    )

    assert result.returncode == 0
    assert result.case_directory == case
    assert result.output_directory == case / "run"
    assert result.command == (str(executable), "--")
    assert not (result.output_directory / "run.prm").exists()
    assert result.stdout_path.read_text(encoding="utf-8").splitlines() == [
        str(case),
        f"include {parameters}",
        f"set Output directory = {result.output_directory}",
    ]
    assert result.stderr_path.read_text(encoding="utf-8") == "diagnostic\n"
    assert (
        parameters.read_text(encoding="utf-8") == "set Output directory = elsewhere\n"
    )


def test_run_aspect_case_reports_failure_and_keeps_logs(prepared_case):
    executable, parameters = prepared_case
    with executable.open("a", encoding="utf-8") as script:
        script.write("exit 7\n")

    result = run_aspect_case(executable, parameters)

    assert result.returncode == 7
    assert result.output_directory == parameters.parent / "output"
    assert result.stderr_path.read_text(encoding="utf-8") == "diagnostic\n"


def test_run_aspect_case_preserves_existing_output(prepared_case):
    executable, parameters = prepared_case
    existing = parameters.parent / "output"
    existing.mkdir()
    marker = existing / "stdout.log"
    marker.write_text("previous run", encoding="utf-8")

    with pytest.raises(FileExistsError):
        run_aspect_case(executable, parameters)

    assert marker.read_text(encoding="utf-8") == "previous run"


def test_run_aspect_case_checks_input_before_creating_output(prepared_case):
    executable, parameters = prepared_case

    with pytest.raises(FileNotFoundError):
        run_aspect_case(executable, parameters.with_name("missing.prm"))

    assert not (parameters.parent / "output").exists()
