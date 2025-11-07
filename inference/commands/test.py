"""
Test command - Run unit and integration tests.
"""
import typer
import subprocess
import sys

test_app = typer.Typer(help="Run tests - unit tests, integration tests, or both")


@test_app.command("all")
def test_all(
    docker: bool = typer.Option(False, "--docker", help="Run tests inside Docker container"),
    verbose: bool = typer.Option(True, "--verbose/--quiet", "-v/-q", help="Verbose output")
):
    """
    Run all tests (unit tests + integration tests).
    
    Matches: make test
    """
    try:
        v_flag = "-v" if verbose else "-q"
        if docker:
            subprocess.run(
                ["docker", "compose", "exec", "api", "python", "-m", "pytest", "tests/unit/", "tests/integration/", v_flag],
                check=True
            )
        else:
            subprocess.run(
                [sys.executable, "-m", "pytest", "tests/unit/", "tests/integration/", v_flag],
                check=True
            )
    except subprocess.CalledProcessError as e:
        typer.echo(f"Tests failed with exit code {e.returncode}", err=True)
        raise typer.Exit(1)


@test_app.command("unit")
def test_unit(
    docker: bool = typer.Option(False, "--docker", help="Run tests inside Docker container"),
    verbose: bool = typer.Option(True, "--verbose/--quiet", "-v/-q", help="Verbose output")
):
    """
    Run unit tests only.
    
    Matches: make unit-tests
    """
    try:
        v_flag = "-v" if verbose else "-q"
        if docker:
            subprocess.run(
                ["docker", "compose", "exec", "api", "python", "-m", "pytest", "tests/unit/", v_flag],
                check=True
            )
        else:
            subprocess.run(
                [sys.executable, "-m", "pytest", "tests/unit/", v_flag],
                check=True
            )
    except subprocess.CalledProcessError as e:
        typer.echo(f"Unit tests failed with exit code {e.returncode}", err=True)
        raise typer.Exit(1)


@test_app.command("integration")
def test_integration(
    docker: bool = typer.Option(False, "--docker", help="Run tests inside Docker container"),
    verbose: bool = typer.Option(True, "--verbose/--quiet", "-v/-q", help="Verbose output")
):
    """
    Run integration tests only.
    
    Matches: make integration-tests
    """
    try:
        v_flag = "-v" if verbose else "-q"
        if docker:
            subprocess.run(
                ["docker", "compose", "exec", "api", "python", "-m", "pytest", "tests/integration/", v_flag],
                check=True
            )
        else:
            subprocess.run(
                [sys.executable, "-m", "pytest", "tests/integration/", v_flag],
                check=True
            )
    except subprocess.CalledProcessError as e:
        typer.echo(f"Integration tests failed with exit code {e.returncode}", err=True)
        raise typer.Exit(1)


def test(
    test_type: str = typer.Argument("all", help="Test type: 'all', 'unit', or 'integration'"),
    docker: bool = typer.Option(False, "--docker", help="Run tests inside Docker container"),
    verbose: bool = typer.Option(True, "--verbose/--quiet", "-v/-q", help="Verbose output")
):
    """
    Run tests - unit tests, integration tests, or both.
    
    Usage:
        deep-rag test all          # Run all tests (unit + integration)
        deep-rag test unit         # Run unit tests only
        deep-rag test integration  # Run integration tests only
    
    Matches: make test, make unit-tests, make integration-tests
    """
    if test_type == "all":
        test_all(docker=docker, verbose=verbose)
    elif test_type == "unit":
        test_unit(docker=docker, verbose=verbose)
    elif test_type == "integration":
        test_integration(docker=docker, verbose=verbose)
    else:
        typer.echo(f"Error: Unknown test type '{test_type}'. Use 'all', 'unit', or 'integration'", err=True)
        raise typer.Exit(1)

