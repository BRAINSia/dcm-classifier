#!/usr/bin/env -S uv run --script
# /// script
# dependencies = ["nox>=2025.2.9"]
# ///
"""Development tasks. Run `nox -l` to list them and `nox` for the defaults."""

from pathlib import Path

import nox

nox.needs_version = ">=2025.2.9"
nox.options.default_venv_backend = "uv|virtualenv"

PYPROJECT = nox.project.load_toml("pyproject.toml")
ROOT = Path(__file__).parent


@nox.session(default=True)
def lint(session: nox.Session) -> None:
    """Run the pre-commit hooks on every file."""
    session.install("pre-commit")
    session.run(
        "pre-commit", "run", "--all-files", "--show-diff-on-failure", *session.posargs
    )


@nox.session(default=True)
def tests(session: nox.Session) -> None:
    """Run the test suite against the installed package."""
    session.install(".", *nox.project.dependency_groups(PYPROJECT, "test"))
    session.run("pytest", *session.posargs)


@nox.session(default=False)
def coverage(session: nox.Session) -> None:
    """Run the test suite under coverage and report the lines it missed."""
    session.install(".", *nox.project.dependency_groups(PYPROJECT, "test"))
    session.run("coverage", "run", "-m", "pytest", *session.posargs)
    session.run("coverage", "report")


@nox.session(python="3.13", reuse_venv=True, default=False)
def docs(session: nox.Session) -> None:
    """Build the Sphinx documentation into docs/_build/html."""
    session.install(".", *nox.project.dependency_groups(PYPROJECT, "docs"))
    session.run(
        "sphinx-build",
        "-b",
        "html",
        "docs/source",
        "docs/_build/html",
        *session.posargs,
    )


@nox.session(default=False)
def build(session: nox.Session) -> None:
    """Build the source distribution and wheel into dist/."""
    session.install("build", "twine")
    dist = ROOT / "dist"
    if dist.exists():
        for item in dist.iterdir():
            item.unlink()
    session.run("python", "-m", "build")
    session.run("twine", "check", "dist/*")


if __name__ == "__main__":
    nox.main()
