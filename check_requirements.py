# -*- coding: utf-8 -*-
"""
check_requirements.py -- check (and optionally install) the required packages.

The packages are listed in requirements.txt (next to this file), one per
line, in the usual pip format, e.g. ``numpy>=1.20``.  Only the standard
library is used, so the script works before anything is installed.

Usage
-----
    python check_requirements.py              check; print one line per package
    python check_requirements.py --install    install the missing or outdated
                                              packages with pip
    python check_requirements.py --missing    print only the pip specifications
                                              of the missing packages

Exit codes
----------
    0   all requirements are satisfied (or the installation succeeded)
    1   some packages are missing or outdated
    2   the installation failed
    3   the Python version is too old or requirements.txt cannot be read

run_Python.bat calls this script: first to check, then (after the user has
agreed) with --install, and then once more in a new process to verify the
installation (a fresh interpreter also sees packages installed into a
user site-packages folder that did not exist before).
"""

from __future__ import annotations

import argparse
import importlib
import os
import re
import subprocess
import sys
from typing import List, Optional, Tuple

MINIMUM_PYTHON = (3, 8)
REQUIREMENTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "requirements.txt")

# module names that differ from the distribution names (extend if necessary)
IMPORT_NAMES = {"scikit-learn": "sklearn", "pyyaml": "yaml", "pillow": "PIL"}

_SPEC = re.compile(r"^\s*([A-Za-z0-9][A-Za-z0-9._-]*)\s*(?:(==|>=|<=|>|<|!=|~=)\s*([0-9][0-9A-Za-z.]*))?\s*$")


class Requirement:
    def __init__(self, line: str):
        match = _SPEC.match(line)
        if match is None:
            raise ValueError("cannot parse the requirement %r" % line)
        self.text = line.strip()
        self.name, self.operator, self.version = match.group(1), match.group(2), match.group(3)
        self.module = IMPORT_NAMES.get(self.name.lower(), self.name.replace("-", "_"))


def _version_tuple(version: str) -> Tuple[int, ...]:
    """Numeric release part of a version string: '1.26.4rc1' -> (1, 26, 4)."""
    parts = []
    for item in version.split("."):
        digits = re.match(r"\d+", item)
        if digits is None:
            break
        parts.append(int(digits.group()))
        if digits.group() != item:
            break
    return tuple(parts)


def _satisfies(installed: str, operator: Optional[str], required: Optional[str]) -> bool:
    if operator is None:
        return True
    a, b = _version_tuple(installed), _version_tuple(required)
    width = max(len(a), len(b))
    a, b = a + (0,) * (width - len(a)), b + (0,) * (width - len(b))
    if operator == "~=":
        return a >= b and a[:len(_version_tuple(required)) - 1] == b[:len(_version_tuple(required)) - 1]
    return {"==": a == b, ">=": a >= b, "<=": a <= b, ">": a > b, "<": a < b, "!=": a != b}[operator]


def read_requirements(path: str = REQUIREMENTS) -> List[Requirement]:
    requirements = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.split("#", 1)[0].strip()
            if line:
                requirements.append(Requirement(line))
    return requirements


def installed_version(requirement: Requirement) -> Optional[str]:
    """Version of an importable package, or None if it cannot be imported."""
    try:
        module = importlib.import_module(requirement.module)
    except Exception:            # ImportError, but also broken installations
        return None
    version = getattr(module, "__version__", None)
    if version is None:
        try:
            from importlib import metadata
            version = metadata.version(requirement.name)
        except Exception:
            version = "0"
    return str(version)


def check(requirements: List[Requirement], verbose: bool = True) -> List[Requirement]:
    """Return the requirements that are missing or outdated."""
    problems = []
    for requirement in requirements:
        version = installed_version(requirement)
        if version is None:
            status = "MISSING"
            problems.append(requirement)
        elif not _satisfies(version, requirement.operator, requirement.version):
            status = "OUTDATED (installed %s, required %s%s)" % (version, requirement.operator,
                                                                 requirement.version)
            problems.append(requirement)
        else:
            status = "OK (%s)" % version
        if verbose:
            print("         %-12s ... %s" % (requirement.text, status))
    return problems


def _pip(arguments: List[str]) -> int:
    command = [sys.executable, "-m", "pip"] + arguments
    print("         > " + " ".join(command))
    sys.stdout.flush()
    return subprocess.call(command)


def install(requirements: List[Requirement]) -> bool:
    """Install the given requirements with pip; returns True on success."""
    if subprocess.call([sys.executable, "-m", "pip", "--version"],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) != 0:
        print("         pip is not available; trying to install it with ensurepip ...")
        if subprocess.call([sys.executable, "-m", "ensurepip", "--upgrade"]) != 0:
            print("         [ERROR] pip could not be installed.")
            return False
    specs = [r.text for r in requirements]
    if _pip(["install", "--upgrade"] + specs) == 0:
        return True
    # e.g. no write access to a system-wide installation: install for the
    # current user (not possible inside a virtual environment)
    in_venv = sys.prefix != getattr(sys, "base_prefix", sys.prefix)
    if not in_venv:
        print("         Retrying the installation for the current user only ...")
        if _pip(["install", "--upgrade", "--user"] + specs) == 0:
            return True
    return False


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Check the Python packages of requirements.txt.")
    parser.add_argument("--install", action="store_true", help="install missing/outdated packages")
    parser.add_argument("--missing", action="store_true",
                        help="print only the specifications of missing/outdated packages")
    parser.add_argument("-r", "--requirements", default=REQUIREMENTS, help="requirements file")
    args = parser.parse_args(argv)

    if sys.version_info < MINIMUM_PYTHON:
        print("         [ERROR] Python %d.%d or newer is required (this is Python %s)."
              % (MINIMUM_PYTHON + (sys.version.split()[0],)))
        return 3
    try:
        requirements = read_requirements(args.requirements)
    except (OSError, ValueError) as error:
        print("         [ERROR] %s" % error)
        return 3

    if args.missing:
        problems = check(requirements, verbose=False)
        print(" ".join(r.text for r in problems))
        return 1 if problems else 0

    problems = check(requirements, verbose=not args.install)
    if not problems:
        return 0
    if not args.install:
        return 1
    return 0 if install(problems) else 2


if __name__ == "__main__":
    sys.exit(main())
