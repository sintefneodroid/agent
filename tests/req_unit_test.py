"""Test availability of required packages."""

import unittest
from pathlib import Path

import pkg_resources
import pytest

_REQUIREMENTS_PATH = Path(__file__).parent.with_name("requirements.txt")
_EXTRA_REQUIREMENTS_PATH = Path(__file__).parent.parent / "requirements"


class TestRequirements(unittest.TestCase):
    """Test availability of required packages."""

    @pytest.mark.xfail(
        strict=False
    )  # DO not successfully parse recursing of reqs using -r
    def test_requirements(self):
        """Test that each required package is available."""
        requirements = pkg_resources.parse_requirements(_REQUIREMENTS_PATH.open())
        for requirement in requirements:
            requirement = str(requirement)
            with self.subTest(requirement=requirement):
                pkg_resources.require(requirement)

    @pytest.mark.xfail(
        strict=False
    )  # DO not successfully parse recursing of reqs using -r
    def test_extra_requirements(self):
        """Test that each required package is available."""
        if _EXTRA_REQUIREMENTS_PATH.exists():
            for extra_req_file in _EXTRA_REQUIREMENTS_PATH.iterdir():
                if extra_req_file.is_file() and extra_req_file.suffix == ".txt":
                    requirements = pkg_resources.parse_requirements(
                        extra_req_file.open()
                    )
                    for requirement in requirements:
                        requirement = str(requirement)
                        with self.subTest(requirement=requirement):
                            pkg_resources.require(requirement)
