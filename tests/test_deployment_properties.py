"""
Property-based tests for deployment requirements.

**Feature: confluence-rag-system, Property 28: Dependency file presence**
**Feature: confluence-rag-system, Property 29: Python version verification**
**Validates: Requirements 8.1, 8.3**
"""

import sys
from pathlib import Path

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st


class TestDeploymentProperties:
    """Property tests for deployment configuration."""

    def test_property_28_dependency_file_presence(self):
        """
        Property 28: Dependency file presence
        *For any* packaged system, a requirements.txt or pyproject.toml file
        should exist at the repository root
        **Validates: Requirements 8.1**
        """
        # Get repository root (parent of tests directory)
        repo_root = Path(__file__).parent.parent

        # Check for requirements.txt
        requirements_txt = repo_root / "requirements.txt"
        pyproject_toml = repo_root / "pyproject.toml"

        # At least one should exist
        assert (
            requirements_txt.exists() or pyproject_toml.exists()
        ), "Neither requirements.txt nor pyproject.toml found at repository root"

        # If requirements.txt exists, verify it's not empty
        if requirements_txt.exists():
            content = requirements_txt.read_text()
            assert len(content.strip()) > 0, "requirements.txt exists but is empty"

            # Verify it contains actual dependencies (not just comments)
            lines = [
                line.strip()
                for line in content.split("\n")
                if line.strip() and not line.strip().startswith("#")
            ]
            assert len(lines) > 0, "requirements.txt contains no dependencies"

        # If pyproject.toml exists, verify it has dependencies
        if pyproject_toml.exists():
            content = pyproject_toml.read_text()
            assert len(content.strip()) > 0, "pyproject.toml exists but is empty"
            assert (
                "dependencies" in content or "[project]" in content
            ), "pyproject.toml missing dependencies section"

    def test_property_29_python_version_verification(self):
        """
        Property 29: Python version verification
        *For any* running system instance, the Python version should be 3.12.x
        **Validates: Requirements 8.3**
        """
        # Get current Python version
        version_info = sys.version_info

        # Verify major and minor version
        assert (
            version_info.major == 3
        ), f"Python major version is {version_info.major}, expected 3"
        assert (
            version_info.minor == 12
        ), f"Python minor version is {version_info.minor}, expected 12"

    def test_pyproject_specifies_python_312(self):
        """
        Verify that pyproject.toml specifies Python 3.12 requirement.
        """
        repo_root = Path(__file__).parent.parent
        pyproject_toml = repo_root / "pyproject.toml"

        if pyproject_toml.exists():
            content = pyproject_toml.read_text()

            # Check for Python version specification
            assert (
                "requires-python" in content or "python_requires" in content
            ), "pyproject.toml missing Python version requirement"

            # Verify it specifies 3.12
            assert (
                "3.12" in content
            ), "pyproject.toml does not specify Python 3.12 requirement"

    @given(st.text(min_size=1, max_size=100))
    @settings(max_examples=10)
    def test_requirements_file_format_robustness(self, package_name: str):
        """
        Property: Requirements file should be parseable
        *For any* valid package name format, the requirements.txt should be
        properly formatted and parseable.
        """
        repo_root = Path(__file__).parent.parent
        requirements_txt = repo_root / "requirements.txt"

        if not requirements_txt.exists():
            pytest.skip("requirements.txt not found")

        content = requirements_txt.read_text()
        lines = content.split("\n")

        # Verify each non-comment, non-empty line has valid format
        for line in lines:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue

            # Should contain package name (may have version specifier)
            # Valid formats: package, package==version, package>=version, etc.
            assert (
                len(stripped) > 0
            ), f"Empty dependency line found: '{line}'"

            # Should not have invalid characters at start
            assert not stripped.startswith(
                ("=", ">", "<", "!")
            ), f"Invalid dependency format: '{stripped}'"

    def test_deployment_files_compatibility(self):
        """
        Verify that deployment files are compatible with Posit Connect.
        """
        repo_root = Path(__file__).parent.parent

        # Check for requirements.txt (preferred for Posit Connect)
        requirements_txt = repo_root / "requirements.txt"
        assert (
            requirements_txt.exists()
        ), "requirements.txt required for Posit Connect deployment"

        # Verify Python version in requirements or pyproject
        pyproject_toml = repo_root / "pyproject.toml"
        if pyproject_toml.exists():
            content = pyproject_toml.read_text()
            # Should specify Python 3.12
            assert (
                ">=3.12" in content or "3.12" in content
            ), "Python 3.12 requirement not clearly specified"
