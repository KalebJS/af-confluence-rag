"""
Property-based tests for documentation presence and completeness.

Feature: confluence-rag-system, Property 37: Documentation presence
Validates: Requirements 10.5
"""

import os
from pathlib import Path
from hypothesis import given, strategies as st
import pytest


class TestDocumentationPresence:
    """Test that required documentation files exist and contain necessary sections."""

    def test_readme_exists(self):
        """
        Property 37: Documentation presence
        For any repository checkout, a README.md file should exist at the root
        containing sections for setup, architecture, and usage.
        
        Validates: Requirements 10.5
        """
        readme_path = Path("README.md")
        assert readme_path.exists(), "README.md file must exist at repository root"
        
        # Read README content
        content = readme_path.read_text()
        
        # Check for required sections
        required_sections = [
            "## Overview",
            "## Architecture",
            "## Installation",
            "## Configuration",
            "## Usage",
        ]
        
        for section in required_sections:
            assert section in content, f"README.md must contain '{section}' section"
    
    def test_readme_has_setup_instructions(self):
        """
        Verify README contains detailed setup instructions.
        
        Validates: Requirements 10.5
        """
        readme_path = Path("README.md")
        content = readme_path.read_text()
        
        # Check for setup-related keywords
        setup_keywords = [
            "install",
            "uv",
            "dependencies",
            "environment",
            "configuration",
        ]
        
        content_lower = content.lower()
        for keyword in setup_keywords:
            assert keyword in content_lower, f"README.md should mention '{keyword}' in setup instructions"
    
    def test_readme_has_architecture_diagram(self):
        """
        Verify README contains architecture information.
        
        Validates: Requirements 10.5
        """
        readme_path = Path("README.md")
        content = readme_path.read_text()
        
        # Check for architecture-related content
        assert "Architecture" in content, "README.md should have Architecture section"
        
        # Check for component descriptions
        components = [
            "Ingestion Service",
            "Query Interface",
            "Vector Database",
        ]
        
        for component in components:
            assert component in content, f"README.md should describe '{component}' component"
    
    def test_readme_has_usage_examples(self):
        """
        Verify README contains usage examples.
        
        Validates: Requirements 10.5
        """
        readme_path = Path("README.md")
        content = readme_path.read_text()
        
        # Check for usage examples
        usage_indicators = [
            "```bash",  # Code blocks
            "uv run",   # Command examples
            "Usage",    # Usage section
        ]
        
        for indicator in usage_indicators:
            assert indicator in content, f"README.md should contain usage examples with '{indicator}'"
    
    def test_project_structure_documented(self):
        """
        Verify README documents the project structure.
        
        Validates: Requirements 10.5
        """
        readme_path = Path("README.md")
        content = readme_path.read_text()
        
        # Check for project structure documentation
        key_directories = [
            "src/",
            "tests/",
            "config/",
            "scripts/",
            "docs/",
        ]
        
        for directory in key_directories:
            assert directory in content, f"README.md should document '{directory}' directory"
    
    def test_additional_documentation_exists(self):
        """
        Verify additional documentation files exist.
        
        Validates: Requirements 10.5
        """
        docs_dir = Path("docs")
        assert docs_dir.exists(), "docs/ directory must exist"
        
        # Check for specific documentation files
        expected_docs = [
            "docs/STREAMLIT_APP.md",
            "docs/POSIT_CONNECT_DEPLOYMENT.md",
        ]
        
        for doc_path in expected_docs:
            path = Path(doc_path)
            assert path.exists(), f"Documentation file '{doc_path}' must exist"
    
    def test_env_example_exists(self):
        """
        Verify .env.example file exists with required variables.
        
        Validates: Requirements 10.5
        """
        env_example_path = Path(".env.example")
        assert env_example_path.exists(), ".env.example file must exist"
        
        content = env_example_path.read_text()
        
        # Check for required environment variables
        required_vars = [
            "CONFLUENCE_BASE_URL",
            "CONFLUENCE_AUTH_TOKEN",
            "CONFLUENCE_SPACE_KEY",
        ]
        
        for var in required_vars:
            assert var in content, f".env.example must contain '{var}' variable"
    
    def test_pyproject_toml_exists(self):
        """
        Verify pyproject.toml exists with project metadata.
        
        Validates: Requirements 10.5
        """
        pyproject_path = Path("pyproject.toml")
        assert pyproject_path.exists(), "pyproject.toml file must exist"
        
        content = pyproject_path.read_text()
        
        # Check for essential sections
        essential_sections = [
            "[project]",
            "name",
            "version",
            "dependencies",
        ]
        
        for section in essential_sections:
            assert section in content, f"pyproject.toml must contain '{section}'"


class TestDocumentationCompleteness:
    """Test that documentation is complete and up-to-date."""
    
    @given(st.sampled_from([
        "README.md",
        "docs/STREAMLIT_APP.md",
        "docs/POSIT_CONNECT_DEPLOYMENT.md",
    ]))
    def test_documentation_files_not_empty(self, doc_file: str):
        """
        Property: Documentation files should not be empty
        For any documentation file, it should contain meaningful content (>100 chars).
        
        Validates: Requirements 10.5
        """
        doc_path = Path(doc_file)
        if doc_path.exists():
            content = doc_path.read_text()
            assert len(content) > 100, f"{doc_file} should contain substantial content"
    
    @given(st.sampled_from([
        "src/ingestion",
        "src/processing",
        "src/storage",
        "src/query",
        "src/models",
        "src/utils",
    ]))
    def test_modules_have_init_files(self, module_path: str):
        """
        Property: All Python modules should have __init__.py files
        For any module directory, it should contain an __init__.py file.
        
        Validates: Requirements 10.5
        """
        module_dir = Path(module_path)
        if module_dir.exists() and module_dir.is_dir():
            init_file = module_dir / "__init__.py"
            assert init_file.exists(), f"{module_path} should have __init__.py file"


class TestScriptsDocumentation:
    """Test that scripts are documented."""
    
    def test_scripts_readme_exists(self):
        """
        Verify scripts directory has a README.
        
        Validates: Requirements 10.5
        """
        scripts_readme = Path("scripts/README.md")
        assert scripts_readme.exists(), "scripts/README.md should exist"
        
        content = scripts_readme.read_text()
        
        # Check for script descriptions
        scripts = [
            "ingest.py",
            "run_app.py",
            "scheduled_sync.py",
        ]
        
        for script in scripts:
            assert script in content, f"scripts/README.md should document '{script}'"
    
    @given(st.sampled_from([
        "scripts/ingest.py",
        "scripts/run_app.py",
        "scripts/scheduled_sync.py",
        "scripts/setup_deployment.py",
        "scripts/health_check.py",
        "scripts/verify_setup.py",
    ]))
    def test_scripts_have_docstrings(self, script_path: str):
        """
        Property: Scripts should have module-level docstrings
        For any script file, it should contain a docstring explaining its purpose.
        
        Validates: Requirements 10.5
        """
        script_file = Path(script_path)
        if script_file.exists():
            content = script_file.read_text()
            
            # Check for docstring (triple quotes near the top)
            lines = content.split('\n')
            # Skip shebang and imports, look for docstring in first 20 lines
            docstring_found = False
            for i, line in enumerate(lines[:20]):
                if '"""' in line or "'''" in line:
                    docstring_found = True
                    break
            
            assert docstring_found, f"{script_path} should have a module-level docstring"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
