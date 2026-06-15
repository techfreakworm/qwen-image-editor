"""L1 structural tests — assert the expected project layout exists.

Source modules (app.py, backend.py, etc.) are created by other agents;
this file only asserts their paths exist and that meta-file contents
are correct.  No heavy dependencies (torch, diffusers) required.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _path(relative: str) -> Path:
    return ROOT / relative


# ---------------------------------------------------------------------------
# Source module existence
# ---------------------------------------------------------------------------


def test_app_py_exists() -> None:
    assert _path("app.py").exists(), "app.py not found — run the app agent first"


def test_backend_py_exists() -> None:
    assert _path("backend.py").exists(), "backend.py not found"


def test_modes_py_exists() -> None:
    assert _path("modes.py").exists(), "modes.py not found"


def test_models_py_exists() -> None:
    assert _path("models.py").exists(), "models.py not found"


def test_ui_py_exists() -> None:
    assert _path("ui.py").exists(), "ui.py not found"


def test_theme_py_exists() -> None:
    assert _path("theme.py").exists(), "theme.py not found"


def test_tooltips_py_exists() -> None:
    assert _path("tooltips.py").exists(), "tooltips.py not found"


# ---------------------------------------------------------------------------
# Meta file existence
# ---------------------------------------------------------------------------


def test_requirements_txt_exists() -> None:
    assert _path("requirements.txt").exists()


def test_pyproject_toml_exists() -> None:
    assert _path("pyproject.toml").exists()


def test_readme_exists() -> None:
    assert _path("README.md").exists(), "README.md not found — run the readme agent first"


def test_setup_sh_exists() -> None:
    assert _path("setup.sh").exists()


def test_license_exists() -> None:
    assert _path("LICENSE").exists()


# ---------------------------------------------------------------------------
# Test infrastructure existence
# ---------------------------------------------------------------------------


def test_tests_init_exists() -> None:
    assert _path("tests/__init__.py").exists()


def test_tests_conftest_exists() -> None:
    assert _path("tests/conftest.py").exists()


def test_tests_test_scaffold_exists() -> None:
    assert _path("tests/test_scaffold.py").exists()


# ---------------------------------------------------------------------------
# pyproject.toml content
# ---------------------------------------------------------------------------


def test_pyproject_contains_py311() -> None:
    text = _path("pyproject.toml").read_text()
    assert "py311" in text, "pyproject.toml must declare target-version = 'py311'"


def test_pyproject_contains_gpu_marker_addopts() -> None:
    text = _path("pyproject.toml").read_text()
    assert "-m 'not gpu'" in text, "pyproject.toml addopts must include \"-m 'not gpu'\""


# ---------------------------------------------------------------------------
# requirements.txt content
# ---------------------------------------------------------------------------


def test_requirements_contains_gradio() -> None:
    req = _path("requirements.txt").read_text()
    assert "gradio" in req


def test_requirements_contains_diffusers() -> None:
    req = _path("requirements.txt").read_text()
    assert "diffusers" in req


def test_requirements_contains_peft() -> None:
    req = _path("requirements.txt").read_text()
    assert "peft" in req


def test_requirements_no_pinned_spaces() -> None:
    req = _path("requirements.txt").read_text()
    assert "spaces==" not in req, (
        "Do not pin 'spaces' — HF Spaces injects it automatically on ZeroGPU; "
        "a pin causes pip resolution failure at build time."
    )


# ---------------------------------------------------------------------------
# LICENSE content
# ---------------------------------------------------------------------------


def test_license_is_mit() -> None:
    text = _path("LICENSE").read_text()
    assert "MIT" in text, "LICENSE must be MIT"


def test_license_author() -> None:
    text = _path("LICENSE").read_text()
    assert "Mayank Gupta" in text, "LICENSE must name 'Mayank Gupta' as copyright holder"


# ---------------------------------------------------------------------------
# README.md content
# ---------------------------------------------------------------------------


def test_readme_frontmatter_sdk_gradio() -> None:
    text = _path("README.md").read_text()
    assert "sdk: gradio" in text, "README.md YAML frontmatter must contain 'sdk: gradio'"


def test_readme_frontmatter_preload_qwen() -> None:
    text = _path("README.md").read_text()
    assert "Qwen/Qwen-Image-Edit-2511" in text, "README.md preload_from_hub must reference 'Qwen/Qwen-Image-Edit-2511'"
