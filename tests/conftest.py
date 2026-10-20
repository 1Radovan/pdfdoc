from __future__ import annotations

import shutil
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import samples  # noqa: E402

HAS_TESSERACT = shutil.which("tesseract") is not None


@pytest.fixture(scope="session")
def sample_pdfs(tmp_path_factory) -> dict[str, Path]:
    return samples.build_all(tmp_path_factory.mktemp("samples"))


@pytest.fixture
def fresh_copy(tmp_path, sample_pdfs):
    """Копия тестового PDF в чистой папке (чтобы JSON писался туда)."""

    def _copy(key: str) -> Path:
        dst = tmp_path / sample_pdfs[key].name
        shutil.copy(sample_pdfs[key], dst)
        return dst

    return _copy


needs_tesseract = pytest.mark.skipif(not HAS_TESSERACT, reason="tesseract не установлен")
