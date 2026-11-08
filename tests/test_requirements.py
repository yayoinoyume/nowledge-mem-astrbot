"""依赖声明必须给 aiohttp 带上主版本上限，避免跟随 4.x 漂移。"""

from __future__ import annotations

from pathlib import Path

REQUIREMENTS = Path(__file__).resolve().parents[1] / "requirements.txt"


def test_requirements_declares_bounded_aiohttp() -> None:
    specs = [
        line.strip()
        for line in REQUIREMENTS.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    assert "aiohttp>=3.11,<4" in specs
    assert "aiohttp" not in specs
