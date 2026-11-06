"""注入内容必须标记为临时内容。"""

from __future__ import annotations

from core.parts import make_temp_parts


class FakePart:
    def __init__(self, text: str) -> None:
        self.text = text
        self.temp = False

    def mark_as_temp(self) -> "FakePart":
        self.temp = True
        return self


def test_make_temp_parts_marks_every_part() -> None:
    parts = make_temp_parts(["a", "", "b"], part_cls=FakePart)

    assert [part.text for part in parts] == ["a", "b"]
    assert all(part.temp for part in parts)
