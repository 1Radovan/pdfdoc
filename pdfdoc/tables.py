"""Внутреннее представление таблицы и её выгрузка в table_cells / table_md / html."""

from __future__ import annotations

import html as _html
from dataclasses import dataclass, field
from typing import Optional, Sequence

from pdfdoc.models import CellSpan


@dataclass
class CellData:
    row: int
    col: int
    rowspan: int = 1
    colspan: int = 1
    text: str = ""
    confidence: Optional[float] = None  # None — пусто или нет оценки (текстовый слой)


@dataclass
class TableData:
    n_rows: int
    n_cols: int
    cells: list[CellData] = field(default_factory=list)
    structure_penalty: float = 1.0  # <1, если сетку пришлось «чинить»

    # ------------------------------------------------------------ выгрузка
    def matrix(self, fill: str = "top_left") -> list[list[str]]:
        m = [["" for _ in range(self.n_cols)] for _ in range(self.n_rows)]
        for c in self.cells:
            if fill == "repeat":
                for r in range(c.row, c.row + c.rowspan):
                    for k in range(c.col, c.col + c.colspan):
                        m[r][k] = c.text
            else:
                m[c.row][c.col] = c.text
        return m

    def confidence_matrix(self) -> list[list[Optional[float]]]:
        m: list[list[Optional[float]]] = [
            [None for _ in range(self.n_cols)] for _ in range(self.n_rows)
        ]
        for c in self.cells:
            if c.confidence is not None:
                m[c.row][c.col] = round(c.confidence, 3)
        return m

    def unreadable_cells(self) -> int:
        """Ячейки, где есть изображение текста, но OCR ничего не прочитал."""
        return sum(1 for c in self.cells if not c.text and c.confidence is not None)

    def spans(self) -> list[CellSpan]:
        return [
            CellSpan(row=c.row, col=c.col, rowspan=c.rowspan, colspan=c.colspan)
            for c in self.cells
            if c.rowspan > 1 or c.colspan > 1
        ]

    def covered(self) -> set[tuple[int, int]]:
        """Позиции сетки, перекрытые объединёнными ячейками (кроме левой верхней)."""
        out: set[tuple[int, int]] = set()
        for c in self.cells:
            for r in range(c.row, c.row + c.rowspan):
                for k in range(c.col, c.col + c.colspan):
                    if (r, k) != (c.row, c.col):
                        out.add((r, k))
        return out

    def to_markdown(self, fill: str = "top_left") -> str:
        m = self.matrix(fill)
        if not m:
            return ""

        def esc(s: str) -> str:
            return s.replace("|", "\\|").replace("\n", "<br>")

        lines = ["| " + " | ".join(esc(x) for x in m[0]) + " |"]
        lines.append("|" + "|".join("---" for _ in m[0]) + "|")
        for row in m[1:]:
            lines.append("| " + " | ".join(esc(x) for x in row) + " |")
        return "\n".join(lines)

    def to_html(self) -> str:
        """HTML с rowspan/colspan — единственное представление, где объединения не теряются."""
        starts = {(c.row, c.col): c for c in self.cells}
        covered = self.covered()
        out = ["<table>"]
        for r in range(self.n_rows):
            tag = "th" if r == 0 else "td"
            tds = []
            for k in range(self.n_cols):
                if (r, k) in covered:
                    continue
                c = starts.get((r, k))
                attrs = ""
                text = ""
                if c is not None:
                    if c.rowspan > 1:
                        attrs += f' rowspan="{c.rowspan}"'
                    if c.colspan > 1:
                        attrs += f' colspan="{c.colspan}"'
                    text = _html.escape(c.text).replace("\n", "<br>")
                tds.append(f"<{tag}{attrs}>{text}</{tag}>")
            out.append("<tr>" + "".join(tds) + "</tr>")
        out.append("</table>")
        return "".join(out)

    def all_text(self) -> str:
        return " ".join(c.text for c in self.cells if c.text)

    def confidence(self) -> Optional[float]:
        """Средняя уверенность по распознанным ячейкам (нечитаемые = 0) с учётом
        штрафа за структуру; None — если оценок нет (таблица из текстового слоя)."""
        vals = [c.confidence for c in self.cells if c.confidence is not None]
        if not vals:
            return None
        return max(0.0, min(1.0, sum(vals) / len(vals) * self.structure_penalty))

    def empty_cells_by_column(self, skip_header: bool = True) -> dict[int, int]:
        """Число пустых (не перекрытых объединением) ячеек в каждой колонке.

        Колонки, пустые целиком, не включаются: это скорее разделитель, чем пропуск.
        """
        m = self.matrix("top_left")
        covered = self.covered()
        start = 1 if skip_header and self.n_rows > 1 else 0
        out: dict[int, int] = {}
        for k in range(self.n_cols):
            cells = [(r, k) for r in range(start, self.n_rows) if (r, k) not in covered]
            empty = sum(1 for r, kk in cells if not m[r][kk].strip())
            if 0 < empty < len(cells):
                out[k] = empty
        return out


# ---------------------------------------------------------------- геометрия


def _cluster(values: Sequence[float], tol: float) -> list[float]:
    """Склеить близкие координаты (границы соседних ячеек совпадают не идеально)."""
    out: list[float] = []
    for v in sorted(values):
        if out and v - out[-1] <= tol:
            out[-1] = (out[-1] + v) / 2
        else:
            out.append(v)
    return out


def _index(bounds: list[float], v: float) -> int:
    return min(range(len(bounds)), key=lambda i: abs(bounds[i] - v))


def table_from_cell_bboxes(
    bboxes: Sequence[tuple[float, float, float, float]],
    texts: Sequence[str],
    confidences: Optional[Sequence[Optional[float]]] = None,
    tol: float = 2.0,
) -> Optional[TableData]:
    """Собрать таблицу из прямоугольников ячеек (объединённые ячейки — один bbox).

    Строки и колонки сетки восстанавливаются по всем границам ячеек; rowspan и
    colspan считаются по тому, сколько границ сетки перекрывает ячейка.
    """
    if not bboxes:
        return None
    xs = _cluster([b[0] for b in bboxes] + [b[2] for b in bboxes], tol)
    ys = _cluster([b[1] for b in bboxes] + [b[3] for b in bboxes], tol)
    n_cols, n_rows = len(xs) - 1, len(ys) - 1
    if n_cols < 1 or n_rows < 1:
        return None
    t = TableData(n_rows=n_rows, n_cols=n_cols)
    occupied: set[tuple[int, int]] = set()
    for i, (b, text) in enumerate(zip(bboxes, texts)):
        c0, c1 = _index(xs, b[0]), _index(xs, b[2])
        r0, r1 = _index(ys, b[1]), _index(ys, b[3])
        if c1 <= c0 or r1 <= r0:
            continue
        span = {(r, k) for r in range(r0, r1) for k in range(c0, c1)}
        if span & occupied:  # пересечение ячеек — берём первую
            t.structure_penalty = min(t.structure_penalty, 0.9)
            continue
        occupied |= span
        conf = confidences[i] if confidences is not None else None
        t.cells.append(
            CellData(row=r0, col=c0, rowspan=r1 - r0, colspan=c1 - c0, text=text, confidence=conf)
        )
    # дыры в сетке (ячейки без bbox) — пустые 1x1
    for r in range(n_rows):
        for k in range(n_cols):
            if (r, k) not in occupied:
                t.cells.append(CellData(row=r, col=k))
    t.cells.sort(key=lambda c: (c.row, c.col))
    return t
