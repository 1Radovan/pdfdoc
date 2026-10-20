"""Генерация тестовых PDF: текстовый слой, сканы (перекос, шум, поворот), смешанный."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pymupdf
from PIL import Image, ImageFilter

FONT_DIRS = [
    Path("/usr/share/fonts/truetype/liberation"),
    Path("/usr/share/fonts/truetype/dejavu"),
    Path("C:/Windows/Fonts"),
]


def _font(name_options: list[str]) -> str:
    for d in FONT_DIRS:
        for n in name_options:
            p = d / n
            if p.exists():
                return str(p)
    raise FileNotFoundError(f"нет шрифта с кириллицей: {name_options}")


REGULAR = _font(["LiberationSerif-Regular.ttf", "DejaVuSerif.ttf", "times.ttf"])
BOLD = _font(["LiberationSerif-Bold.ttf", "DejaVuSerif-Bold.ttf", "timesbd.ttf"])

PEOPLE = [
    ("1", "Иванов И.И.", "770123456701", "Менеджер", "1000.00"),
    ("2", "Петров П.П.", "770123456702", "Бухгалтер", "2000.00"),
    ("3", "Сидоров С.С.", "770123456703", "Инженер", "1500.00"),
]


def _text(page, x, y, s, size=11, bold=False):
    page.insert_text((x, y), s, fontsize=size, fontname="FB" if bold else "FR",
                     fontfile=BOLD if bold else REGULAR)


def _center(page, x0, x1, y0, y1, s, size=10, bold=False):
    font = pymupdf.Font(fontfile=BOLD if bold else REGULAR)
    w = font.text_length(s, fontsize=size)
    _text(page, (x0 + x1) / 2 - w / 2, (y0 + y1) / 2 + size * 0.35, s, size, bold)


def _grid(page, xs, ys, merges=()):
    """Сетка таблицы; merges — [(r0, c0, r1, c1)] объединения (r1/c1 не включительно)."""
    hidden_h = set()  # (row_boundary, col) — горизонтальных линий нет внутри объединения
    hidden_v = set()  # (col_boundary, row)
    for r0, c0, r1, c1 in merges:
        for r in range(r0 + 1, r1):
            for c in range(c0, c1):
                hidden_h.add((r, c))
        for c in range(c0 + 1, c1):
            for r in range(r0, r1):
                hidden_v.add((c, r))
    for ri, y in enumerate(ys):
        for ci in range(len(xs) - 1):
            if (ri, ci) not in hidden_h:
                page.draw_line((xs[ci], y), (xs[ci + 1], y), width=0.8)
    for ci, x in enumerate(xs):
        for ri in range(len(ys) - 1):
            if (ci, ri) not in hidden_v:
                page.draw_line((x, ys[ri]), (x, ys[ri + 1]), width=0.8)


def build_text_pdf(path: Path) -> Path:
    doc = pymupdf.open()
    # ---- страница 1
    p = doc.new_page(width=595, height=842)
    _text(p, 72, 70, "ВЕДОМОСТЬ начисления заработной платы за январь 2025 г.", 14, bold=True)
    _text(p, 72, 110, 'Организация: ООО "Ромашка", ИНН 7701234567, КПП 770101001', 11)
    _text(p, 72, 128, "Подразделение: бухгалтерия, табельный учёт ведётся по форме Т-12.", 11)
    xs = [72, 100, 210, 330, 440, 523]
    ys = [145 + 22 * i for i in range(6)]
    _grid(p, xs, ys)
    head = ["№", "ФИО", "ИНН", "Должность", "Сумма"]
    for c, h in enumerate(head):
        _center(p, xs[c], xs[c + 1], ys[0], ys[1], h, 10, bold=True)
    # у 4-й строки пустая «Должность» — для предупреждения о пустых ячейках
    rows = PEOPLE + [("4", "Козлов К.К.", "770123456704", "", "0.00")]
    for r, row in enumerate(rows, start=1):
        for c, v in enumerate(row):
            if v:
                _center(p, xs[c], xs[c + 1], ys[r], ys[r + 1], v, 10)

    _text(p, 72, 300, "Таблица 2. Удержания по видам", 12, bold=True)
    xs2 = [72, 200, 300, 400, 523]
    ys2 = [310, 332, 354, 376, 398]
    _grid(p, xs2, ys2, merges=[(1, 0, 3, 1), (0, 1, 1, 3)])
    _center(p, xs2[0], xs2[1], ys2[0], ys2[1], "Вид удержания", 10, True)
    _center(p, xs2[1], xs2[3], ys2[0], ys2[1], "Период", 10, True)
    _center(p, xs2[3], xs2[4], ys2[0], ys2[1], "Сумма", 10, True)
    _center(p, xs2[0], xs2[1], ys2[1], ys2[3], "НДФЛ", 10)
    _center(p, xs2[1], xs2[2], ys2[1], ys2[2], "январь", 10)
    _center(p, xs2[2], xs2[3], ys2[1], ys2[2], "2025", 10)
    _center(p, xs2[3], xs2[4], ys2[1], ys2[2], "585.00", 10)
    _center(p, xs2[1], xs2[2], ys2[2], ys2[3], "февраль", 10)
    _center(p, xs2[2], xs2[3], ys2[2], ys2[3], "2025", 10)
    _center(p, xs2[3], xs2[4], ys2[2], ys2[3], "0.00", 10)
    _center(p, xs2[0], xs2[1], ys2[3], ys2[4], "Алименты", 10)
    _center(p, xs2[1], xs2[2], ys2[3], ys2[4], "январь", 10)
    _center(p, xs2[2], xs2[3], ys2[3], ys2[4], "2025", 10)
    _center(p, xs2[3], xs2[4], ys2[3], ys2[4], "250.00", 10)
    _text(p, 72, 430, "1. Начисления произведены согласно штатному расписанию.", 11)
    _text(p, 72, 448, "2. Выплата производится не позднее 10 февраля 2025 г.", 11)
    _text(p, 260, 810, "Стр. 1 из 2", 9)
    # ---- страница 2
    p = doc.new_page(width=595, height=842)
    _text(p, 72, 70, "Итого к выплате: 4500.00 руб.", 12)
    _text(p, 72, 230, "Главный бухгалтер ______________ / Смирнова А.В. /", 11)
    _text(p, 72, 270, "Руководитель ______________ / Орлов Д.Е. /", 11)
    _text(p, 260, 810, "Стр. 2 из 2", 9)
    doc.save(path)
    doc.close()
    return path


def build_content_rotated_pdf(src: Path, path: Path, rotate: int) -> Path:
    """Книжные листы, вставленные в альбомные поворотом содержимого (без /Rotate)."""
    s = pymupdf.open(src)
    doc = pymupdf.open()
    for i in range(len(s)):
        page = doc.new_page(width=842, height=595)
        page.show_pdf_page(page.rect, s, i, rotate=rotate)
    doc.save(path)
    doc.close()
    s.close()
    return path


def build_rotated_table_pdf(path: Path, turn: int = 90) -> Path:
    """Обычный лист с заголовком и абзацем, а таблица на нём повёрнута:
    turn=90 — текст таблицы идёт снизу вверх, 270 — сверху вниз."""
    doc = pymupdf.open()
    p = doc.new_page(width=595, height=842)
    _text(p, 72, 70, "СВЕДЕНИЯ о сотрудниках подразделения", 14, bold=True)
    _text(p, 72, 100, "Таблица приведена в альбомной ориентации.", 11)
    us = [0, 28, 138, 258, 368, 451]  # u — вдоль строк текста таблицы, v — поперёк
    vs = [22 * i for i in range(6)]

    def pt(u, v):
        if turn == 90:
            return pymupdf.Point(150 + v, 650 - u)
        return pymupdf.Point(260 - v, 199 + u)

    for v in vs:
        p.draw_line(pt(us[0], v), pt(us[-1], v), width=0.8)
    for u in us:
        p.draw_line(pt(u, vs[0]), pt(u, vs[-1]), width=0.8)
    font = pymupdf.Font(fontfile=REGULAR)
    rows = [["№", "ФИО", "ИНН", "Должность", "Сумма"]] + [list(r) for r in PEOPLE] + [
        ["4", "Козлов К.К.", "770123456704", "", "0.00"]
    ]
    for r, row in enumerate(rows):
        for c, s in enumerate(row):
            if not s:
                continue
            w = font.text_length(s, fontsize=10)
            u = (us[c] + us[c + 1]) / 2 - w / 2
            v = (vs[r] + vs[r + 1]) / 2 + 10 * 0.35
            p.insert_text(pt(u, v), s, fontsize=10, fontname="FR", fontfile=REGULAR, rotate=turn)
    doc.save(path)
    doc.close()
    return path


def page_images(pdf_path: Path, dpi: int = 300) -> list[Image.Image]:
    doc = pymupdf.open(pdf_path)
    out = []
    for page in doc:
        pix = page.get_pixmap(dpi=dpi, colorspace=pymupdf.csGRAY)
        out.append(Image.frombytes("L", (pix.width, pix.height), pix.samples))
    doc.close()
    return out


def degrade(img: Image.Image, angle: float = 0.0, noise: float = 0.0, blur: float = 0.0,
            seed: int = 1, shear: float = 0.0) -> Image.Image:
    """shear — сдвиг листа (град.): вертикали наклоняются, горизонтали остаются ровными."""
    if angle:
        img = img.rotate(angle, resample=Image.Resampling.BICUBIC, fillcolor=255)
    if shear:
        k = float(np.tan(np.radians(shear)))
        img = img.transform(img.size, Image.Transform.AFFINE, (1, -k, k * img.height / 2, 0, 1, 0),
                            resample=Image.Resampling.BICUBIC, fillcolor=255)
    if blur:
        img = img.filter(ImageFilter.GaussianBlur(blur))
    if noise:
        a = np.asarray(img).astype(np.float32)
        a += np.random.default_rng(seed).normal(0, noise, a.shape)
        img = Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))
    return img


def images_to_pdf(images: list[Image.Image], path: Path, dpi: int = 300) -> Path:
    """PDF только из картинок — как с офисного сканера (без текстового слоя)."""
    doc = pymupdf.open()
    for img in images:
        w_pt, h_pt = img.width * 72 / dpi, img.height * 72 / dpi
        page = doc.new_page(width=w_pt, height=h_pt)
        import io

        buf = io.BytesIO()
        img.convert("L").save(buf, format="PNG")
        page.insert_image(page.rect, stream=buf.getvalue())
    doc.save(path)
    doc.close()
    return path


def build_all(folder: Path) -> dict[str, Path]:
    folder.mkdir(parents=True, exist_ok=True)
    text_pdf = build_text_pdf(folder / "vedomost_text.pdf")
    imgs = page_images(text_pdf)
    scan = images_to_pdf(
        [degrade(im, angle=0.8, noise=8, blur=0.6, seed=i) for i, im in enumerate(imgs)],
        folder / "vedomost_scan.pdf",
    )
    sheared = images_to_pdf(
        [degrade(im, shear=2.0, noise=6, blur=0.6, seed=i) for i, im in enumerate(imgs)],
        folder / "vedomost_sheared.pdf",
    )
    rot = images_to_pdf([imgs[0].rotate(90, expand=True)], folder / "vedomost_rot90.pdf")
    rot180 = images_to_pdf([imgs[0].rotate(180)], folder / "vedomost_rot180.pdf")
    rot270 = images_to_pdf([imgs[0].rotate(-90, expand=True)], folder / "vedomost_rot270.pdf")
    text_rot90 = build_content_rotated_pdf(text_pdf, folder / "vedomost_text_rot90.pdf", 90)
    text_rot270 = build_content_rotated_pdf(text_pdf, folder / "vedomost_text_rot270.pdf", 270)
    rotated_table = build_rotated_table_pdf(folder / "rotated_table.pdf", 90)
    rotated_table_270 = build_rotated_table_pdf(folder / "rotated_table_270.pdf", 270)
    rotated_table_scans = {
        f"rotated_table{sfx}_scan": images_to_pdf(
            [degrade(im, angle=0.5, noise=6, blur=0.6) for im in page_images(src)],
            folder / f"rotated_table{sfx}_scan.pdf",
        )
        for sfx, src in (("", rotated_table), ("_270", rotated_table_270))
    }
    mixed_doc = pymupdf.open(text_pdf)
    scan_doc = pymupdf.open(scan)
    mixed = pymupdf.open()
    mixed.insert_pdf(mixed_doc, from_page=0, to_page=0)
    mixed.insert_pdf(scan_doc, from_page=1, to_page=1)
    mixed.save(folder / "vedomost_mixed.pdf")
    for d in (mixed_doc, scan_doc, mixed):
        d.close()
    return {
        "text": text_pdf,
        "scan": scan,
        "sheared": sheared,
        "rot90": rot,
        "rot180": rot180,
        "rot270": rot270,
        "text_rot90": text_rot90,
        "text_rot270": text_rot270,
        "rotated_table": rotated_table,
        "rotated_table_270": rotated_table_270,
        **rotated_table_scans,
        "mixed": folder / "vedomost_mixed.pdf",
    }


if __name__ == "__main__":
    import sys

    for k, v in build_all(Path(sys.argv[1])).items():
        print(k, v)
