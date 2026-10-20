"""Нормализация текста, язык, оценка «битости» текстового слоя."""

from __future__ import annotations

import re
import unicodedata
from typing import Optional

_SPACE_RE = re.compile(r"[ \t    ]+")
_MULTI_NL_RE = re.compile(r"\n{3,}")
_CYR_RE = re.compile(r"[А-Яа-яЁё]")
_LAT_RE = re.compile(r"[A-Za-z]")
_WORD_RE = re.compile(r"\w+", re.UNICODE)
_DECIMAL_RE = re.compile(r"-?\d{1,3}(?: \d{3})*[.,]\d*|-?\d+[.,]\d*")
_WRAP_DIGITS_RE = re.compile(r"\d{1,3}")

# Латинские буквы, неотличимые от кириллических (частая ошибка OCR с rus+eng)
_LAT2CYR = str.maketrans(
    {
        "A": "А", "B": "В", "C": "С", "E": "Е", "H": "Н", "K": "К", "M": "М",
        "O": "О", "P": "Р", "T": "Т", "X": "Х", "Y": "У",
        "a": "а", "c": "с", "e": "е", "o": "о", "p": "р", "x": "х", "y": "у",
    }
)


def clean_line(text: str) -> str:
    """Схлопнуть пробелы внутри строки."""
    return _SPACE_RE.sub(" ", text).strip()


def clean_text(text: str) -> str:
    """Схлопнуть пробелы, сохранив переводы строк между строками блока."""
    lines = [clean_line(x) for x in text.replace("\r", "").split("\n")]
    out = "\n".join(lines).strip()
    return _MULTI_NL_RE.sub("\n\n", out)


def join_cell_lines(text: str) -> str:
    """Многострочный текст ячейки → одна строка.

    Перенос с дефисом («вертикально-\\nфрезерный», «ГОСТ 17473-\\n80»,
    «Санкт-\\nПетербург») склеивается без пробела, дефис сохраняется: убрать
    его безопасно нельзя (бывают и настоящие дефисы). Дефис после пробела —
    это тире, после него пробел остаётся.
    Число, перенесённое в узкой ячейке («12 183.3\\n3»), склеивается без пробела.
    """
    lines = [clean_line(x) for x in text.replace("\r", "").split("\n")]
    lines = [x for x in lines if x]
    out = ""
    for ln in lines:
        if not out:
            out = ln
        elif out.endswith("-") and (ln[:1].islower() or (out[-2:-1].isalnum() and ln[:1].isalnum())):
            out += ln
        elif _DECIMAL_RE.fullmatch(out) and _WRAP_DIGITS_RE.fullmatch(ln) and len(ln) < len(out) / 2:
            out += ln
        else:
            out += " " + ln
    return out


_LIST_ITEM_RE = re.compile(r"\s*(?:\d{1,2}(?:\.\d{1,2})*[.)]|[•●▪–—-])\s+[А-ЯЁA-Z«\"]")


def starts_list_item(line: str) -> bool:
    """Строка начинается с пункта списка: «2. Срок …», «3.1. Оплата …», «4) …», «• …».

    Номер пункта без точки («5.1 Договора») — не пункт: так заканчивается
    ссылка на пункт, перенесённая на новую строку.
    """
    return bool(_LIST_ITEM_RE.match(line))


def fix_homoglyphs(text: str) -> str:
    """Латинские двойники → кириллица в словах, где кириллицы больше половины букв.

    «Cтанoк» (C и o латинские) → «Станок». Слова на латинице («LLC», «ISO»)
    и смешанные коды («A-100») не трогаются, если кириллицы в них меньше латиницы.
    """
    if not text or not _LAT_RE.search(text) or not _CYR_RE.search(text):
        return text
    out_lines = []
    for line in text.split("\n"):
        cyr_l = len(_CYR_RE.findall(line))
        lat_l = len(_LAT_RE.findall(line))
        line_is_ru = cyr_l and cyr_l >= 0.7 * (cyr_l + lat_l)

        def _fix(m: re.Match) -> str:
            w = m.group(0)
            cyr = len(_CYR_RE.findall(w))
            lat = len(_LAT_RE.findall(w))
            if cyr and lat and cyr >= lat:
                return w.translate(_LAT2CYR)
            # инициалы и короткие слова («A.B.» → «А.В.») в русской строке,
            # если все латинские буквы имеют кириллического двойника
            if line_is_ru and not cyr and 0 < lat <= 3 and w.isalpha() and w.translate(_LAT2CYR) != w \
                    and not _LAT_RE.search(w.translate(_LAT2CYR)):
                return w.translate(_LAT2CYR)
            return w

        out_lines.append(_WORD_RE.sub(_fix, line))
    return "\n".join(out_lines)


_NPP_RE = re.compile(r"^№\s*[пnлтПN/|1l\\]{1,4}\.?$")
_BRACKET_P_RE = re.compile(r"\[1(\d{2,})\b(?!\])")


def fix_ocr_confusions(text: str) -> str:
    """Безопасные исправления частых путаниц Tesseract (rus+eng).

    * «№ пп», «№ плт», «№ n/n» (ячейка целиком) → «№ п/п»;
    * «}» / «{» при незакрытой круглой скобке → «)» / «(»;
    * «[1» перед цифрами в русском тексте → «П» («[16330» → «П6330»).
    """
    if not text:
        return text
    t = text
    if _NPP_RE.match(t.strip()):
        return "№ п/п"
    if "}" in t and t.count("(") > t.count(")"):
        t = t.replace("}", ")", t.count("(") - t.count(")"))
    if "{" in t and t.count(")") > t.count("("):
        t = t.replace("{", "(", t.count(")") - t.count("("))
    if "[1" in t and _CYR_RE.search(t):
        t = _BRACKET_P_RE.sub(lambda m: "П" + m.group(1), t)
    return t


_MONEY_RE = re.compile(r"(?<![\d.,])(\d{1,3})((?:(?:[.,] ?| )\d{3})+)([.,])(\d{2})(?![\d.,])")
_GROUP_SEP_RE = re.compile(r"[.,] ?| ")


def fix_number_separators(text: str) -> str:
    """Разделители тысяч в суммах, испорченные OCR: «1. 550,00», «1, 490,00»,
    «1,490,00», «1.317 500,00» → «1 550,00», «1 490,00», «1 317 500,00».

    Сумма меняется, только если разделители явно не настоящие: после точки или
    запятой стоит пробел, разделитель тысяч совпадает с десятичным или они
    смешаны. Согласованные форматы («1.905,00», «1,905.00», «1 905,00») не трогаются.
    """
    def _fix(m: re.Match) -> str:
        seps = _GROUP_SEP_RE.findall(m.group(2))
        bad = (
            any(s.endswith(" ") and len(s) == 2 for s in seps)
            or any(s.strip() == m.group(3) for s in seps)
            or len(set(seps)) > 1
        )
        if not bad:
            return m.group(0)
        return m.group(1) + _GROUP_SEP_RE.sub(" ", m.group(2)) + m.group(3) + m.group(4)

    return _MONEY_RE.sub(_fix, text)


def detect_language(text: Optional[str]) -> Optional[str]:
    """'ru' / 'en' / 'mixed' по доле кириллицы и латиницы; None — если букв нет."""
    if not text:
        return None
    cyr = len(_CYR_RE.findall(text))
    lat = len(_LAT_RE.findall(text))
    total = cyr + lat
    if total == 0:
        return None
    if cyr / total >= 0.7:
        return "ru"
    if lat / total >= 0.7:
        return "en"
    return "mixed"


def garbage_ratio(text: str) -> float:
    """Доля подозрительных символов в тексте текстового слоя.

    Признаки битого слоя: U+FFFD, символы Private Use Area (шрифты без
    ToUnicode), управляющие символы, «(cid:NN)» и кракозябры из Latin-1
    («Ïðèâåò» вместо «Привет» — кириллица в неверной кодировке).
    """
    chars = [c for c in text if not c.isspace()]
    if not chars:
        return 0.0
    bad = 0
    for c in chars:
        o = ord(c)
        if c == "�" or 0xE000 <= o <= 0xF8FF or unicodedata.category(c) == "Cc":
            bad += 1
        elif 0xC0 <= o <= 0xFF and c not in "×÷":
            bad += 1
    bad += 5 * text.count("(cid:")
    return min(1.0, bad / len(chars))


_LAT_WORD_RE = re.compile(r"[A-Za-z]{2,}")
# без «or», «to», «no», «he»: так выглядят «от», «то», «по», «не», прочитанные латиницей
_EN_STOPWORDS = frozenset(
    "the and of in for is on with by this that be are from as at an it not we you will "
    "all shall".split()
)


def misread_cyrillic_ratio(text: str, min_words: int = 3) -> float:
    """Похож ли текст на кириллицу, прочитанную латиницей («Cqer Ha onnary»).

    Так выглядит невидимый слой чужого OCR, запущенного без русского языка.
    Признак: букв кириллицы почти нет, а латиница не похожа на английский
    (в английском тексте служебные слова — 20–30% слов, в таком слое — единицы).
    Возвращает долю латинских букв (0 — текст нормальный или его меньше min_words слов).
    """
    cyr = len(_CYR_RE.findall(text))
    lat = len(_LAT_RE.findall(text))
    words = [w.lower() for w in _LAT_WORD_RE.findall(text)]
    if len(words) < min_words or lat < 3 * cyr + 1:  # кириллицы ≥ 25% букв — слой русский
        return 0.0
    if sum(w in _EN_STOPWORDS for w in words) / len(words) >= 0.15:
        return 0.0
    return lat / (cyr + lat)


def upper_ratio(text: str) -> float:
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return 0.0
    return sum(1 for c in letters if c.isupper()) / len(letters)


def alnum_count(text: str) -> int:
    return sum(1 for c in text if c.isalnum())


def is_numeric_like(text: str) -> bool:
    """Ячейка похожа на число / сумму / дату («1 250,00», «-», «12.01.2025»)."""
    t = text.strip()
    if not t:
        return False
    digits = sum(c.isdigit() for c in t)
    if digits == 0:
        return False
    if re.fullmatch(r"[\d\s.,:;\-–—+%()/]+", t):
        return True
    return digits / len(t.replace(" ", "")) >= 0.7
