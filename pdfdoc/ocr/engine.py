"""Обёртка над Tesseract: проверка установки, распознавание с координатами слов."""

from __future__ import annotations

import logging
import os
import re
import shutil
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import numpy as np
from PIL import Image

from pdfdoc.config import PipelineConfig
from pdfdoc.errors import TesseractNotAvailableError

log = logging.getLogger("pdfdoc")

# Tesseract по умолчанию запускает OpenMP-потоки на каждый вызов. Для сотен мелких
# вызовов (ячейки таблиц) это медленнее, чем один поток: на 2 ядрах ~25% выигрыша.
# setdefault — чтобы приложение могло задать своё значение.
os.environ.setdefault("OMP_THREAD_LIMIT", "1")

_WINDOWS_CANDIDATES = (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe",
    r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
)


@dataclass
class Word:
    text: str
    conf: float  # 0..100
    left: int
    top: int
    width: int
    height: int
    block: int
    par: int
    line: int

    @property
    def right(self) -> int:
        return self.left + self.width

    @property
    def bottom(self) -> int:
        return self.top + self.height


class Tesseract:
    """Один экземпляр на документ; потокобезопасен (каждый вызов — отдельный процесс)."""

    _checked: dict[tuple, tuple[str, set[str]]] = {}
    _lock = threading.Lock()

    def __init__(self, cfg: PipelineConfig):
        import pytesseract

        self._pt = pytesseract
        self.cfg = cfg
        self.lang = cfg.ocr_lang
        self.timeout = cfg.tesseract_timeout_s or 0
        cmd = cfg.tesseract_cmd or os.environ.get("TESSERACT_CMD")
        if not cmd:
            cmd = shutil.which("tesseract")
        if not cmd:
            cmd = next((c for c in _WINDOWS_CANDIDATES if Path(c).exists()), None)
        self.cmd = cmd
        self.tessdata = cfg.tessdata_dir
        self._ready = False
        self.version = ""
        self.languages: set[str] = set()

    # ------------------------------------------------------------ проверка
    def ensure(self) -> None:
        """Проверить бинарник и языки; при проблеме — понятная ошибка, а не пустой результат."""
        if self._ready:
            return
        if not self.cmd:
            raise TesseractNotAvailableError(
                "Tesseract не найден. Установите tesseract-ocr (см. REQUIREMENTS.md) "
                "или укажите путь в PipelineConfig.tesseract_cmd / переменной TESSERACT_CMD."
            )
        key = (self.cmd, self.tessdata, self.lang)
        with self._lock:
            if key not in self._checked:
                self._pt.pytesseract.tesseract_cmd = self.cmd
                try:
                    version = str(self._pt.get_tesseract_version())
                    langs = set(self._pt.get_languages(config=self._base_config()))
                except Exception as exc:  # noqa: BLE001
                    raise TesseractNotAvailableError(
                        f"Не удалось запустить Tesseract ({self.cmd}): {exc}"
                    ) from exc
                self._checked[key] = (version, langs)
            self.version, self.languages = self._checked[key]
        major = re.match(r"(\d+)", self.version)
        if major and int(major.group(1)) < 4:
            raise TesseractNotAvailableError(
                f"Нужен Tesseract 4+ (LSTM), установлен {self.version}"
            )
        missing = [l for l in self.lang.split("+") if l not in self.languages]
        if missing:
            raise TesseractNotAvailableError(
                f"Нет языковых данных Tesseract: {', '.join(missing)}. Установлены: "
                f"{', '.join(sorted(self.languages)) or 'нет'}. Debian/Ubuntu: "
                f"apt install {' '.join('tesseract-ocr-' + m for m in missing)}"
            )
        self._ready = True

    @property
    def has_osd(self) -> bool:
        return "osd" in self.languages

    def _base_config(self) -> str:
        return f'--tessdata-dir "{self.tessdata}"' if self.tessdata else ""

    # ------------------------------------------------------------ вызовы
    def _config(self, psm: int, whitelist: Optional[str]) -> str:
        parts = [self._base_config(), f"--psm {psm}"]
        if whitelist:
            parts.append(f"-c tessedit_char_whitelist={whitelist}")
        return " ".join(p for p in parts if p)

    def words(
        self,
        img: Union[np.ndarray, Image.Image],
        psm: int,
        whitelist: Optional[str] = None,
        lang: Optional[str] = None,
    ) -> list[Word]:
        """Слова с координатами (пиксели изображения) и уверенностью 0..100."""
        self.ensure()
        pil = Image.fromarray(img) if isinstance(img, np.ndarray) else img
        data = self._pt.image_to_data(
            pil,
            lang=lang or self.lang,
            config=self._config(psm, whitelist),
            output_type=self._pt.Output.DICT,
            timeout=self.timeout,
        )
        out: list[Word] = []
        for i, text in enumerate(data["text"]):
            t = (text or "").strip()
            conf = float(data["conf"][i])
            if not t or conf < 0:
                continue
            out.append(
                Word(
                    text=t,
                    conf=conf,
                    left=int(data["left"][i]),
                    top=int(data["top"][i]),
                    width=int(data["width"][i]),
                    height=int(data["height"][i]),
                    block=int(data["block_num"][i]),
                    par=int(data["par_num"][i]),
                    line=int(data["line_num"][i]),
                )
            )
        return out

    def orientation(self, img: np.ndarray) -> Optional[tuple[int, float]]:
        """(градусов повернуть по часовой, уверенность) или None, если OSD не сработал."""
        self.ensure()
        if not self.has_osd:
            return None
        try:
            txt = self._pt.image_to_osd(
                Image.fromarray(img),
                config=f"{self._base_config()} --psm 0 -c min_characters_to_try=20".strip(),
                timeout=self.timeout,
            )
        except Exception as exc:  # noqa: BLE001 — мало текста для OSD: это нормально
            log.debug("OSD failed: %s", exc)
            return None
        rot = re.search(r"Rotate:\s*(\d+)", txt)
        conf = re.search(r"Orientation confidence:\s*([\d.]+)", txt)
        if not rot:
            return None
        return int(rot.group(1)) % 360, float(conf.group(1)) if conf else 0.0


def words_to_text(words: list[Word], line_sep: str = "\n") -> str:
    """Слова → текст: слова строки через пробел, строки через line_sep."""
    lines: dict[tuple[int, int, int], list[Word]] = {}
    for w in words:
        lines.setdefault((w.block, w.par, w.line), []).append(w)
    out = []
    for key in sorted(lines, key=lambda k: min(w.top for w in lines[k]) if k else 0):
        ws = sorted(lines[key], key=lambda w: w.left)
        out.append(" ".join(w.text for w in ws))
    return line_sep.join(out)


def words_confidence(words: list[Word]) -> Optional[float]:
    """Средняя уверенность слов (0..1), взвешенная по длине слова."""
    total = sum(len(w.text) for w in words)
    if not total:
        return None
    return sum(w.conf * len(w.text) for w in words) / total / 100.0
