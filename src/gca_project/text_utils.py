import re
from typing import Iterable


CHINESE_PUNCTUATION_MAP = str.maketrans(
    {
        "。": ".",
        "，": ",",
        "；": ";",
        "：": ":",
        "？": "?",
        "！": "!",
        "（": "(",
        "）": ")",
        "【": "[",
        "】": "]",
        "《": "<",
        "》": ">",
        "“": '"',
        "”": '"',
        "‘": "'",
        "’": "'",
        "、": ",",
        "—": "-",
        "－": "-",
        "…": "...",
    }
)


def normalize_text(text: object, remove_spaces: bool = False) -> str:
    """Normalize text before tokenization or sentence splitting."""
    value = "" if text is None else str(text)
    value = value.replace("\r", " ").replace("\n", " ").replace("\t", " ")
    value = value.translate(CHINESE_PUNCTUATION_MAP)
    value = re.sub(r"\s+", "" if remove_spaces else " ", value)
    return value.strip()


def remove_after_last_keyword(text: object, keyword: str = "各位代表") -> str:
    """Remove closing report text after the last keyword occurrence."""
    value = normalize_text(text)
    last_pos = value.rfind(keyword)
    if last_pos == -1:
        return value

    backward_match = re.search(r"[.!?]", value[:last_pos][::-1])
    if backward_match:
        end_pos = last_pos - backward_match.start() - 1
        return value[: end_pos + 1]
    return value[:last_pos]


def split_sentences(text: object, min_length: int = 3) -> list[str]:
    """Split Chinese/English text into sentence-like units."""
    value = normalize_text(text)
    sentences = re.split(r"[.!?;。！？；]", value)
    return [sentence.strip() for sentence in sentences if len(sentence.strip()) >= min_length]


def ensure_columns(columns: Iterable[str], required: Iterable[str]) -> None:
    missing = [name for name in required if name not in columns]
    if missing:
        raise ValueError(f"Missing required column(s): {', '.join(missing)}")
