"""Per-background text cleaning applied before chunking.

Called from pipeline.py::row_to_records() after normalize_text().
Each background gets only the cleaning it needs; others pass through unchanged.
"""

from __future__ import annotations

import re

# ── optional: traditional → simplified Chinese ────────────────────────────────
try:
    import zhconv as _zhconv

    def _to_simplified(text: str) -> str:
        return _zhconv.convert(text, "zh-hans")

except ImportError:
    def _to_simplified(text: str) -> str:  # graceful fallback: no-op
        return text


# ── backgrounds whose source is predominantly traditional Chinese ─────────────
_TRADITIONAL_BG = frozenset(
    ["classical_poetry", "zh_wikipedia", "historical_diary", "academic_humanities", "religious_text"]
)


# ── weibo_senti ───────────────────────────────────────────────────────────────
# Raw weibo posts contain @mentions, [emoji] tags, retweet chains, and numeric
# entry separators introduced by the dataset format.

_WEIBO_AT = re.compile(r"@[\w\-·]{1,30}")
_WEIBO_EMOJI = re.compile(r"\[[^\[\]\n]{1,12}\]")
_WEIBO_RETWEET = re.compile(r"//+@[^:：\n]{1,30}[：:]")
_WEIBO_ENTRY_NUM = re.compile(r"^\d{1,3}。\s*")
_WEIBO_HASHTAG = re.compile(r"#[^#\n]{1,30}#")


def _clean_weibo(text: str) -> str:
    text = _WEIBO_RETWEET.sub("", text)       # strip //@user: chains first
    text = _WEIBO_AT.sub("", text)             # strip remaining @mentions
    text = _WEIBO_EMOJI.sub("", text)          # strip [哈哈][赞] etc.
    text = _WEIBO_HASHTAG.sub("", text)        # strip #话题#
    text = _WEIBO_ENTRY_NUM.sub("", text.lstrip())  # strip leading "1。"
    text = re.sub(r"\s{2,}", " ", text).strip()
    return text


# ── lyrics ────────────────────────────────────────────────────────────────────
# The dataset prepends a 16-char hex hash + "#N。" to each song, and includes
# inline production credits (作词/作曲/编曲/演唱…) and section labels
# (主歌/副歌/过渡…).  After normalize_text() the text is a single line, so
# patterns must match inline rather than at line boundaries.

_LYRICS_HASH_PREFIX = re.compile(r"^[0-9a-f]{16}\s*#\d+。\s*", re.IGNORECASE)
# Inline credits: "作曲 郑冰冰" or "演唱：张三" – a credit keyword followed by
# a short name (Chinese chars or Latin word, up to ~20 chars).
_LYRICS_CREDIT = re.compile(
    r"(?:作词|作曲|编曲|演唱|和声|后期|制作人|出品|发行|录音|混音|母带|监制)"
    r"\s*[:：]?\s*[一-鿿\w·]{1,20}"
)
# Inline section markers: "主歌1 " / "副歌 " / "过渡2 " etc.
_LYRICS_SECTION = re.compile(
    r"(?:主歌|副歌|过渡|结尾|尾声|前奏|间奏|桥段|Bridge|Chorus|Verse)\s*\d*\s*[:：]?\s*",
    re.IGNORECASE,
)


def _clean_lyrics(text: str) -> str:
    text = _LYRICS_HASH_PREFIX.sub("", text)
    text = _LYRICS_CREDIT.sub("", text)
    text = _LYRICS_SECTION.sub("", text)
    text = re.sub(r"\s{2,}", " ", text).strip()
    return text


# ── psych_discourse ───────────────────────────────────────────────────────────
# The dataset mixes LLM roleplay system-prompt preambles (现在你是虚拟心理咨询师…)
# with actual counselling dialogues.  After normalize_text() all text is a
# single line, so we search inline for where the real conversation begins
# (first person-to-person greeting or problem statement after a sentence break).

_PSYCH_SYSPROMPT_START = re.compile(r"^(?:现在你是|你是一个|你是一位|你扮演)")
# Conversation starter: a greeting or personal disclosure that follows a
# sentence-ending punctuation or space (to avoid matching mid-sentence occurrences).
_PSYCH_CONV_MARKER = re.compile(
    r"(?<=[。！？ ])"
    r"(?:你好|您好|嗨|我最近|最近我|我想聊|我有点|我感觉|我觉得|我今天|我一直)"
)


def _clean_psych(text: str) -> str:
    if not _PSYCH_SYSPROMPT_START.match(text):
        return text.strip()
    m = _PSYCH_CONV_MARKER.search(text)
    if m and m.start() > 100:   # only cut if preamble is substantial
        text = text[m.start():]
    return text.strip()


# ── children_writing ──────────────────────────────────────────────────────────
# Some entries contain AI-generated JSON/code blocks (```json { … } ```) that
# slipped into the dataset; strip them entirely.

_CODE_BLOCK = re.compile(r"```[a-z]*\s*.*?```", re.DOTALL)
# Also strip bare JSON-looking objects: lines that are "{" or "}" only, or
# lines with "key": value structure
_JSON_ARTIFACT = re.compile(r'^\s*[{}\[\],]\s*$', re.MULTILINE)
_JSON_KV_LINE = re.compile(r'^\s*"[^"]{1,40}"\s*:\s*.{0,120},?\s*$', re.MULTILINE)


def _clean_children(text: str) -> str:
    text = _CODE_BLOCK.sub("", text)
    text = _JSON_ARTIFACT.sub("", text)
    text = _JSON_KV_LINE.sub("", text)
    text = re.sub(r"\s{2,}", " ", text).strip()
    return text


# ── public API ────────────────────────────────────────────────────────────────

def clean_background_text(bg_key: str, text: str) -> str:
    """Return cleaned text for the given background key.

    Keeps source semantics intact — removes only structural noise and artefacts
    introduced by dataset format, not content.  Called once per source row in
    row_to_records(), before chunking.
    """
    if not text:
        return text

    if bg_key == "weibo_senti":
        text = _clean_weibo(text)
    elif bg_key == "lyrics":
        text = _clean_lyrics(text)
    elif bg_key == "psych_discourse":
        text = _clean_psych(text)
    elif bg_key == "children_writing":
        text = _clean_children(text)

    if bg_key in _TRADITIONAL_BG:
        text = _to_simplified(text)

    return text
