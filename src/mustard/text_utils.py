"""Text cleaning. Sarcasm-preserving tokenization vs. classical TF-IDF cleanup."""

from __future__ import annotations

import re
import string

INTENSIFIERS = frozenset(
    """
    so very really extremely totally literally actually obviously clearly
    definitely absolutely certainly surely completely utterly highly
    incredibly amazingly ridiculously perfectly wonderful great fantastic
    awesome brilliant lovely nice just
    """.split()
)

HYPERBOLE = frozenset(
    """
    always never ever everyone nobody everything nothing forever
    million billion infinite dying dead kill love hate worst best
    """.split()
)

# Compact stoplist for the classical baseline only (sarcasm models keep function words).
STOPWORDS = frozenset(
    """
    a an the and or but if while of at by for to in on from with as is are
    was were be been being it this that these those i you he she we they
    me him her us them my your his our their
    """.split()
)


def preserve_text(text: str) -> str:
    """Keep casing and punctuation — both carry sarcasm signal."""
    return re.sub(r"\s+", " ", (text or "")).strip()


def aggressive_clean(text: str) -> str:
    """Lowercase, strip punctuation, drop stopwords, Porter-like suffix stem."""
    text = (text or "").lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    tokens = [t for t in text.split() if t and t not in STOPWORDS]
    return " ".join(_crude_stem(t) for t in tokens)


def _crude_stem(token: str) -> str:
    for suffix in ("ing", "ingly", "edly", "ed", "ly", "es", "s"):
        if len(token) > len(suffix) + 3 and token.endswith(suffix):
            return token[: -len(suffix)]
    return token


def tokenize_words(text: str) -> list[str]:
    return re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?|[0-9]+|[^\sA-Za-z0-9]", text or "")
