from __future__ import annotations


def needs_screen_context(text: str) -> bool:
    """Return True when an utterance explicitly refers to the desktop/screen.

    This is only a cheap routing hint: it decides whether to spend work on a
    fresh screen observation. It does not infer the meaning of the screen.
    """
    normalized = " ".join(str(text).lower().split())
    if not normalized:
        return False

    terms = (
        # Vietnamese
        "màn hình",
        "trên màn",
        "trên screen",
        "ảnh chụp màn hình",
        "screenshot",
        "m đang thấy",
        "mày đang thấy",
        "thấy gì",
        "nhìn màn hình",
        "nhìn cái này",
        "cái trên màn",
        "cái này trên màn",
        # English
        "screen",
        "desktop",
        "on my monitor",
        "on the monitor",
        "what do you see",
        "look at this",
        # Japanese
        "画面",
        "スクリーン",
        "デスクトップ",
        "何が見える",
    )
    return any(term in normalized for term in terms)
