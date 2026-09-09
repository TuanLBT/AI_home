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
        # Explicit monitor references. These are intentionally included even
        # without the words "màn hình" because natural queries are often like
        # "màn bên phải có gì?" or "màn trái đang lỗi gì?".
        "màn bên trái",
        "màn bên phải",
        "màn trái",
        "màn phải",
        "monitor bên trái",
        "monitor bên phải",
        "screen bên trái",
        "screen bên phải",
        "màn 1",
        "màn 2",
        "monitor 1",
        "monitor 2",
        # English
        "screen",
        "desktop",
        "monitor",
        "on my monitor",
        "on the monitor",
        "what do you see",
        "look at this",
        "left display",
        "right display",
        # Japanese
        "画面",
        "スクリーン",
        "デスクトップ",
        "モニター",
        "何が見える",
    )
    return any(term in normalized for term in terms)
