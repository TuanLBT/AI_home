from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class IntentResult:
    type: str
    text: str
    confidence: float
    data: dict


class IntentEngine:
    """
    Minimal rule-based intent layer for Japanese ASR text.

    This is deliberately small:
      - GREETING
      - CALL_ATTENTION
      - SIT_COMMAND
      - STOP_COMMAND
      - UNKNOWN

    Later this can be replaced by an LLM/NLU classifier without changing ASR.
    """

    def parse(self, text: str) -> IntentResult:
        normalized = self._normalize(text)

        if not normalized:
            return IntentResult(
                type="UNKNOWN",
                text=text,
                confidence=0.0,
                data={},
            )

        if any(
            phrase in normalized
            for phrase in (
                "こんにちは",
                "こんばんは",
                "おはよう",
                "やあ",
                "もしもし",
            )
        ):
            return IntentResult(
                type="GREETING",
                text=text,
                confidence=0.95,
                data={},
            )

        if any(
            phrase in normalized
            for phrase in (
                "ねぇ",
                "おい",
                "ちょっと",
                "こっち",
                "聞いて",
            )
        ):
            return IntentResult(
                type="CALL_ATTENTION",
                text=text,
                confidence=0.85,
                data={},
            )

        if any(
            phrase in normalized
            for phrase in (
                "座って",
                "すわって",
                "座れ",
            )
        ):
            return IntentResult(
                type="SIT_COMMAND",
                text=text,
                confidence=0.9,
                data={},
            )

        if any(
            phrase in normalized
            for phrase in (
                "止まって",
                "とまって",
                "止まれ",
                "ストップ",
            )
        ):
            return IntentResult(
                type="STOP_COMMAND",
                text=text,
                confidence=0.9,
                data={},
            )

        return IntentResult(
            type="UNKNOWN",
            text=text,
            confidence=0.2,
            data={},
        )

    @staticmethod
    def _normalize(text: str) -> str:
        return (
            text.strip()
            .replace(" ", "")
            .replace("　", "")
            .replace("。", "")
            .replace("、", "")
            .replace("！", "")
            .replace("？", "")
        )
