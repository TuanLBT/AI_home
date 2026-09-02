from __future__ import annotations

import random
from typing import Any


class SpeechPolicy:
    """
    Memory-aware speech policy.

    It uses the current world snapshot plus short-term interaction memory.
    The goal is not full conversation yet; it simply avoids sounding as if
    every interaction is the first one.
    """

    def __init__(
        self,
        language: str = "ja",
        randomize: bool = True,
    ):
        self.language = language
        self.randomize = randomize

    def get_text(
        self,
        action: str,
        gesture: str | None = None,
        context: dict[str, Any] | None = None,
    ) -> str | None:
        context = context or {}

        if self.language == "ja":
            return self._get_japanese(action, gesture, context)

        if self.language == "vi":
            return self._get_vietnamese(action, gesture, context)

        return self._get_english(action, gesture, context)

    def _pick(
        self,
        options: list[str],
        avoid: str | None = None,
    ) -> str:
        if not options:
            return ""

        filtered = [
            option
            for option in options
            if option != avoid
        ]

        if not filtered:
            filtered = options

        if self.randomize and len(filtered) > 1:
            return random.choice(filtered)

        return filtered[0]

    @staticmethod
    def _recent_same_gesture(
        memory: dict[str, Any],
        gesture: str | None,
        within_s: float = 20.0,
    ) -> bool:
        if gesture is None:
            return False

        for item in reversed(memory.get("recent_actions", [])):
            if (
                item.get("action") == "acknowledge_gesture"
                and item.get("gesture") == gesture
                and item.get("age_s", 9999.0) <= within_s
            ):
                return True

        return False

    @staticmethod
    def _recent_action(
        memory: dict[str, Any],
        action: str,
        within_s: float,
    ) -> bool:
        for item in reversed(memory.get("recent_actions", [])):
            if (
                item.get("action") == action
                and item.get("age_s", 9999.0) <= within_s
            ):
                return True

        return False

    def _get_japanese(
        self,
        action: str,
        gesture: str | None,
        context: dict[str, Any],
    ) -> str | None:
        motion = context.get("motion")
        posture = context.get("posture")
        behavior_state = context.get("behavior_state")
        memory = context.get("memory") or {}
        last_speech = memory.get("last_speech")

        if action == "greet":
            if self._recent_action(
                memory,
                "greet",
                within_s=30.0,
            ):
                return self._pick(
                    [
                        "また来たね。",
                        "おかえり。",
                    ],
                    avoid=last_speech,
                )

            if motion in (
                "forward",
                "forward_left",
                "forward_right",
            ):
                return self._pick(
                    [
                        "こんにちは。こっちに来たんだね。",
                        "やあ。近くに来たね。",
                    ],
                    avoid=last_speech,
                )

            return self._pick(
                [
                    "こんにちは。",
                    "やあ。",
                ],
                avoid=last_speech,
            )

        if action == "acknowledge_gesture":
            repeated = self._recent_same_gesture(
                memory,
                gesture,
                within_s=20.0,
            )

            if gesture == "BOTH_HANDS_RAISED":
                if repeated:
                    return self._pick(
                        [
                            "また両手だね。",
                            "両手、また見えたよ。",
                        ],
                        avoid=last_speech,
                    )

                return self._pick(
                    [
                        "はい、両手が見えてるよ。",
                        "両手、見えてるよ。",
                    ],
                    avoid=last_speech,
                )

            if gesture == "LEFT_HAND_RAISED":
                if repeated:
                    return self._pick(
                        [
                            "また左手だね。",
                            "左手、また見えたよ。",
                        ],
                        avoid=last_speech,
                    )

                return self._pick(
                    [
                        "はい、左手が見えてるよ。",
                        "左手、見えてるよ。",
                    ],
                    avoid=last_speech,
                )

            if gesture == "RIGHT_HAND_RAISED":
                if repeated:
                    return self._pick(
                        [
                            "また右手だね。",
                            "右手、また見えたよ。",
                        ],
                        avoid=last_speech,
                    )

                return self._pick(
                    [
                        "はい、右手が見えてるよ。",
                        "右手、見えてるよ。",
                    ],
                    avoid=last_speech,
                )

            if behavior_state == "attending":
                return self._pick(
                    [
                        "はい、見えてるよ。",
                        "うん、わかったよ。",
                    ],
                    avoid=last_speech,
                )

            return self._pick(
                [
                    "見えてるよ。",
                    "わかったよ。",
                ],
                avoid=last_speech,
            )

        if action == "reply_greeting":
            return self._pick(
                [
                    "こんにちは。",
                    "やあ。",
                    "うん、こんにちは。",
                ],
                avoid=last_speech,
            )

        if action == "acknowledge_call":
            return self._pick(
                [
                    "なに？",
                    "うん、聞いてるよ。",
                    "はい、どうしたの？",
                ],
                avoid=last_speech,
            )

        if action == "sit_command":
            return self._pick(
                [
                    "わかったよ。",
                    "はい。",
                ],
                avoid=last_speech,
            )

        if action == "stop_command":
            return self._pick(
                [
                    "はい、止まるね。",
                    "わかった。止まるね。",
                ],
                avoid=last_speech,
            )

        if action == "settle_idle":
            if self._recent_action(
                memory,
                "settle_idle",
                within_s=30.0,
            ):
                return self._pick(
                    [
                        "まだゆっくりしてるんだね。",
                        "そのままゆっくりしてね。",
                    ],
                    avoid=last_speech,
                )

            if posture == "sitting":
                return self._pick(
                    [
                        "座ったんだね。ゆっくりしてね。",
                        "ゆっくりしてね。",
                    ],
                    avoid=last_speech,
                )

            return "ゆっくりしてね。"

        return None

    def _get_english(
        self,
        action: str,
        gesture: str | None,
        context: dict[str, Any],
    ) -> str | None:
        memory = context.get("memory") or {}
        last_speech = memory.get("last_speech")

        if action == "greet":
            return self._pick(
                ["Hello.", "Hi there."],
                avoid=last_speech,
            )

        if action == "acknowledge_gesture":
            return self._pick(
                ["I see you.", "Got it."],
                avoid=last_speech,
            )

        if action == "settle_idle":
            return "Make yourself comfortable."

        return None

    def _get_vietnamese(
        self,
        action: str,
        gesture: str | None,
        context: dict[str, Any],
    ) -> str | None:
        memory = context.get("memory") or {}
        last_speech = memory.get("last_speech")

        if action == "greet":
            return self._pick(
                ["Chào.", "Xin chào."],
                avoid=last_speech,
            )

        if action == "acknowledge_gesture":
            return self._pick(
                ["Thấy rồi.", "Biết rồi."],
                avoid=last_speech,
            )

        if action == "settle_idle":
            return "Cứ ngồi thoải mái đi."

        return None
