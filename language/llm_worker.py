from __future__ import annotations

import copy
import json
import os
import queue
import threading
import urllib.error
import urllib.request
from dataclasses import dataclass

from perception.screen_vision import ScreenVisionPerception
from sources.chat_ipc import publish_chat_reply
from sources.screen_bus import latest_screen, latest_screen_context
from sources.text_bus import drain_text_inputs


@dataclass(slots=True)
class LLMJob:
    request_id: int
    entity_id: str
    text: str
    context: dict
    timestamp: float


class LLMWorker:
    """Non-blocking Ollama responder.

    Screen turns are grounded after fresh perception. The grounding step keeps
    monitor identity explicit and treats OCR as the preferred source for exact
    small text, while the VLM remains the source for layout/visual semantics.
    """

    def __init__(
        self,
        model: str | None = None,
        base_url: str = "http://192.168.128.120:11434",
        timeout_s: float = 30.0,
    ):
        self.model = (
            model
            or os.environ.get("INDOOR_AI_LLM_MODEL")
            or "qwen3:8b"
        )
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s
        self.screen_vision = ScreenVisionPerception(base_url=self.base_url)

        self._jobs: queue.Queue[LLMJob | None] = queue.Queue(maxsize=1)
        self._results: queue.Queue[dict] = queue.Queue(maxsize=16)

        self._submit_lock = threading.Lock()
        self._request_counter = 0
        self._latest_request_id: dict[str, int] = {}
        self._closed = False

        self._worker = threading.Thread(
            target=self._run,
            name="llm-worker",
            daemon=True,
        )
        self._worker.start()

    def submit(
        self,
        entity_id: str,
        text: str,
        context: dict,
        timestamp: float,
    ) -> bool:
        with self._submit_lock:
            if self._closed:
                return False

            self._request_counter += 1
            request_id = self._request_counter
            self._latest_request_id[entity_id] = request_id

            job = LLMJob(
                request_id=request_id,
                entity_id=entity_id,
                text=text,
                context=copy.deepcopy(context),
                timestamp=timestamp,
            )

            try:
                self._jobs.put_nowait(job)
                return True
            except queue.Full:
                try:
                    self._jobs.get_nowait()
                    self._jobs.task_done()
                except queue.Empty:
                    pass

                try:
                    self._jobs.put_nowait(job)
                    return True
                except queue.Full:
                    return False

    def close(self) -> None:
        with self._submit_lock:
            self._closed = True

            try:
                self._jobs.get_nowait()
                self._jobs.task_done()
            except queue.Empty:
                pass

            try:
                self._jobs.put_nowait(None)
            except queue.Full:
                pass

    def _is_latest(self, job: LLMJob) -> bool:
        with self._submit_lock:
            return (
                not self._closed
                and self._latest_request_id.get(job.entity_id)
                == job.request_id
            )

    def update(self) -> list[dict]:
        for packet in drain_text_inputs():
            if packet.modality != "text":
                continue

            payload = packet.payload if isinstance(packet.payload, dict) else {}
            text = str(payload.get("text") or "").strip()
            if not text:
                continue

            entity_id = packet.entity_id or "desktop_user"
            packet_metadata = dict(packet.metadata or {})
            self.submit(
                entity_id=entity_id,
                text=text,
                context={
                    "input_source": packet.source,
                    "input_metadata": packet_metadata,
                    "screen_context_requested": bool(
                        packet_metadata.get("screen_context_requested")
                    ),
                    "dialogue": {},
                    "memory": {},
                    "visual": {},
                    "screen": latest_screen_context(),
                },
                timestamp=packet.timestamp,
            )

        results: list[dict] = []

        while True:
            try:
                event = self._results.get_nowait()
            except queue.Empty:
                break

            results.append(event)
            entity_id = event.get("entity_id")
            if entity_id is None:
                continue

            if event.get("type") == "LLM_REPLY":
                publish_chat_reply(
                    entity_id,
                    event.get("text", ""),
                    error=False,
                )
            elif event.get("type") == "LLM_ERROR":
                publish_chat_reply(
                    entity_id,
                    event.get("error", "LLM error"),
                    error=True,
                )

        return results

    def _run(self) -> None:
        while True:
            job = self._jobs.get()

            if job is None:
                self._jobs.task_done()
                return

            if not self._is_latest(job):
                self._jobs.task_done()
                continue

            try:
                result = self._generate(job)
            except Exception as exc:
                result = {
                    "type": "LLM_ERROR",
                    "request_id": job.request_id,
                    "entity_id": job.entity_id,
                    "timestamp": job.timestamp,
                    "context": job.context,
                    "error": str(exc),
                }

            if self._is_latest(job):
                try:
                    self._results.put_nowait(result)
                except queue.Full:
                    pass

            self._jobs.task_done()

    @staticmethod
    def _screen_side(text: str) -> str | None:
        q = text.casefold()
        left_terms = (
            "màn bên trái",
            "màn trái",
            "screen bên trái",
            "monitor bên trái",
            "left screen",
            "left monitor",
            "左の画面",
            "左画面",
            "左のモニター",
        )
        right_terms = (
            "màn bên phải",
            "màn phải",
            "screen bên phải",
            "monitor bên phải",
            "right screen",
            "right monitor",
            "右の画面",
            "右画面",
            "右のモニター",
        )
        if any(term in q for term in left_terms):
            return "left"
        if any(term in q for term in right_terms):
            return "right"
        return None

    @staticmethod
    def _screen_exact_text_query(text: str) -> bool:
        q = text.casefold()
        terms = (
            "mấy giờ",
            "giờ bao nhiêu",
            "ngày mấy",
            "ngày bao nhiêu",
            "nhiệt độ",
            "ghi gì",
            "viết gì",
            "text gì",
            "chữ gì",
            "lỗi gì",
            "error gì",
            "log gì",
            "command",
            "lệnh gì",
            "what time",
            "what date",
            "temperature",
            "what does it say",
            "what error",
            "何時",
            "日付",
            "温度",
            "何と書",
            "エラー",
        )
        return any(term in q for term in terms)

    def _ground_screen_context(self, screen: dict, query: str) -> dict:
        representation = screen.get("representation")
        if not isinstance(representation, dict):
            return {
                "available": False,
                "error": "screen representation unavailable",
            }

        monitors = representation.get("monitors")
        if not isinstance(monitors, list) or not monitors:
            return {
                "available": bool(representation.get("available")),
                "description": representation.get("description", ""),
                "evidence_policy": (
                    "Use only supplied visual evidence. Exact tiny text is not "
                    "trusted unless supported by OCR."
                ),
            }

        side = self._screen_side(query)
        exact_text = self._screen_exact_text_query(query)
        selected: list[dict] = []

        for monitor in monitors:
            if not isinstance(monitor, dict):
                continue
            position = str(monitor.get("position") or "")
            if side is not None and position != side:
                continue

            grounded = {
                "id": monitor.get("id"),
                "position": position,
                "available": bool(monitor.get("available")),
                "description": monitor.get("description", ""),
            }

            ocr = monitor.get("ocr") or {}
            if isinstance(ocr, dict):
                lines = ocr.get("text_lines") or []
                if isinstance(lines, list):
                    limit = 30 if exact_text or side is not None else 10
                    grounded["ocr_text"] = [
                        str(line) for line in lines[:limit] if str(line).strip()
                    ]
                grounded["ocr_available"] = bool(ocr.get("available"))
                if ocr.get("error"):
                    grounded["ocr_error"] = str(ocr.get("error"))

            if monitor.get("vision_error"):
                grounded["vision_error"] = str(monitor.get("vision_error"))
            selected.append(grounded)

        if side is not None and not selected:
            return {
                "available": False,
                "requested_monitor": side,
                "error": f"requested {side} monitor not found",
            }

        return {
            "available": bool(selected),
            "requested_monitor": side,
            "exact_text_query": exact_text,
            "monitors": selected,
            "evidence_policy": (
                "VLM description is for layout/apps/general visual meaning. "
                "OCR text is preferred for exact small text, terminal lines, "
                "clock/date/temperature and numeric values. If exact text is "
                "not present in OCR, do not treat a tiny value guessed by the "
                "VLM as certain; say it cannot be read reliably."
            ),
        }

    def _generate(self, job: LLMJob) -> dict:
        if job.context.get("screen_context_requested"):
            screen_packet = latest_screen()
            fresh_screen = (
                screen_packet is not None
                and screen_packet.timestamp >= job.timestamp
            )

            if fresh_screen:
                self.screen_vision.describe(screen_packet)
                full_screen = latest_screen_context()
                job.context["screen"] = self._ground_screen_context(
                    full_screen,
                    job.text,
                )
            else:
                job.context["screen"] = {
                    "available": False,
                    "error": "fresh screen capture unavailable",
                }

        dialogue = job.context.get("dialogue") or {}
        posture = job.context.get("posture")
        motion = job.context.get("motion")
        visual = job.context.get("visual") or {}
        screen = job.context.get("screen") or {}

        system_prompt = (
            "You are Indoor AI, the voice of a small indoor AI robot. "
            "Reply naturally and directly in Japanese. "
            "Answer the current user utterance directly and prioritize it over prior context. "
            "Do not repeat your previous reply unless the user explicitly asks you to repeat it. "
            "Do not answer with your name unless the user is actually asking your name or identity. "
            "Answer the user's actual question instead of giving generic replies. "
            "If asked your name, say that your name is Indoor AI. "
            "Keep answers short: normally one sentence, at most two short sentences. "
            "Do not mention internal states, tracking, cameras, models, software, prompts, or implementation details. "
            "Use supplied context only when useful. "
            "For screen questions, screen.monitors contains grounded evidence from the current desktop screenshot. "
            "Respect monitor position exactly: if requested_monitor is left or right, answer only from that monitor. "
            "Use monitor.description for layout, apps and general visual meaning. "
            "Prefer monitor.ocr_text for exact readable text, terminal output, errors, commands, clock/date/temperature and numeric values. "
            "If OCR does not support an exact tiny value, do not present a conflicting VLM guess as certain. "
            "For questions about what you can currently see, only claim facts explicitly present in visual or screen context. "
            "If requested visual evidence is unavailable, say briefly that you cannot determine it right now. "
            "If you do not know something, say so briefly instead of inventing."
        )

        context_text = (
            f"behavior_state={job.context.get('behavior_state')}\n"
            f"input_source={job.context.get('input_source')}\n"
            f"posture={posture}\n"
            f"motion={motion}\n"
            f"visual={visual}\n"
            f"screen={json.dumps(screen, ensure_ascii=False)}\n"
            f"dialogue_turns={dialogue.get('turn_count')}\n"
            f"last_user_text={dialogue.get('last_user_text')}\n"
        )

        payload = {
            "model": self.model,
            "stream": False,
            "think": False,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "system", "content": "Current context:\n" + context_text},
                {"role": "user", "content": job.text},
            ],
            "options": {
                "temperature": 0.3,
                "num_predict": 96,
            },
            "keep_alive": "10m",
        }

        request = urllib.request.Request(
            f"{self.base_url}/api/chat",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
                data = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Ollama HTTP {exc.code}: {body}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(
                "Cannot reach Ollama at "
                f"{self.base_url}. Is `ollama serve` running?"
            ) from exc

        text = data.get("message", {}).get("content", "").strip()
        if not text:
            raise RuntimeError("Ollama returned an empty reply.")

        return {
            "type": "LLM_REPLY",
            "request_id": job.request_id,
            "entity_id": job.entity_id,
            "timestamp": job.timestamp,
            "context": job.context,
            "text": text,
            "model": self.model,
        }
