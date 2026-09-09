from __future__ import annotations

import copy
import json
import os
import queue
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass

from language.semantic_router import SemanticObservationRouter
from perception.screen_vision import ScreenVisionPerception
from sources.chat_ipc import publish_chat_reply
from sources.screen_bus import (
    latest_screen,
    latest_screen_context,
    request_screen_capture,
)
from sources.text_bus import drain_text_inputs


@dataclass(slots=True)
class LLMJob:
    request_id: int
    entity_id: str
    text: str
    context: dict
    timestamp: float


class LLMWorker:
    """Non-blocking Ollama responder with semantic observation routing.

    Every turn is first planned into zero or more observation requests. Source
    acquisition is then performed only for requested evidence. The planner is
    generic: screen is merely one supported source today, not a special case in
    ChatSource or a keyword trigger.
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
        self.router = SemanticObservationRouter(
            base_url=self.base_url,
            model=os.environ.get("INDOOR_AI_ROUTER_MODEL") or self.model,
        )
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
            self.submit(
                entity_id=entity_id,
                text=text,
                context={
                    "input_source": packet.source,
                    "input_metadata": dict(packet.metadata or {}),
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

    def _wait_for_fresh_screen(self, timeout_s: float = 3.0):
        requested_at = time.monotonic()
        request_screen_capture()
        deadline = requested_at + max(0.2, float(timeout_s))

        while time.monotonic() < deadline:
            packet = latest_screen()
            if packet is not None and packet.timestamp >= requested_at:
                return packet
            time.sleep(0.02)

        return None

    @staticmethod
    def _screen_request_info(requests: list[dict]) -> tuple[str | None, bool]:
        targets: list[str] = []
        exact = False

        for request in requests:
            if str(request.get("source") or "") not in {"screen", "app_state"}:
                continue
            target = request.get("target")
            if target is not None:
                value = str(target).strip().casefold()
                if value:
                    targets.append(value)
            need = str(request.get("need") or "").casefold()
            if need in {"exact_text", "error", "current_value"}:
                exact = True

        target = targets[0] if targets else None
        return target, exact

    def _ground_screen_context(
        self,
        screen: dict,
        *,
        target: str | None,
        exact_text: bool,
    ) -> dict:
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
                "target": target,
            }

        selected: list[dict] = []
        canonical_target = (target or "all").casefold()

        for monitor in monitors:
            if not isinstance(monitor, dict):
                continue

            position = str(monitor.get("position") or "").casefold()
            monitor_id = str(monitor.get("id") or "").casefold()
            if canonical_target not in {"", "all", "screen", "desktop"}:
                if canonical_target not in {position, monitor_id}:
                    continue

            grounded = {
                "id": monitor.get("id"),
                "position": monitor.get("position"),
                "available": bool(monitor.get("available")),
                "description": monitor.get("description", ""),
            }

            ocr = monitor.get("ocr") or {}
            if isinstance(ocr, dict):
                lines = ocr.get("text_lines") or []
                if isinstance(lines, list):
                    limit = 30 if exact_text else 12
                    grounded["ocr_text"] = [
                        str(line) for line in lines[:limit] if str(line).strip()
                    ]
                grounded["ocr_available"] = bool(ocr.get("available"))
                if ocr.get("error"):
                    grounded["ocr_error"] = str(ocr.get("error"))

            if monitor.get("vision_error"):
                grounded["vision_error"] = str(monitor.get("vision_error"))
            selected.append(grounded)

        if not selected and canonical_target not in {"", "all", "screen", "desktop"}:
            return {
                "available": False,
                "target": target,
                "error": f"requested screen target not found: {target}",
            }

        return {
            "available": bool(selected),
            "target": target,
            "exact_text_query": exact_text,
            "monitors": selected,
            "evidence_policy": (
                "VLM description is for layout/apps/general visual meaning. "
                "OCR text is preferred for exact small text, terminal output, "
                "errors, commands, clock/date/temperature and numeric values. "
                "If OCR does not support an exact tiny value, do not present a "
                "conflicting VLM guess as certain."
            ),
        }

    def _acquire_observations(self, job: LLMJob, route: dict) -> dict:
        requests = route.get("requests") or []
        if not isinstance(requests, list):
            requests = []

        observations: dict = {}
        unsupported: list[dict] = []

        screen_requests = [
            request
            for request in requests
            if isinstance(request, dict)
            and str(request.get("source") or "") in {"screen", "app_state"}
        ]
        if screen_requests:
            target, exact_text = self._screen_request_info(screen_requests)
            screen_packet = self._wait_for_fresh_screen()
            if screen_packet is None:
                observations["screen"] = {
                    "available": False,
                    "target": target,
                    "error": "fresh screen capture unavailable",
                }
            else:
                self.screen_vision.describe(screen_packet)
                observations["screen"] = self._ground_screen_context(
                    latest_screen_context(),
                    target=target,
                    exact_text=exact_text,
                )

        for request in requests:
            if not isinstance(request, dict):
                continue
            source = str(request.get("source") or "")
            if source in {"screen", "app_state"}:
                continue
            if source == "visual":
                observations["visual"] = job.context.get("visual") or {
                    "available": False,
                    "error": "current visual context unavailable",
                }
            elif source == "memory":
                observations["memory"] = job.context.get("memory") or {
                    "available": False,
                    "error": "memory context unavailable",
                }
            else:
                unsupported.append(
                    {
                        "source": source,
                        "target": request.get("target"),
                        "need": request.get("need"),
                        "available": False,
                        "error": "observation source not connected yet",
                    }
                )

        if unsupported:
            observations["unavailable_requests"] = unsupported

        return observations

    def _generate(self, job: LLMJob) -> dict:
        try:
            route = self.router.route(job.text)
        except Exception as exc:
            route = {
                "model": self.router.model,
                "requests": [],
                "error": str(exc),
            }

        observations = self._acquire_observations(job, route)
        job.context["observation_route"] = route
        job.context["observations"] = observations

        dialogue = job.context.get("dialogue") or {}
        posture = job.context.get("posture")
        motion = job.context.get("motion")
        visual = job.context.get("visual") or {}

        system_prompt = (
            "You are Indoor AI, the voice of a small indoor AI robot. "
            "Reply naturally and directly in Japanese. "
            "Answer the current user utterance directly and prioritize it over prior context. "
            "Do not repeat your previous reply unless the user explicitly asks you to repeat it. "
            "Do not answer with your name unless the user is actually asking your name or identity. "
            "Answer the user's actual question instead of giving generic replies. "
            "If asked your name, say that your name is Indoor AI. "
            "Keep answers short: normally one sentence, at most two short sentences. "
            "Do not mention internal states, tracking, cameras, models, software, prompts, routing, or implementation details. "
            "Observation context contains only evidence acquired because the planner determined it was needed. "
            "Use screen monitor description for layout/apps/general visual meaning and OCR text for exact small text. "
            "Respect requested targets in observation evidence and do not mix evidence from other targets. "
            "If requested evidence is unavailable, say briefly that you cannot determine it right now. "
            "For current-world questions, only claim facts supported by supplied observations. "
            "If you do not know something, say so briefly instead of inventing."
        )

        context_text = (
            f"behavior_state={job.context.get('behavior_state')}\n"
            f"input_source={job.context.get('input_source')}\n"
            f"posture={posture}\n"
            f"motion={motion}\n"
            f"visual={json.dumps(visual, ensure_ascii=False)}\n"
            f"observation_route={json.dumps(route, ensure_ascii=False)}\n"
            f"observations={json.dumps(observations, ensure_ascii=False)}\n"
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
