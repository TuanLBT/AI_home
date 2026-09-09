from __future__ import annotations

import copy
import json
import os
import queue
import threading
import urllib.error
import urllib.request
from dataclasses import dataclass

from sources.chat_ipc import publish_chat_reply
from sources.screen_bus import latest_screen_context
from sources.text_bus import drain_text_inputs


@dataclass(slots=True)
class LLMJob:
    request_id: int
    entity_id: str
    text: str
    context: dict
    timestamp: float


class LLMWorker:
    """
    Non-blocking Ollama responder.

    Input:
        free-form user utterance + dialogue/world context

    Output:
        LLM_REPLY
        LLM_ERROR

    The camera/perception loop never waits for the LLM.
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

        self._jobs: queue.Queue[LLMJob | None] = queue.Queue(
            maxsize=1
        )
        self._results: queue.Queue[dict] = queue.Queue(
            maxsize=16
        )

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

    def _generate(self, job: LLMJob) -> dict:
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
            "Keep answers short: normally one sentence, "
            "at most two short sentences. "
            "Do not mention internal states, tracking, cameras, models, "
            "software, prompts, or implementation details. "
            "Use supplied context only when useful. "
            "Screen availability metadata only means a desktop frame exists; "
            "it does not describe the frame contents. "
            "For questions about what you can currently see, only claim visual facts "
            "that are explicitly present in the supplied visual context. "
            "Never guess a visual detail that is missing or unavailable. "
            "If the requested visual fact is unavailable, say briefly that you cannot determine it right now. "
            "If you do not know something, say so briefly instead of inventing."
        )

        context_text = (
            f"behavior_state={job.context.get('behavior_state')}\n"
            f"input_source={job.context.get('input_source')}\n"
            f"posture={posture}\n"
            f"motion={motion}\n"
            f"visual={visual}\n"
            f"screen={screen}\n"
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
            with urllib.request.urlopen(
                request,
                timeout=self.timeout_s,
            ) as response:
                data = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(
                f"Ollama HTTP {exc.code}: {body}"
            ) from exc
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
