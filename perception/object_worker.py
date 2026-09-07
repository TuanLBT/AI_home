from __future__ import annotations

import queue
import threading
from dataclasses import dataclass

import numpy as np
from ultralytics import YOLO


@dataclass(slots=True)
class ObjectJob:
    frame: np.ndarray
    timestamp: float


class ObjectWorker:
    def __init__(
        self,
        model_name: str = "yolo11n.pt",
        confidence: float = 0.35,
        imgsz: int = 320,
        device: str = "cpu",
    ):
        self.model = YOLO(model_name)
        self.confidence = confidence
        self.imgsz = imgsz
        self.device = device

        self._jobs: queue.Queue[ObjectJob | None] = queue.Queue(maxsize=1)
        self._results: queue.Queue[dict] = queue.Queue(maxsize=2)

        self._worker = threading.Thread(
            target=self._run,
            name="object-worker",
            daemon=True,
        )
        self._worker.start()

    def submit(self, frame, timestamp: float) -> bool:
        try:
            self._jobs.put_nowait(
                ObjectJob(
                    frame=np.asarray(frame).copy(),
                    timestamp=timestamp,
                )
            )
            return True
        except queue.Full:
            return False

    def update(self) -> list[dict]:
        results = []

        while True:
            try:
                results.append(self._results.get_nowait())
            except queue.Empty:
                break

        return results

    def close(self) -> None:
        try:
            self._jobs.put_nowait(None)
        except queue.Full:
            pass

    def _run(self) -> None:
        while True:
            job = self._jobs.get()

            if job is None:
                self._jobs.task_done()
                return

            try:
                result = self.model.predict(
                    source=job.frame,
                    conf=self.confidence,
                    imgsz=self.imgsz,
                    device=self.device,
                    verbose=False,
                )

                detections = []

                if result:
                    first = result[0]
                    boxes = first.boxes
                    names = first.names

                    if boxes is not None and len(boxes) > 0:
                        xyxy = boxes.xyxy.cpu().tolist()
                        confs = boxes.conf.cpu().tolist()
                        classes = boxes.cls.int().cpu().tolist()

                        for bbox, conf, class_id in zip(
                            xyxy,
                            confs,
                            classes,
                        ):
                            if int(class_id) == 0:
                                continue

                            detections.append({
                                "label": str(names[int(class_id)]),
                                "confidence": float(conf),
                                "bbox": [float(v) for v in bbox],
                            })

                payload = {
                    "type": "OBJECT_DETECTIONS",
                    "timestamp": job.timestamp,
                    "detections": detections,
                }

            except Exception as exc:
                payload = {
                    "type": "OBJECT_ERROR",
                    "timestamp": job.timestamp,
                    "error": str(exc),
                }

            try:
                self._results.put_nowait(payload)
            except queue.Full:
                pass

            self._jobs.task_done()
