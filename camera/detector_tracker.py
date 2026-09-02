from __future__ import annotations

from ultralytics import YOLO

from perception.observation import Observation


class PersonDetectorTracker:
    def __init__(
        self,
        model_name: str,
        confidence: float = 0.45,
        imgsz: int = 320,
        device: str = "cpu",
    ):
        self.model = YOLO(model_name)
        self.confidence = confidence
        self.imgsz = imgsz
        self.device = device

    def process(self, frame) -> list[Observation]:
        results = self.model.track(
            source=frame,
            persist=True,
            tracker="bytetrack.yaml",
            classes=[0],
            conf=self.confidence,
            imgsz=self.imgsz,
            device=self.device,
            verbose=False,
        )

        observations: list[Observation] = []
        if not results:
            return observations

        result = results[0]
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            return observations

        xyxy = boxes.xyxy.cpu().tolist()
        confs = boxes.conf.cpu().tolist()

        ids = boxes.id
        track_ids = ids.int().cpu().tolist() if ids is not None else [None] * len(xyxy)

        for bbox, conf, track_id in zip(xyxy, confs, track_ids):
            if track_id is None:
                continue

            x1, y1, x2, y2 = bbox
            observations.append(
                Observation(
                    source="camera",
                    type="person_detected",
                    entity_id=f"person_{track_id}",
                    confidence=float(conf),
                    data={
                        "bbox": [float(x1), float(y1), float(x2), float(y2)],
                        "track_id": int(track_id),
                    },
                )
            )

        return observations
