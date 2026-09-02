from __future__ import annotations

from ultralytics import YOLO

from perception.observation import Observation


class PersonPoseTracker:
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
        keypoints = result.keypoints

        if boxes is None or len(boxes) == 0:
            return observations

        xyxy = boxes.xyxy.cpu().tolist()
        confs = boxes.conf.cpu().tolist()
        ids = boxes.id
        track_ids = ids.int().cpu().tolist() if ids is not None else [None] * len(xyxy)

        kp_xy = None
        kp_conf = None

        if keypoints is not None:
            if keypoints.xy is not None:
                kp_xy = keypoints.xy.cpu().tolist()
            if keypoints.conf is not None:
                kp_conf = keypoints.conf.cpu().tolist()

        for i, (bbox, conf, track_id) in enumerate(zip(xyxy, confs, track_ids)):
            if track_id is None:
                continue

            x1, y1, x2, y2 = bbox

            data = {
                "bbox": [float(x1), float(y1), float(x2), float(y2)],
                "track_id": int(track_id),
                "keypoints": [],
                "keypoint_confidences": [],
            }

            if kp_xy is not None and i < len(kp_xy):
                data["keypoints"] = [
                    [float(x), float(y)] for x, y in kp_xy[i]
                ]

            if kp_conf is not None and i < len(kp_conf):
                data["keypoint_confidences"] = [
                    float(v) for v in kp_conf[i]
                ]

            observations.append(
                Observation(
                    source="camera",
                    type="person_pose",
                    entity_id=f"person_{track_id}",
                    confidence=float(conf),
                    data=data,
                )
            )

        return observations
