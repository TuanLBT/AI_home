from __future__ import annotations

from typing import Any

from ultralytics import YOLO


class StaticPoseExtractor:
    """Pose extraction for still images without requiring tracker IDs."""

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

    def process(self, image) -> list[dict[str, Any]]:
        results = self.model.predict(
            source=image,
            classes=[0],
            conf=self.confidence,
            imgsz=self.imgsz,
            device=self.device,
            verbose=False,
        )

        if not results:
            return []

        result = results[0]
        boxes = result.boxes
        keypoints = result.keypoints

        if boxes is None or len(boxes) == 0:
            return []

        xyxy = boxes.xyxy.cpu().tolist()
        confs = boxes.conf.cpu().tolist()

        kp_xy = None
        kp_conf = None
        if keypoints is not None:
            if keypoints.xy is not None:
                kp_xy = keypoints.xy.cpu().tolist()
            if keypoints.conf is not None:
                kp_conf = keypoints.conf.cpu().tolist()

        poses: list[dict[str, Any]] = []

        for i, (bbox, conf) in enumerate(zip(xyxy, confs)):
            x1, y1, x2, y2 = bbox
            pose = {
                "confidence": float(conf),
                "bbox": [float(x1), float(y1), float(x2), float(y2)],
                "keypoints": [],
                "keypoint_confidences": [],
            }

            if kp_xy is not None and i < len(kp_xy):
                pose["keypoints"] = [
                    [float(x), float(y)] for x, y in kp_xy[i]
                ]

            if kp_conf is not None and i < len(kp_conf):
                pose["keypoint_confidences"] = [
                    float(v) for v in kp_conf[i]
                ]

            poses.append(pose)

        return poses
