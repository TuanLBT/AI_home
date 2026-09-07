from __future__ import annotations

import cv2
import numpy as np


def enrich_person_visual_facts(
    frame,
    observations,
    *,
    keypoint_confidence: float = 0.35,
) -> None:
    if frame is None or frame.size == 0:
        return

    height, width = frame.shape[:2]

    for obs in observations:
        if getattr(obs, "type", None) != "person_pose":
            continue

        data = obs.data
        torso_crop = _torso_crop(
            data,
            width=width,
            height=height,
            keypoint_confidence=keypoint_confidence,
        )

        lower_crop = _lower_body_crop(
            data,
            width=width,
            height=height,
            keypoint_confidence=keypoint_confidence,
        )

        shirt_color = None
        shirt_confidence = 0.0

        if torso_crop is not None:
            x1, y1, x2, y2 = torso_crop
            region = frame[y1:y2, x1:x2]
            shirt_color, shirt_confidence = _estimate_color(region)

        pants_color = None
        pants_confidence = 0.0

        if lower_crop is not None:
            x1, y1, x2, y2 = lower_crop
            region = frame[y1:y2, x1:x2]
            pants_color, pants_confidence = _estimate_color(region)

        data["visual_facts"] = {
            "shirt_color": shirt_color,
            "shirt_color_confidence": shirt_confidence,
            "pants_color": pants_color,
            "pants_color_confidence": pants_confidence,
        }


def _torso_crop(
    data: dict,
    *,
    width: int,
    height: int,
    keypoint_confidence: float,
):
    points = data.get("keypoints") or []
    confs = data.get("keypoint_confidences") or []

    ids = (5, 6, 11, 12)

    if (
        len(points) > 12
        and len(confs) > 12
        and all(confs[i] >= keypoint_confidence for i in ids)
    ):
        left_shoulder = points[5]
        right_shoulder = points[6]
        left_hip = points[11]
        right_hip = points[12]

        shoulder_y = (left_shoulder[1] + right_shoulder[1]) * 0.5
        hip_y = (left_hip[1] + right_hip[1]) * 0.5

        xs = [
            left_shoulder[0],
            right_shoulder[0],
            left_hip[0],
            right_hip[0],
        ]

        raw_x1 = min(xs)
        raw_x2 = max(xs)
        raw_y1 = min(shoulder_y, hip_y)
        raw_y2 = max(shoulder_y, hip_y)

        torso_w = raw_x2 - raw_x1
        torso_h = raw_y2 - raw_y1

        if torso_w >= 12 and torso_h >= 12:
            x1 = raw_x1 + 0.18 * torso_w
            x2 = raw_x2 - 0.18 * torso_w
            y1 = raw_y1 + 0.12 * torso_h
            y2 = raw_y2 - 0.10 * torso_h

            return _clip_rect(x1, y1, x2, y2, width, height)

    bbox = data.get("bbox")
    if not bbox or len(bbox) != 4:
        return None

    bx1, by1, bx2, by2 = bbox
    bw = bx2 - bx1
    bh = by2 - by1

    if bw < 20 or bh < 30:
        return None

    x1 = bx1 + 0.28 * bw
    x2 = bx1 + 0.72 * bw
    y1 = by1 + 0.24 * bh
    y2 = by1 + 0.58 * bh

    return _clip_rect(x1, y1, x2, y2, width, height)


def _lower_body_crop(
    data: dict,
    *,
    width: int,
    height: int,
    keypoint_confidence: float,
):
    points = data.get("keypoints") or []
    confs = data.get("keypoint_confidences") or []

    # COCO pose:
    # 11/12 = hips, 13/14 = knees
    ids = (11, 12, 13, 14)

    if (
        len(points) > 14
        and len(confs) > 14
        and all(confs[i] >= keypoint_confidence for i in ids)
    ):
        left_hip = points[11]
        right_hip = points[12]
        left_knee = points[13]
        right_knee = points[14]

        hip_y = (left_hip[1] + right_hip[1]) * 0.5
        knee_y = (left_knee[1] + right_knee[1]) * 0.5

        xs = [
            left_hip[0],
            right_hip[0],
            left_knee[0],
            right_knee[0],
        ]

        raw_x1 = min(xs)
        raw_x2 = max(xs)
        raw_y1 = min(hip_y, knee_y)
        raw_y2 = max(hip_y, knee_y)

        leg_w = raw_x2 - raw_x1
        leg_h = raw_y2 - raw_y1

        if leg_w >= 12 and leg_h >= 12:
            x1 = raw_x1 + 0.14 * leg_w
            x2 = raw_x2 - 0.14 * leg_w
            y1 = raw_y1 + 0.10 * leg_h
            y2 = raw_y2 - 0.12 * leg_h

            return _clip_rect(x1, y1, x2, y2, width, height)

    bbox = data.get("bbox")
    if not bbox or len(bbox) != 4:
        return None

    bx1, by1, bx2, by2 = bbox
    bw = bx2 - bx1
    bh = by2 - by1

    if bw < 20 or bh < 40:
        return None

    x1 = bx1 + 0.30 * bw
    x2 = bx1 + 0.70 * bw
    y1 = by1 + 0.56 * bh
    y2 = by1 + 0.82 * bh

    return _clip_rect(x1, y1, x2, y2, width, height)


def _clip_rect(x1, y1, x2, y2, width: int, height: int):
    x1 = max(0, min(width - 1, int(round(x1))))
    x2 = max(0, min(width, int(round(x2))))
    y1 = max(0, min(height - 1, int(round(y1))))
    y2 = max(0, min(height, int(round(y2))))

    if x2 - x1 < 8 or y2 - y1 < 8:
        return None

    return x1, y1, x2, y2


def _estimate_color(region):
    if region is None or region.size == 0:
        return None, 0.0

    small = cv2.resize(
        region,
        (32, 32),
        interpolation=cv2.INTER_AREA,
    )

    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    pixels = hsv.reshape(-1, 3).astype(np.float32)

    h = pixels[:, 0]
    s = pixels[:, 1]
    v = pixels[:, 2]

    usable = (v >= 18) & (v <= 245)

    if usable.sum() < 80:
        usable = np.ones(len(pixels), dtype=bool)

    h = h[usable]
    s = s[usable]
    v = v[usable]

    if len(h) == 0:
        return None, 0.0

    med_s = float(np.median(s))
    med_v = float(np.median(v))

    if med_v < 58:
        return "black", 0.90

    if med_s < 38:
        if med_v >= 190:
            return "white", 0.86
        return "gray", 0.78

    chromatic = s >= max(45.0, med_s * 0.55)
    if chromatic.sum() >= 40:
        hue = float(np.median(h[chromatic]))
        sat = float(np.median(s[chromatic]))
        val = float(np.median(v[chromatic]))
    else:
        hue = float(np.median(h))
        sat = med_s
        val = med_v

    if hue < 5 or hue >= 172:
        label = "red"
    elif hue < 12:
        label = "orange"
    elif hue < 24:
        label = "yellow"
    elif hue < 44:
        label = "brown" if val < 145 else "yellow"
    elif hue < 78:
        label = "green"
    elif hue < 96:
        label = "cyan"
    elif hue < 132:
        label = "blue"
    elif hue < 155:
        label = "purple"
    elif hue < 172:
        label = "pink"
    else:
        label = "red"

    confidence = min(
        0.92,
        0.58 + 0.34 * min(1.0, sat / 180.0),
    )

    return label, round(float(confidence), 3)
