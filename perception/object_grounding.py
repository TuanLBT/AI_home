from __future__ import annotations

import math


def apply_held_object_grounding(
    world,
    detections: list[dict],
    *,
    wrist_confidence: float = 0.35,
) -> None:
    for person in world.people.values():
        if not person.present:
            continue

        facts = dict(getattr(person, "visual_facts", {}) or {})

        held = _find_held_object(
            person,
            detections,
            wrist_confidence=wrist_confidence,
        )

        if held is None:
            facts["held_object"] = None
            facts["held_object_confidence"] = 0.0
        else:
            facts["held_object"] = held["label"]
            facts["held_object_confidence"] = round(
                float(held["confidence"]),
                3,
            )

        person.visual_facts = facts


def _find_held_object(
    person,
    detections: list[dict],
    *,
    wrist_confidence: float,
):
    points = getattr(person, "keypoints", []) or []
    confs = getattr(person, "keypoint_confidences", []) or []
    bbox = getattr(person, "bbox", None)

    if not bbox or len(bbox) != 4:
        return None

    wrists = []

    for idx in (9, 10):
        if (
            idx < len(points)
            and idx < len(confs)
            and confs[idx] >= wrist_confidence
        ):
            x, y = points[idx]

            if x > 0 and y > 0:
                wrists.append((float(x), float(y)))

    if not wrists:
        return None

    px1, py1, px2, py2 = map(float, bbox)
    person_w = max(1.0, px2 - px1)
    person_h = max(1.0, py2 - py1)
    person_diag = math.hypot(person_w, person_h)

    best = None
    best_score = float("inf")

    for det in detections:
        box = det.get("bbox")
        if not box or len(box) != 4:
            continue

        ox1, oy1, ox2, oy2 = map(float, box)
        ow = max(1.0, ox2 - ox1)
        oh = max(1.0, oy2 - oy1)

        if ow > person_w * 0.85 or oh > person_h * 0.85:
            continue

        margin = max(14.0, 0.35 * max(ow, oh))

        expanded = (
            ox1 - margin,
            oy1 - margin,
            ox2 + margin,
            oy2 + margin,
        )

        cx = (ox1 + ox2) * 0.5
        cy = (oy1 + oy2) * 0.5

        for wx, wy in wrists:
            inside = (
                expanded[0] <= wx <= expanded[2]
                and expanded[1] <= wy <= expanded[3]
            )

            center_dist = math.hypot(wx - cx, wy - cy)
            norm_dist = center_dist / person_diag

            if not inside and norm_dist > 0.28:
                continue

            score = norm_dist - 0.20 * float(
                det.get("confidence", 0.0)
            )

            if score < best_score:
                best_score = score
                best = det

    return best
