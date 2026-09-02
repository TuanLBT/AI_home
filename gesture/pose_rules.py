import math

NOSE = 0
LEFT_EYE = 1
RIGHT_EYE = 2
LEFT_EAR = 3
RIGHT_EAR = 4
LEFT_SHOULDER = 5
RIGHT_SHOULDER = 6
LEFT_ELBOW = 5#7
RIGHT_ELBOW = 5#8
LEFT_WRIST = 9
RIGHT_WRIST = 10
LEFT_HIP = 11
RIGHT_HIP = 12
LEFT_KNEE = 13
RIGHT_KNEE = 14
LEFT_ANKLE = 15
RIGHT_ANKLE = 16


def _valid(points, confs, idx, threshold):
    return (
        idx < len(points)
        and idx < len(confs)
        and confs[idx] >= threshold
        and points[idx][0] > 0
        and points[idx][1] > 0
    )


def _angle(a, b, c):
    ba = (a[0] - b[0], a[1] - b[1])
    bc = (c[0] - b[0], c[1] - b[1])

    dot = ba[0] * bc[0] + ba[1] * bc[1]
    mag_ba = math.hypot(ba[0], ba[1])
    mag_bc = math.hypot(bc[0], bc[1])

    if mag_ba == 0 or mag_bc == 0:
        return None

    cos_angle = dot / (mag_ba * mag_bc)
    cos_angle = max(-1.0, min(1.0, cos_angle))
    return math.degrees(math.acos(cos_angle))


def infer_gestures(points, confs, threshold=0.35) -> set[str]:
    gestures: set[str] = set()

    if not points or not confs:
        return gestures

    # Estimate torso size so "hand raised" requires a meaningful margin,
    # not just wrist_y being 1-2 noisy pixels above shoulder_y.
    torso = None
    torso_needed = [LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_HIP, RIGHT_HIP]

    if all(_valid(points, confs, i, threshold) for i in torso_needed):
        shoulder_y = (
            points[LEFT_SHOULDER][1] + points[RIGHT_SHOULDER][1]
        ) / 2
        hip_y = (
            points[LEFT_HIP][1] + points[RIGHT_HIP][1]
        ) / 2
        torso = abs(hip_y - shoulder_y)

    # Allow a horizontally raised arm to count too.
    # The previous rule required the wrist to be clearly ABOVE the shoulder,
    # so it only triggered when the hand was near head height.
    torso_size = torso or 50.0
    shoulder_tolerance = max(6.0, torso_size * 0.10)
    elbow_tolerance = max(8.0, torso_size * 0.18)
    wrist_elbow_tolerance = max(6.0, torso_size * 0.10)

    left_ok = (
        _valid(points, confs, LEFT_WRIST, threshold)
        and _valid(points, confs, LEFT_SHOULDER, threshold)
        and _valid(points, confs, LEFT_ELBOW, threshold)
    )

    right_ok = (
        _valid(points, confs, RIGHT_WRIST, threshold)
        and _valid(points, confs, RIGHT_SHOULDER, threshold)
        and _valid(points, confs, RIGHT_ELBOW, threshold)
    )

    left_raised = False
    right_raised = False

    if left_ok:
        wrist_y = points[LEFT_WRIST][1]
        shoulder_y = points[LEFT_SHOULDER][1]
        elbow_y = points[LEFT_ELBOW][1]

        left_raised = (
            wrist_y <= shoulder_y + shoulder_tolerance
            and elbow_y <= shoulder_y + elbow_tolerance
            and wrist_y <= elbow_y + wrist_elbow_tolerance
        )

    if right_ok:
        wrist_y = points[RIGHT_WRIST][1]
        shoulder_y = points[RIGHT_SHOULDER][1]
        elbow_y = points[RIGHT_ELBOW][1]

        right_raised = (
            wrist_y <= shoulder_y + shoulder_tolerance
            and elbow_y <= shoulder_y + elbow_tolerance
            and wrist_y <= elbow_y + wrist_elbow_tolerance
        )

    if left_raised:
        gestures.add("LEFT_HAND_RAISED")

    if right_raised:
        gestures.add("RIGHT_HAND_RAISED")

    if left_raised and right_raised:
        gestures.add("BOTH_HANDS_RAISED")

    return gestures


def infer_posture(points, confs, threshold=0.35) -> str:
    needed = [
        LEFT_SHOULDER,
        RIGHT_SHOULDER,
        LEFT_HIP,
        RIGHT_HIP,
        LEFT_KNEE,
        RIGHT_KNEE,
    ]

    if not all(_valid(points, confs, i, threshold) for i in needed):
        return "unknown"

    shoulder_y = (
        points[LEFT_SHOULDER][1]
        + points[RIGHT_SHOULDER][1]
    ) / 2

    hip_y = (
        points[LEFT_HIP][1]
        + points[RIGHT_HIP][1]
    ) / 2

    knee_y = (
        points[LEFT_KNEE][1]
        + points[RIGHT_KNEE][1]
    ) / 2

    torso = max(1.0, hip_y - shoulder_y)
    hip_to_knee = knee_y - hip_y
    ratio = hip_to_knee / torso

    left_angle = _angle(
        points[LEFT_SHOULDER],
        points[LEFT_HIP],
        points[LEFT_KNEE],
    )

    right_angle = _angle(
        points[RIGHT_SHOULDER],
        points[RIGHT_HIP],
        points[RIGHT_KNEE],
    )

    angles = [a for a in (left_angle, right_angle) if a is not None]

    if not angles:
        return "unknown"

    hip_angle = sum(angles) / len(angles)

    # Calibrated from your current camera test.
    if ratio >= 0.58 and hip_angle >= 168:
        return "standing"

    if ratio <= 0.45 and hip_angle <= 160:
        return "sitting"

    return "unknown"
