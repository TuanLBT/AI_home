def fuse_motion(horizontal_state: str, depth_state: str) -> str:
    """
    Fuse two independent motion axes into one higher-level motion state.

    Horizontal:
        stable / moving_left / moving_right

    Depth:
        stable / approaching / moving_away

    Examples:
        moving_left + approaching -> forward_left
        moving_right + moving_away -> backward_right
    """

    horizontal = {
        "stable": "",
        "moving_left": "left",
        "moving_right": "right",
    }.get(horizontal_state, "")

    depth = {
        "stable": "",
        "approaching": "forward",
        "moving_away": "backward",
    }.get(depth_state, "")

    if not horizontal and not depth:
        return "stationary"

    if depth and horizontal:
        return f"{depth}_{horizontal}"

    if depth:
        return depth

    return horizontal
