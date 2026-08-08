import math


def map_planar_stick(
    raw_horizontal,
    raw_vertical,
    *,
    axis_max=32767.0,
    deadzone=0.08,
    invert_horizontal=True,
    invert_vertical=False,
    swap_xy=True,
):
    """Map two raw joystick axes to a planar heading-frame velocity fraction.

    The returned planar components and magnitude are bounded by the unit
    circle.  A radial deadzone is removed and the remaining travel is
    rescaled to [0, 1], so motion is continuous from zero to maximum speed.
    """
    axis_max = float(axis_max)
    deadzone = float(deadzone)
    if axis_max <= 0.0:
        raise ValueError("axis_max must be positive")
    if not 0.0 <= deadzone < 1.0:
        raise ValueError("deadzone must be in [0, 1)")

    horizontal = max(-1.0, min(1.0, float(raw_horizontal) / axis_max))
    vertical = max(-1.0, min(1.0, float(raw_vertical) / axis_max))
    if invert_horizontal:
        horizontal = -horizontal
    if invert_vertical:
        vertical = -vertical

    if swap_xy:
        body_x, body_y = vertical, horizontal
    else:
        body_x, body_y = horizontal, vertical

    raw_magnitude = math.hypot(body_x, body_y)
    if raw_magnitude <= deadzone:
        return 0.0, 0.0, 0.0

    command_magnitude = (min(raw_magnitude, 1.0) - deadzone) / (1.0 - deadzone)
    scale = command_magnitude / raw_magnitude
    return body_x * scale, body_y * scale, command_magnitude
