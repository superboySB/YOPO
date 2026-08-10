import math


def map_centered_axis(raw_value, *, axis_max=32767.0, deadzone=0.08, invert=False):
    """Map one self-centering Linux joystick axis to ``[-1, 1]``.

    The deadzone is removed and the remaining travel is rescaled, so the
    command is continuous at center and still reaches exactly full scale.
    """
    axis_max = float(axis_max)
    deadzone = float(deadzone)
    if axis_max <= 0.0:
        raise ValueError("axis_max must be positive")
    if not 0.0 <= deadzone < 1.0:
        raise ValueError("deadzone must be in [0, 1)")

    value = max(-1.0, min(1.0, float(raw_value) / axis_max))
    if invert:
        value = -value
    magnitude = abs(value)
    if magnitude <= deadzone:
        return 0.0
    mapped = (magnitude - deadzone) / (1.0 - deadzone)
    return math.copysign(mapped, value)


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


def map_dual_sticks(
    raw_right_horizontal,
    raw_right_vertical,
    raw_left_vertical,
    raw_left_horizontal,
    *,
    axis_max=32767.0,
    deadzone=0.08,
    invert_right_horizontal=True,
    invert_right_vertical=False,
    swap_right_xy=True,
    invert_left_vertical=False,
    invert_left_horizontal=True,
):
    """Map the two sticks to translation and yaw fractions.

    The returned translation is in heading coordinates: ``[forward, left,
    up]``.  Positive yaw is a left/counter-clockwise turn.  Each stick keeps
    its own center behavior: the right stick uses a radial deadzone, while the
    two left-stick axes are mapped independently so climb and yaw can be
    commanded together.
    """
    forward, left, planar_magnitude = map_planar_stick(
        raw_right_horizontal,
        raw_right_vertical,
        axis_max=axis_max,
        deadzone=deadzone,
        invert_horizontal=invert_right_horizontal,
        invert_vertical=invert_right_vertical,
        swap_xy=swap_right_xy,
    )
    up = map_centered_axis(
        raw_left_vertical,
        axis_max=axis_max,
        deadzone=deadzone,
        invert=invert_left_vertical,
    )
    yaw = map_centered_axis(
        raw_left_horizontal,
        axis_max=axis_max,
        deadzone=deadzone,
        invert=invert_left_horizontal,
    )
    translation = (forward, left, up)
    translation_magnitude = math.sqrt(forward * forward + left * left + up * up)
    return translation, yaw, planar_magnitude, translation_magnitude
