"""Range-limited safety veto for joystick altitude-lock trajectory rewrites.

This module deliberately has no ROS, Torch, or project-local dependencies.  It
answers one narrow question: did a *material* rewrite of a YOPO-selected 3-D
trajectory put the final trajectory into an obstacle (or into unobserved
space) in the four depth frames captured at the planning pose?

``SafetyResult.safe`` means "do not veto the rewrite", not "the flight is
globally collision-free".  In particular:

* rewrites no larger than ``rewrite_threshold_m`` retain the original YOPO
  score without re-judging it;
* observations are valid only inside the supplied camera fields of view and
  x-depth range (normally 4 m); and
* this is a single-frame, static-obstacle check.

Coordinate convention
---------------------
The simulator's ToF images store camera-forward x-depth.  Camera coordinates
are ``+X`` forward, ``+Y`` left, and ``+Z`` up, hence

``u = cx - fx * y/x`` and ``v = cy - fy * z/x``.

Each camera yaw rotates camera coordinates into body coordinates.  The default
order matches the ROS topics: front, left, right, back.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any, Callable, Dict, Optional, Sequence, Tuple, Union

import numpy as np


DEFAULT_CAMERA_YAWS_RAD = (0.0, math.pi / 2.0, -math.pi / 2.0, math.pi)


@dataclass(frozen=True)
class BodyPose:
    """Body pose at the timestamp shared by the four depth images.

    ``rotation_world_from_body`` maps a vector expressed in body coordinates
    into world coordinates.
    """

    position_world: Sequence[float]
    rotation_world_from_body: Sequence[Sequence[float]]


@dataclass(frozen=True)
class CameraIntrinsics:
    """Pinhole intrinsics for all four equally calibrated ToF cameras."""

    fx: float
    fy: float
    cx: float
    cy: float


@dataclass(frozen=True)
class CollisionDetail:
    """First trajectory sample with observed obstacle overlap."""

    dense_sample_index: int
    source_segment_index: int
    source_segment_fraction: float
    world_point: Tuple[float, float, float]
    camera_point: Tuple[float, float, float]
    camera_index: int
    camera_yaw_rad: float
    pixel_row: int
    pixel_col: int
    measured_x_depth_m: float
    obstacle_distance_to_center_m: float
    required_clearance_m: float


@dataclass(frozen=True)
class UnknownDetail:
    """First materially changed sample that cannot be certified by ToF."""

    dense_sample_index: int
    source_segment_index: int
    source_segment_fraction: float
    world_point: Tuple[float, float, float]
    reason: str
    camera_index: Optional[int] = None
    pixel_row: Optional[int] = None
    pixel_col: Optional[int] = None


@dataclass(frozen=True)
class SafetyResult:
    """Structured result of :func:`evaluate_altitude_lock_safety`."""

    safe: bool
    reason: str
    first_collision: Optional[CollisionDetail]
    unknown: bool
    first_unknown: Optional[UnknownDetail]
    max_rewrite_deviation_m: float
    rewrite_threshold_m: float
    checked_samples: int
    skipped_unmodified_samples: int
    skipped_start_samples: int

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serializable representation."""

        return asdict(self)


TrajectoryInput = Union[
    Sequence[Sequence[float]],
    np.ndarray,
    Callable[[float], Sequence[float]],
    Any,
]


@dataclass(frozen=True)
class _DenseSample:
    point: np.ndarray
    deviation_m: float
    segment_index: int
    segment_fraction: float


@dataclass(frozen=True)
class _CameraObservation:
    state: str
    collision: Optional[CollisionDetail] = None
    unknown_reason: Optional[str] = None
    unknown_pixel: Optional[Tuple[int, int]] = None
    partial_footprint: bool = False


def sample_world_trajectory(
    trajectory: TrajectoryInput,
    sample_times_s: Optional[Sequence[float]] = None,
) -> np.ndarray:
    """Return ``(N, 3)`` world points from samples or a Poly5-like interface.

    Supported inputs are:

    * an existing ``(N, 3)`` array;
    * a callable ``trajectory(t) -> (x, y, z)``;
    * an object with ``get_position(t) -> (x, y, z)``; or
    * three axis objects, each exposing scalar ``get_position(t)`` (the
      interface used by three :class:`Poly5Solver` instances).

    ``sample_times_s`` is required for every form except an existing array.
    """

    try:
        points = np.asarray(trajectory, dtype=np.float64)
    except (TypeError, ValueError):
        points = None
    if points is not None and points.ndim == 2 and points.shape[1] == 3:
        return _validated_points(points)

    if sample_times_s is None:
        raise ValueError("sample_times_s is required for a trajectory sampler")
    times = np.asarray(sample_times_s, dtype=np.float64)
    if times.ndim != 1 or times.size == 0 or not np.all(np.isfinite(times)):
        raise ValueError("sample_times_s must be a non-empty finite 1-D sequence")
    if np.any(np.diff(times) < 0.0):
        raise ValueError("sample_times_s must be monotonically non-decreasing")

    axis_solvers = None
    if isinstance(trajectory, (tuple, list)) and len(trajectory) == 3:
        if all(hasattr(axis, "get_position") for axis in trajectory):
            axis_solvers = trajectory

    sampled = []
    for timestamp in times:
        t = float(timestamp)
        if axis_solvers is not None:
            point = [axis.get_position(t) for axis in axis_solvers]
        elif callable(trajectory):
            point = trajectory(t)
        elif hasattr(trajectory, "get_position"):
            point = trajectory.get_position(t)
        else:
            raise TypeError(
                "trajectory must be (N,3) samples, a callable, a vector "
                "get_position object, or three scalar get_position objects"
            )
        sampled.append(np.asarray(point, dtype=np.float64).reshape(-1))

    points = np.asarray(sampled, dtype=np.float64)
    return _validated_points(points)


def evaluate_altitude_lock_safety(
    flattened_trajectory: TrajectoryInput,
    body_pose: BodyPose,
    depth_images_m: Sequence[Sequence[Sequence[float]]],
    valid_masks: Sequence[Sequence[Sequence[bool]]],
    camera_intrinsics: CameraIntrinsics,
    *,
    original_trajectory: Optional[TrajectoryInput] = None,
    rewrite_deviation_m: Optional[Union[float, Sequence[float], np.ndarray]] = None,
    sample_times_s: Optional[Sequence[float]] = None,
    camera_yaws_rad: Sequence[float] = DEFAULT_CAMERA_YAWS_RAD,
    max_depth_m: float = 4.0,
    vehicle_radius_m: float = 0.30,
    safety_margin_m: float = 0.15,
    rewrite_threshold_m: float = 0.20,
    start_exclusion_m: Optional[float] = None,
    max_sample_spacing_m: Optional[float] = None,
    require_full_footprint: bool = True,
) -> SafetyResult:
    """Decide whether an altitude-lock rewrite should be vetoed.

    Exactly one of ``original_trajectory`` and ``rewrite_deviation_m`` must be
    supplied.  Supplying the original samples is preferred: it lets the helper
    ignore locally unchanged parts of a globally significant rewrite.  A
    scalar deviation is accepted when only the already-computed maximum is
    available; in that case it applies to every non-start sample.

    The trajectory is spatially densified before testing.  Every materially
    With the default ``require_full_footprint=True``, every materially changed
    sphere must be fully inside at least one camera footprint and x-depth
    range, with valid pixels over its projected footprint.  When explicitly
    disabled, a sphere whose *center* remains inside a camera image may use
    the valid, visible part of its footprint; this leaves the clipped portion
    uncertified and is therefore a deliberate residual-risk mode rather than
    complete sphere clearance.  A valid depth pixel is treated as a
    rectangular angular zone at its measured x-depth; this preserves the
    simulator's x-depth convention and is more conservative than treating an
    8x8 zone as a single infinitesimal ray.

    The initial sphere is excluded because it represents the already occupied
    vehicle volume at the depth timestamp.  Locally unchanged samples retain
    YOPO's original collision score and are not reclassified by this veto.
    """

    points = sample_world_trajectory(flattened_trajectory, sample_times_s)
    deviations = _resolve_deviations(
        points,
        original_trajectory,
        rewrite_deviation_m,
        sample_times_s,
    )
    max_deviation = float(np.max(deviations))

    threshold = _finite_nonnegative(rewrite_threshold_m, "rewrite_threshold_m")
    if max_deviation <= threshold + 1e-12:
        return SafetyResult(
            safe=True,
            reason="rewrite_within_yopo_trust_threshold",
            first_collision=None,
            unknown=False,
            first_unknown=None,
            max_rewrite_deviation_m=max_deviation,
            rewrite_threshold_m=threshold,
            checked_samples=0,
            skipped_unmodified_samples=int(points.shape[0]),
            skipped_start_samples=0,
        )

    position_w, rotation_wb = _validated_pose(body_pose)
    depths, masks, yaws = _validated_depth_inputs(
        depth_images_m,
        valid_masks,
        camera_yaws_rad,
    )
    intrinsics = _validated_intrinsics(camera_intrinsics)
    max_depth = _finite_positive(max_depth_m, "max_depth_m")
    radius = _finite_positive(vehicle_radius_m, "vehicle_radius_m")
    margin = _finite_nonnegative(safety_margin_m, "safety_margin_m")
    clearance = radius + margin

    if start_exclusion_m is None:
        start_exclusion = clearance
    else:
        start_exclusion = _finite_nonnegative(start_exclusion_m, "start_exclusion_m")
    if max_sample_spacing_m is None:
        max_spacing = max(0.025, min(0.20, 0.5 * clearance))
    else:
        max_spacing = _finite_positive(max_sample_spacing_m, "max_sample_spacing_m")
    if not isinstance(require_full_footprint, (bool, np.bool_)):
        raise ValueError("require_full_footprint must be boolean")
    require_full_footprint = bool(require_full_footprint)

    dense_samples = _densify(points, deviations, max_spacing)
    first_unknown = None
    checked = 0
    skipped_unmodified = 0
    skipped_start = 0
    partial_footprint_samples = 0

    rotation_bw = rotation_wb.T
    for dense_index, dense in enumerate(dense_samples):
        # Only judge the portion changed enough to invalidate trust in YOPO's
        # original score.  This also avoids interpreting the sensor origin as
        # an obstacle-bearing projected sphere at t=0.
        if dense.deviation_m <= threshold + 1e-12:
            skipped_unmodified += 1
            continue

        point_b = rotation_bw.dot(dense.point - position_w)
        if float(np.linalg.norm(point_b)) <= start_exclusion + 1e-12:
            skipped_start += 1
            continue

        checked += 1
        sample_clear = False
        sample_full_clear = False
        sample_partial_clear = False
        sample_unknown = None

        for camera_index, yaw in enumerate(yaws):
            point_c = _body_to_camera(point_b, yaw)
            observation = _observe_sphere(
                point_c,
                depths[camera_index],
                masks[camera_index],
                intrinsics,
                max_depth,
                clearance,
                dense_index,
                dense,
                camera_index,
                yaw,
                require_full_footprint,
            )
            if observation.collision is not None:
                return SafetyResult(
                    safe=False,
                    reason="material_rewrite_observed_collision",
                    first_collision=observation.collision,
                    unknown=first_unknown is not None,
                    first_unknown=first_unknown,
                    max_rewrite_deviation_m=max_deviation,
                    rewrite_threshold_m=threshold,
                    checked_samples=checked,
                    skipped_unmodified_samples=skipped_unmodified,
                    skipped_start_samples=skipped_start,
                )
            if observation.state == "clear":
                sample_clear = True
                if observation.partial_footprint:
                    sample_partial_clear = True
                else:
                    sample_full_clear = True
            elif observation.state == "unknown" and sample_unknown is None:
                row = col = None
                if observation.unknown_pixel is not None:
                    row, col = observation.unknown_pixel
                sample_unknown = UnknownDetail(
                    dense_sample_index=dense_index,
                    source_segment_index=dense.segment_index,
                    source_segment_fraction=dense.segment_fraction,
                    world_point=_tuple3(dense.point),
                    reason=observation.unknown_reason or "unobserved",
                    camera_index=camera_index,
                    pixel_row=row,
                    pixel_col=col,
                )

        if not sample_clear and first_unknown is None:
            if sample_unknown is None:
                sample_unknown = UnknownDetail(
                    dense_sample_index=dense_index,
                    source_segment_index=dense.segment_index,
                    source_segment_fraction=dense.segment_fraction,
                    world_point=_tuple3(dense.point),
                    reason="outside_all_camera_fields_of_view",
                )
            first_unknown = sample_unknown
        elif sample_partial_clear and not sample_full_clear:
            partial_footprint_samples += 1

    if first_unknown is not None:
        return SafetyResult(
            safe=False,
            reason="material_rewrite_enters_unknown_space",
            first_collision=None,
            unknown=True,
            first_unknown=first_unknown,
            max_rewrite_deviation_m=max_deviation,
            rewrite_threshold_m=threshold,
            checked_samples=checked,
            skipped_unmodified_samples=skipped_unmodified,
            skipped_start_samples=skipped_start,
        )

    if partial_footprint_samples > 0:
        reason = "material_rewrite_partial_footprint_observed_clear"
    elif checked > 0:
        reason = "material_rewrite_observed_clear"
    else:
        reason = "material_rewrite_only_in_initial_occupied_volume"
    return SafetyResult(
        safe=True,
        reason=reason,
        first_collision=None,
        unknown=False,
        first_unknown=None,
        max_rewrite_deviation_m=max_deviation,
        rewrite_threshold_m=threshold,
        checked_samples=checked,
        skipped_unmodified_samples=skipped_unmodified,
        skipped_start_samples=skipped_start,
    )


def _observe_sphere(
    point_c: np.ndarray,
    depth: np.ndarray,
    mask: np.ndarray,
    intrinsics: CameraIntrinsics,
    max_depth: float,
    clearance: float,
    dense_index: int,
    dense: _DenseSample,
    camera_index: int,
    camera_yaw: float,
    require_full_footprint: bool,
) -> _CameraObservation:
    height, width = depth.shape
    footprint = _project_sphere_footprint(point_c, intrinsics, clearance)
    if footprint is None:
        return _CameraObservation(state="not_visible")

    u_min, u_max, v_min, v_max = footprint
    image_u_min, image_u_max = -0.5, width - 0.5
    image_v_min, image_v_max = -0.5, height - 0.5
    footprint_crosses_fov = (
        u_min < image_u_min - 1e-9
        or u_max > image_u_max + 1e-9
        or v_min < image_v_min - 1e-9
        or v_max > image_v_max + 1e-9
    )
    if (
        u_max < image_u_min
        or u_min > image_u_max
        or v_max < image_v_min
        or v_min > image_v_max
    ):
        return _CameraObservation(state="not_visible")

    center_u = intrinsics.cx - intrinsics.fx * float(point_c[1]) / float(point_c[0])
    center_v = intrinsics.cy - intrinsics.fy * float(point_c[2]) / float(point_c[0])
    center_inside_image = (
        image_u_min <= center_u <= image_u_max
        and image_v_min <= center_v <= image_v_max
    )
    row_first = max(0, int(math.ceil(v_min - 0.5 - 1e-12)))
    row_last = min(height - 1, int(math.floor(v_max + 0.5 + 1e-12)))
    col_first = max(0, int(math.ceil(u_min - 0.5 - 1e-12)))
    col_last = min(width - 1, int(math.floor(u_max + 0.5 + 1e-12)))
    if row_first > row_last or col_first > col_last:
        return _CameraObservation(state="not_visible")

    # A collision is useful evidence even if another part of the sphere lies
    # beyond range.  Range/validity unknown is returned only after examining
    # all observed footprint pixels.
    first_invalid = None
    for row in range(row_first, row_last + 1):
        for col in range(col_first, col_last + 1):
            measured = float(depth[row, col])
            valid = bool(mask[row, col])
            if (
                not valid
                or not math.isfinite(measured)
                or measured <= 0.0
                or measured > max_depth + 1e-6
            ):
                if first_invalid is None:
                    first_invalid = (row, col)
                continue

            # A max-range return means no observed surface inside range.
            if measured >= max_depth - 1e-6:
                continue

            obstacle_distance = _distance_to_depth_zone(
                point_c,
                measured,
                row,
                col,
                intrinsics,
            )
            if obstacle_distance <= clearance + 1e-9:
                collision = CollisionDetail(
                    dense_sample_index=dense_index,
                    source_segment_index=dense.segment_index,
                    source_segment_fraction=dense.segment_fraction,
                    world_point=_tuple3(dense.point),
                    camera_point=_tuple3(point_c),
                    camera_index=camera_index,
                    camera_yaw_rad=float(camera_yaw),
                    pixel_row=row,
                    pixel_col=col,
                    measured_x_depth_m=measured,
                    obstacle_distance_to_center_m=obstacle_distance,
                    required_clearance_m=clearance,
                )
                return _CameraObservation(state="collision", collision=collision)

    # Inspect every visible overlap for collision before classifying a clipped
    # footprint as unknown.  This preserves collision veto authority even if
    # the center belongs to an adjacent camera (or to a blind gap).
    if footprint_crosses_fov and not center_inside_image:
        return _CameraObservation(
            state="unknown",
            unknown_reason="sphere_center_outside_camera_fov",
        )
    if footprint_crosses_fov and require_full_footprint:
        return _CameraObservation(
            state="unknown",
            unknown_reason="sphere_footprint_crosses_camera_fov",
        )
    if point_c[0] + clearance > max_depth + 1e-9:
        return _CameraObservation(
            state="unknown",
            unknown_reason="sphere_extends_beyond_x_depth_range",
        )
    if first_invalid is not None:
        return _CameraObservation(
            state="unknown",
            unknown_reason="invalid_depth_in_sphere_footprint",
            unknown_pixel=first_invalid,
        )
    return _CameraObservation(
        state="clear",
        partial_footprint=footprint_crosses_fov,
    )


def _project_sphere_footprint(
    point_c: np.ndarray,
    intrinsics: CameraIntrinsics,
    radius: float,
) -> Optional[Tuple[float, float, float, float]]:
    """Project the conservative rectangular angular footprint of a sphere."""

    x, y, z = [float(value) for value in point_c]
    # A sphere touching/crossing the camera plane has no bounded pinhole
    # projection and cannot be certified by this camera.
    if x <= radius + 1e-12:
        return None

    horizontal_range = math.hypot(x, y)
    vertical_range = math.hypot(x, z)
    if horizontal_range <= radius or vertical_range <= radius:
        return None

    theta = math.atan2(y, x)
    phi = math.atan2(z, x)
    theta_pad = math.asin(min(1.0, radius / horizontal_range))
    phi_pad = math.asin(min(1.0, radius / vertical_range))
    theta_lo, theta_hi = theta - theta_pad, theta + theta_pad
    phi_lo, phi_hi = phi - phi_pad, phi + phi_pad
    half_pi = 0.5 * math.pi
    if (
        theta_lo <= -half_pi
        or theta_hi >= half_pi
        or phi_lo <= -half_pi
        or phi_hi >= half_pi
    ):
        return None

    u_a = intrinsics.cx - intrinsics.fx * math.tan(theta_lo)
    u_b = intrinsics.cx - intrinsics.fx * math.tan(theta_hi)
    v_a = intrinsics.cy - intrinsics.fy * math.tan(phi_lo)
    v_b = intrinsics.cy - intrinsics.fy * math.tan(phi_hi)
    return min(u_a, u_b), max(u_a, u_b), min(v_a, v_b), max(v_a, v_b)


def _distance_to_depth_zone(
    point_c: np.ndarray,
    x_depth: float,
    row: int,
    col: int,
    intrinsics: CameraIntrinsics,
) -> float:
    """Distance from a point to an 8x8 pixel's slab at measured x-depth."""

    u_edges = (col - 0.5, col + 0.5)
    v_edges = (row - 0.5, row + 0.5)
    y_values = [-(u - intrinsics.cx) * x_depth / intrinsics.fx for u in u_edges]
    z_values = [-(v - intrinsics.cy) * x_depth / intrinsics.fy for v in v_edges]
    y_min, y_max = min(y_values), max(y_values)
    z_min, z_max = min(z_values), max(z_values)
    dx = float(point_c[0]) - x_depth
    dy = _distance_to_interval(float(point_c[1]), y_min, y_max)
    dz = _distance_to_interval(float(point_c[2]), z_min, z_max)
    return math.sqrt(dx * dx + dy * dy + dz * dz)


def _distance_to_interval(value: float, lower: float, upper: float) -> float:
    if value < lower:
        return lower - value
    if value > upper:
        return value - upper
    return 0.0


def _body_to_camera(point_b: np.ndarray, yaw: float) -> np.ndarray:
    """Apply R(body<-camera)^T for a yaw-only camera extrinsic."""

    cosine = math.cos(yaw)
    sine = math.sin(yaw)
    x_b, y_b, z_b = [float(value) for value in point_b]
    return np.asarray(
        [cosine * x_b + sine * y_b, -sine * x_b + cosine * y_b, z_b],
        dtype=np.float64,
    )


def _resolve_deviations(
    flattened_points: np.ndarray,
    original_trajectory: Optional[TrajectoryInput],
    rewrite_deviation_m: Optional[Union[float, Sequence[float], np.ndarray]],
    sample_times_s: Optional[Sequence[float]],
) -> np.ndarray:
    if (original_trajectory is None) == (rewrite_deviation_m is None):
        raise ValueError(
            "supply exactly one of original_trajectory and rewrite_deviation_m"
        )
    if original_trajectory is not None:
        original_points = sample_world_trajectory(original_trajectory, sample_times_s)
        if original_points.shape != flattened_points.shape:
            raise ValueError("original and flattened trajectory samples must have equal shape")
        deviations = np.linalg.norm(flattened_points - original_points, axis=1)
    else:
        raw = np.asarray(rewrite_deviation_m, dtype=np.float64)
        if raw.ndim == 0:
            deviations = np.full(flattened_points.shape[0], float(raw), dtype=np.float64)
        elif raw.ndim == 1 and raw.shape[0] == flattened_points.shape[0]:
            deviations = raw.copy()
        else:
            raise ValueError("rewrite_deviation_m must be scalar or have one value per sample")
    if not np.all(np.isfinite(deviations)) or np.any(deviations < 0.0):
        raise ValueError("rewrite deviations must be finite and non-negative")
    return deviations


def _densify(
    points: np.ndarray,
    deviations: np.ndarray,
    max_spacing: float,
) -> Sequence[_DenseSample]:
    if points.shape[0] == 1:
        return [_DenseSample(points[0].copy(), float(deviations[0]), 0, 0.0)]

    samples = []
    for segment in range(points.shape[0] - 1):
        start = points[segment]
        finish = points[segment + 1]
        distance = float(np.linalg.norm(finish - start))
        subdivisions = max(1, int(math.ceil(distance / max_spacing)))
        first_step = 0 if segment == 0 else 1
        for step in range(first_step, subdivisions + 1):
            fraction = float(step) / float(subdivisions)
            point = (1.0 - fraction) * start + fraction * finish
            deviation = (
                (1.0 - fraction) * float(deviations[segment])
                + fraction * float(deviations[segment + 1])
            )
            samples.append(
                _DenseSample(
                    point=point,
                    deviation_m=deviation,
                    segment_index=segment,
                    segment_fraction=fraction,
                )
            )
    return samples


def _validated_points(points: np.ndarray) -> np.ndarray:
    if points.ndim != 2 or points.shape[1] != 3 or points.shape[0] == 0:
        raise ValueError("trajectory samples must have shape (N, 3), N >= 1")
    if not np.all(np.isfinite(points)):
        raise ValueError("trajectory samples must be finite")
    return np.asarray(points, dtype=np.float64)


def _validated_pose(body_pose: BodyPose) -> Tuple[np.ndarray, np.ndarray]:
    position = np.asarray(body_pose.position_world, dtype=np.float64)
    rotation = np.asarray(body_pose.rotation_world_from_body, dtype=np.float64)
    if position.shape != (3,) or not np.all(np.isfinite(position)):
        raise ValueError("body pose position must be a finite 3-vector")
    if rotation.shape != (3, 3) or not np.all(np.isfinite(rotation)):
        raise ValueError("body pose rotation must be a finite 3x3 matrix")
    if not np.allclose(rotation.T.dot(rotation), np.eye(3), atol=1e-6):
        raise ValueError("body pose rotation must be orthonormal")
    if not math.isclose(float(np.linalg.det(rotation)), 1.0, abs_tol=1e-6):
        raise ValueError("body pose rotation must have determinant +1")
    return position, rotation


def _validated_depth_inputs(
    depth_images_m: Sequence[Sequence[Sequence[float]]],
    valid_masks: Sequence[Sequence[Sequence[bool]]],
    camera_yaws_rad: Sequence[float],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    depths = np.asarray(depth_images_m, dtype=np.float64)
    masks = np.asarray(valid_masks, dtype=bool)
    yaws = np.asarray(camera_yaws_rad, dtype=np.float64)
    if depths.ndim != 3 or depths.shape[1] <= 0 or depths.shape[2] <= 0:
        raise ValueError("depth_images_m must have shape (cameras, height, width)")
    if masks.shape != depths.shape:
        raise ValueError("valid_masks must have the same shape as depth_images_m")
    if yaws.ndim != 1 or yaws.shape[0] != depths.shape[0]:
        raise ValueError("camera_yaws_rad must contain one yaw per depth image")
    if not np.all(np.isfinite(yaws)):
        raise ValueError("camera yaws must be finite")
    return depths, masks, yaws


def _validated_intrinsics(intrinsics: CameraIntrinsics) -> CameraIntrinsics:
    fx = _finite_positive(intrinsics.fx, "camera fx")
    fy = _finite_positive(intrinsics.fy, "camera fy")
    cx = float(intrinsics.cx)
    cy = float(intrinsics.cy)
    if not math.isfinite(cx) or not math.isfinite(cy):
        raise ValueError("camera principal point must be finite")
    return CameraIntrinsics(fx=fx, fy=fy, cx=cx, cy=cy)


def _finite_positive(value: float, name: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return result


def _finite_nonnegative(value: float, name: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
    return result


def _tuple3(values: np.ndarray) -> Tuple[float, float, float]:
    return float(values[0]), float(values[1]), float(values[2])


# Synthetic regression cases live here so the pure helper remains testable even
# when ROS and the simulator are unavailable.  Run this file directly.
def _run_synthetic_tests() -> None:
    intrinsics = CameraIntrinsics(
        fx=4.0 / math.tan(math.radians(22.5)),
        fy=4.0 / math.tan(math.radians(22.5)),
        cx=3.5,
        cy=3.5,
    )
    pose = BodyPose(np.zeros(3), np.eye(3))

    def fixture():
        return np.full((4, 8, 8), 4.0), np.ones((4, 8, 8), dtype=bool)

    # Material, fully observed, obstacle-free rewrite.
    depths, masks = fixture()
    flat = np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 0.0, 0.0]])
    raw = flat.copy()
    raw[:, 2] = [0.0, 0.35, 0.55, 0.55]
    clear = evaluate_altitude_lock_safety(
        flat,
        pose,
        depths,
        masks,
        intrinsics,
        original_trajectory=raw,
        vehicle_radius_m=0.10,
        safety_margin_m=0.05,
    )
    assert clear.safe and clear.reason == "material_rewrite_observed_clear", clear

    # YOPO went above a centered obstacle; flattening that path must collide.
    depths, masks = fixture()
    depths[0, 3:5, 3:5] = 2.0
    collision = evaluate_altitude_lock_safety(
        flat,
        pose,
        depths,
        masks,
        intrinsics,
        original_trajectory=raw,
        vehicle_radius_m=0.10,
        safety_margin_m=0.05,
    )
    assert not collision.safe and collision.first_collision is not None, collision
    assert collision.reason == "material_rewrite_observed_collision", collision
    assert collision.first_collision.camera_index == 0, collision
    assert collision.first_collision.measured_x_depth_m == 2.0, collision

    raw_below = flat.copy()
    raw_below[:, 2] = [0.0, -0.35, -0.55, -0.55]
    collision_from_below = evaluate_altitude_lock_safety(
        flat,
        pose,
        depths,
        masks,
        intrinsics,
        original_trajectory=raw_below,
        vehicle_radius_m=0.10,
        safety_margin_m=0.05,
    )
    assert collision_from_below.first_collision is not None, collision_from_below

    # The same obstacle is safe when the material portion of the final path
    # passes horizontally beside its measured 8x8 zone.
    horizontal_flat = np.asarray(
        [[0.0, 0.0, 0.0], [1.0, 0.2, 0.0], [2.0, 0.4, 0.0], [3.0, 0.6, 0.0]]
    )
    horizontal_raw = horizontal_flat.copy()
    horizontal_raw[:, 2] = [0.0, 0.0, 0.45, 0.55]
    horizontal = evaluate_altitude_lock_safety(
        horizontal_flat,
        pose,
        depths,
        masks,
        intrinsics,
        original_trajectory=horizontal_raw,
        vehicle_radius_m=0.06,
        safety_margin_m=0.03,
    )
    assert horizontal.safe, horizontal

    # A diagonal material rewrite falls into the 45-degree gap between the
    # front and left 45-degree-FOV cameras.
    depths, masks = fixture()
    blind_flat = np.asarray([[0.0, 0.0, 0.0], [1.5, 1.5, 0.0], [2.0, 2.0, 0.0]])
    blind_raw = blind_flat.copy()
    blind_raw[:, 2] = [0.0, 0.4, 0.6]
    blind = evaluate_altitude_lock_safety(
        blind_flat,
        pose,
        depths,
        masks,
        intrinsics,
        original_trajectory=blind_raw,
        vehicle_radius_m=0.08,
        safety_margin_m=0.02,
    )
    assert not blind.safe and blind.unknown and blind.first_collision is None, blind

    # Invalid/NaN data under the projected sphere is unknown, never clear.
    depths, masks = fixture()
    depths[0, 3, 3] = np.nan
    masks[0, 3, 3] = False
    invalid = evaluate_altitude_lock_safety(
        flat,
        pose,
        depths,
        masks,
        intrinsics,
        original_trajectory=raw,
        vehicle_radius_m=0.10,
        safety_margin_m=0.05,
    )
    assert not invalid.safe and invalid.unknown, invalid
    assert invalid.first_unknown is not None, invalid
    assert invalid.first_unknown.reason == "invalid_depth_in_sphere_footprint", invalid

    # Optional residual-risk mode: the sphere center is still inside the front
    # image, but its footprint is clipped by the FOV edge.  Visible clear
    # pixels may pass only when full-footprint certification is disabled.
    edge_slope = math.tan(math.radians(20.0))
    partial_flat = np.asarray(
        [[0.0, 0.0, 0.0], [1.0, edge_slope, 0.0], [2.0, 2.0 * edge_slope, 0.0], [3.0, 3.0 * edge_slope, 0.0]]
    )
    partial_raw = partial_flat.copy()
    partial_raw[:, 2] = [0.0, 0.35, 0.55, 0.55]
    depths, masks = fixture()
    partial_strict = evaluate_altitude_lock_safety(
        partial_flat,
        pose,
        depths,
        masks,
        intrinsics,
        original_trajectory=partial_raw,
        vehicle_radius_m=0.10,
        safety_margin_m=0.05,
    )
    assert not partial_strict.safe and partial_strict.unknown, partial_strict
    assert partial_strict.first_unknown is not None, partial_strict
    assert partial_strict.first_unknown.reason == "sphere_footprint_crosses_camera_fov", partial_strict

    partial_clear = evaluate_altitude_lock_safety(
        partial_flat,
        pose,
        depths,
        masks,
        intrinsics,
        original_trajectory=partial_raw,
        vehicle_radius_m=0.10,
        safety_margin_m=0.05,
        require_full_footprint=False,
    )
    assert partial_clear.safe, partial_clear
    assert partial_clear.reason == "material_rewrite_partial_footprint_observed_clear", partial_clear

    # Visible pixels retain full veto authority in partial-footprint mode.
    partial_obstacle_depths, masks = fixture()
    partial_obstacle_depths[0, 3:5, 0] = 2.0
    partial_collision = evaluate_altitude_lock_safety(
        partial_flat,
        pose,
        partial_obstacle_depths,
        masks,
        intrinsics,
        original_trajectory=partial_raw,
        vehicle_radius_m=0.10,
        safety_margin_m=0.05,
        require_full_footprint=False,
    )
    assert not partial_collision.safe and partial_collision.first_collision is not None, partial_collision

    # If the sphere center itself lies in the gap between cameras, clipping is
    # not enough: it remains unknown even in residual-risk mode.
    blind_slope = math.tan(math.radians(30.0))
    center_out_flat = np.asarray(
        [[0.0, 0.0, 0.0], [1.0, blind_slope, 0.0], [2.0, 2.0 * blind_slope, 0.0]]
    )
    center_out_raw = center_out_flat.copy()
    center_out_raw[:, 2] = [0.0, 0.4, 0.6]
    depths, masks = fixture()
    center_out = evaluate_altitude_lock_safety(
        center_out_flat,
        pose,
        depths,
        masks,
        intrinsics,
        original_trajectory=center_out_raw,
        vehicle_radius_m=0.10,
        safety_margin_m=0.05,
        require_full_footprint=False,
    )
    assert not center_out.safe and center_out.unknown, center_out
    assert center_out.first_unknown is not None, center_out
    assert center_out.first_unknown.reason in (
        "sphere_center_outside_camera_fov",
        "outside_all_camera_fields_of_view",
    ), center_out

    # Small rewrites retain YOPO's score even in unknown input; this helper is
    # a rewrite veto, not a replacement planner/collision checker.
    small = evaluate_altitude_lock_safety(
        blind_flat,
        pose,
        np.full((4, 8, 8), np.nan),
        np.zeros((4, 8, 8), dtype=bool),
        intrinsics,
        rewrite_deviation_m=0.20,
        vehicle_radius_m=0.08,
        safety_margin_m=0.02,
    )
    assert small.safe and not small.unknown, small
    assert small.reason == "rewrite_within_yopo_trust_threshold", small

    # Three scalar Poly5-like axes are accepted, and +pi/2 really is the left
    # camera.  The depth value remains x-depth in that camera frame.
    class Axis:
        def __init__(self, scale):
            self.scale = scale

        def get_position(self, timestamp):
            return self.scale * timestamp

    depths, masks = fixture()
    depths[1, 3:5, 3:5] = 2.0
    sampled_left = evaluate_altitude_lock_safety(
        (Axis(0.0), Axis(1.0), Axis(0.0)),
        pose,
        depths,
        masks,
        intrinsics,
        rewrite_deviation_m=[0.0, 0.4, 0.5, 0.5],
        sample_times_s=[0.0, 1.0, 2.0, 3.0],
        vehicle_radius_m=0.10,
        safety_margin_m=0.05,
    )
    assert sampled_left.first_collision is not None, sampled_left
    assert sampled_left.first_collision.camera_index == 1, sampled_left

    print("joystick_safety synthetic tests: PASS")


if __name__ == "__main__":
    _run_synthetic_tests()
