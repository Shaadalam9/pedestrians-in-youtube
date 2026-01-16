import os
import glob
from typing import Dict, List, Tuple, Any, Optional

import cv2
import numpy as np
import polars as pl
import common
from utils.core.metadata import MetaData
from custom_logger import CustomLogger
from logmod import logs

metadata = MetaData()


logs(show_level=common.get_configs("logger_level"), show_color=True)
logger = CustomLogger(__name__)  # use custom logger


# ============================================================
# CSV discovery + parsing
# ============================================================
def find_csv_for_video(video_id: str, data_dir: str = "data/bbox") -> str:
    """
    Return the newest CSV matching {data_dir}/{video_id}_*.csv.
    """
    pattern = os.path.join(data_dir, f"{video_id}_*.csv")
    matches = glob.glob(pattern)
    if not matches:
        raise FileNotFoundError(f"No CSV found for video_id='{video_id}' with pattern: {pattern}")
    matches.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return matches[0]


def parse_csv_filename(csv_path: str) -> Tuple[str, float, str]:
    """
    Expected: {video_id}_{start_seconds}_{fps}.csv
    Example: 3ai7SUaPoHM_660_30.csv -> start_seconds=660
    video_id may contain underscores; we split from the end.
    """
    base = os.path.basename(csv_path)
    stem = os.path.splitext(base)[0]
    parts = stem.split("_")
    if len(parts) < 3:
        raise ValueError(f"CSV filename does not match expected pattern: {base}")

    fps_str = parts[-1]
    start_str = parts[-2]
    vid = "_".join(parts[:-2])

    try:
        start_seconds = float(start_str)
    except Exception:
        raise ValueError(f"Could not parse start_seconds from '{start_str}' in {base}")

    return vid, start_seconds, fps_str


# ============================================================
# Geometry + drawing helpers (NORMALISED ONLY)
# ============================================================
def xywh_to_xyxy_norm(xc: float, yc: float, w: float, h: float, W: int, H: int) -> Tuple[int, int, int, int]:
    """
    Convert normalised (xc, yc, w, h) in [0,1] to pixel (x1, y1, x2, y2).
    """
    xc_px, yc_px = xc * W, yc * H
    w_px, h_px = w * W, h * H

    x1 = int(round(xc_px - w_px / 2.0))
    y1 = int(round(yc_px - h_px / 2.0))
    x2 = int(round(xc_px + w_px / 2.0))
    y2 = int(round(yc_px + h_px / 2.0))

    x1 = max(0, min(W - 1, x1))
    y1 = max(0, min(H - 1, y1))
    x2 = max(0, min(W - 1, x2))
    y2 = max(0, min(H - 1, y2))

    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1

    return x1, y1, x2, y2


# ============================================================
# Class IDs (COCO-ish + static refs used in crossing validity)
# ============================================================
PERSON_CLASS = 0
BICYCLE_CLASS = 1
CAR_CLASS = 2
MOTORCYCLE_CLASS = 3
BUS_CLASS = 5
TRUCK_CLASS = 7

STATIC_CLASS_IDS = (
    9,   # traffic light
    10,  # fire hydrant
    11,  # stop sign
    12,  # parking meter
    13,  # bench
)


def class_name(yolo_id: int) -> str:
    mapping = {
        PERSON_CLASS: "person",
        BICYCLE_CLASS: "bicycle",
        CAR_CLASS: "car",
        MOTORCYCLE_CLASS: "motorcycle",
        BUS_CLASS: "bus",
        TRUCK_CLASS: "truck",
        9: "traffic light",
        10: "fire hydrant",
        11: "stop sign",
        12: "parking meter",
        13: "bench",
    }
    return mapping.get(int(yolo_id), f"class{int(yolo_id)}")


# ============================================================
# Colors (BGR)
# ============================================================
COLOR_PEDESTRIAN = (0, 255, 0)         # green
COLOR_BICYCLIST = (0, 215, 255)        # orange/yellow-ish
COLOR_MOTORCYCLIST = (255, 0, 255)     # magenta
COLOR_PASSENGER = (255, 255, 255)      # white

COLOR_BICYCLE = (255, 0, 0)            # blue
COLOR_MOTORCYCLE = (0, 0, 255)         # red
COLOR_CAR = (0, 165, 255)              # orange
COLOR_BUS = (0, 140, 0)                # darker green
COLOR_TRUCK = (140, 0, 0)              # dark blue-ish
COLOR_OTHER = (255, 255, 0)            # cyan-ish

# Crossing validity overlay colors (kept exactly)
COLOR_CROSS_VALID = (0, 0, 0)          # (comment in original said bright green, but value was (0,0,0))
COLOR_CROSS_FAKE = (0, 0, 255)         # bright red


def det_color(yolo_id: int, person_id=None, person_type_map=None) -> Tuple[int, int, int]:
    """
    - Person boxes colored by classification:
        bicycle -> bicyclist
        motorcycle -> motorcyclist
        car/bus/truck -> passenger
        None -> pedestrian
    - Vehicle boxes colored by class.
    """
    if yolo_id == PERSON_CLASS and person_type_map is not None:
        ptype = person_type_map.get(person_id)  # "bicycle"|"motorcycle"|"car"|"bus"|"truck"|None
        if ptype == "bicycle":
            return COLOR_BICYCLIST
        if ptype == "motorcycle":
            return COLOR_MOTORCYCLIST
        if ptype in ("car", "bus", "truck"):
            return COLOR_PASSENGER
        return COLOR_PEDESTRIAN

    if yolo_id == BICYCLE_CLASS:
        return COLOR_BICYCLE
    if yolo_id == MOTORCYCLE_CLASS:
        return COLOR_MOTORCYCLE
    if yolo_id == CAR_CLASS:
        return COLOR_CAR
    if yolo_id == BUS_CLASS:
        return COLOR_BUS
    if yolo_id == TRUCK_CLASS:
        return COLOR_TRUCK
    return COLOR_OTHER


# ============================================================
# Data cleaning / indexing
# ============================================================
def _dedup_per_frame(df: pl.DataFrame) -> pl.DataFrame:
    """Keep the highest-confidence detection for each (yolo-id, unique-id, frame-count)."""
    if "confidence" not in df.columns:
        return df.unique(subset=["yolo-id", "unique-id", "frame-count"], keep="first")

    return (
        df.sort(
            ["yolo-id", "unique-id", "frame-count", "confidence"],
            descending=[False, False, False, True],
        )
        .unique(subset=["yolo-id", "unique-id", "frame-count"], keep="first")
    )


def build_frame_index(df: pl.DataFrame) -> Dict[int, List[Tuple[int, Any, float, float, float, float, float]]]:
    """
    frame-count -> list of (yolo_id, unique_id, conf, x_center, y_center, width, height)
    Assumes df already deduped.
    """
    df2 = df.select([
        pl.col("frame-count").cast(pl.Int64),
        pl.col("yolo-id").cast(pl.Int64),
        pl.col("unique-id"),
        pl.col("confidence").cast(pl.Float64),
        pl.col("x-center").cast(pl.Float64),
        pl.col("y-center").cast(pl.Float64),
        pl.col("width").cast(pl.Float64),
        pl.col("height").cast(pl.Float64),
    ])

    frame_map: Dict[int, List[Tuple[int, Any, float, float, float, float, float]]] = {}
    for r in df2.iter_rows(named=True):
        f = int(r["frame-count"])
        frame_map.setdefault(f, []).append((
            int(r["yolo-id"]),
            r["unique-id"],
            float(r["confidence"]),
            float(r["x-center"]),
            float(r["y-center"]),
            float(r["width"]),
            float(r["height"]),
        ))
    return frame_map


# ============================================================
# Person classification: rider (bicycle/motorcycle) + passenger (car/bus/truck)
# (keeps signature/params; avg_height is now sourced via df_mapping in RUN block)
# ============================================================
def classify_rider_type(
    df: pl.DataFrame,
    person_id,
    *,
    avg_height: Optional[float] = None,
    min_shared_frames: int = 4,
    dist_rel_thresh: float = 0.8,
    prox_req: float = 0.7,
    alpha_x: float = 1.0,
    beta_y: float = 0.03,
    gamma_y: float = 1.4,
    coloc_req: float = 0.7,
    sim_thresh: float = 0.4,
    sim_req: float = 0.5,
    min_motion_steps: int = 3,
    motion_coloc_min: float = 0.5,
    short_shared_frames: int = 8,
    short_sim_req: float = 0.8,
    short_disp_req: float = 0.12,
    eps: float = 1e-9,
    person_class: int = PERSON_CLASS,
    bicycle_class: int = BICYCLE_CLASS,
    motorcycle_class: int = MOTORCYCLE_CLASS,
    car_class: int = CAR_CLASS,
    bus_class: int = BUS_CLASS,
    truck_class: int = TRUCK_CLASS,
) -> dict:
    if avg_height is not None:
        try:
            if float(avg_height) <= 0.0:
                return {
                    "is_rider": False, "rider_type": None, "role": None, "vehicle_id": None,
                    "score": 0.0, "shared_frames": 0
                }
        except Exception:
            return {
                "is_rider": False, "rider_type": None, "role": None, "vehicle_id": None,
                "score": 0.0, "shared_frames": 0
            }

    df = _dedup_per_frame(df)

    p = (
        df.filter((pl.col("yolo-id") == person_class) & (pl.col("unique-id") == person_id))
          .sort("frame-count")
    )
    if p.height == 0:
        return {
            "is_rider": False, "rider_type": None, "role": None, "vehicle_id": None,
            "score": 0.0, "shared_frames": 0
        }

    p_frames = p.get_column("frame-count").to_numpy()
    if p_frames.size < min_shared_frames:
        return {
            "is_rider": False, "rider_type": None, "role": None, "vehicle_id": None,
            "score": 0.0, "shared_frames": 0
        }

    first_frame = int(p_frames.min())
    last_frame = int(p_frames.max())

    supported_vehicle_classes = [bicycle_class, motorcycle_class, car_class, bus_class, truck_class]

    vehicles = df.filter(
        (pl.col("frame-count") >= first_frame)
        & (pl.col("frame-count") <= last_frame)
        & (pl.col("yolo-id").is_in(supported_vehicle_classes))
    )
    if vehicles.height == 0:
        return {
            "is_rider": False, "rider_type": None, "role": None, "vehicle_id": None,
            "score": 0.0, "shared_frames": 0
        }

    vehicle_ids = vehicles.select("unique-id").unique().to_series().to_list()
    p1 = p.unique(subset=["frame-count"], keep="first")

    best = None

    for vid in vehicle_ids:
        v = vehicles.filter(pl.col("unique-id") == vid).sort("frame-count")
        if v.height == 0:
            continue

        v_class = int(v.get_column("yolo-id")[0])
        vtype = (
            "bicycle" if v_class == bicycle_class else
            "motorcycle" if v_class == motorcycle_class else
            "car" if v_class == car_class else
            "bus" if v_class == bus_class else
            "truck" if v_class == truck_class else
            None
        )
        if vtype is None:
            continue

        role = "rider" if v_class in (bicycle_class, motorcycle_class) else "passenger"

        v1 = v.unique(subset=["frame-count"], keep="first")
        j = p1.join(v1, on="frame-count", how="inner", suffix="_v")
        shared = j.height
        if shared < min_shared_frames:
            continue

        p_xy = j.select(["x-center", "y-center"]).to_numpy()
        v_xy = j.select(["x-center_v", "y-center_v"]).to_numpy()

        p_w = j.get_column("width").to_numpy()
        p_h = j.get_column("height").to_numpy()
        v_w = j.get_column("width_v").to_numpy()
        v_h = j.get_column("height_v").to_numpy()

        dist = np.linalg.norm(p_xy - v_xy, axis=1)
        if role == "rider":
            dist_rel = dist / np.maximum(p_h, eps)
        else:
            dist_rel = dist / np.maximum(v_h, eps)

        prox = dist_rel < dist_rel_thresh
        prox_ratio = float(prox.mean())
        if prox_ratio < prox_req:
            continue

        relx = v_xy[:, 0] - p_xy[:, 0]
        rely = v_xy[:, 1] - p_xy[:, 1]

        if role == "rider":
            spatial = (np.abs(relx) < alpha_x * p_w) & (rely > beta_y * p_h) & (rely < gamma_y * p_h)
        else:
            inside = (np.abs(relx) <= 0.5 * v_w) & (np.abs(rely) <= 0.5 * v_h)
            spatial = inside

        coloc = prox & spatial
        coloc_ratio = float(coloc.mean())

        p_mov = np.diff(p_xy, axis=0)
        v_mov = np.diff(v_xy, axis=0)

        sim_ratio = 0.0
        if p_mov.shape[0] > 0:
            na = np.linalg.norm(p_mov, axis=1)
            nb = np.linalg.norm(v_mov, axis=1)
            move_mask = (na > eps) & (nb > eps)

            cos = np.zeros_like(na, dtype=float)
            cos[move_mask] = (p_mov[move_mask] * v_mov[move_mask]).sum(axis=1) / (na[move_mask] * nb[move_mask])

            prox_steps = prox[1:]
            m = min(len(prox_steps), len(cos), len(move_mask))
            prox_steps = prox_steps[:m]
            cos = cos[:m]
            move_mask = move_mask[:m]

            denom_mask = prox_steps & move_mask
            denom = int(denom_mask.sum())
            if denom >= min_motion_steps:
                sim_ratio = float(((cos > sim_thresh) & denom_mask).sum() / denom)

        if shared < short_shared_frames:
            if shared > 1:
                p_disp = float(np.linalg.norm(p_xy[-1] - p_xy[0]))
                p_disp_rel = p_disp / float(np.maximum(np.mean(p_h), eps))
            else:
                p_disp_rel = 0.0

            if not (sim_ratio >= short_sim_req or p_disp_rel >= short_disp_req):
                continue

        ok = (coloc_ratio >= coloc_req) or (sim_ratio >= sim_req and coloc_ratio >= motion_coloc_min)
        if not ok:
            continue

        score = 0.7 * coloc_ratio + 0.2 * prox_ratio + 0.1 * float(sim_ratio)
        cand = {
            "is_rider": True,
            "rider_type": vtype,
            "role": role,
            "vehicle_id": vid,
            "score": float(score),
            "shared_frames": int(shared),
            "prox_ratio": prox_ratio,
            "coloc_ratio": coloc_ratio,
            "sim_ratio": float(sim_ratio),
        }

        if best is None or cand["score"] > best["score"]:
            best = cand

    if best is None:
        return {
            "is_rider": False, "rider_type": None, "role": None, "vehicle_id": None,
            "score": 0.0, "shared_frames": 0
        }

    return best


def classify_all_persons_and_print(
    df: pl.DataFrame,
    **rider_params
) -> Tuple[Dict[Any, Optional[str]], Dict[Any, dict]]:
    df = _dedup_per_frame(df)

    person_ids = (
        df.filter(pl.col("yolo-id") == PERSON_CLASS)
        .select("unique-id").unique()
        .to_series().to_list()
    )
    try:
        person_ids = sorted(person_ids)
    except Exception:
        pass

    assoc = []
    non_assoc = []

    person_type_map: Dict[Any, Optional[str]] = {}
    full_results: Dict[Any, dict] = {}

    for pid in person_ids:
        res = classify_rider_type(df, pid, **rider_params)
        full_results[pid] = res
        person_type_map[pid] = res.get("rider_type", None)

        if res["is_rider"]:
            assoc.append(
                (pid, res.get("role"), res.get("rider_type"), res.get("vehicle_id"),
                 res.get("score", 0.0), res.get("shared_frames", 0),
                 res.get("prox_ratio", 0.0), res.get("coloc_ratio", 0.0), res.get("sim_ratio", 0.0))
            )
        else:
            non_assoc.append(pid)

    logger.info("\n=== ASSOCIATED PERSONS (RIDER/PASSENGER) ===")
    logger.info(f"Count: {len(assoc)}")
    for pid, role, vtype, vid, score, shared, pr, cr, sr in assoc:
        logger.info(
            f"person_id={pid}  role={role}  vehicle_type={vtype}  vehicle_id={vid}  "
            f"score={score:.3f}  shared_frames={shared}  prox={pr:.2f}  coloc={cr:.2f}  sim={sr:.2f}"
        )

    logger.info("\n=== NON-ASSOCIATED PERSONS ===")
    logger.info(f"Count: {len(non_assoc)}")
    for pid in non_assoc:
        logger.info(f"person_id={pid}")

    return person_type_map, full_results


# ============================================================
# Crossing validity (camera-turn rejection) + maps
# ============================================================
def _robust_range(series: pl.Series, q: float = 0.05) -> float:
    try:
        lo = series.quantile(q, "nearest")
        hi = series.quantile(1.0 - q, "nearest")
        return float(hi - lo)  # type: ignore
    except Exception:
        return 0.0


def is_valid_crossing_debug(
    df: pl.DataFrame,
    person_uid,
    *,
    person_class: int = PERSON_CLASS,
    static_class_ids: tuple[int, ...] = STATIC_CLASS_IDS,
    min_shared_frames: int = 8,
    ratio_thresh: float = 0.6,
    rel_disp_thresh: float = 0.05,
    q: float = 0.05,
    eps: float = 1e-9
) -> tuple[bool, dict]:
    df = _dedup_per_frame(df)

    person = (
        df.filter((pl.col("yolo-id") == person_class) & (pl.col("unique-id") == person_uid))
          .sort("frame-count")
    )
    if person.height == 0:
        return False, {"reason": "no_person_track"}

    frames = person.get_column("frame-count").to_numpy()
    first_f = int(frames.min())
    last_f = int(frames.max())

    static = df.filter(
        (pl.col("frame-count") >= first_f) &
        (pl.col("frame-count") <= last_f) &
        (pl.col("yolo-id").is_in(list(static_class_ids)))
    )
    if static.height == 0:
        return True, {"reason": "no_static_reference"}

    person_x_rng = _robust_range(person.get_column("x-center"), q=q)
    if person_x_rng <= eps:
        return False, {"reason": "person_x_no_motion", "person_x_rng": float(person_x_rng)}

    static_uids = static.select("unique-id").unique().to_series().to_list()

    best = None
    for sid in static_uids:
        s = static.filter(pl.col("unique-id") == sid).sort("frame-count")
        if s.height == 0:
            continue

        j = person.join(s, on="frame-count", how="inner", suffix="_s")
        shared = j.height
        if shared < min_shared_frames:
            continue

        px = j.get_column("x-center")
        sx = j.get_column("x-center_s")
        relx = px - sx

        px_rng = _robust_range(px, q=q)
        sx_rng = _robust_range(sx, q=q)
        relx_rng = _robust_range(relx, q=q)

        ratio = float(sx_rng / max(px_rng, eps))

        cand = {
            "static_uid": sid,
            "shared_frames": int(shared),
            "person_x_rng": float(px_rng),
            "static_x_rng": float(sx_rng),
            "relx_rng": float(relx_rng),
            "ratio": float(ratio),
        }

        if best is None:
            best = cand
        else:
            if (cand["shared_frames"], -cand["ratio"], cand["relx_rng"]) > (
                best["shared_frames"], -best["ratio"], best["relx_rng"]
            ):
                best = cand

    if best is None:
        return True, {"reason": "no_static_overlap"}

    if best["ratio"] >= ratio_thresh:
        return False, {"reason": "camera_turn_ratio", **best}

    if best["relx_rng"] < rel_disp_thresh:
        return False, {"reason": "tiny_relative_motion", **best}

    return True, {"reason": "ok", **best}


def compute_crossing_candidate_ids(
    df: pl.DataFrame,
    *,
    person_class: int = PERSON_CLASS,
    min_x: float = 0.45,
    max_x: float = 0.55
) -> list:
    df = _dedup_per_frame(df)
    persons = df.filter(pl.col("yolo-id") == person_class)
    if persons.height == 0:
        return []

    crossed_ids = (
        persons.group_by("unique-id")
        .agg([
            pl.col("x-center").min().alias("_x_min"),
            pl.col("x-center").max().alias("_x_max"),
        ])
        .filter((pl.col("_x_min") <= float(min_x)) & (pl.col("_x_max") >= float(max_x)))
        .select("unique-id")
        .to_series()
        .to_list()
    )
    return crossed_ids


def compute_crossing_validity_maps(
    df: pl.DataFrame,
    candidate_ids: list,
    *,
    person_type_map: dict,
    ratio_thresh: float = 0.6,
    rel_disp_thresh: float = 0.05,
    min_shared_frames: int = 8
) -> tuple[dict, dict]:
    df = _dedup_per_frame(df)

    crossing_valid_map: Dict[Any, bool] = {}
    crossing_debug_map: Dict[Any, dict] = {}

    for uid in candidate_ids:
        ok, dbg = is_valid_crossing_debug(
            df,
            uid,
            ratio_thresh=ratio_thresh,
            rel_disp_thresh=rel_disp_thresh,
            min_shared_frames=min_shared_frames,
        )
        crossing_valid_map[uid] = bool(ok)
        crossing_debug_map[uid] = dbg

    return crossing_valid_map, crossing_debug_map


# ============================================================
# Overlay (trim by start_seconds; align by frame-count only)
# ============================================================
def overlay_video(
    video_path: str,
    csv_path: str,
    output_path: str,
    start_seconds: float,
    person_type_map: Dict[Any, Optional[str]],
    *,
    draw_classes=None,
    show_ids: bool = True,
    show_conf: bool = True,
    frame_offset: int = 0,
    cross_min_x: float = 0.3,
    cross_max_x: float = 0.7,
    crossing_valid_map: Optional[Dict[Any, bool]] = None,
    crossing_debug_map: Optional[Dict[Any, dict]] = None
):
    df = pl.read_csv(csv_path)

    required = {"yolo-id", "x-center", "y-center", "width", "height", "unique-id", "confidence", "frame-count"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing required columns: {sorted(missing)}")

    df = _dedup_per_frame(df)
    frame_map = build_frame_index(df)
    if not frame_map:
        raise ValueError("No detections found in CSV.")

    fc_min = min(frame_map.keys())
    fc_max = max(frame_map.keys())

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, float(start_seconds)) * 1000.0)

    ret, frame = cap.read()
    if not ret:
        cap.release()
        raise RuntimeError(f"Could not read video after seeking to {start_seconds}s.")

    H, W = frame.shape[:2]

    out_fps = cap.get(cv2.CAP_PROP_FPS)
    if not out_fps or out_fps <= 0:
        out_fps = 30.0

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # type: ignore
    out = cv2.VideoWriter(output_path, fourcc, float(out_fps), (W, H))
    if not out.isOpened():
        cap.release()
        raise RuntimeError(f"Could not open VideoWriter for: {output_path}")

    seg_i = 0

    bx1 = int(round(float(cross_min_x) * W))
    bx2 = int(round(float(cross_max_x) * W))

    def draw_boundaries(frm):
        cv2.line(frm, (bx1, 0), (bx1, H - 1), (255, 255, 255), 2)
        cv2.line(frm, (bx2, 0), (bx2, H - 1), (255, 255, 255), 2)
        cv2.putText(frm, f"min_x={cross_min_x:.2f}", (max(0, bx1 - 60), 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frm, f"max_x={cross_max_x:.2f}", (max(0, bx2 - 60), 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

    def draw_frame(frm, csv_fc: int):
        dets = frame_map.get(csv_fc, [])
        for (yid, uid, conf, xc, yc, w, h) in dets:
            if draw_classes is not None and yid not in draw_classes:
                continue

            x1, y1, x2, y2 = xywh_to_xyxy_norm(xc, yc, w, h, W, H)

            color = det_color(yid, person_id=uid, person_type_map=person_type_map)

            cross_tag = ""
            dbg2 = None
            if yid == PERSON_CLASS and crossing_valid_map is not None and uid in crossing_valid_map:
                ok = bool(crossing_valid_map[uid])
                color = COLOR_CROSS_VALID if ok else COLOR_CROSS_FAKE
                cross_tag = "CROSS:VALID" if ok else "CROSS:FAKE"
                if crossing_debug_map is not None:
                    dbg2 = crossing_debug_map.get(uid)

            cv2.rectangle(frm, (x1, y1), (x2, y2), color, 2)

            label = class_name(yid)

            if yid == PERSON_CLASS:
                ptype = person_type_map.get(uid)
                if ptype == "bicycle":
                    label += " (bicyclist)"
                elif ptype == "motorcycle":
                    label += " (motorcyclist)"
                elif ptype in ("car", "bus", "truck"):
                    label += f" (passenger:{ptype})"

            if cross_tag:
                label += f"  {cross_tag}"

            if show_ids:
                label += f" id={uid}"
            if show_conf:
                label += f" {conf:.2f}"

            ty = max(0, y1 - 7)
            cv2.putText(frm, label, (x1, ty),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

            if dbg2 and cross_tag:
                ratio = dbg2.get("ratio", None)
                relx = dbg2.get("relx_rng", None)
                sid = dbg2.get("static_uid", None)
                sh = dbg2.get("shared_frames", None)
                reason = dbg2.get("reason", "")
                if ratio is not None and relx is not None:
                    dbg_line = f"{reason} ratio={ratio:.2f} relx={relx:.3f} static={sid} shared={sh}"
                else:
                    dbg_line = f"{reason}"
                cv2.putText(frm, dbg_line, (x1, min(H - 5, y2 + 15)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

        if crossing_valid_map is not None:
            draw_boundaries(frm)

        dbg = f"trim_start={start_seconds}s  seg_i={seg_i}  csv_frame={csv_fc} (min={fc_min}, max={fc_max})"
        cv2.putText(frm, dbg, (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (240, 240, 240), 2, cv2.LINE_AA)

    csv_fc = fc_min + frame_offset + seg_i
    draw_frame(frame, csv_fc)
    out.write(frame)
    written = 1

    while True:
        seg_i += 1
        csv_fc = fc_min + frame_offset + seg_i
        if csv_fc > fc_max:
            break

        ret, frame = cap.read()
        if not ret:
            break

        draw_frame(frame, csv_fc)
        out.write(frame)
        written += 1

    cap.release()
    out.release()

    logger.info("\nOverlay complete.")
    logger.info(f"Video in: {video_path}")
    logger.info(f"CSV used: {csv_path}")
    logger.info(f"Output: {output_path}")
    logger.info(f"Trim start: {start_seconds} seconds")
    logger.info(f"CSV frames: {fc_min} .. {fc_max}")
    logger.info("Coord mode:  normalised(0-1)")
    logger.info(f"Written: {written} frames")


# ============================================================
# EDIT ONLY THIS SECTION (single video/csv; runs BOTH analyses)
# ============================================================
video_id = "gYKMob1pfcQ"

video_path = os.path.join("videos", f"{video_id}.mp4")
csv_path = find_csv_for_video(video_id, data_dir="data/bbox")

_, start_seconds, _fps_str = parse_csv_filename(csv_path)
output_path = os.path.join("videos", f"{video_id}_overlay_{int(start_seconds)}s.mp4")

# IMPORTANT for metadata lookup:
# video_id for mapping must be COMPLETE: {video}_{start_time}_{fps}
# safest is to use the CSV stem (exactly matches {video_id}_{start}_{fps}.csv):
video_id_full = os.path.splitext(os.path.basename(csv_path))[0]

# Mapping CSV (you will provide this file)
# Set this path to your df_mapping CSV file:
DF_MAPPING_CSV_PATH = common.get_configs("mapping")

# If you want to see static refs (traffic light/bench/etc), keep None.
# If you want only people + vehicles, use the list.
draw_classes = None  # or: [PERSON_CLASS, BICYCLE_CLASS, MOTORCYCLE_CLASS, CAR_CLASS, BUS_CLASS, TRUCK_CLASS]

RIDER_PARAMS = dict(
    min_shared_frames=4,
    dist_rel_thresh=0.8,
    prox_req=0.7,
    alpha_x=1.0,
    beta_y=0.03,
    gamma_y=1.4,
    coloc_req=0.7,
    sim_thresh=0.4,
    sim_req=0.5,
    min_motion_steps=3,
    motion_coloc_min=0.5,
    short_shared_frames=8,
    short_sim_req=0.8,
    short_disp_req=0.12,
    # avg_height will be injected from df_mapping below (not defaulted)
)

FRAME_OFFSET = 0

CROSS_MIN_X = common.get_configs("boundary_left")
CROSS_MAX_X = common.get_configs("boundary_right")

CROSS_RATIO_THRESH = 0.60
CROSS_RELX_THRESH = 0.05
CROSS_MIN_SHARED = 8


# ============================================================
# RUN (single pipeline: riders+passengers + crossing candidates/validity + overlay)
# ============================================================
df = pl.read_csv(csv_path)
df = _dedup_per_frame(df)

# ------------------------------------------------------------
# Metadata lookup for average height (used by the existing rider filter)
# avg_height is sourced from df_mapping using the FULL id: {video}_{start_time}_{fps}
# ------------------------------------------------------------
avg_height = None
try:
    df_mapping = pl.read_csv(common.get_configs("mapping"))
    result = metadata.find_values_with_video_id(df_mapping, video_id_full)
    if result is not None:
        avg_height = result[15]
except Exception:
    logger.error(f"avg_height lookup failed for video_id_full='{video_id_full}' using mapping='{DF_MAPPING_CSV_PATH}'")

# normalise avg_height into float if possible
if avg_height is not None:
    try:
        avg_height = float(avg_height)
        if avg_height <= 0.0:
            avg_height = None
    except Exception:
        avg_height = None

logger.info(f"avg_height (from mapping) for video_id_full='{video_id_full}': {avg_height}")

# inject into rider params (no default human height)
if avg_height:
    RIDER_PARAMS["avg_height"] = avg_height
else:
    RIDER_PARAMS["avg_height"] = 170

# 1) Riders + passengers
person_type_map, _full_results = classify_all_persons_and_print(df, **RIDER_PARAMS)

# 2) Crossing candidates
candidate_ids = compute_crossing_candidate_ids(df, min_x=CROSS_MIN_X, max_x=CROSS_MAX_X)

logger.info("\n=== IDS DETECTED AS CROSSING (by x-boundaries) ===")
logger.info(f"Count: {len(candidate_ids)}")
for uid in candidate_ids:
    logger.info(f"crossing_detected unique-id={uid} rider_type={person_type_map.get(uid)}")

# 3) Validity on crossing candidates
crossing_valid_map, crossing_debug_map = compute_crossing_validity_maps(
    df,
    candidate_ids,
    person_type_map=person_type_map,
    ratio_thresh=CROSS_RATIO_THRESH,
    rel_disp_thresh=CROSS_RELX_THRESH,
    min_shared_frames=CROSS_MIN_SHARED,
)

logger.info("\n=== CROSSING VALIDITY (non-rider only) ===")
valid_ids = [uid for uid, ok in crossing_valid_map.items() if ok]
fake_ids = [uid for uid, ok in crossing_valid_map.items() if not ok]
logger.info(f"VALID crossing ids: {valid_ids}")
logger.info(f"FAKE  crossing ids: {fake_ids}")

# 4) Overlay
overlay_video(
    video_path=video_path,
    csv_path=csv_path,
    output_path=output_path,
    start_seconds=start_seconds,
    person_type_map=person_type_map,
    draw_classes=draw_classes,
    show_ids=True,
    show_conf=True,
    frame_offset=FRAME_OFFSET,
    cross_min_x=CROSS_MIN_X,
    cross_max_x=CROSS_MAX_X,
    crossing_valid_map=crossing_valid_map,
    crossing_debug_map=crossing_debug_map,
)
