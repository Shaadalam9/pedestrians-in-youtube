# by Shadab Alam <md_shadab_alam@outlook.com> and Pavlo Bazilinskyy <pavlo.bazilinskyy@gmail.com>
"""Main pipeline runner.

This script orchestrates a multi-stage video processing pipeline that:
  - Parses a mapping CSV describing videos and segment time ranges.
  - Discovers/skips already-processed segments by indexing existing outputs once per pass.
  - Prefetches videos (cache/FTP/YouTube fallback depending on configuration).
  - Trims video segments and runs tracking/segmentation on each segment.
  - Optionally writes annotated videos (Ultralytics plot overlays) per segment.
  - Writes outputs using a write-then-commit strategy to avoid partial final artifacts.

Annotated video behavior:
  - When config.save_annotated_video is True, the pipeline will create
    data/annotated/{vid}_{start}_{fps}_ann.mp4 for each segment it processes.
  - If bbox/seg CSVs already exist but the annotated MP4 is missing, the segment
    will be scheduled again to (re)run YOLO and produce the annotated MP4, while
    *not* overwriting existing CSVs.

HPC/Snellius notes:
  - When snellius_mode is enabled, concurrency is intentionally constrained to avoid
    oversubscription and to align with Slurm scheduling (one task/process per GPU).
  - Sharding can be done by video or by segment (config.snellius_shard_mode).
  - Node-local scratch ($TMPDIR) may be used to stage videos/trims for performance.
"""

import ast
import inspect
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from datetime import datetime
from types import SimpleNamespace
from typing import Optional, Dict, Any, List, Tuple, Set
import pandas as pd
from tqdm import tqdm
import common
from dataclasses import dataclass

from logmod import logs
from custom_logger import CustomLogger
from video_io import VideoIO
from patches import Patches
from config_utils import ConfigUtils
from processing.output_paths import OutputPaths
from maintenence.maintenence import Maintenance
from processing.tracking_runner import TrackingRunner
from hpc.snellius import HPC
from dataset_enrichment import DatasetEnrichment
from processing.output_index import OutputIndex
from parsing_utils import ParsingUtils


# Initialize global logging configuration early so downstream modules inherit it.
logs(show_level=common.get_configs("logger_level"), show_color=True)
logger = CustomLogger(__name__)

# Instantiate helper classes (shared singletons for this run).
patches_class = Patches()
configutils_class = ConfigUtils()
videoio_class = VideoIO()
outputpath_class = OutputPaths()
maintenance_class = Maintenance()
tracckingrunner_class = TrackingRunner()
hpc_class = HPC()
datasetenrich_class = DatasetEnrichment()
outputindex_class = OutputIndex()
parsingutils_class = ParsingUtils()

# Make tqdm updates thread-safe across worker threads.
tqdm.set_lock(threading.RLock())


@dataclass
class VideoReq:
    """Per-video request built from the mapping."""
    vid: str
    segments: List[Tuple[int, int]]
    city: str = ""
    state: str = ""
    country: str = ""
    iso3: str = ""


@dataclass
class VideoCtx:
    """Runtime context for a video currently being processed."""
    vid: str
    base_video_path: str
    fps: int
    resolution: str
    ftp_download: bool
    output_path: str
    external_ssd: bool
    delete_youtube_video: bool
    pending: int = 0
    processed_any: bool = False


# Segment job tuple:
# (vid, st, et, base_video_path, do_bbox, do_seg,
#  bbox_final, seg_final, bbox_tmp, seg_tmp,
#  ann_final, ann_tmp, fps)
SegmentJob = Tuple[
    str, int, int, str, bool, bool,
    Optional[str], Optional[str], Optional[str], Optional[str],
    Optional[str], Optional[str], int
]


@dataclass
class DownloadResult:
    """Result returned by the download-and-prepare stage."""
    vid: str
    base_video_path: str
    fps: int
    resolution: str
    ftp_download: bool
    segment_jobs: List[SegmentJob]
    elapsed_sec: float


def _cleanup_partial_files_any(tmp_dir: str, token: str = ".partial") -> None:
    """Best-effort cleanup for partial files.

    Unlike OutputPaths._cleanup_stale_partials (which targets '*.partial'),
    this helper removes any file whose name contains `token` so it also covers
    partial MP4s like '*.partial.mp4'.
    """
    try:
        with os.scandir(tmp_dir) as it:
            for entry in it:
                if not entry.is_file():
                    continue
                if token not in entry.name:
                    continue
                try:
                    os.remove(entry.path)
                except Exception:
                    pass
    except Exception:
        pass


def _index_existing_annotated_outputs(data_folders: List[str]) -> Set[Tuple[str, int]]:
    """Indexes existing annotated MP4 outputs once per pass.

    Expected naming:
      annotated/{vid}_{start}_{fps}_ann.mp4

    We index by (vid, start) and ignore fps, matching the CSV "DONE" policy.
    """
    out: Set[Tuple[str, int]] = set()

    for base in data_folders:
        d = os.path.join(base, "annotated")
        if not os.path.isdir(d):
            continue
        try:
            with os.scandir(d) as it:
                for entry in it:
                    if not entry.is_file():
                        continue
                    name = entry.name
                    if not name.endswith(".mp4"):
                        continue
                    stem = name[:-4]
                    try:
                        # rsplit preserves underscores in video_id
                        vid_part, st_str, _fps_str, tag = stem.rsplit("_", 3)
                        if tag != "ann":
                            continue
                        out.add((vid_part, int(st_str)))
                    except Exception:
                        continue
        except Exception:
            continue

    return out


if __name__ == "__main__":
    counter_processed = 0

    try:
        patches_class.patch_ultralytics_botsort_numpy_cpu_bug(logger)
        patches_class.patch_scipy_cdist_accept_1d(logger)

        # ---------------------------------------------------------------------
        # Load configuration
        # ---------------------------------------------------------------------
        max_workers = max(1, int(configutils_class._safe_get_config("max_workers", 3)))  # type: ignore
        download_workers = max(1, int(configutils_class._safe_get_config("download_workers", 1)))  # type: ignore
        max_active_segments_per_video = max(
            1, int(configutils_class._safe_get_config("max_active_segments_per_video", 1))  # type: ignore
        )

        prefetch_raw = int(configutils_class._safe_get_config("prefetch_videos", 0) or 0)
        prefetch_videos = prefetch_raw if prefetch_raw > 0 else (max_workers + 1)
        prefetch_videos = max(prefetch_videos, max_workers + 1)  # enforce N+1

        config = SimpleNamespace(
            mapping=common.get_configs("mapping"),
            videos=common.get_configs("videos"),
            data=common.get_configs("data"),
            tracking_model=common.get_configs("tracking_model"),
            segment_model=common.get_configs("segment_model"),
            bbox_tracker=common.get_configs("bbox_tracker"),
            seg_tracker=common.get_configs("seg_tracker"),
            countries_analyse=configutils_class._safe_get_config("countries_analyse", []),
            update_pop_country=configutils_class._safe_get_config("update_pop_country", False),
            update_gini_value=configutils_class._safe_get_config("update_gini_value", False),
            segmentation_mode=configutils_class._safe_get_config("segmentation_mode", False),
            tracking_mode=configutils_class._safe_get_config("tracking_mode", False),
            delete_youtube_video=configutils_class._safe_get_config("delete_youtube_video", False),
            external_ssd=configutils_class._safe_get_config("external_ssd", False),
            track_buffer_sec=configutils_class._safe_get_config("track_buffer_sec", 1),
            save_annotated_video=bool(configutils_class._safe_get_config("save_annotated_video", False)),
            sleep_sec=configutils_class._safe_get_config("sleep_sec", 0),
            git_pull=configutils_class._safe_get_config("git_pull", False),
            machine_name=configutils_class._safe_get_config("machine_name", "unknown"),
            email_send=configutils_class._safe_get_config("email_send", False),
            email_sender=configutils_class._safe_get_config("email_sender", ""),
            email_recipients=configutils_class._safe_get_config("email_recipients", []),
            ftp_server=common.get_configs("ftp_server"),
            max_workers=int(max_workers),
            download_workers=int(download_workers),
            prefetch_videos=int(prefetch_videos),
            max_active_segments_per_video=int(max_active_segments_per_video),
            snellius_mode=bool(configutils_class._safe_get_config("snellius_mode", False)),
            snellius_shard_mode=str(configutils_class._safe_get_config("snellius_shard_mode", "video") or "video").strip().lower(),  # noqa: E501
            snellius_tmp_root=str(configutils_class._safe_get_config("snellius_tmp_root", "") or ""),
            snellius_disable_youtube_fallback=bool(configutils_class._safe_get_config("snellius_disable_youtube_fallback", False)),  # noqa: E501
        )

        # Check TrackingRunner support for annotated_video_out
        _sig = inspect.signature(tracckingrunner_class.tracking_mode_threadsafe)
        _supports_annot = ("annotated_video_out" in _sig.parameters)
        if bool(config.save_annotated_video) and not _supports_annot:
            raise RuntimeError(
                "save_annotated_video=True but TrackingRunner.tracking_mode_threadsafe(...) "
                "does not accept parameter 'annotated_video_out'. "
                "Update tracking_runner.py to add annotated video writing."
            )

        # ---------------------------------------------------------------------
        # Snellius/Slurm context + runtime overrides
        # ---------------------------------------------------------------------
        config.slurm_task_id, config.slurm_task_count = hpc_class._snellius_task_identity()
        config.tmpdir = hpc_class._resolve_tmp_root(config)

        if bool(config.snellius_mode):
            config.max_workers = 1
            config.download_workers = 1
            config.max_active_segments_per_video = 1
            config.prefetch_videos = max(2, int(config.prefetch_videos or 2))

            config.sleep_sec = 0
            config.git_pull = False
            config.email_send = False
            config.delete_youtube_video = False
            config.external_ssd = False

            hpc_class._maybe_bind_gpu_from_slurm_localid(config)

        # ---------------------------------------------------------------------
        # Load secrets
        # ---------------------------------------------------------------------
        secret = SimpleNamespace(
            email_smtp=common.get_secrets("email_smtp"),
            email_account=common.get_secrets("email_account"),
            email_password=common.get_secrets("email_password"),
            ftp_username=common.get_secrets("ftp_username"),
            ftp_password=common.get_secrets("ftp_password"),
        )

        hpc_class._log_run_banner(config)

        pass_index = 0
        while True:
            pass_index += 1
            pass_start_ts = time.time()

            mapping = pd.read_csv(config.mapping)
            video_paths = config.videos

            if config.external_ssd:
                internal_ssd = config.videos[-1]
                os.makedirs(internal_ssd, exist_ok=True)
                output_path = config.videos[-1]
            else:
                internal_ssd = None
                output_path = config.videos[-1]

            scratch_videos_dir: Optional[str] = None
            if bool(config.snellius_mode) and getattr(config, "tmpdir", ""):
                scratch_videos_dir = os.path.join(str(config.tmpdir), "videos")
                os.makedirs(scratch_videos_dir, exist_ok=True)

            data_folders = config.data
            data_path = config.data[-1]
            countries_analyse = config.countries_analyse or []
            counter_processed = 0

            bbox_mode_cfg = bool(config.tracking_mode)
            seg_mode_cfg = bool(config.segmentation_mode)
            ann_mode_cfg = bool(config.save_annotated_video)

            # If the user wants annotated videos, we must run YOLO in at least one mode.
            # If both bbox/seg are disabled, we still run bbox tracking solely for annotation.
            need_any_processing = bool(bbox_mode_cfg or seg_mode_cfg or ann_mode_cfg)

            logger.info(
                f"=== Pass {pass_index} started === rows={mapping.shape[0]} "
                f"max_workers={config.max_workers} prefetch_videos={config.prefetch_videos} "
                f"download_workers={config.download_workers} "
                f"tracking_mode={bbox_mode_cfg} seg_mode={seg_mode_cfg} save_annotated_video={ann_mode_cfg} "
                f"shard_mode={getattr(config, 'snellius_shard_mode', 'video')}"
            )

            if config.update_pop_country:
                logger.info("Updating population in mapping file...")
                datasetenrich_class.update_population_in_csv(mapping)

            if config.update_gini_value:
                logger.info("Updating GINI values in mapping file...")
                datasetenrich_class.fill_gini_data(mapping)

            maintenance_class.delete_folder(folder_path="runs")
            outputpath_class._ensure_dirs(data_path)

            # Ensure additional dirs for annotated output
            annot_dir = os.path.join(data_path, "annotated")
            os.makedirs(annot_dir, exist_ok=True)

            tmp_bbox_dir, tmp_seg_dir = outputpath_class._ensure_tmp_dirs(data_path)
            outputpath_class._cleanup_stale_partials(tmp_bbox_dir)
            outputpath_class._cleanup_stale_partials(tmp_seg_dir)

            tmp_annot_dir = os.path.join(data_path, "__tmp__", "annotated")
            os.makedirs(tmp_annot_dir, exist_ok=True)
            _cleanup_partial_files_any(tmp_annot_dir)

            if not need_any_processing:
                logger.info("tracking_mode, segmentation_mode and save_annotated_video are all disabled; nothing to do.")  # noqa: E501
                break

            # -----------------------------------------------------------------
            # Index existing outputs ONCE per pass (skip work + skip downloads)
            # -----------------------------------------------------------------
            existing_idx = outputindex_class._index_existing_outputs(
                data_folders=data_folders,
                want_bbox=bbox_mode_cfg,
                want_seg=seg_mode_cfg,
            )
            bbox_done_start = existing_idx["bbox_start"]
            seg_done_start = existing_idx["seg_start"]

            ann_done_start: Set[Tuple[str, int]] = set()
            if ann_mode_cfg:
                ann_done_start = _index_existing_annotated_outputs(data_folders)

            done_lock = threading.Lock()

            logger.info(
                f"Existing outputs indexed: bbox_start={len(bbox_done_start)} seg_start={len(seg_done_start)} "
                f"ann_start={len(ann_done_start)} (done by (vid,start) ignoring fps)"
            )

            # -----------------------------------------------------------------
            # Stage 1: Build per-video requests
            # -----------------------------------------------------------------
            req_by_vid: Dict[str, VideoReq] = {}

            shard_mode = str(getattr(config, "snellius_shard_mode", "video") or "video").strip().lower()
            use_sharding = bool(config.snellius_mode) and int(getattr(config, "slurm_task_count", 1)) > 1

            pbar_rows = tqdm(
                mapping.iterrows(),
                total=mapping.shape[0],
                desc="Mapping rows",
                dynamic_ncols=True,
                position=0,
                leave=True,
            )

            for _, row in pbar_rows:
                video_ids = parsingutils_class._parse_bracket_list(str(row.get("videos", "")))
                if not video_ids:
                    continue

                iso3 = str(row.get("iso3", ""))
                if countries_analyse and iso3 and iso3 not in countries_analyse:
                    continue

                city = str(row.get("city", ""))
                state = str(row.get("state", ""))
                country = str(row.get("country", ""))

                try:
                    start_times = ast.literal_eval(row["start_time"])
                    end_times = ast.literal_eval(row["end_time"])
                except Exception:
                    logger.warning("Failed to parse start_time/end_time for a row; skipping row.")
                    continue

                if not isinstance(start_times, list) or not isinstance(end_times, list):
                    continue

                for i, vid in enumerate(video_ids):
                    st_list = start_times[i] if i < len(start_times) else []
                    et_list = end_times[i] if i < len(end_times) else []
                    if not isinstance(st_list, list) or not isinstance(et_list, list):
                        continue

                    for st, et in zip(st_list, et_list):
                        st_i = int(st)
                        et_i = int(et)

                        bbox_done = (not bbox_mode_cfg) or ((vid, st_i) in bbox_done_start)
                        seg_done = (not seg_mode_cfg) or ((vid, st_i) in seg_done_start)
                        ann_done = (not ann_mode_cfg) or ((vid, st_i) in ann_done_start)

                        # Schedule if ANY required artifact is missing.
                        if bbox_done and seg_done and ann_done:
                            continue

                        if use_sharding and shard_mode == "segment":
                            owner = hpc_class._shard_to_task(f"{vid}:{st_i}", int(config.slurm_task_count))
                            if owner != int(config.slurm_task_id):
                                continue

                        if vid not in req_by_vid:
                            req_by_vid[vid] = VideoReq(
                                vid=vid,
                                segments=[],
                                city=city,
                                state=state,
                                country=country,
                                iso3=iso3,
                            )

                        seg = (st_i, et_i)
                        if seg not in req_by_vid[vid].segments:
                            req_by_vid[vid].segments.append(seg)

            pbar_rows.close()

            video_reqs = list(req_by_vid.values())
            logger.info(f"Pass {pass_index}: videos needing work={len(video_reqs)} (after done-index pre-check).")

            if use_sharding and shard_mode == "video":
                task_id = int(getattr(config, "slurm_task_id", 0))
                task_count = int(getattr(config, "slurm_task_count", 1))
                video_reqs = sorted(video_reqs, key=lambda r: r.vid)
                video_reqs = [vr for i, vr in enumerate(video_reqs) if (i % task_count) == task_id]
                logger.info(
                    "Snellius video-sharding applied: task_id=%d task_count=%d videos_assigned=%d",
                    task_id,
                    task_count,
                    len(video_reqs),
                )

            # -----------------------------------------------------------------
            # Stage 2: Bounded prefetch + global segment pool with per-video cap
            # -----------------------------------------------------------------
            ctx_lock = threading.Lock()
            ctx_by_vid: Dict[str, VideoCtx] = {}

            ready_jobs_by_vid: Dict[str, List[SegmentJob]] = {}
            active_segments_by_vid: Dict[str, int] = {}
            rr_vids: List[str] = []
            rr_state = {"idx": 0}

            pbar_segs = tqdm(total=0, desc="Segments completed", unit="seg",
                             dynamic_ncols=True, position=1, leave=True)

            def _maybe_add_rr_vid(vid: str) -> None:
                if vid in rr_vids:
                    return
                if ready_jobs_by_vid.get(vid):
                    rr_vids.append(vid)

            def _maybe_remove_rr_vid(vid: str) -> None:
                if vid not in rr_vids:
                    return
                try:
                    idx = rr_vids.index(vid)
                except ValueError:
                    return
                rr_vids.pop(idx)
                rr_state["idx"] = (rr_state["idx"] % len(rr_vids)) if rr_vids else 0

            def _get_or_create_ctx(vid: str, base_video_path: str, fps: int, resolution: str,
                                   ftp_download: bool) -> VideoCtx:
                with ctx_lock:
                    if vid not in ctx_by_vid:
                        ctx_by_vid[vid] = VideoCtx(
                            vid=vid,
                            base_video_path=base_video_path,
                            fps=int(fps),
                            resolution=resolution,
                            ftp_download=bool(ftp_download),
                            output_path=output_path,
                            external_ssd=bool(
                                config.external_ssd or (bool(config.snellius_mode) and bool(scratch_videos_dir))
                            ),
                            delete_youtube_video=bool(config.delete_youtube_video),
                            pending=0,
                            processed_any=False,
                        )
                        active_segments_by_vid[vid] = 0
                    else:
                        ctx_by_vid[vid].base_video_path = base_video_path
                        ctx_by_vid[vid].fps = int(fps)
                        ctx_by_vid[vid].resolution = resolution
                        ctx_by_vid[vid].ftp_download = bool(ftp_download)
                    return ctx_by_vid[vid]

            def _inflight_video_count(inflight_downloads: Dict[Any, VideoReq]) -> int:
                with ctx_lock:
                    return len(inflight_downloads) + len(ctx_by_vid)

            def _finalize_video_if_done(vid: str) -> None:
                with ctx_lock:
                    ctx = ctx_by_vid.get(vid)
                    if not ctx:
                        return

                    pending_now = ctx.pending
                    processed_any = ctx.processed_any
                    active_now = active_segments_by_vid.get(vid, 0)
                    has_ready = bool(ready_jobs_by_vid.get(vid))

                    if pending_now > 0 or active_now > 0 or has_ready:
                        return

                    ctx_snapshot = VideoCtx(**ctx.__dict__)

                    ctx_by_vid.pop(vid, None)
                    active_segments_by_vid.pop(vid, None)
                    ready_jobs_by_vid.pop(vid, None)
                    _maybe_remove_rr_vid(vid)

                if not processed_any:
                    return

                if ctx_snapshot.external_ssd:
                    try:
                        os.remove(ctx_snapshot.base_video_path)
                        logger.info(f"{vid}: cleaned SSD/scratch working copy: {ctx_snapshot.base_video_path}")
                    except FileNotFoundError:
                        pass
                    except Exception as e:
                        logger.warning(f"{vid}: failed SSD/scratch cleanup: {e}")

                if ctx_snapshot.ftp_download:
                    try:
                        os.remove(ctx_snapshot.base_video_path)
                        logger.info(f"{vid}: removed FTP-downloaded working copy: {ctx_snapshot.base_video_path}")
                    except FileNotFoundError:
                        pass
                    except Exception as e:
                        logger.warning(f"{vid}: failed FTP cleanup: {e}")

                if ctx_snapshot.delete_youtube_video:
                    try:
                        os.remove(os.path.join(ctx_snapshot.output_path, f"{vid}.mp4"))
                        logger.info(f"{vid}: deleted YouTube video due to delete_youtube_video=True")
                    except FileNotFoundError:
                        pass
                    except Exception as e:
                        logger.warning(f"{vid}: failed delete_youtube_video cleanup: {e}")

            def _download_and_prepare(req: VideoReq) -> DownloadResult:
                t0 = time.time()
                vid = req.vid

                base_video_path, _title, resolution, video_fps, ftp_download = videoio_class._ensure_video_available(
                    vid=vid,
                    config=config,
                    secret=secret,
                    output_path=output_path,
                    video_paths=video_paths,
                )

                if bool(config.snellius_mode) and scratch_videos_dir:
                    base_video_path = videoio_class._copy_to_ssd_if_needed(base_video_path, scratch_videos_dir, vid)
                elif config.external_ssd and internal_ssd:
                    base_video_path = videoio_class._copy_to_ssd_if_needed(base_video_path, internal_ssd, vid)

                fps_i = int(video_fps)

                segment_jobs: List[SegmentJob] = []

                for (st, et) in req.segments:
                    st_i = int(st)
                    et_i = int(et)

                    need_bbox_csv = bool(bbox_mode_cfg and ((vid, st_i) not in bbox_done_start))
                    need_seg_csv = bool(seg_mode_cfg and ((vid, st_i) not in seg_done_start))
                    need_ann = bool(ann_mode_cfg and ((vid, st_i) not in ann_done_start))

                    if not (need_bbox_csv or need_seg_csv or need_ann):
                        continue

                    do_seg = bool(seg_mode_cfg and (need_seg_csv or need_ann))
                    do_bbox = bool(
                        (bbox_mode_cfg and (need_bbox_csv or (need_ann and not do_seg)))
                        or ((not bbox_mode_cfg and not seg_mode_cfg) and need_ann)
                    )

                    if need_ann and not (do_bbox or do_seg):
                        do_bbox = True

                    segment_csv = f"{vid}_{st_i}_{fps_i}.csv"

                    bbox_final = os.path.join(data_path, "bbox", segment_csv) if need_bbox_csv else None
                    seg_final = os.path.join(data_path, "seg", segment_csv) if need_seg_csv else None

                    bbox_tmp = os.path.join(tmp_bbox_dir, segment_csv + ".partial") if do_bbox else None
                    seg_tmp = os.path.join(tmp_seg_dir, segment_csv + ".partial") if do_seg else None

                    ann_final = None
                    ann_tmp = None
                    if need_ann:
                        ann_name = f"{vid}_{st_i}_{fps_i}_ann.mp4"
                        ann_final = os.path.join(annot_dir, ann_name)
                        ann_tmp = os.path.join(tmp_annot_dir, f"{vid}_{st_i}_{fps_i}_ann.partial.mp4")

                    segment_jobs.append((
                        vid, st_i, et_i, base_video_path, do_bbox, do_seg,
                        bbox_final, seg_final, bbox_tmp, seg_tmp,
                        ann_final, ann_tmp, fps_i
                    ))

                dt = time.time() - t0
                return DownloadResult(
                    vid=vid,
                    base_video_path=base_video_path,
                    fps=fps_i,
                    resolution=resolution,
                    ftp_download=bool(ftp_download),
                    segment_jobs=segment_jobs,
                    elapsed_sec=dt,
                )

            def _segment_worker(job: SegmentJob) -> str:
                vid, st, et, base_path, do_bbox, do_seg, bbox_final, seg_final, bbox_tmp, seg_tmp, ann_final, ann_tmp, fps = job  # noqa: E501
                th = threading.current_thread().name
                t0 = time.time()

                mode_str = f"{'bbox' if do_bbox else ''}{'&' if do_bbox and do_seg else ''}{'seg' if do_seg else ''}"
                job_label = f"{vid} [{st}-{et}s] ({mode_str})"

                try:
                    worker_idx = int(th.split("_")[-1])
                except Exception:
                    worker_idx = 0
                tqdm_pos = 2 + (worker_idx % max(1, int(config.max_workers)))

                if bool(config.snellius_mode) and scratch_videos_dir:
                    work_dir = scratch_videos_dir
                else:
                    work_dir = internal_ssd if (config.external_ssd and internal_ssd) else output_path

                trimmed_path = os.path.join(work_dir, f"{vid}_{st}_{et}_mod.mp4")

                logger.info(
                    f"[worker-start] thread={th} job={job_label} fps={int(fps)} "
                    f"trimmed={os.path.basename(trimmed_path)} "
                    f"bbox_final={os.path.basename(bbox_final) if bbox_final else None} "
                    f"seg_final={os.path.basename(seg_final) if seg_final else None} "
                    f"ann_final={os.path.basename(ann_final) if ann_final else None}"
                )

                for p in (bbox_tmp, seg_tmp, ann_tmp):
                    if p:
                        os.makedirs(os.path.dirname(p) or ".", exist_ok=True)
                        if os.path.exists(p):
                            try:
                                os.remove(p)
                            except Exception:
                                pass

                try:
                    logger.info(f"[trim-start] thread={th} job={job_label}")
                    end_time_adj = max(st, et - 1)

                    videoio_class._trim_video_with_progress(
                        input_path=base_path,
                        output_path=trimmed_path,
                        start_time=st,
                        end_time=end_time_adj,
                        job_label=job_label,
                        tqdm_position=tqdm_pos,
                    )
                    logger.info(f"[trim-done] thread={th} job={job_label}")

                    logger.info(f"[track-start] thread={th} job={job_label} fps={int(fps)} input={trimmed_path}")

                    tracckingrunner_class.tracking_mode_threadsafe(
                        input_video_path=trimmed_path,
                        video_fps=int(fps),
                        bbox_mode=bool(do_bbox),
                        seg_mode=bool(do_seg),
                        bbox_csv_out=bbox_tmp,
                        seg_csv_out=seg_tmp,
                        annotated_video_out=ann_tmp,
                        job_label=job_label,
                        tqdm_position=tqdm_pos,
                        show_frame_pbar=(False if bool(config.snellius_mode) else True),
                        postfix_every_n=30,
                    )

                    logger.info(f"[track-done] thread={th} job={job_label}")

                    # If bbox/seg CSVs were not requested as final artifacts, remove their tmp files.
                    if bbox_tmp and not bbox_final and os.path.exists(bbox_tmp):
                        try:
                            os.remove(bbox_tmp)
                        except Exception:
                            pass

                    if seg_tmp and not seg_final and os.path.exists(seg_tmp):
                        try:
                            os.remove(seg_tmp)
                        except Exception:
                            pass

                    if bbox_tmp and bbox_final and os.path.exists(bbox_tmp):
                        os.makedirs(os.path.dirname(bbox_final), exist_ok=True)
                        os.replace(bbox_tmp, bbox_final)
                        logger.info(f"[commit] job={job_label} bbox -> {bbox_final}")
                        with done_lock:
                            bbox_done_start.add((vid, st))

                    if seg_tmp and seg_final and os.path.exists(seg_tmp):
                        os.makedirs(os.path.dirname(seg_final), exist_ok=True)
                        os.replace(seg_tmp, seg_final)
                        logger.info(f"[commit] job={job_label} seg -> {seg_final}")
                        with done_lock:
                            seg_done_start.add((vid, st))

                    if ann_tmp and ann_final and os.path.exists(ann_tmp):
                        os.makedirs(os.path.dirname(ann_final), exist_ok=True)
                        os.replace(ann_tmp, ann_final)
                        logger.info(f"[commit] job={job_label} ann -> {ann_final}")
                        with done_lock:
                            ann_done_start.add((vid, st))

                    return vid

                except Exception:
                    for p in (bbox_tmp, seg_tmp, ann_tmp):
                        if p and os.path.exists(p):
                            try:
                                os.remove(p)
                            except Exception:
                                pass
                    raise

                finally:
                    try:
                        os.remove(trimmed_path)
                    except FileNotFoundError:
                        pass
                    dt = time.time() - t0
                    logger.info(f"[worker-done] thread={th} job={job_label} elapsed_sec={dt:.2f}")

            def _dispatch_segments(inflight_process: Dict[Any, str], process_pool: ThreadPoolExecutor) -> None:
                free_slots = int(config.max_workers) - len(inflight_process)
                if free_slots <= 0:
                    return

                scheduled = 0
                for _ in range(free_slots):
                    if not rr_vids:
                        break

                    tried = 0
                    picked_vid = None
                    while tried < len(rr_vids):
                        idx = rr_state["idx"] % len(rr_vids)
                        vid = rr_vids[idx]
                        rr_state["idx"] = (idx + 1) % len(rr_vids)
                        tried += 1

                        if not ready_jobs_by_vid.get(vid):
                            continue

                        active_now = active_segments_by_vid.get(vid, 0)
                        if active_now >= int(config.max_active_segments_per_video):
                            continue

                        picked_vid = vid
                        break

                    if picked_vid is None:
                        break

                    job = ready_jobs_by_vid[picked_vid].pop(0)
                    if not ready_jobs_by_vid[picked_vid]:
                        _maybe_remove_rr_vid(picked_vid)

                    active_segments_by_vid[picked_vid] = active_segments_by_vid.get(picked_vid, 0) + 1

                    pf = process_pool.submit(_segment_worker, job)
                    inflight_process[pf] = picked_vid
                    scheduled += 1

                if scheduled:
                    logger.info(
                        f"Dispatched {scheduled} segment(s). "
                        f"active_segments={len(inflight_process)}/{config.max_workers} rr_videos={len(rr_vids)}"
                    )

            download_pool = ThreadPoolExecutor(max_workers=int(config.download_workers), thread_name_prefix="DL")
            process_pool = ThreadPoolExecutor(max_workers=int(config.max_workers), thread_name_prefix="SEG")

            inflight_downloads: Dict[Any, VideoReq] = {}
            inflight_process: Dict[Any, str] = {}

            flags = {"done_submitting": False}
            state_lock = threading.Lock()

            try:
                req_iter = iter(video_reqs)

                def _submit_downloads_up_to_prefetch() -> None:
                    target_inflight = int(config.prefetch_videos)
                    with state_lock:
                        while (
                            (not flags["done_submitting"])
                            and (len(inflight_downloads) < int(config.download_workers))
                            and (_inflight_video_count(inflight_downloads) < target_inflight)
                        ):
                            try:
                                req = next(req_iter)
                            except StopIteration:
                                flags["done_submitting"] = True
                                break

                            logger.info(
                                f"{req.vid}: queued for download+prepare "
                                f"(segments_requested={len(req.segments)}) "
                                f"inflight_videos={_inflight_video_count(inflight_downloads)+1}/{target_inflight} "
                                f"downloads_inflight={len(inflight_downloads)+1}/{config.download_workers}"
                            )
                            fut = download_pool.submit(_download_and_prepare, req)
                            inflight_downloads[fut] = req

                _submit_downloads_up_to_prefetch()
                first_exception: Optional[BaseException] = None

                while inflight_downloads or inflight_process or (not flags["done_submitting"]):
                    _submit_downloads_up_to_prefetch()
                    _dispatch_segments(inflight_process, process_pool)

                    wait_set = set(inflight_downloads.keys()) | set(inflight_process.keys())
                    if not wait_set:
                        time.sleep(0.05)
                        continue

                    done, _ = wait(wait_set, return_when=FIRST_COMPLETED)

                    for fut in done:
                        if fut in inflight_downloads:
                            req = inflight_downloads.pop(fut)
                            try:
                                dr = fut.result()

                                logger.info(
                                    f"{dr.vid}: download+prepare done fps={dr.fps} res={dr.resolution} "
                                    f"ftp_download={dr.ftp_download} segment_jobs={len(dr.segment_jobs)} "
                                    f"elapsed_sec={dr.elapsed_sec:.2f}"
                                )

                                if not dr.segment_jobs:
                                    logger.info(f"{dr.vid}: after prepare, nothing to run; skipping scheduling.")
                                    continue

                                _get_or_create_ctx(dr.vid, dr.base_video_path, dr.fps, dr.resolution, dr.ftp_download)

                                with ctx_lock:
                                    ctx_by_vid[dr.vid].pending += len(dr.segment_jobs)

                                pbar_segs.total += len(dr.segment_jobs)
                                pbar_segs.refresh()

                                ready_jobs_by_vid.setdefault(dr.vid, []).extend(dr.segment_jobs)
                                _maybe_add_rr_vid(dr.vid)

                                _dispatch_segments(inflight_process, process_pool)

                            except BaseException as e:
                                if first_exception is None:
                                    first_exception = e
                                logger.error(f"Download/prepare failed for {req.vid}: {e!r}")
                                break

                        elif fut in inflight_process:
                            vid_done = inflight_process.pop(fut)
                            try:
                                _ = fut.result()
                                counter_processed += 1
                                pbar_segs.update(1)

                                with ctx_lock:
                                    if vid_done in ctx_by_vid:
                                        ctx_by_vid[vid_done].processed_any = True
                                        ctx_by_vid[vid_done].pending = max(0, ctx_by_vid[vid_done].pending - 1)

                                active_segments_by_vid[vid_done] = max(0, active_segments_by_vid.get(vid_done, 0) - 1)

                                _finalize_video_if_done(vid_done)

                                _submit_downloads_up_to_prefetch()
                                _dispatch_segments(inflight_process, process_pool)

                            except BaseException as e:
                                if first_exception is None:
                                    first_exception = e
                                logger.error(f"Worker failed for {vid_done}: {e!r}")
                                break

                    if first_exception is not None:
                        break

                if first_exception is not None:
                    for f in list(inflight_downloads.keys()):
                        try:
                            f.cancel()
                        except Exception:
                            pass
                    for f in list(inflight_process.keys()):
                        try:
                            f.cancel()
                        except Exception:
                            pass
                    raise first_exception

            finally:
                try:
                    pbar_segs.close()
                except Exception:
                    pass
                try:
                    download_pool.shutdown(wait=True, cancel_futures=False)
                except Exception:
                    pass
                try:
                    process_pool.shutdown(wait=True, cancel_futures=False)
                except Exception:
                    pass

            if config.email_send and counter_processed:
                time_now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                common.send_email(
                    subject=f"✅ Processing job finished on machine {config.machine_name}",
                    content=(
                        f"Processing job finished on {config.machine_name} at {time_now}. "
                        f"{counter_processed} segments were processed."
                    ),
                    sender=config.email_sender,
                    recipients=config.email_recipients,
                )

            dt_pass = time.time() - pass_start_ts
            logger.info(
                f"=== Pass {pass_index} completed === segments_processed={counter_processed} elapsed_sec={dt_pass:.2f}"
            )

            if config.sleep_sec and int(config.sleep_sec) > 0:
                maintenance_class.delete_youtube_mod_videos(video_paths)
                logger.info(f"Sleeping for {config.sleep_sec} s before attempting to go over mapping again.")
                time.sleep(config.sleep_sec)
                if config.git_pull:
                    common.git_pull()
                continue

            if config.git_pull:
                common.git_pull()

            break

    except Exception as e:
        try:
            if "config" in locals() and getattr(config, "email_send", False):
                time_now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                common.send_email(
                    subject=f"‼️ Processing job crashed on machine {getattr(config, 'machine_name', 'unknown')}",
                    content=(
                        f"Processing job crashed on {getattr(config, 'machine_name', 'unknown')} at {time_now}. "
                        f"{counter_processed} segments were processed. Error message: {e}."
                    ),
                    sender=getattr(config, "email_sender", ""),
                    recipients=getattr(config, "email_recipients", []),
                )
        except Exception:
            pass
        raise
