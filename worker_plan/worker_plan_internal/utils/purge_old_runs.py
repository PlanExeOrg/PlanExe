import datetime
import logging
import os
import shutil
import threading
import time
import uuid

logger = logging.getLogger(__name__)

_START_TIME_SUFFIX = "start_time.json"
_PLAN_SUFFIX = "plan.txt"


def _is_uuid_name(name: str) -> bool:
    try:
        parsed = uuid.UUID(name)
    except ValueError:
        return False
    return str(parsed) == name


def _looks_like_plan_run_dir(dirname: str, path: str) -> bool:
    """A UUID-style run directory with required marker files."""
    if not _is_uuid_name(dirname):
        return False
    if not os.path.isdir(path):
        return False
    try:
        filenames = os.listdir(path)
    except OSError:
        return False
    has_start_time = any(name.endswith(_START_TIME_SUFFIX) for name in filenames)
    has_plan = any(name.endswith(_PLAN_SUFFIX) for name in filenames)
    return has_start_time and has_plan


def _looks_like_legacy_prefixed_run_dir(dirname: str, path: str, prefix: str) -> bool:
    """Compatibility mode for legacy non-UUID run directory names when prefix filtering is explicitly used."""
    if not prefix:
        return False
    if not dirname.startswith(prefix):
        return False
    if _is_uuid_name(dirname):
        return False
    if not os.path.isdir(path):
        return False
    try:
        filenames = os.listdir(path)
    except OSError:
        return False
    has_start_time = any(name.endswith(_START_TIME_SUFFIX) for name in filenames)
    has_plan = any(name.endswith(_PLAN_SUFFIX) for name in filenames)
    return has_start_time and has_plan


def purge_old_runs(run_dir: str, max_age_hours: float = 1.0, prefix: str = "") -> None:
    """
    Delete UUID-named run directories older than max_age_hours.

    Compatibility shim: when prefix is explicitly provided, legacy prefixed non-UUID run
    directory names are also considered purge candidates.
    """
    if not os.path.isabs(run_dir):
        raise ValueError(f"run_dir must be an absolute path: {run_dir}")

    if not os.path.exists(run_dir):
        logger.error(f"run_dir does not exist: {run_dir} -- skipping purge")
        return

    logger.info("Running purge...")
    now = datetime.datetime.now()
    cutoff = now - datetime.timedelta(hours=max_age_hours)

    count_deleted = 0
    count_skip_prefix_filter = 0
    count_skip_recent = 0
    count_skip_non_run_shape = 0
    count_error = 0
    for item in os.listdir(run_dir):
        if prefix and not item.startswith(prefix) and not _is_uuid_name(item):
            count_skip_prefix_filter += 1
            continue

        item_path = os.path.join(run_dir, item)
        is_dir = os.path.isdir(item_path)
        if not is_dir:
            # Never delete files from run root. Users may place arbitrary files there.
            count_skip_non_run_shape += 1
            continue

        is_uuid_run = _looks_like_plan_run_dir(item, item_path)
        is_legacy_prefixed_run = _looks_like_legacy_prefixed_run_dir(item, item_path, prefix)
        if not (is_uuid_run or is_legacy_prefixed_run):
            count_skip_non_run_shape += 1
            continue

        try:
            # Get the modification time of the item (file or directory)
            mtime = datetime.datetime.fromtimestamp(os.path.getmtime(item_path))

            if mtime < cutoff:
                logger.debug(f"Deleting old data: {item} from {run_dir}")
                if is_dir:
                    shutil.rmtree(item_path)  # Delete the directory and all its contents
                else:
                    os.remove(item_path)  # Delete the file
                count_deleted += 1
            else:
                logger.debug(f"Skipping {item} in {run_dir}, last modified: {mtime}")
                count_skip_recent += 1

        except Exception as e:
            logger.error(f"Error processing {item} in {run_dir}: {e}")
            count_error += 1
    logger.info(
        "Purge complete: %s deleted, %s skipped (recent), %s skipped (prefix filter), %s skipped (not run artifacts), %s errors",
        count_deleted,
        count_skip_recent,
        count_skip_prefix_filter,
        count_skip_non_run_shape,
        count_error,
    )


def start_purge_scheduler(
    run_dir: str,
    purge_interval_seconds: float = 3600,
    max_age_hours: float = 1.0,
    prefix: str = "",
) -> None:
    """
    Start the purge scheduler in a background thread.
    """
    logger.info(
        "Starting purge scheduler for %s every %s seconds. Prefix: %s. Max age hours: %s",
        run_dir,
        purge_interval_seconds,
        prefix,
        max_age_hours,
    )

    if not os.path.isabs(run_dir):
        raise ValueError(f"run_dir must be an absolute path: {run_dir}")

    def purge_scheduler():
        """
        Schedules the purge_old_runs function to run periodically.
        """
        while True:
            purge_old_runs(run_dir, max_age_hours=max_age_hours, prefix=prefix)
            time.sleep(purge_interval_seconds)

    purge_thread = threading.Thread(target=purge_scheduler, daemon=True)
    purge_thread.start()
