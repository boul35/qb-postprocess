#!/usr/bin/env python3
"""
afterseed.py

A zero-dependency Python post-processing script for qBittorrent.
Handles hardlinking media files to library directories without breaking seeding.

Constraints:
    - Standard Library ONLY.
    - Single file executable.
    - Idempotent.
    - Safe (no deletions, no moves of source).
"""

import argparse
import errno
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ==========================================
# Core Data Structures
# ==========================================


@dataclass
class TorrentContext:
    """
    Represents the context provided by qBittorrent environment variables.
    Does not assume valid data; must be validated.
    """

    torrent_name: str
    save_path: Path
    category: Optional[str]
    tags: List[str]
    content_path: Path  # The actual absolute path to the content (file or dir)
    info_hash: str

    @classmethod
    def from_env(cls) -> "TorrentContext":
        """
        Factories a context instance reading standard qBittorrent params
        exported as environment variables.

        Expected Env Vars:
            - TORRENT_NAME
            - TORRENT_CONTENT_PATH (Path to specific file or root folder)
            - TORRENT_SAVE_PATH (Parent download directory)
            - TORRENT_CATEGORY
            - TORRENT_TAGS (Comma separated)
            - TORRENT_HASH
        """
        name = os.getenv("TORRENT_NAME", "")
        content_path_str = os.getenv("TORRENT_CONTENT_PATH", "")
        save_path_str = os.getenv("TORRENT_SAVE_PATH", "")
        category = os.getenv("TORRENT_CATEGORY", "")
        tags_str = os.getenv("TORRENT_TAGS", "")
        info_hash = os.getenv("TORRENT_HASH", "")

        # Fallback for manual testing or missing env
        if not name:
            logging.debug("No TORRENT_NAME in env, context will be empty/invalid.")

        return cls(
            torrent_name=name,
            content_path=(
                Path(content_path_str).resolve()
                if content_path_str
                else Path(".").resolve()
            ),
            save_path=(
                Path(save_path_str).resolve() if save_path_str else Path(".").resolve()
            ),
            category=category if category else None,
            tags=[t.strip() for t in tags_str.split(",") if t.strip()],
            info_hash=info_hash,
        )

    def is_valid(self) -> bool:
        """Checks if minimum required info is present."""
        return bool(self.torrent_name and self.content_path)


class MediaType(Enum):
    """Types of media recognized by the system."""

    UNKNOWN = auto()
    MOVIE = auto()
    EPISODE = auto()
    MUSIC = auto()
    IGNORE = auto()  # Explicitly ignored (e.g. samples, txt files)


@dataclass
class FileCandidate:
    """
    A file found within the torrent output that might need processing.
    """

    source_path: Path
    relative_path: Path  # Relative to the content root
    size: int
    extension: str
    is_video: bool = False
    extra_category: Optional[str] = None


@dataclass
class ScanResult:
    """
    The structured result of a scan operation.
    """

    main_video_files: List[FileCandidate] = field(default_factory=list)
    extras: List[FileCandidate] = field(default_factory=list)
    ignored: List[FileCandidate] = field(default_factory=list)
    total_size: int = 0


@dataclass
class Classification:
    """
    The result of analyzing a FileCandidate.
    Contains metadata derived from the filename/path.
    """

    candidate: FileCandidate
    media_type: MediaType
    detected_title: Optional[str] = None
    detected_year: Optional[int] = None
    season: Optional[int] = None
    episode: Optional[int] = None
    episode_title: Optional[str] = None
    edition: Optional[str] = None  # e.g. "Extended", "Director's Cut"
    resolution: Optional[str] = None  # e.g. "1080p"
    source: Optional[str] = None  # e.g. "BluRay", "WEB-DL"
    video_codec: Optional[str] = None  # e.g. "x264", "h265"
    confidence: float = 0.0  # 0.0 to 1.0


class ActionType(Enum):
    """The kind of side-effect to perform."""

    LINK = auto()  # Hardlink file
    SKIP = auto()  # Do nothing
    NOTIFY = auto()  # API call (future use)


@dataclass
class PlannedAction:
    """
    A specific side-effect calculated by the DecisionEngine.
    This object is what the ExecutionEngine consumes.
    """

    action_type: ActionType
    reason: str
    source: Optional[Path] = None
    destination: Optional[Path] = None


class ProcessingStats:
    """Tracks execution statistics for final summary."""

    def __init__(self) -> None:
        self.scanned = 0
        self.main_videos = 0
        self.extras = 0
        self.subtitles = 0
        self.ignored = 0
        self.hardlinks_created = 0
        self.hardlinks_failed = 0
        self.api_calls = {"radarr": None, "sonarr": None, "plex": None}

    def print_summary(self, dry_run: bool) -> None:
        UI.header("Summary")

        # Determine total files handled
        total_files = self.main_videos + self.extras + self.ignored  # approximate
        if total_files == 0:
            total_files = self.scanned  # usage fallback

        processed_str = f"{self.hardlinks_created} files {'would be' if dry_run else 'were'} processed"

        if dry_run:
            UI.item(processed_str, icon=UI.INFO, color=UI.BLUE)
            logging.info("")
            logging.info("  Run with --live to execute these changes.")
        else:
            UI.item(
                f"Processed: {self.hardlinks_created}", icon=UI.CHECK, color=UI.GREEN
            )
            if self.hardlinks_failed:
                UI.item(
                    f"Failed:    {self.hardlinks_failed}", icon=UI.WARN, color=UI.RED
                )

            # API Summary inline for Live mode
            # Check if any API interaction happened
            if any(v is not None for v in self.api_calls.values()):
                logging.info("")
                UI.header("External Services Results")
                self._print_api_stat("Radarr", self.api_calls["radarr"])
                self._print_api_stat("Sonarr", self.api_calls["sonarr"])
                self._print_api_stat("Plex", self.api_calls["plex"])

        logging.info("")
        logging.info(f"{UI.CYAN}{'='*80}{UI.RESET}")

    def _print_api_stat(self, name: str, status: Optional[bool]) -> None:
        if status is True:
            UI.item(f"{name:<7}: Triggered/Requested", icon=UI.CHECK, color=UI.GREEN)
        elif status is False:
            UI.item(f"{name:<7}: Failed or Not Configured", icon=UI.WARN, color=UI.RED)
        # Skip None (not applicable) to keep clean


# ==========================================
# Components
# ==========================================


class Notifier:
    """
    Handles API calls to Radarr, Sonarr, and Plex.
    Supports batching notifications to avoid API spam.
    """

    def __init__(self, config: Dict[str, Any], dry_run: bool) -> None:
        self.config = config
        self.dry_run = dry_run

        self.radarr_cfg = config.get("radarr", {})
        self.sonarr_cfg = config.get("sonarr", {})
        self.plex_cfg = config.get("plex", {})
        self.path_mapping = config.get(
            "path_mapping", {}
        )  # Optional remote path mapping

        # Batching queues
        self.pending_radarr_paths: set = set()
        self.pending_sonarr_paths: set = set()
        self.trigger_plex = False

    def _map_path(self, path: str) -> str:
        """Applies remote path mapping if configured."""
        if not self.path_mapping:
            return path

        local_root = self.path_mapping.get("local")
        remote_root = self.path_mapping.get("remote")

        if local_root and remote_root and path.startswith(local_root):
            # Replace start of path
            mapped = path.replace(local_root, remote_root, 1)
            logging.debug(f"Mapped path: {path} -> {mapped}")
            return mapped

        return path

    def queue_radarr(self, folder_path: Path) -> None:
        """Queues a path for Radarr scanning."""
        if self.radarr_cfg.get("url") and self.radarr_cfg.get("key"):
            self.pending_radarr_paths.add(self._map_path(str(folder_path)))

    def queue_sonarr(self, folder_path: Path) -> None:
        """Queues a path for Sonarr scanning."""
        if self.sonarr_cfg.get("url") and self.sonarr_cfg.get("key"):
            self.pending_sonarr_paths.add(self._map_path(str(folder_path)))

    def queue_plex(self) -> None:
        """Flags Plex for a library refresh."""
        if self.plex_cfg.get("url") and self.plex_cfg.get("token"):
            self.trigger_plex = True

    def flush(self) -> Dict[str, Optional[bool]]:
        """Executes all pending notifications. Returns status map."""
        results: Dict[str, Optional[bool]] = {}

        if self.dry_run:
            if self.pending_radarr_paths:
                logging.info(
                    f"[DRY RUN] Would notify Radarr for {len(self.pending_radarr_paths)} paths"
                )
            if self.pending_sonarr_paths:
                logging.info(
                    f"[DRY RUN] Would notify Sonarr for {len(self.pending_sonarr_paths)} paths"
                )
            if self.trigger_plex:
                logging.info("[DRY RUN] Would trigger Plex scan")
            return results

        # 1. Radarr Batch
        if self.pending_radarr_paths:
            UI.header("Radarr Notifications")
            success = True
            for path in self.pending_radarr_paths:
                if not self._send_command(
                    self.radarr_cfg,
                    "DownloadedMoviesScan",
                    {"path": path},
                    f"Scanning {Path(path).name}",
                ):
                    success = False
            results["radarr"] = success

        # 2. Sonarr Batch
        if self.pending_sonarr_paths:
            UI.header("Sonarr Notifications")
            success = True
            for path in self.pending_sonarr_paths:
                if not self._send_command(
                    self.sonarr_cfg,
                    "DownloadedEpisodesScan",
                    {"path": path},
                    f"Scanning {Path(path).name}",
                ):
                    success = False
            results["sonarr"] = success

        # 3. Plex
        if self.trigger_plex:
            UI.header("External Services")
            results["plex"] = self._notify_plex()

        return results

    def _send_command(
        self, cfg: Dict, cmd_name: str, body_args: Dict, log_msg: str
    ) -> bool:
        """Generic command sender for *arrs with retry logic."""
        url = cfg.get("url")
        key = cfg.get("key")
        if not url or not key:
            return False

        api_url = f"{url.rstrip('/')}/api/v3/command"
        payload = {"name": cmd_name, **body_args}

        max_retries = 3
        for attempt in range(max_retries):
            try:
                req = urllib.request.Request(
                    api_url, data=json.dumps(payload).encode(), method="POST"
                )
                req.add_header("X-Api-Key", key)
                req.add_header("Content-Type", "application/json")

                with urllib.request.urlopen(req, timeout=15) as resp:
                    if 200 <= resp.status < 300:
                        logging.info(f"✓ {log_msg}")
                        return True
                    else:
                        logging.warning(f"⚠ API returned {resp.status} for {cmd_name}")
                        return False
            except Exception as e:
                if attempt < max_retries - 1:
                    wait_time = 2**attempt  # Exponential backoff
                    logging.warning(
                        f"⚠ {cmd_name} failed (attempt {attempt+1}/{max_retries}): {e}. Retrying in {wait_time}s..."
                    )
                    time.sleep(wait_time)
                else:
                    logging.warning(
                        f"⚠ {cmd_name} failed after {max_retries} attempts: {e}"
                    )
                    return False

    def _notify_plex(self) -> bool:
        url = self.plex_cfg.get("url")
        token = self.plex_cfg.get("token")
        section_id = self.plex_cfg.get("section_id")

        if not url or not token:
            logging.warning("Plex URL or Token missing in config.")
            return False

        if not section_id:
            logging.warning(
                "Plex 'section_id' not configured - skipping library refresh."
            )
            return False

        logging.info(f"Triggering Plex Library Refresh for section {section_id}...")

        base_url = url.rstrip("/")
        endpoint = f"/library/sections/{section_id}/refresh"
        # Only strictly required param is the token
        full_url = f"{base_url}{endpoint}?X-Plex-Token={token}"

        # Log masked URL
        masked_url = f"{base_url}{endpoint}?X-Plex-Token=******"
        logging.info(f"Plex Request: POST {masked_url}")

        try:
            # Method must be POST
            req = urllib.request.Request(full_url, method="POST")

            with urllib.request.urlopen(req, timeout=10) as resp:
                status = resp.status
                logging.info(f"Plex HTTP Status: {status}")

                if 200 <= status < 300:
                    logging.info("✓ Plex refresh triggered successfully.")
                    return True
                else:
                    body = resp.read().decode("utf-8", errors="ignore")
                    logging.warning("⚠ Plex returned non-200 status.")
                    logging.warning(f"Response: {body}")
                    return False

        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", errors="ignore")
            logging.error(f"Plex HTTP Error {e.code}: {e.reason}")

            if e.code == 401:
                logging.error(
                    "Hint: HTTP 401 Unauthorized usually means the Plex token is invalid or expired."
                )

            logging.error(f"Response Body: {body}")
            return False
        except Exception as e:
            logging.error(f"Plex notification error: {e}")
            return False


class ArchiveHandler:
    """
    Handles extraction of compressed archives.
    Supports RAR (via unrar) and 7z/Zip (via 7z).
    """

    ARCHIVE_EXTS = {".rar", ".7z", ".zip"}

    def __init__(self, dry_run: bool = False) -> None:
        self.dry_run = dry_run
        self.unrar_exe = shutil.which("unrar")
        self.sevenzip_exe = shutil.which("7z") or shutil.which("7za")

    def has_dependencies(self) -> bool:
        return bool(self.unrar_exe or self.sevenzip_exe)

    def extract_all(self, root_path: Path) -> int:
        """
        Recursively finds and extracts archives.
        Returns number of archives extracted.
        """
        if not self.has_dependencies():
            # If archives exist but no tools, we could warn, but this method just returns 0
            # Warning logic will be in main()
            return 0

        count = 0
        # Walk and find archives
        for root, _, files in os.walk(root_path):
            # Sort to handle archives in order if needed
            for file in sorted(files):
                file_path = Path(root) / file

                # Check extension
                if file_path.suffix.lower() not in self.ARCHIVE_EXTS:
                    continue

                # Extract
                if self._extract_archive(file_path):
                    count += 1

        return count

    def _extract_archive(self, file_path: Path) -> bool:
        ext = file_path.suffix.lower()
        cmd = []

        # Decide tool
        if ext == ".rar" and self.unrar_exe:
            # unrar x -o- -y "file.rar" "destination/"
            # -o- : do not overwrite
            # -y : assume yes on all queries
            # Using parent dir as destination (in-place extraction)
            cmd = [
                self.unrar_exe,
                "x",
                "-o-",
                "-y",
                str(file_path),
                str(file_path.parent),
            ]
        elif (ext in {".7z", ".zip"}) and self.sevenzip_exe:
            # 7z x -aos "file.7z" -o"destination/"
            # -aos : skip existing files
            cmd = [
                self.sevenzip_exe,
                "x",
                "-aos",
                f"-o{file_path.parent}",
                str(file_path),
            ]

        if not cmd:
            return False

        try:
            if self.dry_run:
                logging.info(f"[DRY RUN] Would extract: {file_path.name}")
                return True

            logging.info(f"Extracting: {file_path.name}...")
            # Suppress output of tool unless error
            result = subprocess.run(cmd, capture_output=True, text=True)

            if result.returncode == 0:
                logging.info(f"Extracted: {file_path.name}")
                return True
            else:
                # Log stderr if failed
                logging.error(
                    f"Extraction failed for {file_path.name}: {result.stderr.strip()}"
                )
                return False

        except Exception as e:
            logging.error(f"Error executing extraction: {e}")
            return False


class Scanner:
    """
    Responsible for identifying relevant files in the torrent output.
    Does NOT modify filesystem.
    """

    # Constants for file detection
    VIDEO_EXTENSIONS = {".mkv", ".mp4", ".avi", ".m4v", ".mov", ".ts", ".m2ts"}
    SUBTITLE_EXTENSIONS = {".srt", ".ass", ".ssa", ".sub", ".idx", ".sup"}
    IGNORED_EXTENSIONS = {".txt", ".nfo", ".exe", ".bat", ".sh", ".sample"}
    MIN_VIDEO_SIZE_MB = 50
    SAMPLE_SIZE_LIMIT_MB = 150  # ID samples if small video

    EXTRAS_CATEGORIES = {
        "Extras": ["extras", "extra", "bonus"],
        "Behind The Scenes": ["behind the scenes", "bts", "making of"],
        "Featurettes": ["featurettes", "featurette"],
        "Deleted Scenes": ["deleted scenes", "deleted"],
        "Interviews": ["interviews", "interview"],
        "Trailers": ["trailers", "trailer"],
    }

    def parse_subtitle_metadata(self, filename: str) -> Dict[str, Any]:
        """
        Parses subtitle filename for language, forced, and sdh flags.
        MOVED from Classifier.
        """
        lower = filename.lower()
        forced = "forced" in lower
        sdh = "sdh" in lower

        # Language detection (simple scan)
        # Using Classifier's Lang Map if available or simple check
        lang = None
        # Simplified for now as we don't have easy access to Classifier.LANG_MAP here
        # unless we duplicate or check Classifier.
        # But wait, Classifier is defined BELOW Scanner?
        # Python classes are objects, Scanner can verify Classifier.LANG_MAP only if Classifier is defined.
        # But file structure is ordered. Classifier is AFTER Scanner.
        # So we can't use Classifier.LANG_MAP statically here easily if we run top-down.
        # We should move parse_subtitle_metadata BACK to Classifier?
        # But we need it in Scanner.scan summary?
        # Summary uses `Classifier` static access. Wait, summary runs AFTER everything is defined.
        # So it is safe to access Classifier.LANG_MAP inside a method if called later.

        # Let's defer map access
        for part in reversed(re.split(r"[._\-\s]", lower)):
            if hasattr(Classifier, "LANG_MAP") and part in Classifier.LANG_MAP:
                lang = Classifier.LANG_MAP[part]
                break

        return {"language": lang, "forced": forced, "sdh": sdh}

    @staticmethod
    def get_related_extras(
        video_candidate: FileCandidate, all_extras: List[FileCandidate]
    ) -> List[FileCandidate]:
        """
        Finds extras that belong to the given video file.
        Logic:
        1. Same directory
        2. 'Subs' or 'Subtitles' subdirectory of the video's directory
        """
        related = []
        video_dir = video_candidate.source_path.parent

        for extra in all_extras:
            extra_dir = extra.source_path.parent

            # Same Dir
            if extra_dir == video_dir:
                related.append(extra)
                continue

            # Subtitles subdir
            if extra_dir.parent == video_dir and extra_dir.name.lower() in [
                "subs",
                "subtitles",
            ]:
                related.append(extra)
                continue

        return related

    def __init__(self, min_size_mb: int = 50) -> None:
        self.min_size_bytes = min_size_mb * 1024 * 1024

    def scan(self, root_path: Path) -> ScanResult:
        """
        Walks the root_path.
        Returns grouped candidates: Main Videos, Extras, Ignored.
        """
        logging.info(f"Scanning directory: {root_path}")
        result = ScanResult()

        if not root_path.exists():
            logging.error(f"Scan path does not exist: {root_path}")
            return result

        # Handle single file case
        if root_path.is_file():
            self._process_file_stat(root_path, root_path, result, root_path.stat())
            return result

        # Optimized recursive scan
        self._scan_recursive(root_path, root_path, result)

        return result

    def _scan_recursive(
        self, current_dir: Path, root_path: Path, result: ScanResult
    ) -> None:
        """Recursively scans directory using os.scandir for performance."""
        try:
            with os.scandir(current_dir) as it:
                for entry in it:
                    # 1. Early skip for hidden files
                    if entry.name.startswith(".") or entry.name in {
                        "Thumbs.db",
                        "desktop.ini",
                        "@eaDir",
                    }:
                        continue

                    if entry.is_dir(follow_symlinks=False):
                        self._scan_recursive(Path(entry.path), root_path, result)

                    elif entry.is_file(follow_symlinks=False):
                        # Pass entry.stat() to avoid re-statting
                        self._process_file_stat(
                            Path(entry.path), root_path, result, entry.stat()
                        )

        except PermissionError:
            logging.warning(f"Permission denied scanning: {current_dir}")
        except OSError as e:
            logging.error(f"Error scanning {current_dir}: {e}")

    def _process_file_stat(
        self, file_path: Path, root_path: Path, result: ScanResult, stat: os.stat_result
    ) -> None:
        """Analyze a single file and sort into result groups using cached stat."""
        try:
            size = stat.st_size
            result.total_size += size

            relative_path = file_path.relative_to(
                root_path.parent if root_path.is_file() else root_path
            )
            extension = file_path.suffix.lower()

            # Detect category based on path
            category = self._detect_extra_category(relative_path)

            candidate = FileCandidate(
                source_path=file_path,
                relative_path=relative_path,
                size=size,
                extension=extension,
                extra_category=category,
            )

            # 1. Check for hard ignores first
            if self._is_ignored(
                filename=file_path.name, extension=extension, size=size
            ):
                # logging.debug(f"Ignored file: {relative_path}") # Reduce noise
                result.ignored.append(candidate)
                return

            # 2. Identify Video Files
            if extension in self.VIDEO_EXTENSIONS and not category:
                candidate.is_video = True
                if size >= self.min_size_bytes:
                    logging.info(
                        f"Found main video: {relative_path} ({self._format_size(size)})"
                    )
                    result.main_video_files.append(candidate)
                else:
                    logging.info(
                        f"Found video extra/sample: {relative_path} ({self._format_size(size)})"
                    )
                    result.extras.append(candidate)
            else:
                if category:
                    logging.info(f"Found {category}: {relative_path}")
                else:
                    logging.debug(f"Found extra: {relative_path}")
                result.extras.append(candidate)

        except Exception as e:
            logging.error(f"Error scanning file {file_path}: {e}")

    # Legacy wrapper for compatibility if needed, though internal usage is updated
    def _process_file(
        self, file_path: Path, root_path: Path, result: ScanResult
    ) -> None:
        self._process_file_stat(file_path, root_path, result, file_path.stat())

    def _detect_extra_category(self, rel_path: Path) -> Optional[str]:
        """Checks if any parent folder matches a known extra category."""
        # Split path into parts, ignoring the filename
        parts = rel_path.parent.parts

        for part in parts:
            part_lower = part.lower()
            for category, variants in self.EXTRAS_CATEGORIES.items():
                if part_lower in variants:
                    return category
        return None

    def _is_ignored(self, filename: str, extension: str, size: int) -> bool:
        """Determine if a file should be completely ignored."""
        if extension in self.IGNORED_EXTENSIONS:
            return True
        if "sample" in filename.lower() and size < (
            self.SAMPLE_SIZE_LIMIT_MB * 1024 * 1024
        ):
            return True
        if size == 0:
            return True
        return False

    @staticmethod
    def _format_size(size_bytes: float) -> str:
        """Helper to format bytes for logging."""
        for unit in ["B", "KB", "MB", "GB"]:
            if size_bytes < 1024.0:
                return f"{size_bytes:.1f} {unit}"
            size_bytes /= 1024.0
        return f"{size_bytes:.1f} TB"

    @staticmethod
    def _print_scan_summary(result: ScanResult) -> None:
        UI.header("Scanning files")

        # Main Videos
        if result.main_video_files:
            for f in result.main_video_files:
                sz = Scanner._format_size(f.size)
                UI.item(
                    f"Main video: {f.relative_path.name} ({sz})",
                    icon=UI.CHECK,
                    color=UI.GREEN,
                )
        else:
            UI.item("No main video files found!", icon=UI.CROSS, color=UI.RED)

        # Extras logic for summary
        extras_count = len(result.extras)
        if extras_count > 0:
            sub_count = sum(
                1 for e in result.extras if e.extension in Scanner.SUBTITLE_EXTENSIONS
            )
            other_count = extras_count - sub_count

            if sub_count:
                # Extract languages for display
                langs = set()
                for e in result.extras:
                    if e.extension in Scanner.SUBTITLE_EXTENSIONS:
                        # Quick parse to find lang
                        parts = e.source_path.stem.split(".")
                        found = False
                        for p in parts:
                            if p in Classifier.LANG_MAP:
                                langs.add(Classifier.LANG_MAP[p])
                                found = True
                        if not found:
                            langs.add("unk")
                lang_str = ", ".join(sorted(langs))
                UI.item(
                    f"Subtitles: {sub_count} files ({lang_str})",
                    icon=UI.CHECK,
                    color=UI.GREEN,
                )

            if other_count:
                UI.item(f"Extras: {other_count} files", icon=UI.INFO, color=UI.BLUE)

        # Ignored
        if result.ignored:
            exts = set(f.extension for f in result.ignored)
            ext_str = ", ".join(sorted(exts))
            UI.item(
                f"Ignored: {len(result.ignored)} files ({ext_str})",
                icon=UI.INFO,
                color=UI.DIM,
            )

        logging.info("")  # Spacer

        logging.info(f"\n[IGNORED] ({len(result.ignored)})")
        for f in result.ignored:
            logging.info(
                f"  • {f.relative_path} ({Scanner._format_size(f.size)}) - Ignored"
            )
        logging.info("-" * 40)


class Classifier:
    """
    Responsible for pure-function analysis of filenames to determine content.
    """

    # Pre-compile regex for performance
    # 1. TV Patterns
    # Matches: S01E01, s1e1, S01.E01, 1x01
    RE_SEASON_EPISODE = re.compile(
        r"(?P<season_marker>[sS]?)(?P<season>\d{1,2})[\.\s\_-]?"
        r"(?P<episode_marker>[eExX])(?P<episode>\d{1,2})",
        re.IGNORECASE,
    )

    # Matches: S01, Season 01 (Season Pack)
    # Added \b to prevent matching ends of words like "Guys.2010"
    RE_SEASON_PACK = re.compile(
        r"\b(?P<season_marker>S|Season)[\.\s\_-]?(?P<season>\d{1,2})\b", re.IGNORECASE
    )

    # 2. Year Pattern: classic (1999) or .2020.
    RE_YEAR = re.compile(r"[\(\[\.\s_](?P<year>19\d{2}|20\d{2})($|[\)\]\.\s_])")

    # 3. Resolution Patterns
    RE_RES = re.compile(r"(?P<res>480p|576p|720p|1080[pi]|2160p|4k|uhd)", re.IGNORECASE)

    # 4. Source Patterns
    RE_SOURCE = re.compile(
        r"(?P<source>bluray|remux|web-?dl|hdtv|dvd|bdrip|brrip)", re.IGNORECASE
    )

    # 5. Codec Patterns
    RE_CODEC = re.compile(
        r"(?P<codec>x264|x265|h\.?264|h\.?265|hevc|av1|divx|xvid|avc)", re.IGNORECASE
    )

    # 6. Tech/Source Tags (for cleaning episode titles)
    RE_TECH_JUNK = re.compile(
        r"\b(x264|x265|h264|h265|hevc|web-?dl|bluray|remux|hdtv|amzn|nf|dsnp|repack|proper)\b",
        re.IGNORECASE,
    )

    # 7. Language Codes (Simplified map)
    LANG_MAP = {
        "en": "en",
        "eng": "en",
        "english": "en",
        "fr": "fr",
        "fre": "fr",
        "french": "fr",
        "es": "es",
        "spa": "es",
        "spanish": "es",
        "de": "de",
        "ger": "de",
        "german": "de",
        "it": "it",
        "ita": "it",
        "italian": "it",
        "pt": "pt",
        "por": "pt",
        "portuguese": "pt",
        "ru": "ru",
        "rus": "ru",
        "russian": "ru",
        "ja": "ja",
        "jpn": "ja",
        "japanese": "ja",
        "zh": "zh",
        "chi": "zh",
        "chinese": "zh",
    }

    def classify(
        self, candidate: FileCandidate, context: TorrentContext
    ) -> Classification:
        """
        Analyzes a candidate file.
        Returns a Classification object with derived metadata.
        """
        classification = Classification(
            candidate=candidate, media_type=MediaType.UNKNOWN, detected_title=None
        )

        filename = candidate.source_path.stem

        # 1. Extract Tech Properties
        self._extract_tech_metadata(filename, classification)

        # 2. Strategy Selection based on Category Hint
        check_movie_first = False
        if context.category and "movie" in context.category.lower():
            check_movie_first = True

        if check_movie_first:
            # MOVIE PRIORITY
            if self._apply_movie_parsing(filename, classification):
                return classification
            if self._apply_tv_parsing(filename, classification):
                return classification
        else:
            # TV PRIORITY (Default)
            if self._apply_tv_parsing(filename, classification):
                return classification
            if self._apply_movie_parsing(filename, classification):
                return classification

        # 3. Unknown/Fallback based on context
        self._apply_fallback_logic(filename, context, classification)
        return classification

    def _apply_tv_parsing(self, filename: str, classification: Classification) -> bool:
        """Attempts to parse as TV Show. Updates classification if successful."""
        tv_info = self.parse_tv_name(filename)
        if tv_info:
            classification.media_type = MediaType.EPISODE
            classification.detected_title = tv_info.get("show", tv_info.get("title"))
            classification.season = tv_info["season"]
            classification.episode = tv_info["episode"]
            classification.episode_title = tv_info["episode_title"]
            classification.confidence = 0.9

            s_str = f"S{classification.season:02d}"
            e_str = (
                f"E{classification.episode:02d}"
                if classification.episode is not None
                else " (Season Pack)"
            )
            logging.debug(
                f"Classified as TV: {classification.detected_title} {s_str}{e_str}"
            )
            return True
        return False

    def _apply_movie_parsing(
        self, filename: str, classification: Classification
    ) -> bool:
        """Attempts to parse as Movie. Updates classification if successful."""
        movie_info = self.parse_movie_name(filename)
        if movie_info:
            classification.media_type = MediaType.MOVIE
            classification.detected_title = movie_info["title"]
            classification.detected_year = movie_info["year"]
            if movie_info.get("quality"):
                classification.resolution = movie_info["quality"]
            classification.confidence = 0.8
            logging.debug(
                f"Classified as Movie: {classification.detected_title} ({classification.detected_year})"
            )
            return True
        return False

    def parse_tv_name(self, filename: str) -> Optional[Dict[str, Any]]:
        """Extracts show, season, episode, and ep_title from filename string."""

        # 1. Try SxxExx / 1x01 Pattern
        match = self.RE_SEASON_EPISODE.search(filename)
        if match:
            try:
                season = int(match.group("season"))
                episode = int(match.group("episode"))

                # Title is typically before match
                raw_title = filename[: match.start()]
                clean_title = self._clean_title_string(raw_title)

                # Ep Title is after match
                raw_suffix = filename[match.end() :]
                ep_title = self._extract_episode_title(raw_suffix)

                return {
                    "show": clean_title,
                    "season": season,
                    "episode": episode,
                    "episode_title": ep_title,
                }
            except ValueError:
                pass

        # 2. Try Season Pack Pattern
        match = self.RE_SEASON_PACK.search(filename)
        if match:
            try:
                season = int(match.group("season"))

                raw_title = filename[: match.start()]
                clean_title = self._clean_title_string(raw_title)

                return {
                    "show": clean_title,
                    "season": season,
                    "episode": None,
                    "episode_title": None,
                }
            except ValueError:
                pass

        return None

    def parse_movie_name(self, filename: str) -> Optional[Dict[str, Any]]:
        """
        Extracts title, year, and quality from filename string.
        Enhancement: Handles cases without years and removed tech tags.
        """
        # 1. Detect Year
        year = None
        year_match = self.RE_YEAR.search(filename)
        if year_match:
            year = int(year_match.group("year"))

        # 2. Detect Quality/Resolution
        quality = None
        res_match = self.RE_RES.search(filename)
        if res_match:
            quality = res_match.group("res").lower()
            # Normalize 4k/uhd to 2160p
            if quality in ["4k", "uhd"]:
                quality = "2160p"

        # 3. Determine Title End Position
        # Title ends at the first occurrence of: Year, Resolution, or specific Tech Tags
        end_indices = []
        if year_match:
            end_indices.append(year_match.start())
        if res_match:
            end_indices.append(res_match.start())

        for regex in [self.RE_SOURCE, self.RE_CODEC, self.RE_TECH_JUNK]:
            m = regex.search(filename)
            if m:
                end_indices.append(m.start())

        if end_indices:
            raw_title = filename[: min(end_indices)]
        else:
            raw_title = filename

        clean_title = self._clean_title_string(raw_title)

        # If title is empty, we can't classify
        if not clean_title:
            return None

        return {"title": clean_title, "year": year, "quality": quality}

    def _extract_episode_title(self, raw_suffix: str) -> Optional[str]:
        """Isolates episode title from suffix by stripping tech tags."""
        indices = []
        for regex in [self.RE_RES, self.RE_TECH_JUNK, self.RE_SOURCE, self.RE_CODEC]:
            m = regex.search(raw_suffix)
            if m:
                indices.append(m.start())

        if indices:
            raw_suffix = raw_suffix[: min(indices)]

        clean = self._clean_title_string(raw_suffix)
        return clean if clean and len(clean) > 2 else None

    def _extract_tech_metadata(self, filename: str, cl: Classification) -> None:
        """Helpers to fill resolution, source, codec, etc."""
        # Year (Metadata)
        ym = self.RE_YEAR.search(filename)
        if ym:
            cl.detected_year = int(ym.group("year"))

        # Resolution
        rm = self.RE_RES.search(filename)
        if rm:
            cl.resolution = rm.group("res").lower()

        # Source
        sm = self.RE_SOURCE.search(filename)
        if sm:
            src = sm.group("source").upper()
            if "WEB" in src:
                src = "WEB-DL"
            cl.source = src
        if "REMUX" in filename.upper():
            cl.source = "REMUX"

        # Codec
        cm = self.RE_CODEC.search(filename)
        if cm:
            codec = cm.group("codec").lower()
            if codec in ["h.265", "h265"]:
                codec = "hevc"
            if codec in ["h.264", "h264", "avc"]:
                codec = "x264"
            cl.video_codec = codec

        # Edition
        if re.search(r"\b(repack|proper)\b", filename, re.IGNORECASE):
            cl.edition = "REPACK"

    def _apply_fallback_logic(
        self, filename: str, context: TorrentContext, cl: Classification
    ) -> None:
        """Fallback logic when no clear regex match found."""
        if context.category:
            cat = context.category.lower()
            if "movie" in cat:
                cl.media_type = MediaType.MOVIE
            elif "tv" in cat or "show" in cat:
                cl.media_type = MediaType.EPISODE

        cl.detected_title = self._clean_title_string(filename)

    def _clean_title_string(self, text: str) -> str:
        """Removes dots, underscores, brackets, and common junk."""
        # Replace separators with space
        text = re.sub(r"[._]", " ", text)
        # Remove tags in brackets/parens
        text = re.sub(r"[\[\(].*?[\]\)]", "", text)
        # Compress spaces
        text = re.sub(r"\s+", " ", text)
        # Strip trailing dash or space
        return text.strip(" -")


class DecisionEngine:
    """
    Business logic layer.
    Mapping Classifications to destinations based on Config.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config

        # Default target roots if not specified in config
        # In production, these should come from config file
        self.movie_root = Path(config.get("movie_path", "/data/media/movies"))
        self.tv_root = Path(config.get("tv_path", "/data/media/tv"))

        # Radarr/Sonarr naming templates (simplified implementation for now)
        # {Title} {Year}
        # {Title} - S{season:02d}E{episode:02d} - {EpisodeTitle} [Quality]

    def plan(self, classification: Classification) -> PlannedAction:
        """
        Determines WHERE a file should go, or IF it should be skipped.
        Logic: Use config to map Category/MediaType -> Dest Folder.
        """
        candidate = classification.candidate

        # 0. Safety/Sanity checks
        if not candidate.source_path.exists():
            return PlannedAction(
                ActionType.SKIP, f"Source missing: {candidate.source_path}"
            )

        # 1. Ignored content
        if classification.media_type == MediaType.IGNORE:
            return PlannedAction(ActionType.SKIP, "Ignored content type")

        # 2. Movie Handling
        if classification.media_type == MediaType.MOVIE:
            return self._plan_movie(classification)

        # 3. TV Episode Handling
        if classification.media_type == MediaType.EPISODE:
            return self._plan_episode(classification)

        # 4. Unknown/Fallback
        return PlannedAction(ActionType.SKIP, "Unknown media type")

    def plan_extra(
        self, extra: FileCandidate, parent_plan: PlannedAction
    ) -> PlannedAction:
        """
        Determines where an extra (subtitle, nfo) should go,
        relative to the already-planned main video file.
        """
        if parent_plan.action_type != ActionType.LINK or not parent_plan.destination:
            return PlannedAction(ActionType.SKIP, "Parent video not linked")

        # Handle Subtitles specific renaming
        if extra.extension in Scanner.SUBTITLE_EXTENSIONS:
            return self._plan_subtitle(extra, parent_plan)

        # Strategy: Keep original filename, place in same folder as linked video
        # Exception: Subtitles (.srt). If there is only one, align name with video?
        # For now, SAFE strategy: Keep original filename.

        dest_folder = parent_plan.destination.parent
        dest_path = dest_folder / extra.source_path.name

        return PlannedAction(
            action_type=ActionType.LINK,
            reason=f"Extra for {parent_plan.source.name if parent_plan.source else 'unknown'}",
            source=extra.source_path,
            destination=dest_path,
        )

    def _plan_subtitle(
        self, extra: FileCandidate, parent_plan: PlannedAction
    ) -> PlannedAction:
        """Renames subtitle to match video file with extracted language tags."""
        # 1. Parse metadata from original subtitle filename
        # Moved to Scanner to avoid circular or instance issues, but let's instantiate Scanner or call static.
        # Actually Scanner is instantiated in main. DecisionEngine consumes config.
        # Let's just create a temp Scanner instance or move the helper to a Utils static class.
        # Or instantiate Scanner just for this.
        scanner = Scanner()
        meta = scanner.parse_subtitle_metadata(extra.source_path.stem)

        # 2. Build Suffix
        # Format: .[lang].[forced].[sdh].ext
        suffix_parts = []

        if meta["language"]:
            suffix_parts.append(meta["language"])
        else:
            # If no language found, should we assume 'en' or leave empty?
            # Leaving empty runs risk of collision if multiple subs exist.
            # But "Movie.srt" usually means default lang (often local).
            pass

        if meta["forced"]:
            suffix_parts.append("forced")
        if meta["sdh"]:
            suffix_parts.append("sdh")

        suffix_str = "." + ".".join(suffix_parts) if suffix_parts else ""

        # 3. Construct new name
        # Base is the DESTINATION video name (without extension)
        video_stem = (
            parent_plan.destination.stem if parent_plan.destination else "unknown"
        )
        new_filename = f"{video_stem}{suffix_str}{extra.extension}"

        dest_path = (
            parent_plan.destination.parent / new_filename
            if parent_plan.destination
            else Path("unknown")
        )

        return PlannedAction(
            action_type=ActionType.LINK,
            reason=f"Subtitle ({meta['language'] or 'unk'}) for {parent_plan.source.name if parent_plan.source else 'unknown'}",
            source=extra.source_path,
            destination=dest_path,
        )

    def _plan_movie(self, cl: Classification) -> PlannedAction:
        """Logic for Movie destinations."""
        # 0. Check connection to library
        if not self.movie_root.exists():
            return PlannedAction(
                ActionType.SKIP, f"Movie library path not found: {self.movie_root}"
            )

        title = cl.detected_title or "Unknown Movie"
        year = cl.detected_year

        # Folder Name: Title (Year)
        folder_name = f"{title} ({year})" if year else title

        # Base File Name: Title (Year)
        # Note: We start clean, without tags
        base_filename = f"{folder_name}"

        dest_folder = self.movie_root / folder_name

        # Safe Resolution Logic (Checks for existence and adds quality suffix if needed)
        dest_path, reason = self._resolve_destination(
            dest_folder, base_filename, cl.candidate.extension, cl
        )

        if not dest_path:
            return PlannedAction(ActionType.SKIP, reason)

        return PlannedAction(
            action_type=ActionType.LINK,
            reason=reason,
            source=cl.candidate.source_path,
            destination=dest_path,
        )

    def _plan_episode(self, cl: Classification) -> PlannedAction:
        """Logic for TV destinations."""
        # 0. Check connection to library
        if not self.tv_root.exists():
            return PlannedAction(
                ActionType.SKIP, f"TV library path not found: {self.tv_root}"
            )

        show_title = cl.detected_title or "Unknown Show"
        s_num = cl.season if cl.season is not None else 1
        e_num = cl.episode if cl.episode is not None else 1

        # Folder Structure: Show Name / Season XX
        season_folder = f"Season {s_num:02d}"

        # Base File Name: Show Name - SxxEyy - Episode Title
        base_name = f"{show_title} - S{s_num:02d}E{e_num:02d}"
        if cl.episode_title:
            base_name += f" - {cl.episode_title}"

        dest_folder = self.tv_root / show_title / season_folder

        # Safe Resolution Logic
        dest_path, reason = self._resolve_destination(
            dest_folder, base_name, cl.candidate.extension, cl
        )

        if not dest_path:
            return PlannedAction(ActionType.SKIP, reason)

        return PlannedAction(
            action_type=ActionType.LINK,
            reason=reason,
            source=cl.candidate.source_path,
            destination=dest_path,
        )

    def _resolve_destination(
        self, base_folder: Path, base_filename: str, extension: str, cl: Classification
    ) -> Tuple[Optional[Path], str]:
        """
        Calculates final destination.
        Priority:
        1. Clean Name (if not exists or same file)
        2. Quality Suffixed Name (if clean exists & different)
        Returns (Path, Reason) or (None, Reason) if all options exhausted.
        """
        # 1. Ensure folder exists (logically, for path construction)
        # We don't create it here, ExecutionEngine does. But we check for file existence relative to it.
        # Since we use hardlinks, we can check if file exists even if folder structure isn't fully there (it won't exist).

        # Strategy 1: Clean Name
        clean_path = base_folder / f"{base_filename}{extension}"

        if not clean_path.exists():
            return clean_path, f"Identified: {base_filename}"

        # If exists, check if it's the SAME file (Idempotency)
        if self._is_same_file(cl.candidate.source_path, clean_path):
            return clean_path, f"Identified: {base_filename} (Update/Same File)"

        # Strategy 2: Quality Suffix
        # Construct quality string
        q_parts = []
        # Resolution (1080p, 2160p)
        if cl.resolution:
            q_parts.append(cl.resolution)
        # Source (WEB-DL, REMUX) - Only add if not redundant or implies higher qualify
        if cl.source:
            q_parts.append(cl.source)

        quality_str = " - ".join(q_parts) if q_parts else "Unknown"

        suffixed_name = f"{base_filename} - {quality_str}{extension}"
        suffixed_path = base_folder / suffixed_name

        if not suffixed_path.exists():
            msg = f"⚠ File exists ({clean_path.name}) - creating as: {suffixed_name}"
            logging.info(msg)
            return suffixed_path, msg

        # Check idempotency on suffixed
        if self._is_same_file(cl.candidate.source_path, suffixed_path):
            return suffixed_path, f"Identified: {suffixed_name} (Update/Same File)"

        # Fallback: Both exist and are different
        return None, "Skipped: Both clean and suffixed targets exist"

    def _is_same_file(self, src: Path, dst: Path) -> bool:
        if not src.exists() or not dst.exists():
            return False
        try:
            return src.samefile(dst)
        except OSError:
            # Cross-device check or permissions can cause generic OSError
            return False


class ExecutionEngine:
    """
    Side-effect layer.
    Handles filesystem operations and strictly obeys dry-run/safe-mode.
    """

    def __init__(self, config: Dict[str, Any], dry_run: bool = True) -> None:
        self.config = config
        self.dry_run = dry_run

        # Parse permission settings
        self.normalize_permissions = config.get("normalize_permissions", False)

        # Link Mode: hardlink, symlink, copy
        self.link_mode = config.get("link_mode", "hardlink").lower()
        if self.link_mode not in ["hardlink", "symlink", "copy"]:
            logging.warning(
                f"Invalid link_mode '{self.link_mode}'. Defaulting to 'hardlink'."
            )
            self.link_mode = "hardlink"

        # Parse Octal modes safety
        try:
            self.file_mode = int(config.get("file_mode", "0644"), 8)
            self.dir_mode = int(config.get("dir_mode", "0755"), 8)
        except ValueError:
            logging.error("Invalid octal mode in config. Using defaults 0644/0755")
            self.file_mode = 0o644
            self.dir_mode = 0o755

        # Device ID cache for cross-fs checks
        self.device_cache: Dict[Path, int] = {}

    def execute(self, action: PlannedAction) -> bool:
        """
        Performs the action (e.g., os.link).
        Must log details.
        Returns Success/Fail status.
        """
        if action.action_type == ActionType.SKIP:
            # We don't usually log skips in execution unless verbose,
            # effectively handled by the planner logging usually.
            return True

        if action.action_type == ActionType.LINK:
            return self._dispatch_link_action(action)

        return False

    def _dispatch_link_action(self, action: PlannedAction) -> bool:
        """Dispatches to the correct filesystem operation based on link_mode."""
        if self.link_mode == "symlink":
            return self._symlink(action)
        elif self.link_mode == "copy":
            return self._copy(action)
        else:
            return self._hardlink(action)

    def _ensure_parent_dir(self, dst: Path) -> None:
        """Helper to create parent directory with correct permissions."""
        if not dst.parent.exists():
            logging.info(f"Creating directory: {dst.parent}")
            try:
                dst.parent.mkdir(parents=True, exist_ok=True)
            except PermissionError:
                logging.error(f"✗ Permission denied creating directory: {dst.parent}")
                return
            except OSError as e:
                if e.errno == errno.ENOSPC:
                    logging.error("✗ Disk full - cannot create directory")
                else:
                    logging.error(f"✗ Directory creation failed: {e}")
                return

            # Apply directory permissions if enabled
            if self.normalize_permissions:
                try:
                    dst.parent.chmod(self.dir_mode)
                except Exception as e:
                    logging.warning(
                        f"Failed to set directory permissions on {dst.parent}: {e}"
                    )

    def _get_device_id(self, path: Path) -> Optional[int]:
        """Gets device ID with caching."""
        # For non-existent paths (destinations), we need to check parent
        p = path
        while not p.exists() and p.parent != p:
            p = p.parent

        if p in self.device_cache:
            return self.device_cache[p]

        try:
            dev = p.stat().st_dev
            self.device_cache[p] = dev
            return dev
        except Exception:
            return None

    def _hardlink(self, action: PlannedAction) -> bool:
        src = action.source
        dst = action.destination

        if not src or not dst:
            return False

        if self.dry_run:
            UI.item(f"[CREATE] {dst.name}", icon=UI.ARROW, color=UI.DIM)
            return True

        try:
            self._ensure_parent_dir(dst)

            if not src.exists():
                logging.error(f"Source file disappeared: {src}")
                return False

            # Proactive Cross-Filesystem Check
            src_dev = self._get_device_id(src)
            dst_dev = self._get_device_id(dst.parent)

            if src_dev is not None and dst_dev is not None and src_dev != dst_dev:
                UI.item(
                    "Cannot hardlink across filesystems", icon=UI.WARN, color=UI.RED
                )
                logging.warning(f"  Source: {src} (device: {src_dev})")
                logging.warning(f"  Destination: {dst.parent}/ (device: {dst_dev})")
                logging.warning(
                    "  → Skipping this file. Consider using symlinks or moving torrent to same filesystem."
                )
                return False

            UI.item(f"[CREATE] {dst.name}", icon=UI.CHECK, color=UI.GREEN)
            os.link(src, dst)

            if self.normalize_permissions:
                try:
                    os.chmod(dst, self.file_mode)
                except Exception as e:
                    logging.warning(f"Failed to set permissions on {dst}: {e}")
            return True

        except FileExistsError:
            logging.info(f"ℹ File already exists, skipping: {dst}")
            return self._check_idempotency(src, dst)
        except PermissionError:
            logging.error(f"✗ Permission denied: {dst} - check directory permissions")
            return False
        except OSError as e:
            if e.errno == errno.EXDEV:
                # Catch if proactive check failed or raced
                logging.warning(f"CROSS-DEVICE ERROR: Cannot hardlink {src} -> {dst}")
                logging.warning(
                    "Tip: Use 'link_mode': 'copy' or 'symlink' in config.json to fix this."
                )
                return False
            if e.errno == errno.EEXIST:
                logging.info(f"ℹ File already exists, skipping: {dst}")
                return self._check_idempotency(src, dst)
            if e.errno == errno.ENOSPC:
                logging.error("✗ Disk full - cannot create hardlink")
                return False
            logging.error(f"Filesystem error linking {src} to {dst}: {e}")
            return False
        except Exception:
            logging.exception(f"Unexpected error executing action {action}")
            return False

    def _symlink(self, action: PlannedAction) -> bool:
        src = action.source
        dst = action.destination

        if not src or not dst:
            return False

        if self.dry_run:
            UI.item(f"[SYMLINK] {dst.name}", icon=UI.ARROW, color=UI.DIM)
            return True

        try:
            self._ensure_parent_dir(dst)

            UI.item(f"[SYMLINK] {dst.name}", icon=UI.CHECK, color=UI.GREEN)
            # Use absolute path for symlink to avoid relative link rot
            os.symlink(src.resolve(), dst)

            return True
        except OSError as e:
            if e.errno == errno.EEXIST:
                # For symlinks, idempotency is checking if the link points to the right place
                try:
                    if dst.is_symlink() and dst.resolve() == src.resolve():
                        UI.item(
                            f"Symlink correct: {dst.name}",
                            icon=UI.CHECK,
                            color=UI.GREEN,
                        )
                        return True
                    logging.error(
                        f"Destination exists but is incorrect symlink or file: {dst}"
                    )
                    return False
                except Exception:
                    return False
            logging.error(f"Symlink error {src} -> {dst}: {e}")
            return False

    def _copy(self, action: PlannedAction) -> bool:
        src = action.source
        dst = action.destination

        if not src or not dst:
            return False

        if self.dry_run:
            UI.item(f"[COPY] {dst.name}", icon=UI.ARROW, color=UI.DIM)
            return True

        try:
            self._ensure_parent_dir(dst)

            if not src.exists():
                logging.error(f"Source missing: {src}")
                return False

            logging.info(f"Copying: {src.name} -> {dst}")
            shutil.copy2(src, dst)

            if self.normalize_permissions:
                try:
                    os.chmod(dst, self.file_mode)
                except Exception as e:
                    logging.warning(f"Failed to set permissions: {e}")
            return True
        except OSError as e:
            if e.errno == errno.EEXIST:
                return self._check_idempotency(src, dst)
            logging.error(f"Copy error {src} -> {dst}: {e}")
            return False

    def _check_idempotency(self, src: Path, dst: Path) -> bool:
        """Returns True if dst exists and is 'same' as src, else False."""
        try:
            if src.stat().st_ino == dst.stat().st_ino:
                logging.info(f"Link already exists (idempotent): {dst}")
                return True
            else:
                logging.error(f"Destination exists but is different file: {dst}")
                return False
        except OSError:
            logging.error(f"Destination exists and cannot be verified: {dst}")
            return False


# ==========================================
# Main Logic
# ==========================================


def load_config(config_path: Optional[str]) -> Dict[str, Any]:
    """
    Loads JSON config and applies environment variable overrides.
    Priority:
    1. CLI arg --config
    2. Env Var AFTERSEED_CONFIG
    3. config.json in CWD
    4. config.json in Script Directory
    """
    default_config = {
        "movie_path": "/data/media/movies",
        "tv_path": "/data/media/tv",
        "link_mode": "hardlink",
        "min_size_mb": 50,
        "normalize_permissions": False,
        "file_mode": "0644",
        "dir_mode": "0755",
        "radarr": {"url": "", "key": ""},
        "sonarr": {"url": "", "key": ""},
        "plex": {"url": "", "token": ""},
        "dry_run": True,
    }

    config = default_config.copy()

    # Identify candidate config file
    candidate_file = None

    if config_path:
        candidate_file = Path(config_path)
    else:
        # Try finding it automatically
        candidates = [
            os.getenv("AFTERSEED_CONFIG"),
            Path.cwd() / "config.json",
            Path(__file__).parent / "config.json",
        ]

        for cand in candidates:
            if cand:
                p = Path(cand)
                if p.exists() and p.is_file():
                    candidate_file = p
                    break

    # Load if found
    if candidate_file:
        if candidate_file.exists():
            try:
                with open(candidate_file, "r") as f:
                    user_config = json.load(f)

                    # Schema mapping: Support movies_library/tv_library legacy keys
                    if "movies_library" in user_config:
                        user_config["movie_path"] = user_config["movies_library"]
                    if "tv_library" in user_config:
                        user_config["tv_path"] = user_config["tv_library"]

                    config.update(user_config)
                    logging.info(f"Loaded config from {candidate_file}")
            except json.JSONDecodeError as e:
                logging.error(
                    f"⚠ Config file has invalid JSON syntax - using defaults. Error: {e}"
                )
            except Exception as e:
                logging.error(f"Failed to parse config: {e}. Using defaults.")
        else:
            logging.warning(
                f"Config file specified but not found: {candidate_file}. Using defaults."
            )
    else:
        if config_path:
            # Only warn if user explicitly asked for a file that wasn't found (handled by exists check above largely, but here covers non-matches)
            pass
        else:
            logging.info(
                "No config.json found in search paths. Using built-in defaults."
            )

    # 2. Environment Variables Overrides (Highest Priority for Paths)
    # Allows separate instances to use env vars for different libraries
    env_movie = os.getenv("AFTERSEED_MOVIES")
    if env_movie:
        config["movie_path"] = env_movie
        logging.info(f"Config override (ENV): movie_path = {env_movie}")

    env_tv = os.getenv("AFTERSEED_TV")
    if env_tv:
        config["tv_path"] = env_tv
        logging.info(f"Config override (ENV): tv_path = {env_tv}")

    return config


class UI:
    """Helper for formatted console output."""

    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    CYAN = "\033[36m"

    CHECK = "✓"
    WARN = "⚠"
    INFO = "ℹ"
    ARROW = "→"
    CROSS = "✗"

    _has_colors = False

    @classmethod
    def init(cls) -> None:
        cls._has_colors = sys.stdout.isatty()
        if not cls._has_colors:
            cls.RESET = (
                cls.BOLD
            ) = cls.DIM = cls.RED = cls.GREEN = cls.YELLOW = cls.BLUE = cls.CYAN = ""

    @classmethod
    def header(cls, text: str) -> None:
        # ── Header ────────────────────────
        clean_len = len(text) + 3  # "── text "
        dash_len = max(5, 80 - clean_len)
        line = "─" * dash_len
        logging.info(f"{cls.CYAN}── {text} {line}{cls.RESET}")

    @classmethod
    def main_header(cls, mode_str: str) -> None:
        # Big block header
        logging.info(f"{cls.CYAN}{'='*80}")
        logging.info(f"{'afterseed.py - ' + mode_str:^80}")
        logging.info(f"{'='*80}{cls.RESET}")
        logging.info("")

    @classmethod
    def kv(cls, key: str, value: str) -> None:
        logging.info(f"{cls.BOLD}{key}:{cls.RESET} {value}")

    @classmethod
    def item(
        cls,
        text: str,
        icon: Optional[str] = None,
        color: Optional[str] = None,
        indent: int = 1,
    ) -> None:
        sp = "  " * indent
        c = color if color else ""
        ic = f"{icon} " if icon else ""
        logging.info(f"{sp}{c}{ic}{text}{cls.RESET}")

    @classmethod
    def progress(cls, current: int, total: int, prefix: str = "") -> None:
        """Displays a simple progress bar."""
        if not cls._has_colors and not sys.stdout.isatty():
            # Avoid spamming logs if redirected
            return

        # [=====>    ] 50%
        width = 30
        percent = float(current) / total
        filled = int(width * percent)
        bar = "=" * filled + ">" + " " * (width - filled - 1)
        # Clamp bar
        if filled == width:
            bar = "=" * width

        # \r to overwrite line
        sys.stdout.write(
            f"\r{prefix} [{bar}] {current}/{total} ({int(percent * 100)}%)"
        )
        sys.stdout.flush()

        if current >= total:
            sys.stdout.write("\n")


class ConsoleFormatter(logging.Formatter):
    """Custom formatter to strip timestamps from INFO logs for UI cleanliness."""

    def format(self, record: logging.LogRecord) -> str:
        if record.levelno == logging.INFO:
            return record.getMessage()
        # For warnings/errors, keep standard format
        return super().format(record)


def setup_logging(verbose: bool, log_file: Optional[str] = None) -> None:
    """Configures logging with clean console output and detailed file output."""
    UI.init()

    root_val = logging.DEBUG if verbose else logging.INFO
    root_logger = logging.getLogger()
    root_logger.setLevel(root_val)

    # Reset handlers to avoid duplication during tests/re-runs
    for h in root_logger.handlers[:]:
        root_logger.removeHandler(h)

    console_fmt_str = "%(message)s" if not verbose else "%(levelname)s: %(message)s"
    console_handler = logging.StreamHandler(sys.stdout)
    if not verbose:
        console_handler.setFormatter(ConsoleFormatter(console_fmt_str))
    else:
        console_handler.setFormatter(logging.Formatter(console_fmt_str))

    root_logger.addHandler(console_handler)

    # 2. File Handler (Detailed)
    if log_file:
        file_fmt_str = "%(asctime)s - %(levelname)s - %(message)s"
        try:
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(
                logging.Formatter(file_fmt_str, datefmt="%Y-%m-%d %H:%M:%S")
            )
            root_logger.addHandler(file_handler)
        except Exception as e:
            print(
                f"WARNING: Could not open log file '{log_file}': {e}", file=sys.stderr
            )

    # Suppress lower level lib logging if needed
    # logging.getLogger("urllib3").setLevel(logging.WARNING)


def validate_config(config: Dict[str, Any], live_mode: bool) -> List[str]:
    """
    Validates configuration.
    Returns list of warnings (empty if all OK).
    Never prevents script from running - just warns.
    """
    warnings = []

    # 1. Library Paths
    for key, label in [("movie_path", "Movies"), ("tv_path", "TV")]:
        path_str = config.get(key)
        if not path_str:
            warnings.append(f"Missing config: '{key}' not set")
            continue

        p = Path(path_str)
        if not p.exists():
            warnings.append(
                f"{label} directory not found: {p} (will be created if needed)"
            )
        elif live_mode and not os.access(p, os.W_OK):
            warnings.append(f"{label} directory is not writable: {p}")

    # 2. API Configs
    services = {
        "radarr": ["url", "key"],
        "sonarr": ["url", "key"],
        "plex": ["url", "token", "section_id"],
        "tmdb": ["api_key"],
    }

    for svc, fields in services.items():
        svc_cfg = config.get(svc, {})
        # If section is missing or empty, maybe warn?
        # Requirement: "Warn about missing optional configs".
        if not svc_cfg:
            # warnings.append(f"{svc.capitalize()} is not configured")
            # Let's check based on logic usage elsewhere
            pass
        else:
            for f in fields:
                val = svc_cfg.get(f)

                # Special handling for key aliases
                if f == "key" and svc_cfg.get("api_key"):
                    continue

                if not val:
                    warnings.append(f"{svc.capitalize()} {f} is empty/missing")

                # URL Validation
                if f == "url" and val:
                    parsed = urllib.parse.urlparse(val)
                    if not parsed.scheme or parsed.scheme not in ("http", "https"):
                        warnings.append(f"{svc.capitalize()} URL invalid: {val}")

    return warnings


def health_check(config: Dict[str, Any]) -> List[str]:
    """
    Performs health checks on storage and services.
    Returns list of issues (empty if all healthy).
    """
    issues = []

    # 1. Check disk space
    try:
        for path_key, label in [("movie_path", "Movies"), ("tv_path", "TV")]:
            path_str = config.get(path_key)
            if path_str:
                p = Path(path_str)
                if p.exists():
                    stat = os.statvfs(p)
                    free_gb = (stat.f_bavail * stat.f_frsize) / (1024**3)
                    if free_gb < 10:  # Less than 10GB free
                        issues.append(
                            f"Low disk space on {label}: {free_gb:.1f} GB free"
                        )
    except Exception as e:
        issues.append(f"Disk space check failed: {e}")

    # 2. Test API connectivity (quick ping)
    services = {
        "radarr": ("Radarr", "/api/v3/system/status"),
        "sonarr": ("Sonarr", "/api/v3/system/status"),
        "plex": ("Plex", "/"),
    }

    for svc, (name, endpoint) in services.items():
        svc_cfg = config.get(svc, {})
        url = svc_cfg.get("url")
        key = svc_cfg.get("key") or svc_cfg.get("token")
        if url and key:
            try:
                test_url = f"{url.rstrip('/')}{endpoint}"
                req = urllib.request.Request(test_url)
                if svc != "plex":
                    req.add_header("X-Api-Key", key)
                else:
                    req.add_header("X-Plex-Token", key)
                with urllib.request.urlopen(req, timeout=5) as resp:
                    if resp.status != 200:
                        issues.append(f"{name} API returned {resp.status}")
            except Exception as e:
                issues.append(f"{name} connectivity failed: {e}")

    return issues


def run_tests() -> int:
    """Run internal tests with sample torrent names."""
    print("Running internal validation tests...")

    # Create temporary directory for real FS checks
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_root = Path(tmp_dir)
        movies_root = tmp_root / "movies"
        tv_root = tmp_root / "tv"
        downloads_root = tmp_root / "downloads"

        movies_root.mkdir()
        tv_root.mkdir()
        downloads_root.mkdir()

        config = {
            "movie_path": str(movies_root),
            "tv_path": str(tv_root),
            "link_mode": "hardlink",
        }

        test_cases = [
            {
                "name": "The.Matrix.1999.1080p.BluRay.x264-GROUP",
                "category": "movies",
                "expected": {
                    "type": MediaType.MOVIE,
                    "title": "The Matrix",
                    "year": 1999,
                    "quality": "1080p",
                    "rel_dest": "The Matrix (1999)/The Matrix (1999).mkv",
                },
            },
            {
                "name": "Breaking.Bad.S01E01.Pilot.1080p.WEB-DL",
                "category": "tv",
                "expected": {
                    "type": MediaType.EPISODE,
                    "title": "Breaking Bad",
                    "season": 1,
                    "episode": 1,
                    "rel_dest": "Breaking Bad/Season 01/Breaking Bad - S01E01 - Pilot.mkv",
                },
            },
            {
                "name": "Avengers.Endgame.2019.2160p.UHD.BluRay.x265-IAMGROOT",
                "category": "movies",
                "expected": {
                    "type": MediaType.MOVIE,
                    "title": "Avengers Endgame",
                    "year": 2019,
                    "quality": "2160p",
                    "rel_dest": "Avengers Endgame (2019)/Avengers Endgame (2019).mkv",
                },
            },
            {
                "name": "The.Office.US.S03E15.720p.HDTV.x264-DIMENSION",
                "category": "tv",
                "expected": {
                    "type": MediaType.EPISODE,
                    "title": "The Office US",
                    "season": 3,
                    "episode": 15,
                    "rel_dest": "The Office US/Season 03/The Office US - S03E15.mkv",
                },
            },
        ]

        classifier = Classifier()
        decision_engine = DecisionEngine(config)

        failures = 0

        for test in test_cases:
            # Setup Fake Environment
            fname = test["name"] + ".mkv"
            src_path = downloads_root / fname
            src_path.touch()  # Create dummy file for existence check

            cand = FileCandidate(
                source_path=src_path,
                relative_path=Path(fname),
                size=1024,
                extension=".mkv",
                is_video=True,
            )

            ctx = TorrentContext(
                torrent_name=test["name"],
                save_path=downloads_root,
                content_path=src_path,
                category=test["category"],
                tags=[],
                info_hash="TEST",
            )

            # Execute
            cl = classifier.classify(cand, ctx)
            plan = decision_engine.plan(cl)

            # Verify
            exp = test["expected"]
            errors = []

            # Check Media Type
            if cl.media_type != exp["type"]:
                errors.append(f"Type: got {cl.media_type}, expected {exp['type']}")

            # Check Classifier Metadata
            if cl.media_type == MediaType.MOVIE:
                if cl.detected_title != exp["title"]:
                    errors.append(
                        f"Title: got '{cl.detected_title}', expected '{exp['title']}'"
                    )
                if cl.detected_year != exp["year"]:
                    errors.append(
                        f"Year: got {cl.detected_year}, expected {exp['year']}"
                    )
                if "quality" in exp and cl.resolution != exp["quality"]:
                    errors.append(
                        f"Quality: got '{cl.resolution}', expected '{exp['quality']}'"
                    )

            elif cl.media_type == MediaType.EPISODE:
                if cl.detected_title != exp["title"]:
                    errors.append(
                        f"Title: got '{cl.detected_title}', expected '{exp['title']}'"
                    )
                if cl.season != exp["season"]:
                    errors.append(f"Season: got {cl.season}, expected {exp['season']}")
                if cl.episode != exp["episode"]:
                    errors.append(
                        f"Episode: got {cl.episode}, expected {exp['episode']}"
                    )

            # Check Destination
            if plan.action_type != ActionType.LINK:
                errors.append(
                    f"Action: got {plan.action_type}, expected LINK. Reason: {plan.reason}"
                )
            else:
                # Check relative path from root
                root = movies_root if cl.media_type == MediaType.MOVIE else tv_root
                if plan.destination:
                    try:
                        rel_dest = plan.destination.relative_to(root)
                        if str(rel_dest) != exp["rel_dest"]:
                            errors.append(
                                f"Dest: got '{rel_dest}', expected '{exp['rel_dest']}'"
                            )
                    except ValueError:
                        errors.append(
                            f"Destination {plan.destination} not in expected root {root}"
                        )
                else:
                    errors.append("Destination is None")

            if errors:
                failures += 1
                print(f"✗ Failed: {test['name']}")
                for e in errors:
                    print(f"    - {e}")
            else:
                print(f"✓ Passed: {test['name']}")

    if failures == 0:
        print(f"\nAll {len(test_cases)} tests passed!")
        return 0
    else:
        print(f"\nTests failed: {failures}/{len(test_cases)}")
        return 1


def main() -> int:
    """
    Entry point.
    1. Parse Command Line Args
    2. Setup Logging
    3. Load Config
    4. Build Context
    5. Run Pipeline (Scan -> Classify -> Plan -> Execute)
    """
    parser = argparse.ArgumentParser(
        description="afterseed: qBittorrent post-processor"
    )

    # qBittorrent Positional Arguments (Name, Path, Category)
    parser.add_argument("qb_name", nargs="?", help="Torrent Name (qBit %%N)")
    parser.add_argument("qb_path", nargs="?", help="Torrent Content Path (qBit %%F)")
    parser.add_argument(
        "qb_category", nargs="?", default="", help="Torrent Category (qBit %%L)"
    )

    # Flags
    parser.add_argument(
        "--test", action="store_true", help="Run internal validation tests"
    )
    parser.add_argument(
        "--health-check", action="store_true", help="Run health checks and exit"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Simulate without making changes"
    )
    parser.add_argument("--verbose", action="store_true", help="Debug logging")
    parser.add_argument("--log-file", type=str, help="Append logs to this file")
    parser.add_argument("--config", type=str, help="Path to config file")

    args = parser.parse_args()

    if args.test:
        return run_tests()

    if args.health_check:
        config = load_config(args.config)
        issues = health_check(config)
        if issues:
            print("Health check found issues:")
            for issue in issues:
                print(f"  - {issue}")
            return 1
        else:
            print("All systems healthy!")
            return 0

    # Load config early
    config = load_config(args.config)

    # Determine Dry Run Mode
    # 1. CLI Flag (--dry-run) takes precedence
    # 2. Config "dry_run"
    # 3. Default to True (Safe/Dry)
    if args.dry_run:
        dry_run = True
    else:
        dry_run = config.get("dry_run", True)

    setup_logging(args.verbose, args.log_file)

    # Validation
    cfg_warnings = validate_config(config, live_mode=not dry_run)
    if cfg_warnings:
        logging.warning("⚠ Configuration warnings:")
        for w in cfg_warnings:
            logging.warning(f"  - {w}")

    # Health Check
    health_issues = health_check(config)
    if health_issues:
        logging.warning("⚠ Health check issues:")
        for issue in health_issues:
            logging.warning(f"  - {issue}")

    # 1. Pipeline: Build Context
    # Strict priority: Positional Arguments
    if args.qb_name and args.qb_path:
        UI.main_header("DRY RUN MODE" if dry_run else "LIVE MODE")
        UI.kv("Torrent", args.qb_name)
        UI.kv("Path", args.qb_path)
        UI.kv("Category", args.qb_category if args.qb_category else "(none)")
        logging.info("")

        context = TorrentContext(
            torrent_name=args.qb_name,
            content_path=Path(args.qb_path).resolve(),
            save_path=Path(args.qb_path).parent.resolve(),
            category=args.qb_category if args.qb_category else None,
            tags=[],
            info_hash="",
        )
    else:
        # Fallback to Environment Variables
        context = TorrentContext.from_env()
        if not context.is_valid():
            logging.error(
                "Missing required arguments. Usage: afterseed.py NAME PATH [CATEGORY]"
            )
            return 1

        UI.main_header("DRY RUN MODE (ENV)" if dry_run else "LIVE MODE (ENV)")
        UI.kv("Torrent", context.torrent_name)
        UI.kv("Path", str(context.content_path))
        UI.kv("Category", context.category if context.category else "(none)")
        logging.info("")

    try:
        scan_target = context.content_path

        # Guard: Content path validation
        if not scan_target.exists():
            logging.error(f"Content path does not exist: {scan_target}")
            return 1

        # 1. Archive Extraction
        UI.header("Checking for Archives")
        archive_handler = ArchiveHandler(dry_run=dry_run)

        if not archive_handler.has_dependencies():
            has_archives = False
            # Quick check to see if we should warn
            for root, _, files in os.walk(scan_target):
                if any(
                    Path(f).suffix.lower() in ArchiveHandler.ARCHIVE_EXTS for f in files
                ):
                    has_archives = True
                    break
            if has_archives:
                UI.item(
                    "Archives detected but tools missing!", icon=UI.WARN, color=UI.RED
                )
                logging.warning(
                    "Install 'unrar' and 'p7zip-full' to support extraction."
                )
        else:
            extracted_count = archive_handler.extract_all(scan_target)
            if extracted_count > 0:
                UI.item(
                    f"Extracted {extracted_count} archives.",
                    icon=UI.CHECK,
                    color=UI.GREEN,
                )

        # 2. Scan
        scanner = Scanner()
        result = scanner.scan(scan_target)

        # Always print summary for now
        Scanner._print_scan_summary(result)

        # Initialize Stats
        stats = ProcessingStats()
        stats.main_videos = len(result.main_video_files)
        stats.extras = len(result.extras)
        stats.ignored = len(result.ignored)
        stats.scanned = stats.main_videos + stats.extras + stats.ignored
        # Count subtitles (extension internal to Scanner, but available here implicitly via convention)
        stats.subtitles = sum(
            1
            for e in result.extras
            if e.extension.lower() in Scanner.SUBTITLE_EXTENSIONS
        )

        # 3. Process
        if result.main_video_files:
            # Config is already loaded

            classifier = Classifier()
            decision_engine = DecisionEngine(config)
            execution_engine = ExecutionEngine(config, dry_run=dry_run)
            notifier = Notifier(config, dry_run=dry_run)

            any_success = False

            # Helper to manage visual grouping
            last_destination_parent = None

            total_items = len(result.main_video_files)
            processed_items = 0

            for cand in result.main_video_files:
                processed_items += 1
                UI.progress(processed_items, total_items, prefix="Processing files:")

                # Classify
                cls_result = classifier.classify(cand, context)

                # Plan Main Video
                plan = decision_engine.plan(cls_result)

                # UI Grouping Logic
                # Only print destination header if we have a valid plan and it's a new folder
                if plan.destination:
                    current_parent = plan.destination.parent
                    if current_parent != last_destination_parent:
                        UI.header("Destination")
                        UI.item(f"{current_parent}/", icon=UI.ARROW, color=UI.BLUE)
                        last_destination_parent = current_parent

                        action_label = "DRY RUN Actions" if dry_run else "Actions"
                        UI.header(action_label)

                # Execute Main Video
                success = False
                if plan.action_type == ActionType.LINK:
                    success = execution_engine.execute(plan)
                    if success:
                        any_success = True
                        stats.hardlinks_created += 1
                    else:
                        stats.hardlinks_failed += 1
                elif plan.action_type == ActionType.SKIP:
                    if dry_run or args.verbose:
                        UI.item(
                            f"Skipped: {cand.relative_path} -> {plan.reason}",
                            icon=UI.INFO,
                            color=UI.DIM,
                        )

                # Queue Notifications (batched by Notifier)
                if success and not dry_run:
                    if cls_result.media_type == MediaType.MOVIE and plan.destination:
                        notifier.queue_radarr(plan.destination.parent)
                    elif (
                        cls_result.media_type == MediaType.EPISODE and plan.destination
                    ):
                        notifier.queue_sonarr(plan.destination.parent)

                # Handle Associated Extras
                if (success or dry_run) and plan.action_type == ActionType.LINK:
                    related_extras = Scanner.get_related_extras(cand, result.extras)
                    for extra in related_extras:
                        extra_plan = decision_engine.plan_extra(extra, plan)
                        if extra_plan.action_type == ActionType.LINK:
                            extra_success = execution_engine.execute(extra_plan)
                            if extra_success:
                                stats.hardlinks_created += 1
                            else:
                                stats.hardlinks_failed += 1

            # Finalize Notifications
            if any_success and not dry_run:
                notifier.queue_plex()

            notifier.flush()

        # FINAL SUMMARY
        stats.print_summary(dry_run)

        return 0

    except Exception:
        logging.exception("Fatal error in afterseed execution")
        # In this phase, we don't want to crash the post-processing script pipeline completely
        # but for development it's good to know.
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
