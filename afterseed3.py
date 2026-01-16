#!/usr/bin/env python3
"""
qb-postprocess

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

    @classmethod
    def from_env(cls) -> "TorrentContext":
        """Create context from qBittorrent environment variables."""
        return cls(
            torrent_name=os.environ.get("TORRENT_NAME", ""),
            save_path=Path(os.environ.get("TORRENT_SAVE_PATH", ".")),
            category=os.environ.get("TORRENT_CATEGORY"),
            tags=os.environ.get("TORRENT_TAGS", "").split(",")
            if os.environ.get("TORRENT_TAGS")
            else [],
            content_path=Path(os.environ.get("TORRENT_CONTENT_PATH", ".")),
        )


class MediaType(Enum):
    MOVIE = auto()
    EPISODE = auto()
    UNKNOWN = auto()
    IGNORE = auto()


@dataclass
class FileCandidate:
    """A file that might be processed."""

    source_path: Path
    relative_path: Path
    size: int
    extension: str
    is_video: bool


@dataclass
class ScanResult:
    """Result of scanning a torrent directory."""

    candidates: List[FileCandidate]
    extras: List[FileCandidate]  # Associated files (subtitles, etc.)


@dataclass
class Classification:
    """Structured result of classifying a file candidate."""

    candidate: FileCandidate
    media_type: MediaType
    detected_title: Optional[str] = None
    detected_year: Optional[int] = None
    season: Optional[int] = None
    episode: Optional[int] = None
    episode_title: Optional[str] = None
    resolution: Optional[str] = None
    source: Optional[str] = None
    video_codec: Optional[str] = None
    edition: Optional[str] = None
    confidence: float = 0.0


class ActionType(Enum):
    LINK = auto()
    SKIP = auto()


@dataclass
class PlannedAction:
    """What to do with a file."""

    action_type: ActionType
    destination: Optional[Path] = None
    reason: str = ""


class ProcessingStats:
    """Statistics for the processing run."""

    def __init__(self):
        self.hardlinks_created = 0
        self.hardlinks_failed = 0
        self.start_time = time.time()

    def print_summary(self, dry_run: bool):
        """Print processing summary."""
        duration = time.time() - self.start_time
        mode = "[DRY RUN] " if dry_run else ""

        print(f"\n{mode}Processing Complete")
        print(f"Duration: {duration:.1f}s")
        print(f"Hardlinks Created: {self.hardlinks_created}")
        if self.hardlinks_failed > 0:
            print(f"Hardlinks Failed: {self.hardlinks_failed}")


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
            print("Radarr Notifications")
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
            print("Sonarr Notifications")
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
            print("External Services")
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

    def extract_archive(self, archive_path: Path, extract_to: Path) -> bool:
        """Extract archive to specified directory."""
        if self.dry_run:
            print(f"[DRY RUN] Would extract {archive_path} to {extract_to}")
            return True

        # Implementation would go here
        return False


class Classifier:
    """
    Responsible for pure-function analysis of filenames to determine content.
    """

    # Pre-compile regex for performance
    RE_SEASON_EPISODE = re.compile(
        r"(?P<season_marker>[sS]?)(?P<season>\d{1,2})[\.\s\_-]?"
        r"(?P<episode_marker>[eExX])(?P<episode>\d{1,2})",
        re.IGNORECASE,
    )

    RE_SEASON_PACK = re.compile(
        r"\b(?P<season_marker>S|Season)[\.\s\_-]?(?P<season>\d{1,2})\b", re.IGNORECASE
    )

    RE_YEAR = re.compile(r"[\(\[\.\s_](?P<year>19\d{2}|20\d{2})($|[\)\]\.\s_])")

    RE_RES = re.compile(r"(?P<res>480p|576p|720p|1080[pi]|2160p|4k|uhd)", re.IGNORECASE)

    RE_SOURCE = re.compile(
        r"(?P<source>bluray|remux|web-?dl|hdtv|dvd|bdrip|brrip)", re.IGNORECASE
    )

    RE_CODEC = re.compile(
        r"(?P<codec>x264|x265|h\.?264|h\.?265|hevc|av1|divx|xvid|avc)", re.IGNORECASE
    )

    RE_TECH_JUNK = re.compile(
        r"\b(x264|x265|h264|h265|hevc|web-?dl|bluray|remux|hdtv|amzn|nf|dsnp|repack|proper)\b",
        re.IGNORECASE,
    )

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
        """Analyzes a candidate file."""
        classification = Classification(
            candidate=candidate, media_type=MediaType.UNKNOWN, detected_title=None
        )

        filename = candidate.source_path.stem

        # Extract tech metadata
        self._extract_tech_metadata(filename, classification)

        # Strategy selection
        check_movie_first = False
        if context.category and "movie" in context.category.lower():
            check_movie_first = True

        if check_movie_first:
            if self._apply_movie_parsing(filename, classification):
                return classification
            if self._apply_tv_parsing(filename, classification):
                return classification
        else:
            if self._apply_tv_parsing(filename, classification):
                return classification
            if self._apply_movie_parsing(filename, classification):
                return classification

        # Fallback
        self._apply_fallback_logic(filename, context, classification)
        return classification

    def _apply_tv_parsing(self, filename: str, classification: Classification) -> bool:
        tv_info = self.parse_tv_name(filename)
        if tv_info:
            classification.media_type = MediaType.EPISODE
            classification.detected_title = tv_info.get("show", tv_info.get("title"))
            classification.season = tv_info["season"]
            classification.episode = tv_info["episode"]
            classification.episode_title = tv_info["episode_title"]
            classification.confidence = 0.9
            return True
        return False

    def _apply_movie_parsing(
        self, filename: str, classification: Classification
    ) -> bool:
        movie_info = self.parse_movie_name(filename)
        if movie_info:
            classification.media_type = MediaType.MOVIE
            classification.detected_title = movie_info["title"]
            classification.detected_year = movie_info["year"]
            if movie_info.get("quality"):
                classification.resolution = movie_info["quality"]
            classification.confidence = 0.8
            return True
        return False

    def parse_tv_name(self, filename: str) -> Optional[Dict[str, Any]]:
        match = self.RE_SEASON_EPISODE.search(filename)
        if match:
            try:
                season = int(match.group("season"))
                episode = int(match.group("episode"))
                raw_title = filename[: match.start()]
                clean_title = self._clean_title_string(raw_title)
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
        year = None
        year_match = self.RE_YEAR.search(filename)
        if year_match:
            year = int(year_match.group("year"))

        quality = None
        res_match = self.RE_RES.search(filename)
        if res_match:
            quality = res_match.group("res").lower()
            if quality in ["4k", "uhd"]:
                quality = "2160p"

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

        if not clean_title:
            return None

        return {"title": clean_title, "year": year, "quality": quality}

    def _extract_episode_title(self, raw_suffix: str) -> Optional[str]:
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
        ym = self.RE_YEAR.search(filename)
        if ym:
            cl.detected_year = int(ym.group("year"))

        rm = self.RE_RES.search(filename)
        if rm:
            cl.resolution = rm.group("res").lower()

        sm = self.RE_SOURCE.search(filename)
        if sm:
            src = sm.group("source").upper()
            if "WEB" in src:
                src = "WEB-DL"
            cl.source = src
        if "REMUX" in filename.upper():
            cl.source = "REMUX"

        cm = self.RE_CODEC.search(filename)
        if cm:
            codec = cm.group("codec").lower()
            if codec in ["h.265", "h265"]:
                codec = "hevc"
            if codec in ["h.264", "h264", "avc"]:
                codec = "x264"
            cl.video_codec = codec

        if re.search(r"\b(repack|proper)\b", filename, re.IGNORECASE):
            cl.edition = "REPACK"

    def _apply_fallback_logic(
        self, filename: str, context: TorrentContext, cl: Classification
    ) -> None:
        if context.category:
            cat = context.category.lower()
            if "movie" in cat:
                cl.media_type = MediaType.MOVIE
            elif "tv" in cat or "show" in cat:
                cl.media_type = MediaType.EPISODE

        cl.detected_title = self._clean_title_string(filename)

    def _clean_title_string(self, text: str) -> str:
        text = re.sub(r"[._]", " ", text)
        text = re.sub(r"[\[\(].*?[\]\)]", "", text)
        text = re.sub(r"\s+", " ", text)
        return text.strip(" -")


class DecisionEngine:
    """
    Business logic layer.
    Mapping Classifications to destinations based on Config.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
        self.movie_root = Path(config.get("movie_path", "/data/media/movies"))
        self.tv_root = Path(config.get("tv_path", "/data/media/tv"))

    def plan(self, classification: Classification) -> PlannedAction:
        candidate = classification.candidate

        if not candidate.source_path.exists():
            return PlannedAction(
                ActionType.SKIP, f"Source missing: {candidate.source_path}"
            )

        if classification.media_type == MediaType.IGNORE:
            return PlannedAction(ActionType.SKIP, "Ignored content type")

        if classification.media_type == MediaType.MOVIE:
            return self._plan_movie(classification)
        elif classification.media_type == MediaType.EPISODE:
            return self._plan_episode(classification)

        return PlannedAction(ActionType.SKIP, "Unknown media type")

    def _plan_movie(self, classification: Classification) -> PlannedAction:
        title = classification.detected_title or "Unknown"
        year = classification.detected_year or 0
        folder_name = f"{title} ({year})"
        dest_dir = self.movie_root / folder_name
        filename = f"{folder_name}.mkv"  # Assume mkv for now
        dest_path = dest_dir / filename

        return PlannedAction(ActionType.LINK, dest_path)

    def _plan_episode(self, classification: Classification) -> PlannedAction:
        title = classification.detected_title or "Unknown"
        season = classification.season or 1
        episode = classification.episode

        show_dir = self.tv_root / title
        season_dir = show_dir / f"Season {season:02d}"

        if episode:
            filename = f"{title} - S{season:02d}E{episode:02d}.mkv"
        else:
            filename = f"{title} - S{season:02d} (Season Pack).mkv"

        dest_path = season_dir / filename
        return PlannedAction(ActionType.LINK, dest_path)


class ExecutionEngine:
    """Handles the actual file operations."""

    def __init__(self, dry_run: bool = False) -> None:
        self.dry_run = dry_run

    def execute(self, plan: PlannedAction) -> bool:
        """Execute a planned action."""
        if plan.action_type != ActionType.LINK or not plan.destination:
            return False

        if self.dry_run:
            print(f"[DRY RUN] Would link {plan.destination}")
            return True

        try:
            plan.destination.parent.mkdir(parents=True, exist_ok=True)
            os.link(plan.destination, plan.destination)  # Hardlink
            return True
        except OSError as e:
            logging.error(f"Failed to create hardlink: {e}")
            return False


class Scanner:
    """Scans torrent directories for media files."""

    VIDEO_EXTS = {".mkv", ".mp4", ".avi", ".m4v", ".mov"}
    SUBTITLE_EXTS = {".srt", ".sub", ".idx", ".ass", ".ssa"}

    @staticmethod
    def scan_directory(root_path: Path) -> ScanResult:
        """Scan directory for media files."""
        candidates = []
        extras = []

        for path in root_path.rglob("*"):
            if path.is_file():
                ext = path.suffix.lower()
                rel_path = path.relative_to(root_path)

                candidate = FileCandidate(
                    source_path=path,
                    relative_path=rel_path,
                    size=path.stat().st_size,
                    extension=ext,
                    is_video=ext in Scanner.VIDEO_EXTS,
                )

                if ext in Scanner.VIDEO_EXTS:
                    candidates.append(candidate)
                elif ext in Scanner.SUBTITLE_EXTS:
                    extras.append(candidate)

        return ScanResult(candidates=candidates, extras=extras)


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """Load configuration from file."""
    if not config_path:
        config_path = "config.json"

    try:
        with open(config_path) as f:
            return json.load(f)
    except FileNotFoundError:
        logging.warning(f"Config file {config_path} not found, using defaults")
        return {}


def validate_config(config: Dict[str, Any]) -> List[str]:
    """Validate configuration and return list of issues."""
    issues = []

    # Check required paths
    if not config.get("movie_path"):
        issues.append("movie_path not configured")
    if not config.get("tv_path"):
        issues.append("tv_path not configured")

    return issues


def health_check(config: Dict[str, Any]) -> List[str]:
    """Run health checks."""
    issues = validate_config(config)

    # Check paths exist
    for path_key in ["movie_path", "tv_path"]:
        path = config.get(path_key)
        if path and not Path(path).exists():
            issues.append(f"{path_key} directory does not exist: {path}")

    return issues


def run_tests() -> int:
    """Run internal tests with sample torrent names."""
    print("Running internal validation tests...")

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
                },
            },
        ]

        classifier = Classifier()
        decision_engine = DecisionEngine(config)

        failures = 0

        for test in test_cases:
            fname = test["name"] + ".mkv"
            src_path = downloads_root / fname
            src_path.touch()

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

            cl = classifier.classify(cand, ctx)
            plan = decision_engine.plan(cl)

            exp = test["expected"]
            errors = []

            if cl.media_type != exp["type"]:
                errors.append(f"Type: got {cl.media_type}, expected {exp['type']}")

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

            if plan.action_type != ActionType.LINK:
                errors.append(f"Action: got {plan.action_type}, expected LINK")

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
    """
    parser = argparse.ArgumentParser(
        description="qb-postprocess: qBittorrent post-processor"
    )

    parser.add_argument("qb_name", nargs="?", help="Torrent Name (qBit %%N)")
    parser.add_argument("qb_path", nargs="?", help="Torrent Content Path (qBit %%F)")
    parser.add_argument(
        "qb_category", nargs="?", default="", help="Torrent Category (qBit %%L)"
    )

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

    config = load_config(args.config)

    # Validate config
    issues = validate_config(config)
    if issues:
        print("Configuration issues:")
        for issue in issues:
            print(f"  - {issue}")
        return 1

    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        filename=args.log_file,
    )

    # Build context
    if args.qb_name and args.qb_path:
        # Command line mode
        context = TorrentContext(
            torrent_name=args.qb_name,
            save_path=Path(args.qb_path),
            category=args.qb_category,
            tags=[],
            content_path=Path(args.qb_path),
        )
    else:
        # qBittorrent environment mode
        context = TorrentContext.from_env()

    print(f"Processing: {context.torrent_name}")

    # Scan
    scanner = Scanner()
    result = scanner.scan_directory(context.content_path)

    if not result.candidates:
        print("No video files found")
        return 0

    # Classify and plan
    classifier = Classifier()
    decision_engine = DecisionEngine(config)
    notifier = Notifier(config, args.dry_run)
    execution_engine = ExecutionEngine(args.dry_run)

    stats = ProcessingStats()

    for cand in result.candidates:
        cl = classifier.classify(cand, context)
        plan = decision_engine.plan(cl)

        if plan.action_type == ActionType.LINK:
            success = execution_engine.execute(plan)
            if success:
                stats.hardlinks_created += 1
                # Queue notifications
                if cl.media_type == MediaType.MOVIE and plan.destination:
                    notifier.queue_radarr(plan.destination.parent)
                elif cl.media_type == MediaType.EPISODE and plan.destination:
                    notifier.queue_sonarr(plan.destination.parent)
            else:
                stats.hardlinks_failed += 1

    # Finalize notifications
    notifier.queue_plex()
    notifier.flush()

    stats.print_summary(args.dry_run)

    return 0


if __name__ == "__main__":
    sys.exit(main())
