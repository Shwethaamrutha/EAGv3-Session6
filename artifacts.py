"""Content-addressable artifact store with TTL cleanup."""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

from config import settings
from logger import get_logger
from schemas import Artifact

log = get_logger("artifacts")

ARTIFACTS_DIR = Path(settings.artifacts_dir)
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)


class ArtifactStore:
    def put(self, blob: bytes, *, content_type: str, source: str, descriptor: str) -> str:
        sha = hashlib.sha256(blob).hexdigest()[:16]
        art_id = f"art:{sha}"

        bin_path = ARTIFACTS_DIR / f"{sha}.bin"
        meta_path = ARTIFACTS_DIR / f"{sha}.json"

        if not bin_path.exists():
            bin_path.write_bytes(blob)
            meta = Artifact(
                id=art_id,
                content_type=content_type,
                size_bytes=len(blob),
                source=source,
                descriptor=descriptor[:200],
            )
            meta_path.write_text(meta.model_dump_json(indent=2))

        return art_id

    def get_bytes(self, artifact_id: str) -> bytes:
        sha = artifact_id.replace("art:", "")
        bin_path = ARTIFACTS_DIR / f"{sha}.bin"
        return bin_path.read_bytes()

    def get_meta(self, artifact_id: str) -> Artifact:
        sha = artifact_id.replace("art:", "")
        meta_path = ARTIFACTS_DIR / f"{sha}.json"
        return Artifact.model_validate_json(meta_path.read_text())

    def exists(self, artifact_id: str) -> bool:
        sha = artifact_id.replace("art:", "")
        return (ARTIFACTS_DIR / f"{sha}.bin").exists()

    def cleanup(self, max_age_hours: int = 72):
        """Remove artifacts older than max_age_hours."""
        now = time.time()
        removed = 0
        for meta_path in ARTIFACTS_DIR.glob("*.json"):
            try:
                mtime = meta_path.stat().st_mtime
                if (now - mtime) > max_age_hours * 3600:
                    sha = meta_path.stem
                    meta_path.unlink(missing_ok=True)
                    (ARTIFACTS_DIR / f"{sha}.bin").unlink(missing_ok=True)
                    removed += 1
            except OSError:
                pass
        if removed:
            log.info("artifacts_cleaned", removed=removed, max_age_hours=max_age_hours)


artifact_store = ArtifactStore()
