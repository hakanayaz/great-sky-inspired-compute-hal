"""Small durable, append-only audit log for the HAL MVP.

JSON-lines is used here because each event can be appended independently and
inspected easily during development. Production deployment should add durable
storage guarantees, retention policy, protected access, and authenticated
caller identity before treating this as a compliance-grade audit system.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
from pathlib import Path


@dataclass(frozen=True)
class AuditEvent:
    action: str
    outcome: str
    detail: str = ""
    job_id: str | None = None
    actor: str = "system"


class JsonlAuditLog:
    """Writes one complete JSON object per line, suitable for later ingestion."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def record(self, event: AuditEvent) -> None:
        payload = asdict(event) | {"timestamp": datetime.now(timezone.utc).isoformat()}
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(payload, sort_keys=True) + "\n")
