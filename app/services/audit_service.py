"""Append-only SQLite audit storage; safe concurrent writes on a local laptop.

No updates/deletes are exposed. This is not a tamper-proof regulatory archive.
"""
import json
import sqlite3
from pathlib import Path
from app.models.audit import AuditRecord


class AuditService:
    def __init__(self, path: Path):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(path) as connection:
            connection.execute("CREATE TABLE IF NOT EXISTS audit (id TEXT PRIMARY KEY, request_id TEXT NOT NULL, record TEXT NOT NULL)")
            connection.execute("CREATE INDEX IF NOT EXISTS audit_request ON audit(request_id)")

    def append(self, record: AuditRecord) -> None:
        with sqlite3.connect(self.path, timeout=30) as connection:
            connection.execute("INSERT INTO audit VALUES (?, ?, ?)", (record.audit_id, record.request_id, record.model_dump_json()))

    def replay(self, request_id: str) -> list[dict]:
        """Return every historical decision (including comparisons), never reroute."""
        with sqlite3.connect(self.path) as connection:
            rows = connection.execute("SELECT record FROM audit WHERE request_id = ? ORDER BY rowid", (request_id,)).fetchall()
        return [json.loads(row[0]) for row in rows]
