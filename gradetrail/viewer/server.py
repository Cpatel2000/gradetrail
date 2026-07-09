"""Run discovery, the dataset join, and the viewer's HTTP endpoints.

Stdlib only (docs/design/viewer.md decision 3). All filesystem reading and
join logic lives here, testable without a browser; cli.py only wires the
`view` subcommand to discover_runs() + create_server().

Errors degrade, never 500 on bad data: a malformed manifest becomes an
"error" entry in /api/runs, a corrupted results line becomes parse_errors,
an unreadable dataset becomes dataset_matches=null (tri-state -- see the
design doc's decision 1).
"""

from __future__ import annotations

import hashlib
import json
from functools import partial
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

_INDEX_HTML = Path(__file__).parent / "static" / "index.html"


def discover_runs(root: Path) -> list[Path]:
    """Run directories: direct subdirectories of root holding both results.jsonl
    and manifest.json. Missing or non-directory root yields []."""
    if not root.is_dir():
        return []
    return sorted(
        p
        for p in root.iterdir()
        if p.is_dir() and (p / "results.jsonl").is_file() and (p / "manifest.json").is_file()
    )


def dataset_index(manifest: dict) -> dict[str, dict] | None:
    """Map sample_id -> dataset row for the manifest's dataset, or None if unresolvable.

    Keys mirror EvalSpec.load_samples(): str(row[id_field]) when the id field
    is present (stringified, so integer ids join string sample_ids), else the
    1-based file line number -- blank lines are skipped but still counted.
    Malformed dataset lines are skipped (those rows simply cannot join);
    a manifest without dataset_path (pre-0.2) or an unreadable file is None.
    """
    path_str = manifest.get("dataset_path")
    if not path_str:
        return None
    id_field = manifest.get("dataset_id_field", "id")
    try:
        text = Path(path_str).read_text()
    except OSError:
        return None
    index: dict[str, dict] = {}
    for lineno, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict):
            index[str(row.get(id_field, lineno))] = row
    return index


def _dataset_matches(manifest: dict) -> bool | None:
    """Tri-state live hash check: True = verified match, False = verified
    mismatch, None = unverifiable (no dataset_path, or file unreadable)."""
    path_str = manifest.get("dataset_path")
    recorded = manifest.get("dataset_sha256")
    if not path_str or not recorded:
        return None
    try:
        current = hashlib.sha256(Path(path_str).read_bytes()).hexdigest()
    except OSError:
        return None
    return current == recorded


def _count_parse_errors(results_path: Path) -> int:
    """Non-blank results.jsonl lines that fail to parse as a JSON object."""
    try:
        text = results_path.read_text()
    except OSError:
        return 0
    errors = 0
    for line in text.splitlines():
        if not line.strip():
            continue
        try:
            if not isinstance(json.loads(line), dict):
                errors += 1
        except json.JSONDecodeError:
            errors += 1
    return errors


def _run_entry(run_dir: Path) -> dict:
    """One /api/runs entry. A malformed manifest yields {dir, error} -- the run
    surfaces rather than disappearing. The error key is omitted when healthy."""
    try:
        manifest = json.loads((run_dir / "manifest.json").read_text())
        if not isinstance(manifest, dict):
            raise ValueError("not a JSON object")
    except (OSError, ValueError) as exc:
        return {"dir": run_dir.name, "error": f"manifest.json: {exc}"}
    return {
        "dir": run_dir.name,
        "name": manifest.get("name"),
        "identity_hash": manifest.get("identity_hash"),
        "created_at": manifest.get("created_at"),
        "n_samples": manifest.get("n_samples"),
        "counts": {
            "scored": manifest.get("n_scored"),
            "provider_error": manifest.get("n_provider_error"),
            "judge_error": manifest.get("n_judge_error"),
        },
        "mean_score": manifest.get("mean_score"),
        "total_cost_usd": manifest.get("total_cost_usd"),
        "wall_time_s": manifest.get("wall_time_s"),
        "model": manifest.get("requested_model"),
        "dataset_matches": _dataset_matches(manifest),
        "parse_errors": _count_parse_errors(run_dir / "results.jsonl"),
    }


def list_runs(root: Path) -> list[dict]:
    """All discovered runs as /api/runs entries, newest first; manifest-error
    entries (no created_at) sort last."""
    entries = [_run_entry(run_dir) for run_dir in discover_runs(root)]
    return sorted(entries, key=lambda e: e.get("created_at") or "", reverse=True)


class _ViewerHandler(BaseHTTPRequestHandler):
    """Exact-match routing only; anything else is 404 (traversal included --
    /api/runs/{dir} will validate against the discovered list when it lands)."""

    def __init__(self, *args: object, results_root: Path, **kwargs: object) -> None:
        self.results_root = results_root
        super().__init__(*args, **kwargs)  # handles the request; must come last

    def do_GET(self) -> None:
        """Route GET requests per the design doc's HTTP contract."""
        path = self.path.split("?", 1)[0]
        if path == "/":
            self._send(200, "text/html; charset=utf-8", _INDEX_HTML.read_bytes())
        elif path == "/api/runs":
            body = json.dumps(list_runs(self.results_root)).encode()
            self._send(200, "application/json", body)
        else:
            self._send(404, "application/json", b'{"error": "not found"}')

    def _send(self, status: int, content_type: str, body: bytes) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: object) -> None:
        """Quiet: per-request stderr lines are noise for a local tool."""


def create_server(results_root: Path, port: int = 0) -> ThreadingHTTPServer:
    """A ThreadingHTTPServer bound to 127.0.0.1 only (never 0.0.0.0).

    port=0 binds an ephemeral port; read the real one from server_address.
    """
    handler = partial(_ViewerHandler, results_root=results_root)
    return ThreadingHTTPServer(("127.0.0.1", port), handler)
