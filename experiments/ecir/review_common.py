"""Create-only, resumable analysis utilities; never modify frozen experiments."""
import hashlib
import json
import os
import tempfile
from pathlib import Path

from experiments.revision.protocol import digest, dump, verify_manifest


def payload_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False,
                                    separators=(",", ":")).encode()).hexdigest()


def prepare_output(folder, spec):
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "analysis_spec.json"
    if path.exists():
        if json.loads(path.read_text()) != spec:
            raise ValueError(f"Analysis inputs changed; use a new output: {folder}")
    elif any(folder.iterdir()):
        raise ValueError(f"Nonempty output lacks analysis specification: {folder}")
    else:
        dump(path, spec)
    (folder / "cells").mkdir(exist_ok=True)


def cell(folder, name, compute):
    """Reuse only checksum-verified cells; interrupted writes never become cells."""
    path = folder / "cells" / f"{name}.json"
    if path.exists():
        saved = json.loads(path.read_text())
        if saved["sha256"] != payload_hash(saved["payload"]):
            raise ValueError(f"Damaged cached cell: {path}")
        print(f"Verified cached cell: {name}", flush=True)
        return saved["payload"]
    value = compute()
    # Same-filesystem rename publishes a complete cell after a crash-safe write.
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=folder,
                                     prefix=".partial-", delete=False) as handle:
        json.dump({"payload": value, "sha256": payload_hash(value)}, handle,
                  allow_nan=False)
        handle.flush()
        os.fsync(handle.fileno())
        temporary = Path(handle.name)
    temporary.replace(path)
    print(f"Completed cell: {name}", flush=True)
    return value


def finish(folder, metadata):
    if (folder / "manifest.json").exists():
        return verify_manifest(folder)
    files = sorted(path for path in folder.rglob("*") if path.is_file()
                   and not path.name.startswith(".partial-"))
    dump(folder / "manifest.json", {**metadata, "files": {
        path.relative_to(folder).as_posix(): digest(path) for path in files}})


def write_or_check(path, value):
    if path.exists():
        if json.loads(path.read_text()) != value:
            raise ValueError(f"Existing result differs: {path}")
    else:
        dump(path, value)


def check_sources(record):
    root = Path(__file__).resolve().parents[2]
    for relative, expected in record.items():
        if digest(root / relative) != expected:
            raise ValueError(f"Frozen source changed: {relative}")


def analysis_sources(module):
    root = Path(__file__).resolve().parents[2]
    paths = [Path(module), Path(__file__)]
    return {str(path.relative_to(root)): digest(path) for path in paths}
