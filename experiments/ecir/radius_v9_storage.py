"""Atomic, checksum-verified persistence for the new radius experiment only."""
import contextlib
import fcntl
import hashlib
import json
import os
import tempfile
from pathlib import Path

import torch

from experiments.revision.protocol import digest, verify_manifest


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False,
                                    separators=(",", ":")).encode()).hexdigest()


@contextlib.contextmanager
def output_lock(folder):
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / ".lock").open("a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError(f"Another process is using {folder}") from error
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def json_write(path, payload):
    """Publish once; never replace a completed file with different content."""
    path = Path(path)
    if path.exists():
        if json.loads(path.read_text()) != payload:
            raise ValueError(f"Existing file differs: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                     prefix=".partial-", delete=False) as handle:
        temporary = Path(handle.name)
        json.dump(payload, handle, indent=2, allow_nan=False)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.link(temporary, path)  # Exclusive publication on the same filesystem.
    finally:
        temporary.unlink(missing_ok=True)


def cell_read(path):
    saved = json.loads(Path(path).read_text())
    if saved["sha256"] != fingerprint(saved["payload"]):
        raise ValueError(f"Cached cell checksum mismatch: {path}")
    return saved["payload"]


def text_write(path, text):
    """Atomically publish CSV/Markdown without a partial final file on interruption."""
    if path.exists():
        if path.read_text(encoding="utf-8") != text:
            raise ValueError(f"Existing report differs: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                     prefix=".partial-", delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def cell(path, compute):
    path = Path(path)
    if path.exists():
        return cell_read(path)
    value = compute()
    json_write(path, {"payload": value, "sha256": fingerprint(value)})
    return value


def prepare(folder, spec):
    path = folder / "analysis_spec.json"
    if not path.exists() and any(not p.name.startswith(".") for p in folder.iterdir()):
        raise ValueError(f"Output lacks its analysis specification: {folder}")
    json_write(path, spec)


def cpu_state(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: cpu_state(item) for key, item in value.items()}
    if isinstance(value, list):
        return [cpu_state(item) for item in value]
    if isinstance(value, tuple):
        return tuple(cpu_state(item) for item in value)
    return value


def state_save(folder, payload):
    """Commit checkpoint and checksum together as an atomic directory rename."""
    if folder.exists():
        raise ValueError(f"Checkpoint already exists: {folder}")
    folder.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=folder.parent, prefix=".partial-") as temp:
        stage = Path(temp) / "state"
        stage.mkdir()
        checkpoint = stage / "state.pt"
        with checkpoint.open("wb") as handle:
            torch.save(cpu_state(payload), handle)
            handle.flush()
            os.fsync(handle.fileno())
        json_write(stage / "checksum.json", {"sha256": digest(checkpoint)})
        stage.rename(folder)


def state_load(folder):
    metadata = json.loads((folder / "checksum.json").read_text())
    if digest(folder / "state.pt") != metadata["sha256"]:
        raise ValueError(f"Checkpoint checksum mismatch: {folder}")
    return torch.load(folder / "state.pt", map_location="cpu", weights_only=True)


def finish(folder, spec):
    files = sorted(p for p in folder.rglob("*") if p.is_file()
                   and not any(part.startswith(".") for part in p.relative_to(folder).parts)
                   and p != folder / "manifest.json")
    json_write(folder / "manifest.json", {**spec, "files": {
        p.relative_to(folder).as_posix(): digest(p) for p in files}})


def complete(folder, spec):
    if not (folder / "manifest.json").exists():
        return False
    saved = verify_manifest(folder)
    if any(saved.get(key) != value for key, value in spec.items()):
        raise ValueError(f"Completed output has different inputs: {folder}")
    return True
