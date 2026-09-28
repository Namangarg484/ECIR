"""Build and audit the allowlisted anonymous ECIR review artifact.

This script packages no raw data or checkpoints and runs no experiment.
"""

from __future__ import annotations

import argparse
import getpass
import hashlib
import os
import platform
import re
import shutil
import tempfile
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

EXACT_FILES = (
    "ECIR_ARTIFACT_README.md",
    "ECIR_REVIEW_V8.md",
    "ECIR_RADIUS_V9.md",
    "ECIR_REVIEW_V10.md",
    "ECIR.tex",
    "paper/abstract.txt",
    "paper/IMPLEMENTATION_DETAILS.md",
    "requirements-revision.txt",
    "requirements-revision.lock.txt",
    "experiments/__init__.py",
    "src/__init__.py",
    "src/models/__init__.py",
    "src/models/vce_model.py",
    "src/models/sasrec.py",
    "scripts/run_revision.sh",
    "scripts/run_ecir_experiments.sh",
    "scripts/run_ecir_additional.sh",
    "scripts/run_ecir_frozen_radius.sh",
    "scripts/run_ecir_review_v8.sh",
    "scripts/run_ecir_review_v10.sh",
    "scripts/run_ecir_radius_v9.sh",
    "scripts/build_ecir_anonymous_artifact.py",
    "scripts/audit_ecir_submission.py",
    "artifacts/revision/v1/data/talkplay/manifest.json",
    "artifacts/revision/v1/data/ml1m/manifest.json",
    "artifacts/revision/v1/data/lastfm/manifest.json",
    "artifacts/revision/v1/data/amazon_music/manifest.json",
    "artifacts/revision/mps-v1/runs/talkplay/selection.json",
    "artifacts/revision/mps-v1/runs/ml1m/selection.json",
    "artifacts/revision/mps-v1/runs/lastfm/selection.json",
    "artifacts/revision/mps-v1/runs/amazon_music/selection.json",
    "artifacts/ecir/mps-v2/baseline-runs/talkplay/selection.json",
    "artifacts/ecir/mps-v2/baseline-runs/ml1m/selection.json",
    "artifacts/ecir/mps-v2/baseline-runs/lastfm/selection.json",
    "artifacts/ecir/mps-v2/baseline-runs/amazon_music/selection.json",
    "artifacts/ecir/mps-v2/stochastic-runs/talkplay/selection.json",
    "artifacts/ecir/mps-v2/stochastic-runs/ml1m/selection.json",
    "artifacts/ecir/mps-v2/stochastic-runs/lastfm/selection.json",
    "artifacts/ecir/mps-v2/stochastic-runs/amazon_music/selection.json",
    "artifacts/ecir/additional-v1/session-knn-runs/talkplay/selection.json",
    "artifacts/ecir/additional-v1/session-knn-runs/ml1m/selection.json",
    "artifacts/ecir/additional-v1/session-knn-runs/lastfm/selection.json",
    "artifacts/ecir/additional-v1/session-knn-runs/amazon_music/selection.json",
    "artifacts/ecir/frozen-radius-v1/runs/talkplay/selection.json",
    "artifacts/ecir/frozen-radius-v1/runs/ml1m/selection.json",
    "artifacts/ecir/frozen-radius-v1/runs/lastfm/selection.json",
    "artifacts/ecir/frozen-radius-v1/runs/amazon_music/selection.json",
)

TREE_FILES = {
    "experiments/revision": {".py", ".json"},
    "experiments/ecir": {".py", ".json"},
    "paper/generated/ecir-mps-v2": {".md", ".csv", ".tex", ".json"},
    "paper/generated/ecir-additional-v1": {".md", ".csv", ".tex", ".json"},
    "paper/generated/ecir-direction-sensitivity-v1": {
        ".md",
        ".csv",
        ".tex",
        ".json",
    },
    "paper/generated/ecir-frozen-radius-v1": {".md", ".csv", ".tex", ".json"},
    "paper/generated/ecir-review-v8": {".md", ".csv", ".tex", ".json"},
    "paper/generated/ecir-radius-v9": {".md", ".csv", ".tex", ".json"},
    "paper/generated/ecir-review-v10": {".md", ".csv", ".tex", ".json"},
}

TEXT_SUFFIXES = {".md", ".txt", ".tex", ".csv", ".json", ".py", ".sh"}
FIXED_ZIP_TIME = (2026, 9, 21, 0, 0, 0)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def selected_files() -> list[Path]:
    files = [ROOT / relative for relative in EXACT_FILES]
    for relative, suffixes in TREE_FILES.items():
        base = ROOT / relative
        files.extend(
            path
            for path in base.rglob("*")
            if path.is_file()
            and path.suffix.lower() in suffixes
            and "__pycache__" not in path.parts
        )
    missing = [str(path.relative_to(ROOT)) for path in files if not path.is_file()]
    if missing:
        raise SystemExit("Missing allowlisted files: " + ", ".join(missing))
    return sorted(set(files), key=lambda path: path.relative_to(ROOT).as_posix())


def destination(relative: Path) -> Path:
    if relative.as_posix() == "ECIR_ARTIFACT_README.md":
        return Path("README.md")
    if relative.as_posix() == "ECIR.tex":
        return Path("paper/ECIR.tex")
    parts = relative.parts
    if parts[:4] == ("artifacts", "revision", "v1", "data"):
        return Path("provenance/data") / parts[4] / "manifest.json"
    if parts[:4] == ("artifacts", "revision", "mps-v1", "runs"):
        return Path("provenance/primary-selection") / f"{parts[4]}.json"
    if parts[:4] == ("artifacts", "ecir", "mps-v2", "baseline-runs"):
        return Path("provenance/baseline-selection") / f"{parts[4]}.json"
    if parts[:4] == ("artifacts", "ecir", "mps-v2", "stochastic-runs"):
        return Path("provenance/stochastic-selection") / f"{parts[4]}.json"
    if parts[:4] == ("artifacts", "ecir", "additional-v1", "session-knn-runs"):
        return Path("provenance/session-knn-selection") / f"{parts[4]}.json"
    if parts[:4] == ("artifacts", "ecir", "frozen-radius-v1", "runs"):
        return Path("provenance/frozen-radius-selection") / f"{parts[4]}.json"
    return relative


def copy_payload(stage: Path) -> None:
    for source in selected_files():
        relative = destination(source.relative_to(ROOT))
        target = stage / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        target.chmod(0o755 if target.suffix == ".sh" else 0o644)


def audit_patterns() -> list[tuple[str, re.Pattern[str]]]:
    github_owner = r"(?:https?://)?" + "github" + r"\.com/[^/\s]+"
    git_remote = (
        r"(?:"
        + re.escape("git" + "@")
        + "|"
        + re.escape("ssh" + "://")
        + "|"
        + re.escape("." + "git")
        + r"(?:/|$))"
    )
    patterns = [
        ("POSIX home path", re.compile(r"/(?:Users|home)/[^/\s]+", re.I)),
        ("Windows home path", re.compile(r"[A-Z]:\\Users\\[^\\\s]+", re.I)),
        ("email address", re.compile(r"[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}", re.I)),
        ("GitHub owner URL", re.compile(github_owner, re.I)),
        ("Git remote", re.compile(git_remote, re.I)),
    ]
    dynamic = {
        "local username": getpass.getuser(),
        "local hostname": platform.node(),
        "local home": str(Path.home()),
    }
    for label, value in dynamic.items():
        if value and len(value) >= 3:
            patterns.append((label, re.compile(re.escape(value), re.I)))
    return patterns


def validate_audit_patterns() -> None:
    probes = {
        "POSIX home path": "/" + "Users" + "/reviewer/project",
        "Windows home path": "C:" + "\\" + "Users" + "\\reviewer\\project",
        "email address": "person" + "@" + "example.org",
        "GitHub owner URL": "https://" + "github" + ".com/owner/repository",
        "Git remote": "git" + "@" + "example.org:owner/repository.git",
    }
    patterns = dict(audit_patterns())
    failed = [label for label, probe in probes.items() if not patterns[label].search(probe)]
    if failed:
        raise SystemExit("Broken anonymity audit patterns: " + ", ".join(failed))


def audit(stage: Path) -> list[str]:
    findings: list[str] = []
    patterns = audit_patterns()
    for path in sorted(stage.rglob("*")):
        if not path.is_file() or path.name in {"SHA256SUMS", "ANONYMITY_AUDIT.txt"}:
            continue
        relative = path.relative_to(stage).as_posix()
        for label, pattern in patterns:
            if pattern.search(relative):
                findings.append(f"{relative}: member name matches {label}")
        if path.suffix.lower() not in TEXT_SUFFIXES:
            findings.append(f"{relative}: unexpected binary file")
            continue
        text = path.read_text(encoding="utf-8")
        for line_number, line in enumerate(text.splitlines(), 1):
            for label, pattern in patterns:
                if pattern.search(line):
                    findings.append(f"{relative}:{line_number}: matches {label}")
    return findings


def write_checksums(stage: Path) -> None:
    paths = [
        path
        for path in sorted(stage.rglob("*"))
        if path.is_file() and path.name not in {"SHA256SUMS", "ANONYMITY_AUDIT.txt"}
    ]
    lines = [f"{sha256(path)}  {path.relative_to(stage).as_posix()}" for path in paths]
    (stage / "SHA256SUMS").write_text("\n".join(lines) + "\n", encoding="utf-8")


def make_zip(stage: Path, archive: Path) -> None:
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as out:
        for path in sorted(stage.rglob("*")):
            if not path.is_file():
                continue
            relative = Path(stage.name) / path.relative_to(stage)
            info = zipfile.ZipInfo(relative.as_posix(), FIXED_ZIP_TIME)
            info.create_system = 3
            mode = 0o755 if path.suffix == ".sh" else 0o644
            info.external_attr = mode << 16
            info.compress_type = zipfile.ZIP_DEFLATED
            out.writestr(info, path.read_bytes(), compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)


def verify_zip(stage: Path, archive: Path) -> None:
    expected = {
        (Path(stage.name) / path.relative_to(stage)).as_posix(): sha256(path)
        for path in stage.rglob("*")
        if path.is_file()
    }
    with zipfile.ZipFile(archive) as source:
        names = source.namelist()
        if len(names) != len(set(names)):
            raise SystemExit("Archive contains duplicate member names")
        if set(names) != set(expected):
            raise SystemExit("Archive member list differs from staged payload")
        for name, wanted in expected.items():
            actual = hashlib.sha256(source.read(name)).hexdigest()
            if actual != wanted:
                raise SystemExit(f"Archive checksum mismatch: {name}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "submission/ecir2027_anonymous_artifact",
        help="New artifact directory; an existing path is never overwritten.",
    )
    args = parser.parse_args()
    validate_audit_patterns()
    output = args.out.resolve()
    archive = output.with_suffix(".zip")
    checksum = Path(str(archive) + ".sha256")
    for path in (output, archive, checksum):
        if path.exists():
            raise SystemExit(f"Refusing to overwrite existing path: {path}")
    output.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="ecir-artifact-", dir=output.parent) as temporary:
        stage = Path(temporary) / output.name
        stage.mkdir()
        copy_payload(stage)
        findings = audit(stage)
        if findings:
            raise SystemExit("Anonymity audit failed:\n" + "\n".join(findings))
        (stage / "ANONYMITY_AUDIT.txt").write_text(
            "PASS\n"
            "Allowlisted text payload scanned for local home paths, usernames, "
            "hostnames, email addresses, Git remotes, and repository-owner URLs.\n"
            "Raw data, checkpoints, Git metadata, bytecode, PDFs, PNGs, and "
            "filesystem extended attributes are excluded.\n",
            encoding="utf-8",
        )
        write_checksums(stage)
        findings = audit(stage)
        if findings:
            raise SystemExit("Final anonymity audit failed:\n" + "\n".join(findings))
        shutil.copytree(stage, output, copy_function=shutil.copyfile)
        make_zip(stage, archive)

    verify_zip(output, archive)
    checksum.write_text(f"{sha256(archive)}  {archive.name}\n", encoding="utf-8")
    print(f"Built {output}")
    print(f"Built {archive}")
    print(f"SHA-256 {sha256(archive)}")


if __name__ == "__main__":
    main()
