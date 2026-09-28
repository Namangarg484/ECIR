"""Prepare fixed splits BEFORE embedding eligibility checks; never shift targets."""
import argparse
import csv
import gzip
import io
import json
import pickle
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from .protocol import check_splits, digest, dump, read_jsonl, stable_seed, write_jsonl


def flatten(value):
    if isinstance(value, list):
        return " ".join(map(str, value))
    return str(value or "")


def external_sequences(root, dataset, sources, split_seed):
    texts, users = {}, defaultdict(list)
    if dataset == "ml1m":
        movies = root / "data/ml-1m/ml-1m/movies.dat"
        ratings = root / "data/ml-1m/ml-1m/ratings.dat"
        sources.extend([movies, ratings])
        with movies.open(encoding="latin1") as f:
            for line in f:
                iid, title, genres = line.rstrip().split("::")
                texts[iid] = title + " " + genres.replace("|", " ")
        with ratings.open() as f:
            for line in f:
                uid, iid, rating, ts = line.strip().split("::")
                if float(rating) >= 4:
                    users[uid].append((int(ts), iid))
    elif dataset == "amazon_music":
        meta, reviews = root / "meta_Digital_Music.jsonl.gz", root / "Digital_Music.jsonl.gz"
        sources.extend([meta, reviews])
        with gzip.open(meta, "rt", encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                iid = str(r.get("parent_asin") or r.get("asin") or "")
                if iid:
                    texts[iid] = flatten(r.get("title")) + " " + flatten(r.get("store")).split("Format:")[0]
        with gzip.open(reviews, "rt", encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                if float(r.get("rating", 0)) >= 4:
                    uid, iid = r.get("user_id"), r.get("parent_asin") or r.get("asin")
                    if uid and iid:
                        if r.get("timestamp") is None:
                            raise ValueError("Amazon positive event lacks timestamp")
                        users[str(uid)].append((int(r["timestamp"]), str(iid)))
    else:
        archive = root / "hetrec2011-lastfm-2k.zip"
        sources.append(archive)
        with zipfile.ZipFile(archive) as z:
            def rows(name):
                matches = [n for n in z.namelist() if Path(n).name == name]
                if len(matches) != 1:
                    raise ValueError(f"Expected one ZIP member {name}")
                return list(csv.DictReader(io.StringIO(z.read(matches[0]).decode("utf-8-sig")), delimiter="\t"))
            texts = {r["id"]: r["name"] for r in rows("artists.dat")}
            for r in rows("user_artists.dat"):
                uid, iid = r["userID"], r["artistID"]
                # Listening counts and tag timestamps are NOT an event chronology.
                users[uid].append((stable_seed(split_seed, uid, iid), iid))
    sequences = {}
    for uid, events in sorted(users.items()):
        seen, seq = set(), []
        for _, iid in sorted(events):
            if iid not in seen:
                seq.append(iid)
                seen.add(iid)
        if len(seq) >= 5:
            sequences[uid] = seq
    # Candidate universe is all metadata items, not a target-conditioned subset.
    return texts, sequences


def talk_sequences(path):
    sessions = {}
    for r in read_jsonl(path):
        uid = str(r.get("session_id", ""))
        if not uid or uid in sessions:
            raise ValueError(f"Missing/duplicate session ID in {path}")
        seq = [str(t.get("content", "")).strip() for t in r.get("conversations", [])
               if t.get("role") == "music"]
        sessions[uid] = seq
    return sessions


def make_example(uid, seq, pos, catalog, counts):
    # Choose target in original sequence, then check coverage. Never filter then split.
    history, target = seq[:pos], seq[pos]
    if not history:
        counts["no_context"] += 1
        return None
    if target in history:
        counts["repeated_target"] += 1
        return None
    if target not in catalog:
        counts["missing_target_embedding"] += 1
        return None
    if any(i not in catalog for i in history):
        counts["missing_context_embedding"] += 1
        return None
    return {"event": json.dumps([uid, pos], separators=(",", ":")), "group": uid,
            "context": history, "target": target, "excluded": sorted(set(history))}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", choices=["talkplay", "ml1m", "lastfm", "amazon_music"], required=True)
    p.add_argument("--root", type=Path, default=Path("."))
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--encoder", type=Path, default=Path("models/minilm"))
    p.add_argument("--embedding-device", default="cpu")
    p.add_argument("--split-seed", type=int, default=2026)
    p.add_argument("--validation-fraction", type=float, default=.1)
    p.add_argument("--clap-cache", type=Path, default=Path("cache/embeddings_track_embeddings_track_id_audio-laion_clap_2295310bfe98.pkl"))
    a = p.parse_args()
    if a.out.exists():
        p.error("Output already exists; use a new directory (no overwrite)")
    if not 0 < a.validation_fraction < 1:
        p.error("Validation fraction must be in (0,1)")
    sources, assigned, notes = [], [], {}
    if a.dataset == "talkplay":
        train_path, test_path = a.root / "train_dataset.jsonl", a.root / "test_dataset.jsonl"
        meta_path = a.root / "data/TalkPlayData-Challenge-Track-Metadata/track_metadata.jsonl"
        cache_path = a.root / a.clap_cache
        sources.extend([train_path, test_path, meta_path, cache_path])
        train, test = talk_sequences(train_path), talk_sequences(test_path)
        if train.keys() & test.keys():
            raise ValueError("Official train/test session IDs overlap; resolve upstream")
        test_signatures = {tuple(s) for s in test.values()}
        unique, removed = {}, 0
        for uid, seq in sorted(train.items()):
            sig = tuple(seq)
            if sig in test_signatures or sig in unique:
                removed += 1
                continue
            unique[sig] = uid
        signatures = sorted(unique, key=lambda s: (stable_seed(a.split_seed, s), s))
        nval = max(1, round(len(signatures) * a.validation_fraction))
        for n, sig in enumerate(signatures):
            split, seq, uid = ("validation" if n < nval else "train"), list(sig), unique[sig]
            positions = range(1, len(seq)) if split == "train" else [len(seq) - 1]
            if len(seq) >= 3:
                assigned.extend((split, uid, seq, i) for i in positions)
        # Keep one test example per distinct sequence, without consulting scores.
        seen = set()
        for uid, seq in sorted(test.items()):
            if len(seq) >= 3 and tuple(seq) not in seen:
                assigned.append(("test", uid, seq, len(seq) - 1))
                seen.add(tuple(seq))
        texts = {}
        for r in read_jsonl(meta_path):
            iid = str(r.get("track_id") or r.get("id") or "")
            if iid:
                texts.setdefault(iid, flatten(r.get("track_name")) + " " + flatten(r.get("artist_name")))
        # Local trusted pickle only. Never load an untrusted downloaded pickle.
        with cache_path.open("rb") as f:
            raw = pickle.load(f)
        vectors = {str(k): np.asarray(v, dtype=np.float32) for k, v in raw.items()}
        notes = {"protocol": "official session-disjoint test; validation sessions from train",
                 "train_duplicate_sequences_removed": removed,
                 "raw_train_sessions": len(train), "raw_test_sessions": len(test),
                 "test_unique_eligible_length_sequences": len(seen),
                 "embedding": "frozen supplied CLAP; upstream pretraining provenance not certified"}
    else:
        texts, sequences = external_sequences(a.root, a.dataset, sources, a.split_seed)
        for uid, seq in sequences.items():
            assigned.extend(("train", uid, seq, i) for i in range(1, len(seq) - 2))
            assigned.append(("validation", uid, seq, len(seq) - 2))
            assigned.append(("test", uid, seq, len(seq) - 1))
        encoder_path = a.root / a.encoder
        if not encoder_path.is_dir():
            raise FileNotFoundError(f"Local MiniLM encoder missing: {encoder_path}; no automatic download")
        from sentence_transformers import SentenceTransformer
        encoder = SentenceTransformer(str(encoder_path.resolve()), device=a.embedding_device, local_files_only=True)
        ids = sorted(texts)
        matrix = encoder.encode([texts[i] for i in ids], batch_size=256,
                                show_progress_bar=True, normalize_embeddings=True)
        vectors = dict(zip(ids, matrix))
        sources.extend(sorted(f for f in encoder_path.rglob("*") if f.is_file()))
        notes = {"protocol": "deterministic unordered artist holdout" if a.dataset == "lastfm" else "per-user chronological leave-two-out, positive rating >=4, first unique item occurrence",
                 "eligible_users_before_embedding_checks": len(sequences),
                 "embedding": "local frozen MiniLM; fresh encoding of static metadata only",
                 "metadata": "artist name only; no user tags" if a.dataset == "lastfm" else "static item title/genres or store",
                 "test_context_includes_validation_event": True}
    ids = sorted(vectors)
    if not ids:
        raise ValueError("Empty catalog")
    matrix = np.stack([vectors[i] for i in ids]).astype(np.float32)
    if matrix.ndim != 2 or not np.isfinite(matrix).all() or (np.linalg.norm(matrix, axis=1) == 0).any():
        raise ValueError("Invalid/zero catalog embeddings; repair source rather than silently drop items")
    matrix /= np.linalg.norm(matrix, axis=1, keepdims=True)
    counts = {s: Counter() for s in ("train", "validation", "test")}
    splits = {s: [] for s in counts}
    catalog = set(ids)
    for split, uid, seq, pos in assigned:
        counts[split]["assigned"] += 1
        row = make_example(uid, seq, pos, catalog, counts[split])
        if row is not None:
            splits[split].append(row)
            counts[split]["retained"] += 1
    check_splits(splits)
    if any(not rows for rows in splits.values()):
        raise ValueError("At least one split is empty")
    a.out.mkdir(parents=True)
    dump(a.out / "catalog.json", {"ids": ids, "texts": [texts.get(i, "") for i in ids]})
    np.save(a.out / "embeddings.npy", matrix, allow_pickle=False)
    for split, rows in splits.items():
        write_jsonl(a.out / f"{split}.jsonl", rows)
    files = ["catalog.json", "embeddings.npy", "train.jsonl", "validation.jsonl", "test.jsonl"]
    dump(a.out / "manifest.json", {
        "schema": 1, "dataset": a.dataset, "split_seed": a.split_seed,
        "validation_fraction": a.validation_fraction, "notes": notes,
        "counts": {s: dict(v) for s, v in counts.items()},
        "catalog_size": len(ids), "embedding_dim": matrix.shape[1],
        "missing_item_texts": sum(not texts.get(i, "").strip() for i in ids),
        "sources": {str(f): digest(f) for f in sources},
        "files": {f: digest(a.out / f) for f in files},
        "prepare_source_sha256": digest(Path(__file__)),
        "protocol_source_sha256": digest(Path(__file__).with_name("protocol.py")),
    })
    print(f"Prepared {a.dataset}: {a.out}; inspect manifest.json before training")


if __name__ == "__main__":
    main()
