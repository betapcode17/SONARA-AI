"""Validation-first, resumable retrieval experiments for the v4 checkpoints."""
import argparse
import csv
import hashlib
import json
import math
import os
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
VERSION = "window-dtw-v1"
WINDOWS = (1, 3, 5, 9)
ALPHAS = (0.5, 0.75, 1.0)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def audio_record(path):
    path = Path(path).resolve()
    stat = path.stat()
    return [str(path), stat.st_size, stat.st_mtime_ns]


def bootstrap(model_label="kaggle_v4"):
    """Reuse definitions, never execute the notebook's evaluation/training cells."""
    notebook = ROOT / "notebooks" / 'local_4gb' / 'evaluate' / "conformer_bytehum_qbh_evaluation_local_4gb.ipynb"
    document = json.loads(notebook.read_text(encoding="utf-8"))
    ns = {"__name__": "__ablation_bootstrap__"}
    for cell in document["cells"]:
        if cell["cell_type"] != "code":
            continue
        source = "".join(cell["source"])
        if source.startswith("finetuned_eval_model ="):
            break
        if "PROJECT_ROOT_OVERRIDE = None" in source:
            source = source.replace("PROJECT_ROOT_OVERRIDE = None", "PROJECT_ROOT_OVERRIDE = " + repr(str(ROOT.parent)))
            source = source.replace("COMPARE_ZERO_SHOT = True", "COMPARE_ZERO_SHOT = False")
            source = source.replace("COMPARE_KAGGLE = True", "COMPARE_KAGGLE = False")
            source = source.replace("RUN_FAISS_BENCHMARK = True", "RUN_FAISS_BENCHMARK = False")
            source = source.replace("('evaluation_' + GALLERY_MODE)", "'retrieval_ablation'")
            if model_label == "kaggle_v4":
                checkpoint = ROOT / "checkpoints" / "conformer_bytehum_full_d128_timegrid_lat_v4_best.pth"
                source = source.replace("CHECKPOINT_OVERRIDE = None", "CHECKPOINT_OVERRIDE = " + repr(str(checkpoint)))
        exec(compile(source, str(notebook), "exec"), ns)
    if "score_gallery" not in ns:
        raise RuntimeError("Evaluator bootstrap changed: score_gallery was not defined.")
    model = ns["ConformerByteHumModel"](num_classes=ns["NUM_TRAIN_SONGS"], embedding_dim=128, allow_random_init=True)
    model.load_state_dict(ns["checkpoint_metadata"]["model_state_dict"], strict=True)
    ns["model"] = model.to(ns["device"]).eval()
    ns["checkpoint_hash"] = ns["file_sha256"](ns["finetuned_ckpt_path"])
    del ns["checkpoint_metadata"]["model_state_dict"]
    # Small reductions over windows should not launch dozens of CPU worker threads.
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    return ns


def load_gallery(ns):
    signature = ns["gallery_signature"](ns["gallery_paths"], ns["checkpoint_hash"])
    path = ROOT / "cache" / "evaluation_features" / (signature + ".pt")
    if not path.is_file():
        raise FileNotFoundError("Run the full-gallery evaluator first: " + str(path))
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload["signature"] != signature or list(payload["features"]) != list(ns["gallery_paths"]):
        raise ValueError("Gallery cache identity mismatch.")
    for song, (g, l) in payload["features"].items():
        if g.ndim != 2 or l.shape != (len(g) * 20, 128) or g.shape[1] != 128:
            raise ValueError("Unexpected v4 chunk layout: " + song)
        if not torch.isfinite(g).all() or not torch.isfinite(l).all():
            raise ValueError("Non-finite gallery embeddings: " + song)
    return payload["features"], signature


def audio_duration(path, ns):
    try:
        return float(ns["sf"].info(path).duration)
    except (RuntimeError, ValueError):
        return float(ns["librosa"].get_duration(path=path))


def chunk_starts(duration, count):
    if duration <= 10:
        starts = [0.0]
    else:
        starts = [float(i * 3) for i in range(int((duration - 10) // 3) + 1)]
        if duration - (starts[-1] + 10) > 10 / 3:
            starts.append(duration - 10)
    if len(starts) != count:
        raise ValueError(f"Audio duration/cache chunk mismatch: duration={duration}, expected={len(starts)}, actual={count}")
    return starts


def timeline(local, starts, duration):
    """Average overlapping encoder frames on the nominal v4 0.48-second grid.

    Two kernel-3/stride-2 subsampling layers center frame i at CQT index 3+4i.
    CQT's 3-frame mean adds 0.04 s: centers are 0.40+0.48i. These timestamps
    describe sampling centers, not the Conformer's contextual receptive field.
    """
    chunks = local.reshape(len(starts), 20, -1).numpy()
    centers = np.asarray(starts)[:, None] + (0.40 + 0.48 * np.arange(20))[None, :]
    valid = centers < duration
    if not valid.any():
        raise ValueError("Audio is too short for one valid encoder-frame center.")
    indices = np.rint(centers[valid] / 0.48).astype(int)
    sums = np.zeros((int(indices.max()) + 1, chunks.shape[-1]), dtype=np.float32)
    counts = np.zeros(len(sums), dtype=np.float32)
    np.add.at(sums, indices, chunks[valid])
    np.add.at(counts, indices, 1)
    positions = np.flatnonzero(counts)
    values = sums[positions] / counts[positions, None]
    values /= np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1e-8)
    return values, positions.astype(np.float32) * 0.48


def query_features(ns, record):
    song, fragment, path = record
    identity = {"source": audio_record(path), "checkpoint": ns["checkpoint_hash"],
                "torch": str(torch.__version__), "amp": ns["AMP_ENABLED"], "pipeline": "v4_16k_10s_hop3_96x83"}
    cache = ROOT / "cache" / "ablation_queries" / (digest(identity) + ".pt")
    if cache.is_file():
        data = torch.load(cache, map_location="cpu", weights_only=True)
        if data["identity"] != identity:
            raise ValueError("Query cache identity mismatch.")
    else:
        cqts = ns["test_dataset"]._get_raw_cqts(path)
        g, l = ns["encode_cqts"](ns["model"], cqts)
        data = {"identity": identity, "global": g, "local": l, "duration": audio_duration(path, ns)}
        cache.parent.mkdir(parents=True, exist_ok=True)
        temporary = cache.with_suffix(".tmp")
        torch.save(data, temporary)
        os.replace(temporary, cache)
    g, l = data["global"], data["local"]
    if g.shape[1:] != (128,) or l.shape != (len(g) * 20, 128):
        raise ValueError("Invalid query cache shape: " + path)
    if not torch.isfinite(g).all() or not torch.isfinite(l).all():
        raise ValueError("Invalid query cache values: " + path)
    chunk_starts(data["duration"], len(g))
    return data


def scoring_configs():
    configs = {"baseline": {"kind": "exact", "alpha": 0.5}}
    for width in WINDOWS:
        for alpha in ALPHAS:
            configs[f"window{width}_a{alpha:g}"] = {"kind": "window", "chunks": width, "alpha": alpha}
    return configs


class ChunkScorer:
    """One GPU copy of unpadded gallery; bounded similarity allocations."""
    def __init__(self, gallery, device, chunk_block=512):
        self.songs = list(gallery)
        self.lengths = [len(gallery[s][0]) for s in self.songs]
        self.offsets = np.cumsum([0, *self.lengths])
        self.global_features = torch.cat([gallery[s][0] for s in self.songs]).to(device)
        self.local_features = torch.cat([gallery[s][1] for s in self.songs]).reshape(-1, 20, 128).to(device)
        self.device = device
        self.chunk_block = chunk_block

    @torch.inference_mode()
    def score(self, qg, ql, configs):
        qg, ql = qg.to(self.device), ql.to(self.device)
        maxima, global_scores = [], []
        for start in range(0, len(self.global_features), self.chunk_block):
            refs = self.local_features[start:start + self.chunk_block]
            sim = ql @ refs.flatten(0, 1).T
            maxima.append(sim.reshape(len(ql), len(refs), 20).amax(-1).T.cpu())
            global_scores.append((qg @ self.global_features[start:start + self.chunk_block].T).amax(0).cpu())
        maxima, global_scores = torch.cat(maxima), torch.cat(global_scores)
        results = {name: [] for name in configs}
        for lo, hi in zip(self.offsets[:-1], self.offsets[1:]):
            local, glob = maxima[lo:hi], global_scores[lo:hi]
            for name, cfg in configs.items():
                alpha = cfg["alpha"]
                if cfg["kind"] == "exact":
                    value = alpha * local.amax(0).mean() + (1 - alpha) * glob.max()
                elif cfg["kind"] == "window":
                    width = min(cfg["chunks"], len(local))
                    # The local and global scores must come from the SAME reference window.
                    ls = local.unfold(0, width, 1).amax(-1).mean(-1)
                    gs = glob.unfold(0, width, 1).amax(-1)
                    value = (alpha * ls + (1 - alpha) * gs).max()
                else:
                    raise ValueError("Unknown scoring configuration.")
                results[name].append(float(value))
        return {name: np.asarray(values, dtype=np.float64) for name, values in results.items()}


def dtw_similarity(query, reference, librosa):
    if not len(query) or not len(reference):
        raise ValueError("Empty DTW sequence.")
    cost = np.clip(1 - query @ reference.T, 0, 2).astype(np.float64)
    distance, path = librosa.sequence.dtw(C=cost, subseq=True, backtrack=True)
    if not len(path) or not np.isfinite(distance).any():
        raise ValueError("DTW found no finite alignment.")
    # Report average cosine along the actual path, not a length-biased total cost.
    return float(1 - cost[path[:, 0], path[:, 1]].mean())


def tie_stats(scores, target):
    scores = np.asarray(scores, dtype=np.float64)
    if not np.isfinite(scores).all():
        raise ValueError("Non-finite scores: no query may be silently dropped.")
    tied = np.isclose(scores, scores[target], rtol=1e-6, atol=1e-7)
    greater = int(((scores > scores[target]) & ~tied).sum())
    count = int(tied.sum())
    return {"rank": float(1 + greater + 0.5 * (count - 1)), "ties": count, "greater": greater}


def summarize(rows, config):
    stats = [r["statistics"][config] for r in rows]
    groups = defaultdict(list)
    for row, stat in zip(rows, stats):
        groups[row["song"]].append(1 / stat["rank"])
    return {"queries": len(rows), "mrr": float(np.mean([1 / s["rank"] for s in stats])),
            "macro_mrr": float(np.mean([np.mean(v) for v in groups.values()])),
            **{f"recall_at_{k}": float(np.mean([np.clip((k - s["greater"]) / s["ties"], 0, 1) for s in stats]))
               for k in (1, 5, 10, 20, 50, 100)}}


def select_validation(report):
    protocol = report["protocol"]
    if protocol["split"] != "val" or protocol["subset"] or protocol["test_songs_in_gallery"]:
        raise ValueError("Selection requires ALL validation queries and a train/val-only gallery.")
    # Prefer baseline on exact ties; deterministic secondary ordering uses configuration name.
    return min(report["metrics"], key=lambda key: (-report["metrics"][key]["mrr"], key != "baseline", key))


def records_for_split(ns, split):
    directory = ROOT / "datasets" / "dataset_split" / split
    dataset = ns["ByteHumCachedTripletDataset"](str(directory), ns["prep"], ns["cqt_ext"], is_training=False)
    return list(dataset.humming_samples)


def evaluate(ns, gallery, signature, model_label, split="val", max_queries=None, with_dtw=False,
             selected=None, diagnostic=False):
    if split not in ("val", "test"):
        raise ValueError("Only val and test experiments are supported.")
    if split == "test" and selected is None and not diagnostic:
        raise ValueError("Test evaluation requires a locked validation selection.")
    records = records_for_split(ns, split)
    total = len(records)
    if not records:
        raise ValueError("No queries available; no result can be reported.")
    if max_queries:
        # Interleave songs instead of taking only the first song for smoke checks.
        buckets = defaultdict(list)
        for record in records:
            buckets[record[0]].append(record)
        records = [group[i] for i in range(max(map(len, buckets.values())))
                   for group in buckets.values() if i < len(group)][:max_queries]
    allowed = ns["split_ids"]["train"] | ns["split_ids"]["val"] if split == "val" else set(gallery)
    gallery = {s: features for s, features in gallery.items() if s in allowed}
    if {r[0] for r in records} - gallery.keys():
        raise ValueError("Missing query ground truth.")
    if split == "val" and set(gallery) & ns["split_ids"]["test"]:
        raise ValueError("Test-song gallery leakage.")
    configs = scoring_configs() if split == "val" else {"baseline": scoring_configs()["baseline"]}
    if selected is not None:
        cfg = selected["configuration"]
        if cfg["kind"] != "dtw":
            configs["selected"] = cfg
        with_dtw = cfg["kind"] == "dtw"
    if diagnostic and split == "test":
        configs = {"baseline": scoring_configs()["baseline"]}
    scorer = ChunkScorer(gallery, ns["device"])
    songs = scorer.songs
    song_index = {s: i for i, s in enumerate(songs)}
    heldout_indices = [i for i, s in enumerate(songs) if s in ns["split_ids"][split]]
    protocol = {"split": split, "gallery_songs": len(songs), "reference_source": "full_original_recordings",
                "total_queries": total, "evaluated_queries": len(records), "subset": len(records) != total,
                "test_songs_in_gallery": bool(set(songs) & ns["split_ids"]["test"]),
                "diagnostic_only": diagnostic, "dtw_full_gallery": with_dtw,
                "dtw_timestamps": "nominal_v4_subsampling_centers_overlap_averaged" if with_dtw else None,
                "window_coverage_seconds": {str(w): 10 + 3 * (w - 1) for w in WINDOWS},
                "ties": "average rank; expected Recall@K"}
    source_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    identity = {"version": VERSION, "code": source_hash, "gallery_signature": signature,
                "checkpoint": ns["checkpoint_hash"], "songs": songs, "protocol": protocol,
                "queries": [audio_record(r[2]) for r in records], "configs": configs,
                "selected": selected, "with_dtw": with_dtw}
    run_dir = ROOT / "runs" / "retrieval_ablation" / model_label / (split + "_" + digest(identity)[:16])
    run_dir.mkdir(parents=True, exist_ok=True)
    atomic_json(run_dir / "identity.json", identity)
    reference_sequences = []
    if with_dtw:
        print("Prepare overlap-averaged reference timelines", flush=True)
        for song in songs:
            paths = ns["gallery_paths"][song]
            if len(paths) != 1:
                raise ValueError("DTW timeline expects exactly one full recording per song.")
            duration = audio_duration(paths[0], ns)
            reference_sequences.append(timeline(gallery[song][1], chunk_starts(duration, len(gallery[song][0])), duration)[0])
    rows = []
    print(f"Run {split}: {len(records)}/{total} queries, {len(songs)} songs; resume directory {run_dir}", flush=True)
    for i, record in enumerate(records):
        song, fragment, path = record
        row_path = run_dir / "queries" / (digest(audio_record(path)) + ".json")
        if row_path.is_file():
            row = json.loads(row_path.read_text(encoding="utf-8"))
            if row["query"] != path or row["song"] != song:
                raise ValueError("Resumed query identity mismatch.")
        else:
            start = time.perf_counter()
            features = query_features(ns, record)
            scores = scorer.score(features["global"], features["local"], configs)
            if with_dtw:
                query_sequence = timeline(features["local"], chunk_starts(features["duration"], len(features["global"])), features["duration"])[0]
                dtw = np.asarray([dtw_similarity(query_sequence, ref, ns["librosa"]) for ref in reference_sequences])
                if split == "val":
                    for weight in (0.25, 0.5, 1.0):
                        scores[f"dtw_w{weight:g}"] = (1 - weight) * scores["baseline"] + weight * dtw
                else:
                    weight = selected["configuration"]["weight"]
                    scores["selected"] = (1 - weight) * scores["baseline"] + weight * dtw
            statistics = {name: tie_stats(values, song_index[song]) for name, values in scores.items()}
            reduced = scores["baseline"][heldout_indices]
            statistics["same_audio_heldout_gallery"] = tie_stats(reduced, heldout_indices.index(song_index[song]))
            row = {"song": song, "fragment": fragment, "query": path, "statistics": statistics,
                   "top1": {name: songs[int(values.argmax())] for name, values in scores.items()},
                   "seconds_including_query_cache": time.perf_counter() - start}
            atomic_json(row_path, row)
        rows.append(row)
        if (i + 1) % 25 == 0 or i + 1 == len(records):
            print(f"{split}: {i + 1}/{len(records)} completed (including resumed queries)", flush=True)
    all_configs = list(rows[0]["statistics"])
    metrics = {name: summarize(rows, name) for name in all_configs}
    if with_dtw and split == "val":
        configs.update({f"dtw_w{w:g}": {"kind": "dtw", "weight": w} for w in (0.25, 0.5, 1.0)})
    # The 45-song counterpart is a protocol diagnostic, never a selectable model.
    comparable = {name: metric for name, metric in metrics.items() if name != "same_audio_heldout_gallery"}
    report = {"created_at_utc": datetime.now(timezone.utc).isoformat(), "version": VERSION,
              "checkpoint_sha256": ns["checkpoint_hash"], "gallery_signature": signature,
              "protocol": protocol, "metrics": comparable,
              "same_audio_heldout_gallery": {"gallery_songs": len(heldout_indices), **metrics["same_audio_heldout_gallery"]},
              "configurations": configs, "output_directory": str(run_dir)}
    if split == "test" and not protocol["subset"]:
        original_path = ROOT / "runs" / "local_4gb" / "evaluation_full" / "evaluation_metrics.json"
        if original_path.is_file():
            original = json.loads(original_path.read_text(encoding="utf-8"))
            previous = next((r for r in original["results"] if r["label"] == model_label
                             and r["checkpoint_sha256"] == ns["checkpoint_hash"]), None)
            if previous is not None:
                deltas = {key: metrics["baseline"][key] - previous["modes"]["fused"][key]
                          for key in ("mrr", "recall_at_1", "recall_at_5", "recall_at_10", "recall_at_20")}
                report["baseline_reproduction_delta"] = deltas
                if max(abs(v) for v in deltas.values()) > 1e-4:
                    raise ValueError("Optimized exact baseline did not reproduce the saved evaluator: " + str(deltas))
    atomic_json(run_dir / "metrics.json", report)
    with (run_dir / "summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["scoring", *next(iter(metrics.values())).keys()])
        writer.writeheader()
        for name, metric in metrics.items():
            writer.writerow({"scoring": name, **metric})
    selection = None
    if split == "val" and not protocol["subset"]:
        winner = select_validation(report)
        selection = {"version": VERSION, "code": source_hash, "model": model_label,
                     "checkpoint_sha256": ns["checkpoint_hash"], "validation_report": str(run_dir / "metrics.json"),
                     "winner": winner, "configuration": configs[winner], "validation_metrics": metrics[winner],
                     "baseline_metrics": metrics["baseline"]}
        atomic_json(run_dir.parent / "selection.json", selection)
    del scorer
    if ns["AMP_ENABLED"]:
        torch.cuda.empty_cache()
    print(json.dumps(report["metrics"], indent=2), flush=True)
    return report, selection


def run(model="kaggle_v4", split="val", max_queries=None, dtw=False, diagnostic=False):
    ns = bootstrap(model)
    gallery, signature = load_gallery(ns)
    selected = None
    if split == "test" and not diagnostic:
        selection_path = ROOT / "runs" / "retrieval_ablation" / model / "selection.json"
        selected = json.loads(selection_path.read_text(encoding="utf-8"))
        if selected["checkpoint_sha256"] != ns["checkpoint_hash"]:
            raise ValueError("Selected checkpoint does not match the current checkpoint.")
        if selected["code"] != hashlib.sha256(Path(__file__).read_bytes()).hexdigest():
            raise ValueError("Code changed after validation selection: rerun validation.")
        validation = json.loads(Path(selected["validation_report"]).read_text(encoding="utf-8"))
        if select_validation(validation) != selected["winner"]:
            raise ValueError("Invalid validation selection.")
    return evaluate(ns, gallery, signature, model, split, max_queries, dtw, selected, diagnostic)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("local_4gb", "kaggle_v4"), default="kaggle_v4")
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--max-queries", type=int)
    parser.add_argument("--dtw", action="store_true")
    parser.add_argument("--diagnostic", action="store_true")
    args = parser.parse_args()
    if args.max_queries is not None and args.max_queries < 1:
        parser.error("--max-queries must be positive")
    run(args.model, args.split, args.max_queries, args.dtw, args.diagnostic)
